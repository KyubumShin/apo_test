# rollout.py
import os
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Sequence, Literal

import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from agentlightning import emit_reward

from prompt_template import CLASSIFICATION_SYSTEM_PROMPT, format_judge_prompt

load_dotenv()

# --- API 설정 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# API Provider 타입
APIProvider = Literal["openai", "google"]


# --- 태스크 정의 ---
@dataclass
class Task:
    """
    분류 태스크를 표현하는 데이터 구조입니다.

    - question: 분류 대상 문장
    - expected_labels: 정답으로 기대하는 레이블 리스트(예: ["주행", "미디어"])
    - task_id: (선택) 태스크 식별자
    - system_prompt: (선택) 기본 시스템 프롬프트를 덮어쓰고 싶을 때 사용
    """

    question: str
    expected_labels: List[str]
    task_id: Optional[str] = None
    system_prompt: Optional[str] = None


# --- LLM Judge 응답 형식 ---
class JudgeResponse(BaseModel):
    reason: str = Field(description="점수를 부여한 이유. 100자 이내.")
    score: float = Field(description="0-1 스케일의 점수. 엄격하게 평가하세요.")


# --- LLM 클라이언트 베이스 ---
class BaseLLMClient(ABC):
    """LLM 클라이언트 추상 베이스 클래스"""

    def __init__(self, model: str):
        self.model = model

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

    @abstractmethod
    async def close(self) -> None:
        """클라이언트 리소스 정리"""
        pass

    @abstractmethod
    async def _call_with_retry(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> str:
        """Rate limit을 고려한 LLM 호출"""
        pass

    async def classify(self, task: Task) -> str:
        system_prompt = task.system_prompt or CLASSIFICATION_SYSTEM_PROMPT

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.question},
        ]

        return await self._call_with_retry(messages)


# --- OpenAI 클라이언트 ---
class OpenAIClient(BaseLLMClient):
    """OpenAI API 클라이언트"""

    def __init__(self, model: str = "gpt-4.1-mini"):
        super().__init__(model)
        self.client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def close(self) -> None:
        try:
            await self.client.close()
        except Exception:
            pass

    async def _call_with_retry(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> str:
        """Rate limit을 고려한 LLM 호출"""
        max_retries = 5
        wait_time = 2

        for attempt in range(max_retries):
            try:
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                content = resp.choices[0].message.content
                return content.strip() if content else ""
            except RateLimitError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                    wait_time *= 2
                else:
                    raise
            except Exception:
                if attempt < max_retries - 1:
                    await asyncio.sleep(wait_time)
                    wait_time *= 2
                else:
                    raise

        return ""


# --- Google Gemini 클라이언트 ---
class GeminiClient(BaseLLMClient):
    """Google Gemini API 클라이언트"""

    def __init__(self, model: str = "gemini-2.0-flash"):
        super().__init__(model)
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            self.genai = genai
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError(
                "google-generativeai 패키지가 필요합니다. "
                "pip install google-generativeai 로 설치하세요."
            )

    async def close(self) -> None:
        # Gemini 클라이언트는 별도 정리 불필요
        pass

    async def _call_with_retry(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> str:
        """Rate limit을 고려한 LLM 호출"""
        max_retries = 5
        wait_time = 2

        # OpenAI 형식 메시지를 Gemini 형식으로 변환
        gemini_contents = self._convert_messages(messages)

        for attempt in range(max_retries):
            try:
                # Gemini는 동기 API이므로 asyncio.to_thread 사용
                response = await asyncio.to_thread(
                    self.client.generate_content,
                    gemini_contents
                )
                return response.text.strip() if response.text else ""
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "quota" in error_str or "429" in error_str:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait_time)
                        wait_time *= 2
                    else:
                        raise
                else:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(wait_time)
                        wait_time *= 2
                    else:
                        raise

        return ""

    def _convert_messages(
        self, messages: Sequence[ChatCompletionMessageParam]
    ) -> List[dict]:
        """OpenAI 형식 메시지를 Gemini 형식으로 변환"""
        gemini_contents = []
        system_instruction = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                # 시스템 메시지는 첫 번째 사용자 메시지에 prepend
                system_instruction = content
            elif role == "user":
                if system_instruction:
                    content = f"{system_instruction}\n\n---\n\n{content}"
                    system_instruction = ""
                gemini_contents.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_contents.append({"role": "model", "parts": [content]})

        return gemini_contents


# --- 클라이언트 팩토리 함수 ---
def create_llm_client(
    provider: APIProvider = "openai",
    model: Optional[str] = None
) -> BaseLLMClient:
    """
    API provider에 따라 적절한 LLM 클라이언트를 생성합니다.

    Args:
        provider: "openai" 또는 "google"
        model: 모델명 (None이면 기본값 사용)

    Returns:
        BaseLLMClient 인스턴스
    """
    if provider == "openai":
        return OpenAIClient(model=model or "gpt-4.1-mini")
    elif provider == "google":
        return GeminiClient(model=model or "gemini-2.0-flash")
    else:
        raise ValueError(f"지원하지 않는 provider: {provider}. 'openai' 또는 'google'을 사용하세요.")


# --- LLM Judge ---
async def llm_judge(client: BaseLLMClient, task: Task, predicted: str) -> float:
    """
    LLM을 사용하여 분류 결과를 평가합니다.
    apo_custom_algorithm.py의 llm_judge 패턴을 따릅니다.
    """
    judge_prompt = format_judge_prompt(
        question=task.question,
        expected_labels=task.expected_labels,
        predicted=predicted
    )

    messages: List[ChatCompletionMessageParam] = [
        {"role": "user", "content": judge_prompt},
    ]

    result = ""
    try:
        result = await client._call_with_retry(messages)

        # JSON 파싱 시도
        # 마크다운 코드 블록 제거
        result_cleaned = result.strip()
        if result_cleaned.startswith("```"):
            lines = result_cleaned.split("\n")
            result_cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        parsed = json.loads(result_cleaned)
        score = float(parsed.get("score", 0.0))
        reason = parsed.get("reason", "")

        # 점수 범위 제한
        score = max(0.0, min(1.0, score))

        print(f"  [Judge] {reason[:50]}... -> {score:.2f}")
        return score

    except (json.JSONDecodeError, ValueError, KeyError) as e:
        # JSON 파싱 실패 시 숫자만 추출 시도
        try:
            import re

            numbers = re.findall(r"(\d+\.?\d*)", result)
            if numbers:
                score = float(numbers[-1])
                score = max(0.0, min(1.0, score))
                print(f"  [Judge] 파싱 실패, 숫자 추출: {score:.2f}")
                return score
        except Exception:
            pass

        print(f"  [Judge] 평가 실패: {e}")
        return 0.0


# --- 롤아웃 함수 ---
async def run_rollout(task: Task, client: BaseLLMClient) -> tuple[str, float]:
    """
    단일 롤아웃 실행:
    1) 분류 실행
    2) LLM Judge로 보상 계산
    3) 보상 emit
    """
    # LLM 호출로 분류 수행
    predicted = await client.classify(task)

    # LLM Judge로 보상 계산
    reward = await llm_judge(client, task, predicted)

    # Agent Lightning에 보상 emit
    try:
        emit_reward(reward)
    except RuntimeError:
        pass

    return predicted, reward
