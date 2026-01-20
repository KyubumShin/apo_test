# rollout.py
import os
import json
from dataclasses import dataclass
from typing import Optional, List, Sequence

import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from agentlightning import emit_reward

from prompt_template import CLASSIFICATION_SYSTEM_PROMPT, format_judge_prompt

load_dotenv()

# --- OpenAI 설정 ---
API_KEY = os.getenv("OPENAI_API_KEY")


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


# --- LLM 클라이언트 ---
class OpenAIClient:
    def __init__(self, model: str = "gpt-4.1-mini"):
        self.client = AsyncOpenAI(api_key=API_KEY)
        self.model = model

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            await self.client.close()
        except Exception:
            pass
        return False

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

    async def classify(self, task: Task) -> str:
        system_prompt = task.system_prompt or CLASSIFICATION_SYSTEM_PROMPT

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.question},
        ]

        return await self._call_with_retry(messages)


# --- LLM Judge ---
async def llm_judge(client: OpenAIClient, task: Task, predicted: str) -> float:
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
async def run_rollout(task: Task, client: OpenAIClient) -> tuple[str, float]:
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
