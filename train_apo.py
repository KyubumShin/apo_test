# train_apo.py
"""
APO(Automatic Prompt Optimization)를 사용한 분류 에이전트 학습 스크립트.

사용법:
1. 먼저 서버 실행:
   agl store --port 4747

2. 학습 실행:
   python train_apo.py
"""
import os
import random
import asyncio
import logging
from typing import Tuple, cast

from dotenv import load_dotenv
import agentlightning as agl
from openai import AsyncOpenAI

from rollout import Task, run_rollout, OpenAIClient, GeminiClient, create_llm_client, APIProvider
from dataset import Dataset, create_classification_dataset
from prompt_template import CLASSIFICATION_SYSTEM_PROMPT
import apo_ko_setup  # 한국어 POML 패치용

logging.getLogger("agentlightning").setLevel(logging.CRITICAL)
load_dotenv()

# --- 설정 ---
# API Provider 선택: "openai" 또는 "google"
API_PROVIDER: APIProvider = "openai"

# 모델 설정 (None이면 provider 기본값 사용)
# OpenAI: gpt-4.1-mini, gpt-5-mini, gpt-4o 등
# Google: gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash 등
MODEL_NAME = "gpt-5-mini"  # OpenAI 사용 시
# MODEL_NAME = "gemini-2.0-flash"  # Google 사용 시

RANDOM_SEED = 42
BEAM_ROUNDS = 1
BEAM_WIDTH = 1

# --- 데이터셋 설정 ---
# 파일에서 로드할 경우 경로 지정 (None이면 하드코딩된 데이터셋 사용)
# 지원 형식: .csv, .xlsx, .xls, .parquet
DATASET_FILE_PATH = None  # 예: "data/classification_dataset.csv"
DATASET_QUESTION_COLUMN = "question"
DATASET_LABELS_COLUMN = "expected_labels"
DATASET_TASK_ID_COLUMN = "task_id"
DATASET_LABELS_SEPARATOR = ","

# 초기 프롬프트 템플릿 (prompt_template.py에서 가져옴)
INITIAL_PROMPT = CLASSIFICATION_SYSTEM_PROMPT


def load_tasks_from_source() -> list[Task]:
    """설정에 따라 파일 또는 하드코딩 데이터셋에서 Task 로드"""
    if DATASET_FILE_PATH is not None:
        dataset = Dataset(
            file_path=DATASET_FILE_PATH,
            question_column=DATASET_QUESTION_COLUMN,
            labels_column=DATASET_LABELS_COLUMN,
            task_id_column=DATASET_TASK_ID_COLUMN,
            labels_separator=DATASET_LABELS_SEPARATOR,
        )
        return dataset.load()
    else:
        return create_classification_dataset()


def load_train_val_dataset() -> Tuple[agl.types.Dataset[dict], agl.types.Dataset[dict]]:
    """데이터셋을 학습/검증용으로 분할"""
    all_tasks = load_tasks_from_source()
    random.seed(RANDOM_SEED)
    random.shuffle(all_tasks)

    # Task -> dict 변환
    task_dicts = [
        {
            "question": t.question,
            "expected_labels": t.expected_labels,
            "task_id": t.task_id,
        }
        for t in all_tasks
    ]

    total = len(task_dicts)
    train_split = int(total * 0.6)
    val_split = int(total * 0.8)

    dataset_train = task_dicts[:train_split]
    dataset_val = task_dicts[train_split:val_split]

    return cast(agl.types.Dataset[dict], dataset_train), cast(agl.types.Dataset[dict], dataset_val)


def get_test_dataset() -> list[dict]:
    """테스트 데이터셋 반환"""
    all_tasks = load_tasks_from_source()
    random.seed(RANDOM_SEED)
    random.shuffle(all_tasks)

    task_dicts = [
        {
            "question": t.question,
            "expected_labels": t.expected_labels,
            "task_id": t.task_id,
        }
        for t in all_tasks
    ]

    total = len(task_dicts)
    val_split = int(total * 0.8)
    return task_dicts[val_split:]


@agl.rollout
async def classification_agent(task: dict, prompt_template: agl.PromptTemplate) -> float:
    """
    APO에서 호출되는 분류 에이전트.
    room_selector 패턴을 따릅니다.
    """
    task_obj = Task(
        question=task["question"],
        expected_labels=task["expected_labels"],
        task_id=task.get("task_id"),
    )

    # APO가 최적화한 프롬프트를 system_prompt로 주입
    if hasattr(prompt_template, "template"):
        task_obj.system_prompt = prompt_template.template
    else:
        task_obj.system_prompt = str(prompt_template)

    async with create_llm_client(provider=API_PROVIDER, model=MODEL_NAME) as client:
        _, reward = await run_rollout(task_obj, client)

    return reward


async def evaluate_prompt_on_dataset(prompt_template, dataset_tasks: list[dict]) -> float:
    """주어진 프롬프트 템플릿으로 데이터셋 평가"""
    if hasattr(prompt_template, "template"):
        prompt_str = prompt_template.template
    else:
        prompt_str = str(prompt_template)

    rewards = []
    async with create_llm_client(provider=API_PROVIDER, model=MODEL_NAME) as client:
        for task_item in dataset_tasks:
            try:
                task_obj = Task(
                    question=task_item["question"],
                    expected_labels=task_item["expected_labels"],
                    task_id=task_item.get("task_id"),
                    system_prompt=prompt_str,
                )
                _, reward = await run_rollout(task_obj, client)
                rewards.append(reward)
            except Exception as e:
                print(f"평가 오류: {e}")
                rewards.append(0.0)

    return sum(rewards) / len(rewards) if rewards else 0.0


async def extract_version_info_from_client(store_client: agl.LightningStoreClient) -> dict:
    """LightningStoreClient에서 버전별 프롬프트를 추출"""
    initial_prompt = None
    resources_prompts = {}

    try:
        result = await store_client.query_resources(sort_by="resources_id", sort_order="asc")

        for resources_update in result.items:
            version = resources_update.resources_id
            if "prompt_template" not in resources_update.resources:
                continue

            prompt = resources_update.resources["prompt_template"]
            resources_prompts[version] = prompt

            if version == "v0":
                initial_prompt = prompt
    except Exception as e:
        print(f"리소스 조회 오류: {e}")
        return {"resources_prompts": {}, "initial_prompt": None}

    return {
        "resources_prompts": resources_prompts,
        "initial_prompt": initial_prompt,
    }


def setup_apo_logger(file_path: str = "apo.log") -> None:
    """APO 알고리즘 로그를 파일로 저장"""
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] (%(name)s) %(message)s")
    file_handler.setFormatter(formatter)
    logging.getLogger("agentlightning.algorithm.apo").addHandler(file_handler)


def main():
    agl.setup_logging()
    setup_apo_logger()

    # --- APO 알고리즘 설정 ---
    openai_client = AsyncOpenAI()

    algorithm = agl.APO(
        openai_client,
        gradient_model=MODEL_NAME,
        apply_edit_model=MODEL_NAME,
        beam_rounds=BEAM_ROUNDS,
        beam_width=BEAM_WIDTH,
        val_batch_size=10,
        gradient_batch_size=4,
    )

    # --- Store 설정 (서버 사용) ---
    # 먼저 `agl store --port 4747` 실행 필요
    store_client = agl.LightningStoreClient("http://localhost:4747")

    # --- Trainer 설정 ---
    trainer = agl.Trainer(
        algorithm=algorithm,
        strategy=agl.SharedMemoryExecutionStrategy(main_thread="algorithm"),
        tracer=agl.DummyTracer(),
        initial_resources={
            "prompt_template": agl.PromptTemplate(
                template=INITIAL_PROMPT,
                engine="f-string",
            )
        },
        adapter=agl.TraceToMessages(),
        store=store_client,
    )

    # --- 데이터셋 로드 ---
    dataset_train, dataset_val = load_train_val_dataset()
    test_tasks = get_test_dataset()

    print(f"API Provider: {API_PROVIDER}")
    print(f"모델: {MODEL_NAME}")
    print(f"학습 데이터: {len(dataset_train)}개")
    print(f"검증 데이터: {len(dataset_val)}개")
    print(f"테스트 데이터: {len(test_tasks)}개")
    print("=" * 60)

    # --- 학습 실행 ---
    try:
        trainer.fit(
            agent=classification_agent,
            train_dataset=dataset_train,
            val_dataset=dataset_val,
        )
    except Exception as e:
        print(f"\n학습 중 오류: {e}")
        import traceback
        traceback.print_exc()

    # --- 학습 후 평가 ---
    async def run_post_training():
        info = await extract_version_info_from_client(store_client)

        if not info["resources_prompts"]:
            print("프롬프트 추출 실패 - 리소스가 없습니다.")
            print("(서버 대시보드 http://localhost:4747/resources 에서 확인해보세요)")
            return

        print(f"\n발견된 프롬프트 버전: {list(info['resources_prompts'].keys())}")

        # --- 테스트셋 평가 ---
        version_test_results = {}
        for version in sorted(
            info["resources_prompts"].keys(),
            key=lambda v: int(v[1:]) if v[1:].isdigit() else 0
        ):
            prompt = info["resources_prompts"][version]
            print(f"\n평가 중: {version}")
            score = await evaluate_prompt_on_dataset(prompt, test_tasks)
            version_test_results[version] = score
            print(f"  {version} 점수: {score:.3f}")

        if not version_test_results:
            print("평가할 프롬프트가 없습니다.")
            return

        # 최적 버전 선택
        best_version = max(version_test_results.keys(), key=lambda v: version_test_results[v])
        best_score = version_test_results[best_version]
        initial_test_score = version_test_results.get("v0", 0.0)

        print("\n" + "=" * 60)
        print("최종 평가 결과")
        print("=" * 60)
        print(f"  초기 프롬프트(v0): {initial_test_score:.3f}")
        print(f"  최적 프롬프트({best_version}): {best_score:.3f}")

        # --- 프롬프트 히스토리 저장 ---
        with open("prompt_history.txt", "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n프롬프트 최적화 이력\n" + "=" * 80 + "\n\n")
            for version in sorted(
                info["resources_prompts"].keys(),
                key=lambda v: int(v[1:]) if v[1:].isdigit() else 0
            ):
                prompt = info["resources_prompts"][version]
                prompt_str = prompt.template if hasattr(prompt, "template") else str(prompt)
                score = version_test_results[version]
                f.write(f"[{version}] 테스트셋 점수: {score:.3f}\n")
                f.write("-" * 80 + "\n")
                f.write(f"{prompt_str}\n")
                f.write("=" * 80 + "\n\n")

        print("\n✓ prompt_history.txt 저장 완료")

    try:
        asyncio.run(run_post_training())
    except Exception as e:
        print(f"평가 중 오류: {e}")
        import traceback
        traceback.print_exc()
    finally:
        async def cleanup():
            await store_client.close()
        try:
            asyncio.run(cleanup())
        except Exception:
            pass


if __name__ == "__main__":
    main()
