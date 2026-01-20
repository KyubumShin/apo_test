# run_example.py
import asyncio
from rollout import Task, OpenAIClient, run_rollout


async def run_tests():
    # 샘플 태스크 정의
    tasks = [
        Task(
            question="회사까지 가장 빠른 길 안내 시작해줘",
            expected_labels=["주행"],
            task_id="task_01",
        ),
        Task(
            question="타이어 공기압 체크",
            expected_labels=["차량 상태"],
            task_id="task_02",
        ),
        Task(
            question="온열 시트 켜고 출근길에 듣기 좋은 노래 틀어줘",
            expected_labels=["차량 제어", "미디어"],
            task_id="task_03",
        ),
        Task(
            question="엄마한테 전화 좀 걸어줘",
            expected_labels=["개인 비서"],
            task_id="task_04",
        ),
        Task(
            question="1+1은?",
            expected_labels=[],
            task_id="task_05",
        ),
    ]

    client = OpenAIClient()

    for i, task in enumerate(tasks, 1):
        print(f"[Task {i}/{len(tasks)}] {task.task_id}")
        print(f"질의: {task.question}")

        predicted, reward = await run_rollout(task, client)

        print(f"모델 응답: {predicted}")
        print(f"실제 정답: {task.expected_labels}")
        print(f"Reward: {reward:.2f}\n")


if __name__ == "__main__":
    asyncio.run(run_tests())