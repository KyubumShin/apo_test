# AGENTS.md - APO (Automatic Prompt Optimization) Project

## Project Overview

Korean-language classification prompt optimization using the AgentLightning library.
The system trains prompts to classify user utterances into 5 vehicle-related categories:
주행 (Driving), 차량 상태 (Vehicle Status), 차량 제어 (Vehicle Control), 미디어 (Media), 개인 비서 (Personal Assistant).

## Quick Reference

| Command | Purpose |
|---------|---------|
| `agl store --port 4747` | Start AgentLightning store server (required first) |
| `python train_apo.py` | Run full APO training pipeline |
| `python rollout_example.py` | Test rollout with sample tasks |

## Build / Run Commands

### Prerequisites
```bash
# Environment setup
pip install agentlightning openai python-dotenv pydantic

# Required: OPENAI_API_KEY in .env file
echo "OPENAI_API_KEY=sk-..." > .env
```

### Running Training
```bash
# Step 1: Start the store server (separate terminal)
agl store --port 4747

# Step 2: Run training
python train_apo.py
```

### Running Single Test
```bash
# Quick rollout test without full training
python rollout_example.py
```

## Project Structure

```
APO_Test/
├── train_apo.py          # Main training script with APO algorithm
├── rollout.py            # Task execution and LLM judge logic
├── dataset.py            # Classification dataset (100 tasks)
├── apo_ko_setup.py       # Korean prompt patching for AgentLightning
├── rollout_example.py    # Quick test script
├── prompts/
│   ├── apply_edit_ko.poml    # Korean prompt editor template
│   └── text_gradient_ko.poml # Korean gradient analysis template
├── apo.log               # Training logs
└── prompt_history.txt    # Saved prompt versions and scores
```

## Code Style Guidelines

### Language
- **Comments and docstrings**: Korean (한국어)
- **Variable/function names**: English with snake_case
- **User-facing messages**: Korean

### Imports
```python
# Standard library first
import os
import json
import asyncio
from typing import Optional, List, Tuple
from dataclasses import dataclass

# Third-party
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
import agentlightning as agl

# Local modules
from rollout import Task, run_rollout, OpenAIClient
from dataset import create_classification_dataset
```

### Type Annotations
- **Required** for function signatures
- Use `typing` module: `List`, `Optional`, `Tuple`, `cast`
- Use `dataclass` for data structures
- Use `pydantic.BaseModel` for API response schemas

```python
# Good
def load_train_val_dataset() -> Tuple[agl.types.Dataset[dict], agl.types.Dataset[dict]]:
    ...

@dataclass
class Task:
    question: str
    expected_labels: List[str]
    task_id: Optional[str] = None
```

### Async Patterns
- Use `async/await` for all LLM calls
- Use context managers for client lifecycle
- Implement retry logic with exponential backoff for rate limits

```python
class OpenAIClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.close()
        return False

    async def _call_with_retry(self, messages: List[dict]) -> str:
        max_retries = 5
        wait_time = 2
        for attempt in range(max_retries):
            try:
                # ... API call
            except RateLimitError:
                await asyncio.sleep(wait_time)
                wait_time *= 2
```

### Error Handling
- Catch specific exceptions, not bare `except:`
- Log errors with context
- Return safe defaults (0.0 for scores) on failure
- Use `try/except` around external API calls

```python
try:
    parsed = json.loads(result_cleaned)
    score = float(parsed.get("score", 0.0))
except (json.JSONDecodeError, ValueError, KeyError) as e:
    print(f"평가 오류: {e}")
    return 0.0
```

### Naming Conventions
| Type | Convention | Example |
|------|------------|---------|
| Functions | snake_case | `run_rollout`, `load_train_val_dataset` |
| Classes | PascalCase | `Task`, `OpenAIClient`, `JudgeResponse` |
| Constants | UPPER_SNAKE | `MODEL_NAME`, `RANDOM_SEED`, `BEAM_WIDTH` |
| Variables | snake_case | `task_obj`, `prompt_str`, `rewards` |

### Configuration
- Define constants at module top
- Use environment variables for secrets via `dotenv`
- Never hardcode API keys

```python
MODEL_NAME = "gpt-5-mini"
RANDOM_SEED = 42
BEAM_ROUNDS = 1
BEAM_WIDTH = 1

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
```

### AgentLightning Patterns
- Use `@agl.rollout` decorator for training agents
- Use `emit_reward(float)` to report scores
- Use `agl.PromptTemplate` for optimizable prompts

```python
@agl.rollout
async def classification_agent(task: dict, prompt_template: agl.PromptTemplate) -> float:
    # ... classification logic
    emit_reward(reward)
    return reward
```

### POML Templates
- Located in `prompts/` directory
- Use `{{ variable }}` for template substitution
- Structure: `<poml>`, `<cp caption="">`, `<list>`, `<item>`

## Key Patterns

### Task Definition
```python
@dataclass
class Task:
    question: str           # Input utterance
    expected_labels: List[str]  # Ground truth categories
    task_id: Optional[str] = None
    system_prompt: Optional[str] = None
```

### Reward Calculation
- LLM Judge evaluates classification results
- Score range: 0.0 - 1.0
- 1.0 = exact match, 0.0 = complete mismatch
- Partial credit for partial matches

### Data Split
- Training: 60%
- Validation: 20%
- Test: 20%

## Output Files
- `apo.log`: Detailed training logs
- `prompt_history.txt`: Version history of optimized prompts with scores

## Common Issues

1. **Store server not running**: Start `agl store --port 4747` first
2. **Rate limits**: Built-in exponential backoff handles this
3. **Korean encoding**: Ensure UTF-8 encoding for file operations
