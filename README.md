# APO Test - 자동 프롬프트 최적화

AgentLightning 라이브러리를 활용한 한국어 분류 프롬프트 자동 최적화 프로젝트입니다.

## 개요

차량 관련 사용자 발화를 5가지 카테고리로 분류하는 프롬프트를 자동으로 최적화합니다.

| 카테고리 | 설명 | 예시 |
|---------|------|------|
| 주행 | 내비게이션, 경로 안내 | "집까지 가장 빠른 길 안내해줘" |
| 차량 상태 | 점검, 진단 정보 | "타이어 압력 괜찮아?" |
| 차량 제어 | 창문, 에어컨 등 조작 | "에어컨 22도로 맞춰줘" |
| 미디어 | 음악, 라디오 제어 | "아이유 노래 틀어줘" |
| 개인 비서 | 전화, 문자, 일정 | "엄마한테 전화 걸어줘" |

## 설치

### 1. 의존성 설치

```bash
# 기본 의존성
pip install agentlightning openai python-dotenv pydantic pandas

# Google Gemini API 사용 시 추가 설치
pip install google-generativeai
```

### 2. 환경 변수 설정

```bash
# .env 파일 생성
# OpenAI 사용 시
echo "OPENAI_API_KEY=sk-..." > .env

# Google Gemini 사용 시
echo "GOOGLE_API_KEY=AIza..." >> .env
```

## 사용법

### 학습 실행

```bash
# 1단계: AgentLightning 스토어 서버 시작 (별도 터미널)
agl store --port 4747

# 2단계: 학습 실행
python train_apo.py
```

### 단일 테스트 실행

```bash
# 전체 학습 없이 빠른 롤아웃 테스트
python rollout_example.py
```

## 프로젝트 구조

```
APO_Test/
├── train_apo.py          # 메인 학습 스크립트 (APO 알고리즘)
├── rollout.py            # 태스크 실행 및 LLM Judge 로직
├── dataset.py            # 분류 데이터셋 (100개 태스크)
├── apo_ko_setup.py       # AgentLightning 한국어 프롬프트 패치
├── prompt_template.py    # 프롬프트 템플릿 정의
├── rollout_example.py    # 빠른 테스트용 스크립트
├── prompts/
│   ├── apply_edit_ko.poml    # 한국어 프롬프트 편집 템플릿
│   └── text_gradient_ko.poml # 한국어 그래디언트 분석 템플릿
├── .env                  # API 키 설정 (git 제외)
└── AGENTS.md             # 상세 개발 가이드
```

## 설정 옵션

`train_apo.py`에서 주요 설정을 변경할 수 있습니다:

```python
# API Provider 선택: "openai" 또는 "google"
API_PROVIDER = "openai"

# 모델 설정
# OpenAI: gpt-4.1-mini, gpt-5-mini, gpt-4o 등
# Google: gemini-2.0-flash, gemini-1.5-pro, gemini-1.5-flash 등
MODEL_NAME = "gpt-5-mini"      # OpenAI 사용 시
# MODEL_NAME = "gemini-2.0-flash"  # Google 사용 시

# APO 알고리즘 파라미터
BEAM_ROUNDS = 1    # 빔 서치 라운드 수
BEAM_WIDTH = 1     # 빔 너비

# 데이터셋 분할
# - 학습: 60%
# - 검증: 20%
# - 테스트: 20%
```

### 지원 API Provider

| Provider | 환경 변수 | 기본 모델 |
|----------|----------|----------|
| OpenAI | `OPENAI_API_KEY` | gpt-4.1-mini |
| Google | `GOOGLE_API_KEY` | gemini-2.0-flash |

### 외부 데이터셋 사용

CSV, Excel, Parquet 파일에서 데이터를 로드할 수 있습니다:

```python
# train_apo.py에서 설정
DATASET_FILE_PATH = "data/my_dataset.csv"
DATASET_QUESTION_COLUMN = "question"
DATASET_LABELS_COLUMN = "expected_labels"
DATASET_LABELS_SEPARATOR = ","
```

## 출력 파일

| 파일 | 설명 |
|-----|------|
| `apo.log` | 상세 학습 로그 |
| `prompt_history.txt` | 최적화된 프롬프트 버전별 이력 및 점수 |

## 평가 방식

- **LLM Judge**: GPT 모델이 분류 결과를 평가
- **점수 범위**: 0.0 ~ 1.0
  - 1.0: 완벽히 일치
  - 0.0: 완전 불일치
  - 중간값: 부분 일치

## 문제 해결

### 스토어 서버 미실행

```
Error: Connection refused
```
→ `agl store --port 4747` 먼저 실행

### Rate Limit 에러

→ 내장된 지수 백오프 재시도 로직이 자동 처리

### 한국어 인코딩 문제

→ 파일 저장 시 UTF-8 인코딩 확인

## 대시보드

학습 중 리소스 상태를 확인하려면:

```
http://localhost:4747/resources
```

## 라이선스

MIT License
