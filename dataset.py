# dataset.py
import os
from typing import List, Optional
from dataclasses import dataclass

import pandas as pd

from rollout import Task


@dataclass
class DatasetConfig:
    """Dataset 로드 설정"""
    file_path: str
    question_column: str = "question"
    labels_column: str = "expected_labels"
    task_id_column: Optional[str] = "task_id"
    labels_separator: str = ","  # 레이블이 문자열로 저장된 경우 구분자


class Dataset:
    """
    파일에서 데이터를 로드하여 list[Task]로 변환하는 클래스
    
    지원 형식: CSV, Excel (.xlsx, .xls), Parquet
    
    사용 예시:
        # 기본 사용
        dataset = Dataset("data.csv")
        tasks = dataset.load()
        
        # 컬럼명 지정
        dataset = Dataset(
            "data.parquet",
            question_column="text",
            labels_column="categories",
            task_id_column="id"
        )
        tasks = dataset.load()
    """
    
    SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}
    
    def __init__(
        self,
        file_path: str,
        question_column: str = "question",
        labels_column: str = "expected_labels",
        task_id_column: Optional[str] = "task_id",
        labels_separator: str = ",",
    ):
        """
        Args:
            file_path: 데이터 파일 경로 (.csv, .xlsx, .xls, .parquet)
            question_column: 질문이 담긴 컬럼명
            labels_column: 레이블이 담긴 컬럼명
            task_id_column: task_id가 담긴 컬럼명 (None이면 자동 생성)
            labels_separator: 레이블이 문자열로 저장된 경우 구분자
        """
        self.file_path = file_path
        self.question_column = question_column
        self.labels_column = labels_column
        self.task_id_column = task_id_column
        self.labels_separator = labels_separator
        
        self._validate_file_path()
    
    def _validate_file_path(self) -> None:
        """파일 경로 및 확장자 유효성 검사"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {self.file_path}")
        
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"지원하지 않는 파일 형식입니다: {ext}. "
                f"지원 형식: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
    
    def _load_dataframe(self) -> pd.DataFrame:
        """파일 확장자에 따라 적절한 pandas 로더 사용"""
        ext = os.path.splitext(self.file_path)[1].lower()
        
        if ext == ".csv":
            return pd.read_csv(self.file_path)
        elif ext in {".xlsx", ".xls"}:
            return pd.read_excel(self.file_path)
        elif ext == ".parquet":
            return pd.read_parquet(self.file_path)
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {ext}")
    
    def _parse_labels(self, labels_value) -> List[str]:
        """레이블 값을 List[str]로 변환"""
        # None 또는 NaN 처리
        if pd.isna(labels_value):
            return []
        
        # 이미 리스트인 경우
        if isinstance(labels_value, list):
            return [str(label).strip() for label in labels_value if label]
        
        # 문자열인 경우
        if isinstance(labels_value, str):
            # 빈 문자열 처리
            labels_value = labels_value.strip()
            if not labels_value or labels_value == "[]":
                return []
            
            # JSON 리스트 형식 파싱 시도 (예: '["주행", "미디어"]')
            if labels_value.startswith("[") and labels_value.endswith("]"):
                import json
                try:
                    parsed = json.loads(labels_value)
                    return [str(label).strip() for label in parsed if label]
                except json.JSONDecodeError:
                    pass
            
            # 구분자로 분리
            return [label.strip() for label in labels_value.split(self.labels_separator) if label.strip()]
        
        # 기타 타입은 문자열 변환 후 처리
        return [str(labels_value).strip()]
    
    def load(self) -> List[Task]:
        """
        파일을 로드하여 list[Task]로 반환
        
        Returns:
            List[Task]: Task 객체 리스트
        """
        df = self._load_dataframe()
        
        # 필수 컬럼 확인
        if self.question_column not in df.columns:
            raise ValueError(f"질문 컬럼 '{self.question_column}'이 데이터에 없습니다. "
                           f"사용 가능한 컬럼: {list(df.columns)}")
        
        if self.labels_column not in df.columns:
            raise ValueError(f"레이블 컬럼 '{self.labels_column}'이 데이터에 없습니다. "
                           f"사용 가능한 컬럼: {list(df.columns)}")
        
        tasks: List[Task] = []
        
        for idx, row in df.iterrows():
            question = str(row[self.question_column])
            expected_labels = self._parse_labels(row[self.labels_column])
            
            # task_id 처리
            task_id = None
            if self.task_id_column and self.task_id_column in df.columns:
                task_id_value = row[self.task_id_column]
                if pd.notna(task_id_value):
                    task_id = str(task_id_value)
            
            # task_id가 없으면 자동 생성
            if task_id is None:
                task_id = f"task_{idx:04d}"
            
            task = Task(
                task_id=task_id,
                question=question,
                expected_labels=expected_labels,
            )
            tasks.append(task)
        
        return tasks
    
    def __len__(self) -> int:
        """데이터셋 크기 반환 (파일을 읽어야 함)"""
        df = self._load_dataframe()
        return len(df)
    
    def __repr__(self) -> str:
        return f"Dataset(file_path='{self.file_path}')"


def load_tasks_from_file(
    file_path: str,
    question_column: str = "question",
    labels_column: str = "expected_labels",
    task_id_column: Optional[str] = "task_id",
    labels_separator: str = ",",
) -> List[Task]:
    """
    파일에서 Task 리스트를 로드하는 편의 함수
    
    Args:
        file_path: 데이터 파일 경로 (.csv, .xlsx, .xls, .parquet)
        question_column: 질문이 담긴 컬럼명
        labels_column: 레이블이 담긴 컬럼명
        task_id_column: task_id가 담긴 컬럼명 (None이면 자동 생성)
        labels_separator: 레이블이 문자열로 저장된 경우 구분자
    
    Returns:
        List[Task]: Task 객체 리스트
    """
    dataset = Dataset(
        file_path=file_path,
        question_column=question_column,
        labels_column=labels_column,
        task_id_column=task_id_column,
        labels_separator=labels_separator,
    )
    return dataset.load()


def create_classification_dataset() -> List[Task]:
    return [
        # 주행(단일 카테고리) - 15개
        Task(task_id='task_01', question='집까지 가장 빠른 길 안내해줘', expected_labels=['주행']),
        Task(task_id='task_02', question='근처 주유소 가는 경로 안내 시작', expected_labels=['주행']),
        Task(task_id='task_03', question='고속도로 교통상황 알려줘', expected_labels=['주행']),
        Task(task_id='task_04', question='서울역으로 내비게이션 설정해줘', expected_labels=['주행']),
        Task(task_id='task_05', question='회사까지 최단거리로 안내', expected_labels=['주행']),
        Task(task_id='task_06', question='강남역 가는 길 알려줘', expected_labels=['주행']),
        Task(task_id='task_07', question='톨게이트 요금 얼마나 나올까?', expected_labels=['주행']),
        Task(task_id='task_08', question='우회도로 경로 찾아줘', expected_labels=['주행']),
        Task(task_id='task_09', question='목적지까지 남은 시간 알려줘', expected_labels=['주행']),
        Task(task_id='task_10', question='근처 휴게소 어디 있어?', expected_labels=['주행']),
        Task(task_id='task_11', question='인천공항까지 가는 최적의 경로 찾아줘', expected_labels=['주행']),
        Task(task_id='task_12', question='트래픽 없는 우회 경로 안내해줘', expected_labels=['주행']),
        Task(task_id='task_13', question='가장 가까운 전기차 충전소 찾아줘', expected_labels=['주행']),
        Task(task_id='task_14', question='출발지에서 목적지까지 거리 몇 km야?', expected_labels=['주행']),
        Task(task_id='task_15', question='지금 현재 위치에서 강서구청까지 가는 길 보여줘', expected_labels=['주행']),
        
        # 차량 상태(단일 카테고리) - 15개
        Task(task_id='task_16', question='엔진오일 교체가 필요한지 확인해줘', expected_labels=['차량 상태']),
        Task(task_id='task_17', question='지금 타이어 압력 괜찮아?', expected_labels=['차량 상태']),
        Task(task_id='task_18', question='배터리 잔량 알려줘', expected_labels=['차량 상태']),
        Task(task_id='task_19', question='경고등이 켜졌는데 뭐가 문제야?', expected_labels=['차량 상태']),
        Task(task_id='task_20', question='냉각수 부족한지 체크해줘', expected_labels=['차량 상태']),
        Task(task_id='task_21', question='브레이크 패드 상태 확인', expected_labels=['차량 상태']),
        Task(task_id='task_22', question='차량 점검 필요한 항목 알려줘', expected_labels=['차량 상태']),
        Task(task_id='task_23', question='연료 얼마나 남았어?', expected_labels=['차량 상태']),
        Task(task_id='task_24', question='엔진 온도가 정상인지 확인', expected_labels=['차량 상태']),
        Task(task_id='task_25', question='차량 진단 결과 보여줘', expected_labels=['차량 상태']),
        Task(task_id='task_26', question='타이어 마모도는 어느 정도야?', expected_labels=['차량 상태']),
        Task(task_id='task_27', question='엔진 점검 필요한데 언제 해야 돼?', expected_labels=['차량 상태']),
        Task(task_id='task_28', question='배터리 확인해줄래?', expected_labels=['차량 상태']),
        Task(task_id='task_29', question='휠 얼라인먼트 체크 필요해', expected_labels=['차량 상태']),
        Task(task_id='task_30', question='현재 주행거리 몇 km야?', expected_labels=['차량 상태']),
        
        # 차량 제어(단일 카테고리) - 15개
        Task(task_id='task_31', question='운전석 창문 조금 내려줘', expected_labels=['차량 제어']),
        Task(task_id='task_32', question='에어컨을 22도로 맞춰줘', expected_labels=['차량 제어']),
        Task(task_id='task_33', question='시동 꺼줘', expected_labels=['차량 제어']),
        Task(task_id='task_34', question='문 잠금 해제해줘', expected_labels=['차량 제어']),
        Task(task_id='task_35', question='선루프 열어줘', expected_labels=['차량 제어']),
        Task(task_id='task_36', question='뒷좌석 열선 켜줘', expected_labels=['차량 제어']),
        Task(task_id='task_37', question='외부 순환 모드로 바꿔', expected_labels=['차량 제어']),
        Task(task_id='task_38', question='와이퍼 작동시켜줘', expected_labels=['차량 제어']),
        Task(task_id='task_39', question='헤드라이트 켜줘', expected_labels=['차량 제어']),
        Task(task_id='task_40', question='실내등 켜줘', expected_labels=['차량 제어']),
        Task(task_id='task_41', question='뒷유리 열선 켜', expected_labels=['차량 제어']),
        Task(task_id='task_42', question='주차 센서 활성화해줘', expected_labels=['차량 제어']),
        Task(task_id='task_43', question='크루즈 컨트롤 속도 70으로 설정', expected_labels=['차량 제어']),
        Task(task_id='task_44', question='사이드 미러 자동 접기 실행', expected_labels=['차량 제어']),
        Task(task_id='task_45', question='배기구 매연 필터 리셋해줘', expected_labels=['차량 제어']),
        
        # 미디어(단일 카테고리) - 15개
        Task(task_id='task_46', question='라디오 켜', expected_labels=['미디어']),
        Task(task_id='task_47', question='볼륨을 10으로 낮춰봐', expected_labels=['미디어']),
        Task(task_id='task_48', question='아이유 노래 틀어줘', expected_labels=['미디어']),
        Task(task_id='task_49', question='음악 일시정지', expected_labels=['미디어']),
        Task(task_id='task_50', question='다음 곡으로 넘겨', expected_labels=['미디어']),
        Task(task_id='task_51', question='재즈 음악 재생해줘', expected_labels=['미디어']),
        Task(task_id='task_52', question='블루투스 연결해줘', expected_labels=['미디어']),
        Task(task_id='task_53', question='FM 95.1 주파수로 맞춰', expected_labels=['미디어']),
        Task(task_id='task_54', question='이전 곡으로 돌아가', expected_labels=['미디어']),
        Task(task_id='task_55', question='플레이리스트 셔플해줘', expected_labels=['미디어']),
        Task(task_id='task_56', question='방탄소년단 앨범 재생', expected_labels=['미디어']),
        Task(task_id='task_57', question='팟캐스트 틀어줘', expected_labels=['미디어']),
        Task(task_id='task_58', question='클래식 채널로 변경해줘', expected_labels=['미디어']),
        Task(task_id='task_59', question='밸런스 왼쪽으로 조정해', expected_labels=['미디어']),
        Task(task_id='task_60', question='음악 반복 모드 설정해줄래?', expected_labels=['미디어']),
        
        # 개인 비서(단일 카테고리) - 15개
        Task(task_id='task_61', question='엄마한테 전화 걸어줘', expected_labels=['개인 비서']),
        Task(task_id='task_62', question='김철수한테 3시 도착 예정으로 문자 하나 보내줘', expected_labels=['개인 비서']),
        Task(task_id='task_63', question='오늘 일정 알려줘', expected_labels=['개인 비서']),
        Task(task_id='task_64', question='지금 안읽은 메시지 쭉 읽어줘', expected_labels=['개인 비서']),
        Task(task_id='task_65', question='캘린더에 내일 오전 10시 회의 일정 하나 추가해놔', expected_labels=['개인 비서']),
        Task(task_id='task_66', question='지난 3일 동안 메일 온 것들 브리핑 해줘', expected_labels=['개인 비서']),
        Task(task_id='task_67', question='메신저 알림 좀 1시간 동안 잠깐 꺼줘', expected_labels=['개인 비서']),
        Task(task_id='task_68', question='오후 3시에 알람 설정해줘', expected_labels=['개인 비서']),
        Task(task_id='task_69', question='내 핸드폰 배터리 확인하고 50퍼 이하면 저전력 모드 켜주라', expected_labels=['개인 비서']),
        Task(task_id='task_70', question='음성 메모 녹음 시작해줘', expected_labels=['개인 비서']),
        Task(task_id='task_71', question='아버지한테 "곧 도착하겠습니다" 보내줄래?', expected_labels=['개인 비서']),
        Task(task_id='task_72', question='내일 아침 기상 알림 켜놔줘', expected_labels=['개인 비서']),
        Task(task_id='task_73', question='부재중 찍힌거 누구야?', expected_labels=['개인 비서']),
        Task(task_id='task_74', question='최근 스캔한 명함 정보 말해줘', expected_labels=['개인 비서']),
        Task(task_id='task_75', question='내 휴대폰 방해금지 모드로 해줘', expected_labels=['개인 비서']),
        
        # 멀티레이블 - 15개
        Task(task_id='task_76', question='창문 닫고 아이유 신곡 재생', expected_labels=['차량 제어', '미디어']),
        Task(task_id='task_77', question='엉따 2단계 해주고 퇴근길 경로 찾아줘', expected_labels=['차량 제어', '주행']),
        Task(task_id='task_78', question='음악 잠깐 정지하고 엄마한테 전화 걸어줘', expected_labels=['미디어', '개인 비서']),
        Task(task_id='task_79', question='재생목록 틀어주고 실내 공기 순환 모드 켜줘', expected_labels=['미디어', '차량 제어']),
        Task(task_id='task_80', question='타이어 공기압 확인하고 근처 정비소 안내해줘', expected_labels=['차량 상태', '주행']),
        Task(task_id='task_81', question='배터리 상태 보고하고 문자 앱 좀 열어줘', expected_labels=['차량 상태', '개인 비서']),
        Task(task_id='task_82', question='히터 켜고 음악 틀어줘', expected_labels=['차량 제어', '미디어']),
        Task(task_id='task_83', question='연료 잔량 확인하고 근처 주유소 안내', expected_labels=['차량 상태', '주행']),
        Task(task_id='task_84', question='라디오 끄고 오늘 일정 알려줘', expected_labels=['미디어', '개인 비서']),
        Task(task_id='task_85', question='선루프 열고 집 가는 길 안내', expected_labels=['차량 제어', '주행']),
        Task(task_id='task_86', question='에어컨 20도로 맞추고 클래식 재생해줘', expected_labels=['차량 제어', '미디어']),
        Task(task_id='task_87', question='엔진 진단하고 스팸 메시지 삭제해', expected_labels=['차량 상태', '개인 비서']),
        Task(task_id='task_88', question='휠 상태 확인하고 가장 빠른 경로로 안내', expected_labels=['차량 상태', '주행']),
        Task(task_id='task_89', question='시트 히터 켜고 좋아하는 팟캐스트 틀어', expected_labels=['차량 제어', '미디어']),
        Task(task_id='task_90', question='배터리 잔량 확인 후 동료한테 연락해', expected_labels=['차량 상태', '개인 비서']),
        
        # 범위 외 - 10개
        Task(task_id='task_91', question='가벼운 점심 메뉴 뭐가 좋을까?', expected_labels=[]),
        Task(task_id='task_92', question='하얀 푸들 강아지 이름 추천해줘', expected_labels=[]),
        Task(task_id='task_93', question='고양이는 왜 낮잠을 많이 자?', expected_labels=[]),
        Task(task_id='task_94', question='"내비"는 콩글리쉬야?', expected_labels=[]),
        Task(task_id='task_95', question='야간 운전할 때 눈이 덜 피곤하게 하는 팁 같은 거 있어?', expected_labels=[]),
        Task(task_id='task_96', question='파이썬 공부하기 좋은 책 추천', expected_labels=[]),
        Task(task_id='task_97', question='김치찌개 맛있게 끓이는 방법', expected_labels=[]),
        Task(task_id='task_98', question='지하철은 몇 호선이 제일 오래됐어?', expected_labels=[]),
        Task(task_id='task_99', question='세계에서 가장 높은 산은?', expected_labels=[]),
        Task(task_id='task_100', question='재미있는 영화 추천해줘', expected_labels=[])
    ]