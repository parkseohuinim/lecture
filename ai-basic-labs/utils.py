"""
공통 유틸리티 함수 모듈
- 출력 포맷팅 함수
- 시각화 헬퍼 함수
- ASCII 심볼 상수
- 유사도/거리 계산 및 해석 함수
- OpenAI 클라이언트 헬퍼
"""

import os
from typing import List, Tuple
import numpy as np


# ============================================================================
# 해석 기준 상수 (전체 Lab에서 일관되게 사용)
# ============================================================================

# 코사인 유사도 기준 (-1 ~ 1 범위, 1이 가장 유사)
COSINE_THRESHOLDS = {
    'very_similar': 0.8,   # 매우 유사
    'related': 0.6,        # 관련 있음
    'somewhat': 0.4        # 약간 관련
}

# L2 거리 기준 (0 ~ infinity, 0이 가장 유사)
L2_DISTANCE_THRESHOLDS = {
    'high': 1.0,      # 거리 < 1.0 = 높은 관련성
    'medium': 1.8,    # 거리 < 1.8 = 중간 관련성
    # 거리 >= 1.8 = 낮은 관련성
}


# ============================================================================
# 유사도/거리 해석 함수
# ============================================================================

def interpret_cosine_similarity(score: float) -> str:
    """
    코사인 유사도 해석
    
    Args:
        score: 코사인 유사도 값 (-1 ~ 1)
    
    Returns:
        해석 문자열
    
    Example:
        >>> interpret_cosine_similarity(0.85)
        '[v] 매우 유사'
    """
    if score >= COSINE_THRESHOLDS['very_similar']:
        return "[v] 매우 유사"
    elif score >= COSINE_THRESHOLDS['related']:
        return "[~] 관련 있음"
    elif score >= COSINE_THRESHOLDS['somewhat']:
        return "[o] 약간 관련"
    else:
        return "[x] 다른 주제"


def interpret_l2_distance(distance: float) -> str:
    """
    L2 거리 해석 (ChromaDB 기본 거리 메트릭)
    
    Args:
        distance: L2 거리 값 (0 ~ infinity)
    
    Returns:
        해석 문자열
    
    Example:
        >>> interpret_l2_distance(0.8)
        '[v] 높음'
    """
    if distance < L2_DISTANCE_THRESHOLDS['high']:
        return "[v] 높음"
    elif distance < L2_DISTANCE_THRESHOLDS['medium']:
        return "[~] 중간"
    else:
        return "[x] 낮음"


def l2_distance_to_similarity(distance: float) -> float:
    """
    L2 거리를 0~1 유사도 점수로 변환
    
    Args:
        distance: L2 거리 값
    
    Returns:
        0~1 범위의 유사도 점수
    
    Note:
        공식: similarity = 1 / (1 + distance)
        - 거리 0 → 유사도 1.0
        - 거리 1 → 유사도 0.5
        - 거리 ∞ → 유사도 0.0
    """
    return 1 / (1 + distance)


# ============================================================================
# 벡터 연산 함수
# ============================================================================

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    두 벡터 간의 코사인 유사도 계산
    
    Args:
        vec1: 첫 번째 벡터
        vec2: 두 번째 벡터
    
    Returns:
        코사인 유사도 (-1 ~ 1)
    
    Note:
        코사인 유사도 = (A · B) / (||A|| × ||B||)
        - 1: 완전히 같은 방향
        - 0: 직각 (무관)
        - -1: 반대 방향
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def cosine_similarity_normalized(vec1: List[float], vec2: List[float]) -> float:
    """
    정규화된 벡터의 코사인 유사도 계산 (내적만 사용)
    
    Args:
        vec1: L2 정규화된 첫 번째 벡터
        vec2: L2 정규화된 두 번째 벡터
    
    Returns:
        코사인 유사도 (-1 ~ 1)
    
    Note:
        OpenAI 임베딩은 이미 L2 정규화되어 있으므로,
        ||A|| = ||B|| = 1이고, 따라서 cos(A,B) = A · B
        
        이 함수는 정규화된 벡터에 대해 더 효율적입니다.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2))


def is_normalized(vec: List[float], tolerance: float = 1e-6) -> bool:
    """
    벡터가 L2 정규화되어 있는지 확인
    
    Args:
        vec: 확인할 벡터
        tolerance: 허용 오차
    
    Returns:
        정규화 여부
    """
    norm = np.linalg.norm(vec)
    return abs(norm - 1.0) < tolerance


# ============================================================================
# OpenAI 클라이언트 헬퍼
# ============================================================================

def get_openai_client(api_key: str = None):
    """
    OpenAI 클라이언트 생성 (SSL 인증서 검증 우회 포함)
    
    회사 방화벽, 프록시 환경 등에서 SSL 인증서 문제가 발생할 수 있어
    httpx 클라이언트의 verify=False 옵션을 사용합니다.
    
    Args:
        api_key: OpenAI API 키 (None이면 환경변수에서 로드)
    
    Returns:
        OpenAI 클라이언트 인스턴스
    
    Example:
        >>> client = get_openai_client()
        >>> response = client.chat.completions.create(...)
    """
    import httpx
    from openai import OpenAI
    
    # SSL 인증서 검증 우회 설정
    http_client = httpx.Client(verify=False)
    
    return OpenAI(
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        http_client=http_client
    )


def get_langchain_chat_model(model: str = "gpt-4o-mini", temperature: float = 0):
    """
    LangChain ChatOpenAI 모델 생성 (SSL 우회 포함)
    
    Args:
        model: 모델 이름
        temperature: 생성 온도
    
    Returns:
        ChatOpenAI 인스턴스
    """
    import httpx
    from langchain_openai import ChatOpenAI
    
    http_client = httpx.Client(verify=False)
    
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        http_client=http_client
    )


def get_langchain_embeddings(model: str = "text-embedding-3-small"):
    """
    LangChain OpenAIEmbeddings 생성 (SSL 우회 포함)
    
    Args:
        model: 임베딩 모델 이름
    
    Returns:
        OpenAIEmbeddings 인스턴스
    """
    import httpx
    from langchain_openai import OpenAIEmbeddings
    
    http_client = httpx.Client(verify=False)
    
    return OpenAIEmbeddings(
        model=model,
        http_client=http_client
    )


# ============================================================================
# ASCII 심볼 상수 (이모지 대체용)
# ============================================================================

class Symbols:
    """출력용 ASCII 심볼 상수"""
    # 상태/결과
    OK = "[OK]"
    ERROR = "[X]"
    WARNING = "[!]"
    INFO = "[INFO]"
    
    # 작업 유형
    TIP = "[TIP]"
    DOC = "[DOC]"
    FILE = "[FILE]"
    DIR = "[DIR]"
    DATA = "[DATA]"
    
    # 학습/교육
    LAB = "[LAB]"
    BOOK = "[BOOK]"
    READ = "[READ]"
    
    # 검색/분석
    SEARCH = "[SEARCH]"
    TARGET = "[TARGET]"
    CMP = "[CMP]"
    GRAPH = "[GRAPH]"
    
    # 기타
    GO = "[GO]"
    DONE = "[DONE]"
    LINK = "[LINK]"
    TAG = "[TAG]"
    PIN = "[PIN]"
    TOOL = "[TOOL]"
    BUILD = "[BUILD]"
    BEST = "[BEST]"
    
    # 수학/측정
    NUM = "[NUM]"
    MATH = "[MATH]"
    MEASURE = "[MEASURE]"
    VS = "[vs]"
    
    # 기타 텍스트
    ARROW = "-->"
    NEXT = "[>]"
    ELLIPSIS = "[...]"


# ============================================================================
# 출력 포맷팅 함수
# ============================================================================

def print_section_header(title: str, symbol: str = "[INFO]"):
    """
    섹션 헤더 출력
    
    Args:
        title: 섹션 제목
        symbol: 앞에 붙을 심볼 (기본값: [INFO])
    
    Example:
        print_section_header("토큰화 실습", "[LAB]")
        # ────────────────────────────────────────────────────────────
        # [LAB] 토큰화 실습
        # ────────────────────────────────────────────────────────────
    """
    print(f"\n{'─'*60}")
    print(f"{symbol} {title}")
    print(f"{'─'*60}")


def print_subsection(title: str):
    """
    서브섹션 헤더 출력
    
    Args:
        title: 서브섹션 제목
    
    Example:
        print_subsection("단어 토큰화")
        # [>] 단어 토큰화
        #   ──────────────────────────────────────────────────────
    """
    print(f"\n[>] {title}")
    print(f"  {'─'*50}")


def print_key_points(points: List[str], title: str = "핵심 포인트"):
    """
    핵심 포인트 상자 출력
    
    Args:
        points: 핵심 포인트 리스트
        title: 상자 제목 (기본값: "핵심 포인트")
    
    Example:
        print_key_points([
            "[OK] 성공적으로 처리됨",
            "[TIP] 다음 단계 진행 가능"
        ], "결과 요약")
    """
    print(f"\n{'='*60}")
    print(f"[TIP] {title}:")
    print(f"{'='*60}")
    for point in points:
        print(f"  {point}")


def print_info_box(title: str, content: List[str], symbol: str = "[INFO]"):
    """
    정보 상자 출력
    
    Args:
        title: 상자 제목
        content: 내용 리스트
        symbol: 앞에 붙을 심볼
    """
    print(f"\n{'+'*60}")
    print(f"{symbol} {title}")
    print(f"{'+'*60}")
    for line in content:
        print(f"  {line}")
    print(f"{'+'*60}")


def print_comparison(item1: str, item2: str, description: str = ""):
    """
    비교 출력
    
    Args:
        item1: 첫 번째 항목
        item2: 두 번째 항목
        description: 설명
    """
    print(f"\n[vs] {item1} vs {item2}")
    if description:
        print(f"     {description}")


# ============================================================================
# 시각화 함수
# ============================================================================

def visualize_similarity_bar(score: float, width: int = 40) -> str:
    """
    유사도를 막대 그래프로 시각화
    
    Args:
        score: 0.0 ~ 1.0 사이의 점수
        width: 막대 전체 너비 (기본값: 40)
    
    Returns:
        막대 문자열 (예: "========================================--------")
    
    Example:
        bar = visualize_similarity_bar(0.75)
        print(f"|{bar}| 75%")
        # |==============================----------| 75%
    """
    filled = int(score * width)
    bar = "=" * filled + "-" * (width - filled)
    return bar


def visualize_score_comparison(scores: List[tuple], width: int = 30):
    """
    여러 점수를 비교 시각화
    
    Args:
        scores: (이름, 점수) 튜플의 리스트
        width: 막대 너비
    
    Example:
        visualize_score_comparison([
            ("문서 A", 0.95),
            ("문서 B", 0.72),
            ("문서 C", 0.45)
        ])
    """
    max_name_len = max(len(name) for name, _ in scores)
    
    print()
    for name, score in scores:
        bar = visualize_similarity_bar(score, width)
        print(f"  {name:<{max_name_len}} |{bar}| {score:.2%}")


def format_token_output(tokens: List[str], max_display: int = 10) -> str:
    """
    토큰 리스트를 보기 좋게 포맷팅
    
    Args:
        tokens: 토큰 리스트
        max_display: 최대 표시 개수
    
    Returns:
        포맷팅된 문자열
    """
    if len(tokens) <= max_display:
        return str(tokens)
    
    displayed = tokens[:max_display]
    remaining = len(tokens) - max_display
    return f"{displayed} [...외 {remaining}개]"


# ============================================================================
# 진행 상태 출력
# ============================================================================

def print_step(step_num: int, total_steps: int, description: str):
    """
    단계별 진행 상태 출력
    
    Args:
        step_num: 현재 단계 번호
        total_steps: 전체 단계 수
        description: 단계 설명
    """
    progress = "=" * step_num + "-" * (total_steps - step_num)
    print(f"\n[{step_num}/{total_steps}] [{progress}] {description}")


def print_result(success: bool, message: str):
    """
    결과 출력
    
    Args:
        success: 성공 여부
        message: 결과 메시지
    """
    symbol = "[OK]" if success else "[X]"
    print(f"{symbol} {message}")


def print_next_step(message: str):
    """
    다음 단계 안내 출력
    
    Args:
        message: 안내 메시지
    """
    print(f"\n[GO] 다음 단계: {message}")

