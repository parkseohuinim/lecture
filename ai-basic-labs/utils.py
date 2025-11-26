"""
공통 유틸리티 함수 모듈
- 출력 포맷팅 함수
- 시각화 헬퍼 함수
- ASCII 심볼 상수
"""

from typing import List


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

