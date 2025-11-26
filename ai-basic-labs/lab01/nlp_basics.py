"""
NLP 기초 실습
- 토큰화, 불용어 제거, lemmatization
- OpenAI 임베딩 생성 및 코사인 유사도 계산
- 간단한 문장 검색기 구현

실습 항목:
1. tiktoken으로 토큰 이해하기 - GPT가 텍스트를 어떻게 보는가
2. NLTK 전처리 파이프라인 - 토큰화, 불용어, 표제어 추출
3. OpenAI 임베딩 생성 - 텍스트를 벡터로 변환
4. 코사인 유사도 계산 - 벡터 간 유사성 측정
5. 간단한 검색 엔진 - 의미 기반 문장 검색
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import tiktoken
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from openai import OpenAI
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 공통 유틸리티 import를 위한 경로 추가
sys.path.insert(0, str(project_root))
from utils import (
    print_section_header, 
    print_subsection, 
    print_key_points, 
    visualize_similarity_bar
)


# ============================================================================
# NLTK 데이터 다운로드
# ============================================================================

def download_nltk_data():
    """필요한 NLTK 데이터 다운로드"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        print("NLTK 데이터 다운로드 중...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        print("다운로드 완료!")


# ============================================================================
# 1. tiktoken으로 토큰 이해하기
# ============================================================================

def count_tokens_with_tiktoken(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    tiktoken을 사용하여 텍스트의 토큰 수를 계산
    
    Args:
        text: 토큰 수를 계산할 텍스트
        model: 사용할 모델 이름
    
    Returns:
        토큰 수
    """
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)


def demo_tiktoken():
    """실습 1: tiktoken으로 토큰 이해하기"""
    print("\n" + "="*80)
    print("[1] 실습 1: tiktoken으로 토큰 이해하기")
    print("="*80)
    print("목표: GPT가 텍스트를 어떻게 토큰으로 분해하는지 이해")
    print("핵심: 토큰 != 단어, 한글은 영어보다 더 많은 토큰 사용")
    
    # 토큰이란 무엇인가?
    print_section_header("토큰(Token)이란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] 토큰의 개념                                       │
  │  ─────────────────────────────────────────────────────  │
  │  • GPT는 텍스트를 '토큰' 단위로 처리합니다               │
  │  • 토큰 != 단어 (단어보다 작거나 클 수 있음)             │
  │  • 영어: 1 단어 = 1~2 토큰                               │
  │  • 한글: 1 글자 = 1.5~3 토큰 (바이트 단위 분해)          │
  │                                                         │
  │  왜 중요한가?                                            │
  │  • API 비용이 토큰 단위로 계산됨                         │
  │  • 컨텍스트 윈도우 제한이 토큰 기준                      │
  │  • 예: GPT-4 Turbo = 128K 토큰 제한                     │
  └─────────────────────────────────────────────────────────┘
    """)
    
    texts = [
        "Hello, how are you?",
        "안녕하세요, 반갑습니다!",
        "This is a longer sentence with more words to demonstrate token counting.",
        "AI와 머신러닝은 현대 기술의 핵심입니다."
    ]
    
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    print_section_header("영어 vs 한글 토큰 비교", "[CMP]")
    
    for text in texts:
        token_count = count_tokens_with_tiktoken(text)
        char_count = len(text)
        ratio = token_count / char_count
        
        print(f"\n{'─'*60}")
        print(f"텍스트: {text}")
        print(f"문자 수: {char_count}자 | 토큰 수: {token_count}개")
        print(f"토큰/문자 비율: {ratio:.2f} (1보다 크면 비효율적)")
        
        # 실제 토큰 ID 확인
        tokens = encoding.encode(text)
        print(f"\n토큰 ID: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
        
        # 개별 토큰을 디코딩하고 바이트 정보도 표시
        print(f"\n토큰 분석:")
        for i, token_id in enumerate(tokens[:8]):  # 처음 8개만
            decoded = encoding.decode([token_id])
            byte_repr = decoded.encode('utf-8', errors='replace')
            display = decoded if decoded.isprintable() else repr(decoded)
            print(f"  [{i+1}] ID:{token_id:6d} | '{display}' | bytes: {byte_repr}")
        
        if len(tokens) > 8:
            print(f"  ... (나머지 {len(tokens) - 8}개 토큰 생략)")
        
        # 한글 텍스트에 대한 추가 설명
        if any(ord(c) > 127 for c in text):
            print(f"\n  [!] 한글 토큰화 특징:")
            print(f"     - 한글은 UTF-8에서 3바이트/글자")
            print(f"     - BPE 알고리즘이 바이트 단위로 분해")
            print(f"     - 불완전한 UTF-8 바이트 조각이 생길 수 있음")
            print(f"     - 모든 토큰 합치면 원본 완벽 복원!")
    
    # 핵심 포인트
    print_key_points([
        "- tiktoken: OpenAI 공식 토큰 계산 라이브러리",
        "- 모델마다 다른 인코더 사용 (gpt-3.5-turbo, gpt-4 등)",
        "- 한글은 영어보다 2~3배 더 많은 토큰 소모",
        "- API 비용 추정: 1K 토큰 = $0.001~0.01 (모델별 상이)",
        "- 실무 팁: 긴 한글 문서는 토큰 비용 미리 계산!"
    ], "tiktoken 핵심 포인트")


# ============================================================================
# 2. NLTK 전처리 파이프라인
# ============================================================================

class TextPreprocessor:
    """텍스트 전처리 파이프라인"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def tokenize(self, text: str) -> List[str]:
        """텍스트를 토큰으로 분리"""
        return word_tokenize(text.lower())
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """불용어 제거"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """표제어 추출 (lemmatization)"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess(self, text: str, remove_stopwords: bool = True, 
                   lemmatize: bool = True) -> List[str]:
        """
        전체 전처리 파이프라인 실행
        
        Args:
            text: 전처리할 텍스트
            remove_stopwords: 불용어 제거 여부
            lemmatize: 표제어 추출 여부
        
        Returns:
            전처리된 토큰 리스트
        """
        # 1. 토큰화
        tokens = self.tokenize(text)
        
        # 2. 알파벳만 남기기
        tokens = [token for token in tokens if token.isalpha()]
        
        # 3. 불용어 제거
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)
        
        # 4. 표제어 추출
        if lemmatize:
            tokens = self.lemmatize(tokens)
        
        return tokens


def demo_preprocessing():
    """실습 2: NLTK 전처리 파이프라인"""
    print("\n" + "="*80)
    print("[2] 실습 2: NLTK 전처리 파이프라인")
    print("="*80)
    print("목표: 텍스트 정규화의 필요성과 방법 이해")
    print("핵심: 토큰화 -> 정규화 -> 불용어 제거 -> 표제어 추출")
    
    # 전처리란?
    print_section_header("텍스트 전처리란?", "[DOC]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] 왜 전처리가 필요한가?                             │
  │  ─────────────────────────────────────────────────────  │
  │  • "Running", "runs", "ran" -> 모두 "run"의 변형         │
  │  • "the", "is", "a" -> 의미 없는 단어 (불용어)           │
  │  • 대소문자 통일 -> "AI" = "ai" = "Ai"                   │
  │                                                         │
  │  전처리 없이 검색하면?                                   │
  │  • "cats" 검색 시 "cat" 문서 놓침                        │
  │  • "THE CAT" vs "the cat" 다르게 인식                    │
  └─────────────────────────────────────────────────────────┘
    """)
    
    preprocessor = TextPreprocessor()
    
    text = "The cats are running quickly through the beautiful gardens and jumping over fences."
    
    print_section_header("단계별 전처리 과정", "[STEP]")
    print(f"\n원본 텍스트: {text}")
    
    # 1단계: 토큰화
    print_subsection("1단계: 토큰화 (Tokenization)")
    tokens = preprocessor.tokenize(text)
    print(f"  결과: {tokens}")
    print(f"  설명: 문장을 단어 단위로 분리, 소문자 변환")
    
    # 2단계: 알파벳만 남기기
    print_subsection("2단계: 알파벳 필터링")
    alpha_tokens = [token for token in tokens if token.isalpha()]
    print(f"  결과: {alpha_tokens}")
    print(f"  설명: 구두점(., !) 제거")
    
    # 3단계: 불용어 제거
    print_subsection("3단계: 불용어 제거 (Stopword Removal)")
    no_stop = preprocessor.remove_stopwords(alpha_tokens)
    removed = [t for t in alpha_tokens if t not in no_stop]
    print(f"  결과: {no_stop}")
    print(f"  제거됨: {removed}")
    print(f"  설명: 'the', 'are', 'and' 등 의미 없는 단어 제거")
    
    # 4단계: 표제어 추출
    print_subsection("4단계: 표제어 추출 (Lemmatization)")
    lemmatized = preprocessor.lemmatize(no_stop)
    
    # 변화된 단어 강조
    changes = []
    for orig, lem in zip(no_stop, lemmatized):
        if orig != lem:
            changes.append(f"'{orig}' -> '{lem}'")
    
    print(f"  결과: {lemmatized}")
    if changes:
        print(f"  변환됨: {', '.join(changes)}")
    print(f"  설명: 단어를 기본형으로 변환 (cats->cat, running->run)")
    
    # Lemmatization vs Stemming 비교
    print_section_header("Lemmatization vs Stemming", "[vs]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [CMP] 비교                                              │
  │  ─────────────────────────────────────────────────────  │
  │  단어        │ Stemming     │ Lemmatization            │
  │  ────────────┼──────────────┼─────────────────────────  │
  │  running     │ runn         │ run (동사 기본형)         │
  │  better      │ better       │ good (형용사 원형)        │
  │  studies     │ studi        │ study (명사 기본형)       │
  │  ────────────┼──────────────┼─────────────────────────  │
  │  특징        │ 빠름, 규칙 기반│ 정확, 사전 기반          │
  │  단점        │ 비문법적 결과  │ 느림, 품사 필요          │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 전체 파이프라인 결과
    print_subsection("전체 파이프라인 결과")
    result = preprocessor.preprocess(text)
    print(f"  원본: {text}")
    print(f"  결과: {result}")
    print(f"  토큰 수: {len(text.split())} -> {len(result)} (약 {(1-len(result)/len(text.split()))*100:.0f}% 감소)")
    
    # 핵심 포인트
    print_key_points([
        "- 토큰화: 텍스트를 의미 단위로 분리",
        "- 불용어 제거: 의미 없는 고빈도 단어 제거 (the, is, a...)",
        "- 표제어 추출: 단어를 사전 기본형으로 변환",
        "- 실무 팁: 임베딩 모델은 보통 전처리 불필요 (내부 처리)",
        "- 용도: 키워드 추출, BM25 검색, 텍스트 분석"
    ], "전처리 핵심 포인트")


# ============================================================================
# 3. OpenAI 임베딩 생성
# ============================================================================

class EmbeddingGenerator:
    """OpenAI 임베딩 생성기"""
    
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = "text-embedding-3-small"
    
    def get_embedding(self, text: str) -> List[float]:
        """
        단일 텍스트의 임베딩 생성
        
        Args:
            text: 임베딩을 생성할 텍스트
        
        Returns:
            임베딩 벡터
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        여러 텍스트의 임베딩을 배치로 생성
        
        Args:
            texts: 임베딩을 생성할 텍스트 리스트
        
        Returns:
            임베딩 벡터 리스트
        """
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [data.embedding for data in response.data]


def demo_embeddings():
    """실습 3: OpenAI 임베딩 생성"""
    print("\n" + "="*80)
    print("[3] 실습 3: OpenAI 임베딩 생성")
    print("="*80)
    print("목표: 텍스트가 어떻게 숫자 벡터로 변환되는지 이해")
    print("핵심: 의미가 비슷한 텍스트 -> 비슷한 벡터 -> 가까운 거리")
    
    # 임베딩이란?
    print_section_header("임베딩(Embedding)이란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] 임베딩의 개념                                     │
  │  ─────────────────────────────────────────────────────  │
  │  • 텍스트 -> 고정 길이 숫자 벡터로 변환                  │
  │  • 예: "고양이" -> [0.1, -0.3, 0.5, ..., 0.2] (1536차원) │
  │                                                         │
  │  왜 벡터로 변환하는가?                                   │
  │  • 컴퓨터는 숫자만 연산 가능                             │
  │  • 벡터 공간에서 의미적 유사성 측정 가능                 │
  │  • "왕 - 남자 + 여자 = 여왕" 같은 연산 가능             │
  │                                                         │
  │  OpenAI 임베딩 모델:                                     │
  │  • text-embedding-3-small: 1536차원, 저렴, 빠름          │
  │  • text-embedding-3-large: 3072차원, 고성능              │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    generator = EmbeddingGenerator()
    
    # 단일 임베딩
    print_section_header("단일 텍스트 임베딩", "[DOC]")
    
    text = "Artificial intelligence is transforming the world."
    embedding = generator.get_embedding(text)
    
    print(f"\n텍스트: '{text}'")
    print(f"\n임베딩 결과:")
    print(f"  • 벡터 차원: {len(embedding)}")
    print(f"  • 처음 5개 값: {[round(v, 4) for v in embedding[:5]]}")
    print(f"  • 마지막 5개 값: {[round(v, 4) for v in embedding[-5:]]}")
    print(f"  • 값의 범위: [{min(embedding):.4f}, {max(embedding):.4f}]")
    
    # 벡터 시각화 (간단한 히스토그램)
    print(f"\n  값 분포 시각화:")
    bins = [0, 0, 0, 0, 0]  # -0.1~-0.05, -0.05~0, 0~0.05, 0.05~0.1, 기타
    for v in embedding:
        if v < -0.05:
            bins[0] += 1
        elif v < 0:
            bins[1] += 1
        elif v < 0.05:
            bins[2] += 1
        elif v < 0.1:
            bins[3] += 1
        else:
            bins[4] += 1
    
    labels = ["< -0.05", "-0.05~0", "0~0.05", "0.05~0.1", "> 0.1"]
    max_bin = max(bins)
    for label, count in zip(labels, bins):
        bar_len = int(count / max_bin * 30)
        print(f"    {label:>10}: {'#' * bar_len} ({count})")
    
    # 배치 임베딩
    print_section_header("배치 임베딩 (효율적인 방법)", "[BATCH]")
    
    texts = [
        "I love machine learning.",
        "Deep learning is a subset of AI.",
        "Python is a great programming language."
    ]
    
    print("\n[DOC] 임베딩 생성 코드:")
    print("  ┌─────────────────────────────────────────────────────")
    print("  │ # 비효율적: 개별 호출")
    print("  │ for text in texts:")
    print("  │     emb = client.embeddings.create(input=text)  # API 3번 호출")
    print("  │")
    print("  │ # 효율적: 배치 호출")
    print("  │ embs = client.embeddings.create(input=texts)  # API 1번 호출")
    print("  └─────────────────────────────────────────────────────")
    
    embeddings = generator.get_embeddings_batch(texts)
    
    print(f"\n배치 임베딩 결과:")
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        print(f"  {i+1}. '{text}'")
        print(f"     차원: {len(emb)}, 처음 5개: {[round(v, 4) for v in emb[:5]]}")
    
    # 핵심 포인트
    print_key_points([
        "- 임베딩: 텍스트 -> 고차원 벡터 (의미를 숫자로 인코딩)",
        "- text-embedding-3-small: 1536차원, 대부분의 용도에 충분",
        "- 배치 처리: 여러 텍스트를 한 번에 -> API 호출 최소화, 비용 절약",
        "- 비용: ~$0.00002 / 1K 토큰 (매우 저렴)",
        "- 용도: 유사도 검색, 클러스터링, 분류, RAG"
    ], "임베딩 핵심 포인트")


# ============================================================================
# 4. 코사인 유사도 계산
# ============================================================================

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    두 벡터 간의 코사인 유사도 계산
    
    Args:
        vec1: 첫 번째 벡터
        vec2: 두 번째 벡터
    
    Returns:
        코사인 유사도 (-1 ~ 1)
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    # 벡터 내적
    dot_product = np.dot(vec1, vec2)
    # 벡터 크기 (유클리드 놈)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    return dot_product / (norm1 * norm2)


def one_to_many_similarity(query_embedding: List[float], 
                          document_embeddings: List[List[float]]) -> List[float]:
    """
    1:N 유사도 계산 (하나의 쿼리와 여러 문서)
    
    Args:
        query_embedding: 쿼리 임베딩
        document_embeddings: 문서 임베딩 리스트
    
    Returns:
        각 문서와의 유사도 리스트
    """
    similarities = []
    for doc_emb in document_embeddings:
        sim = cosine_similarity(query_embedding, doc_emb)
        similarities.append(sim)
    return similarities


def many_to_many_similarity(embeddings1: List[List[float]], 
                           embeddings2: List[List[float]]) -> np.ndarray:
    """
    N:M 유사도 계산 (여러 쿼리와 여러 문서)
    
    Args:
        embeddings1: 첫 번째 임베딩 리스트
        embeddings2: 두 번째 임베딩 리스트
    
    Returns:
        유사도 행렬 (N x M)
    """
    matrix = np.zeros((len(embeddings1), len(embeddings2)))
    
    for i, emb1 in enumerate(embeddings1):
        for j, emb2 in enumerate(embeddings2):
            matrix[i][j] = cosine_similarity(emb1, emb2)
    
    return matrix


def demo_similarity():
    """실습 4: 코사인 유사도 계산"""
    print("\n" + "="*80)
    print("[4] 실습 4: 코사인 유사도 계산")
    print("="*80)
    print("목표: 벡터 간 유사성을 측정하는 방법 이해")
    print("핵심: 코사인 유사도 = 벡터 방향의 유사성 (크기 무관)")
    
    # 코사인 유사도란?
    print_section_header("코사인 유사도란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] 코사인 유사도 공식                                │
  │  ─────────────────────────────────────────────────────  │
  │                      A . B                              │
  │  cos(theta) = ─────────────────────                    │
  │               ||A|| x ||B||                            │
  │                                                         │
  │  • A . B : 두 벡터의 내적 (dot product)                 │
  │  • ||A|| : 벡터 A의 크기 (norm)                         │
  │                                                         │
  │  값의 범위:                                              │
  │  • +1 : 완전히 같은 방향 (매우 유사)                     │
  │  •  0 : 직각 (관련 없음)                                │
  │  • -1 : 반대 방향 (반대 의미) <- 실제론 드묾             │
  │                                                         │
  │  [TIP] 임베딩에서는 보통 0.3~1.0 범위에서 비교           │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    generator = EmbeddingGenerator()
    
    # 문장 준비
    sentences = [
        "I love programming in Python.",
        "Python is my favorite programming language.",
        "I enjoy cooking Italian food.",
        "Machine learning is fascinating.",
    ]
    
    # 임베딩 생성
    embeddings = generator.get_embeddings_batch(sentences)
    
    # 1:N 유사도 계산
    print_section_header("1:N 유사도 계산", "[>>>]")
    
    query = "I like coding with Python."
    query_embedding = generator.get_embedding(query)
    
    print(f"\n쿼리: '{query}'")
    print(f"\n각 문장과의 코사인 유사도:")
    print(f"{'─'*60}")
    
    similarities = one_to_many_similarity(query_embedding, embeddings)
    
    # 결과를 유사도 순으로 정렬하여 표시
    sorted_results = sorted(zip(sentences, similarities), key=lambda x: x[1], reverse=True)
    
    for sentence, sim in sorted_results:
        bar = visualize_similarity_bar(sim, 30)
        
        # 유사도 해석
        if sim >= 0.8:
            interpretation = "[v] 매우 유사"
        elif sim >= 0.6:
            interpretation = "[~] 관련 있음"
        elif sim >= 0.4:
            interpretation = "[o] 약간 관련"
        else:
            interpretation = "[x] 다른 주제"
        
        print(f"\n  {bar} {sim:.4f} {interpretation}")
        print(f"  '{sentence}'")
    
    # 가장 유사한 문장
    most_similar_idx = np.argmax(similarities)
    print(f"\n[#1] 가장 유사한 문장: '{sentences[most_similar_idx]}'")
    print(f"     유사도: {similarities[most_similar_idx]:.4f}")
    
    # N:M 유사도 계산
    print_section_header("N:M 유사도 행렬", "[INFO]")
    
    queries = [
        "Programming languages",
        "Food and cooking"
    ]
    query_embeddings = generator.get_embeddings_batch(queries)
    
    similarity_matrix = many_to_many_similarity(query_embeddings, embeddings)
    
    print("\n유사도 행렬:")
    print(f"{'─'*80}")
    
    # 헤더 출력
    print(f"{'쿼리 \\ 문서':<20}", end="")
    for i in range(len(sentences)):
        print(f"Doc{i+1:2d}  ", end="")
    print()
    print(f"{'─'*80}")
    
    # 각 쿼리별 유사도 출력
    for i, query in enumerate(queries):
        print(f"{query:<20}", end="")
        for j in range(len(sentences)):
            score = similarity_matrix[i][j]
            # 높은 점수 강조
            if score >= 0.5:
                print(f"[{score:.3f}]", end="")
            else:
                print(f" {score:.3f} ", end="")
        print()
    
    print(f"{'─'*80}")
    
    # 문서 목록 출력
    print("\n문서 목록:")
    for i, sentence in enumerate(sentences):
        print(f"  Doc{i+1:2d}: {sentence}")
    
    # 각 쿼리별 가장 유사한 문서
    print("\n[*] 각 쿼리별 가장 유사한 문서:")
    for i, query in enumerate(queries):
        most_similar_idx = np.argmax(similarity_matrix[i])
        score = similarity_matrix[i][most_similar_idx]
        print(f"  '{query}'")
        print(f"    -> Doc{most_similar_idx+1}: '{sentences[most_similar_idx]}' ({score:.4f})")
    
    # 핵심 포인트
    print_key_points([
        "- 코사인 유사도: 벡터 방향의 유사성 측정 (-1 ~ 1)",
        "- 임베딩에서: 0.8+ (매우 유사), 0.5~0.8 (관련), 0.5 미만 (다른 주제)",
        "- 1:N 검색: 쿼리 vs 모든 문서 -> 가장 유사한 문서 찾기",
        "- N:M 검색: 여러 쿼리 vs 여러 문서 -> 행렬 형태 결과",
        "- 실무 팁: Vector DB는 내부적으로 이 계산을 최적화"
    ], "유사도 계산 핵심 포인트")


# ============================================================================
# 5. 간단한 검색 엔진
# ============================================================================

class SimpleSearchEngine:
    """간단한 의미 기반 검색 엔진"""
    
    def __init__(self, api_key: str = None):
        self.generator = EmbeddingGenerator(api_key)
        self.documents = []
        self.embeddings = []
    
    def add_documents(self, documents: List[str]):
        """
        문서들을 검색 엔진에 추가
        
        Args:
            documents: 추가할 문서 리스트
        """
        print(f"\n{len(documents)}개의 문서를 인덱싱 중...")
        self.documents.extend(documents)
        new_embeddings = self.generator.get_embeddings_batch(documents)
        self.embeddings.extend(new_embeddings)
        print("인덱싱 완료!")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        쿼리와 가장 유사한 문서 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 결과 개수
        
        Returns:
            (문서, 유사도) 튜플의 리스트
        """
        if not self.documents:
            return []
        
        # 쿼리 임베딩 생성
        query_embedding = self.generator.get_embedding(query)
        
        # 유사도 계산
        similarities = one_to_many_similarity(query_embedding, self.embeddings)
        
        # 상위 k개 결과 추출
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.documents[idx], similarities[idx]))
        
        return results
    
    def print_search_results(self, query: str, top_k: int = 3):
        """검색 결과를 보기 좋게 출력"""
        print(f"\n[>>>] 검색 쿼리: '{query}'")
        print("─" * 60)
        
        results = self.search(query, top_k)
        
        if not results:
            print("검색 결과가 없습니다.")
            return
        
        print(f"\n상위 {len(results)}개 결과:\n")
        for i, (doc, score) in enumerate(results, 1):
            bar = visualize_similarity_bar(score, 25)
            
            # 점수 해석
            if score >= 0.7:
                interpretation = "[v] 높은 관련성"
            elif score >= 0.5:
                interpretation = "[~] 중간 관련성"
            else:
                interpretation = "[o] 낮은 관련성"
            
            print(f"[{i}] {bar} {score:.4f} {interpretation}")
            print(f"    {doc}\n")


def demo_search_engine():
    """실습 5: 간단한 검색 엔진"""
    print("\n" + "="*80)
    print("[5] 실습 5: 간단한 검색 엔진 (의미 기반)")
    print("="*80)
    print("목표: 임베딩 기반 검색 시스템의 작동 원리 이해")
    print("핵심: 문서 인덱싱 -> 쿼리 임베딩 -> 유사도 검색 -> 순위 정렬")
    
    # 검색 엔진 구조
    print_section_header("의미 기반 검색 엔진 구조", "[ARCH]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [STEP 1] 인덱싱 단계 (오프라인)                         │
  │  ─────────────────────────────────────────────────────  │
  │  문서들 -> 임베딩 생성 -> 벡터 저장                       │
  │                                                         │
  │  [STEP 2] 검색 단계 (온라인)                             │
  │  ─────────────────────────────────────────────────────  │
  │  1. 쿼리 입력                                           │
  │  2. 쿼리 임베딩 생성                                    │
  │  3. 저장된 벡터들과 유사도 계산                          │
  │  4. 상위 k개 결과 반환                                  │
  │                                                         │
  │  [!] 이것이 RAG의 "Retrieval" 부분!                      │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 검색 엔진 초기화
    search_engine = SimpleSearchEngine()
    
    # 샘플 문서 추가
    print_section_header("문서 인덱싱", "[LIST]")
    
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand human language.",
        "Computer vision enables machines to interpret visual information.",
        "Data science involves extracting insights from data.",
        "JavaScript is commonly used for web development.",
        "SQL is used for managing relational databases.",
        "Cloud computing provides on-demand computing resources.",
        "Cybersecurity protects systems from digital attacks.",
        "The weather is beautiful today with clear skies.",
        "I love eating pizza and pasta for dinner.",
        "Exercise and healthy eating are important for wellness.",
        "Traveling to new places broadens your perspective.",
        "Reading books is a great way to learn new things.",
    ]
    
    print("\n인덱싱할 문서:")
    for i, doc in enumerate(documents[:5], 1):
        print(f"  {i}. {doc}")
    print(f"  ... ({len(documents) - 5}개 더)")
    
    search_engine.add_documents(documents)
    print(f"\n[OK] 총 {len(documents)}개 문서 인덱싱 완료")
    
    # 다양한 쿼리로 검색
    print_section_header("검색 테스트", "[>>>]")
    
    queries = [
        "What is AI and machine learning?",
        "Tell me about programming languages",
        "How can I stay healthy?",
        "I want to learn about databases",
    ]
    
    for query in queries:
        search_engine.print_search_results(query, top_k=3)
    
    # 키워드 검색 vs 의미 검색 비교
    print_section_header("키워드 검색 vs 의미 검색", "[vs]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [CMP] 비교                                              │
  │  ─────────────────────────────────────────────────────  │
  │  키워드 검색 (BM25)         │ 의미 검색 (임베딩)          │
  │  ───────────────────────────┼────────────────────────── │
  │  "Python" 검색 시           │ "Python" 검색 시           │
  │  -> "Python" 포함 문서만    │ -> 프로그래밍 관련 문서도  │
  │                             │    (JavaScript, SQL 등)    │
  │  ───────────────────────────┼────────────────────────── │
  │  장점: 빠름, 정확한 키워드  │ 장점: 동의어, 유사 개념    │
  │  단점: 동의어 못 찾음       │ 단점: 임베딩 비용 필요     │
  │  ───────────────────────────┼────────────────────────── │
  │  [TIP] 실무: 둘을 결합한 Hybrid 검색 사용 (lab03 학습)  │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- 의미 검색: 키워드가 달라도 의미가 비슷하면 검색됨",
        "- 인덱싱: 문서를 임베딩으로 변환하여 저장 (1회)",
        "- 검색: 쿼리 임베딩 -> 유사도 계산 -> 순위 정렬",
        "- 한계: 대용량 데이터에서 느림 -> Vector DB 필요 (lab02)",
        "- 발전: RAG = 검색 + LLM 답변 생성 (lab03)"
    ], "검색 엔진 핵심 포인트")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """모든 데모 실행"""
    print("\n" + "="*80)
    print("[LAB] NLP 기초 실습")
    print("="*80)
    
    print("\n[LIST] 실습 항목:")
    print("  1. tiktoken으로 토큰 이해하기 - GPT가 텍스트를 어떻게 보는가")
    print("  2. NLTK 전처리 파이프라인 - 토큰화, 불용어, 표제어 추출")
    print("  3. OpenAI 임베딩 생성 - 텍스트를 벡터로 변환")
    print("  4. 코사인 유사도 계산 - 벡터 간 유사성 측정")
    print("  5. 간단한 검색 엔진 - 의미 기반 문장 검색")
    
    # NLTK 데이터 다운로드
    download_nltk_data()
    
    try:
        # 1. tiktoken 데모
        demo_tiktoken()
        
        # 2. 전처리 데모
        demo_preprocessing()
        
        # 3. 임베딩 데모
        demo_embeddings()
        
        # 4. 유사도 계산 데모
        demo_similarity()
        
        # 5. 검색 엔진 데모
        demo_search_engine()
        
        # 완료 메시지
        print("\n" + "="*80)
        print("[OK] 모든 실습 완료!")
        print("="*80)
        
        print("\n[INFO] 오늘 배운 내용 요약:")
        print("  ┌─────────────────────────────────────────────────────")
        print("  │ 1. 토큰: GPT가 텍스트를 처리하는 단위 (비용 계산 기준)")
        print("  │ 2. 전처리: 텍스트 정규화로 검색 품질 향상")
        print("  │ 3. 임베딩: 텍스트를 숫자 벡터로 변환")
        print("  │ 4. 코사인 유사도: 벡터 간 유사성 측정 (-1 ~ 1)")
        print("  │ 5. 의미 검색: 임베딩 기반으로 유사 문서 찾기")
        print("  └─────────────────────────────────────────────────────")
        
        print("\n[TIP] 다음 단계:")
        print("   - lab02/vector_db.py : Vector DB (ChromaDB)로 대용량 검색")
        print("   - lab03/rag_basic.py : RAG 시스템 구축 (검색 + LLM)")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
