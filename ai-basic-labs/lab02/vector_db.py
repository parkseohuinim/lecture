"""
Vector Database 실습 (ChromaDB)
- 임베딩의 개념과 생성
- Vector DB의 필요성과 역할
- 유사도 검색의 원리
- 메타데이터 필터링

실습 항목:
1. 임베딩(Embedding) 이해하기 - 텍스트를 벡터로 변환
2. Vector DB 기본 작업 - 저장과 검색
3. 거리와 유사도 이해하기 - 스코어 해석
4. 메타데이터 필터링 - 조건부 검색
5. 실전 예제 - 문서 관리 시스템
6. ANN 인덱스 알고리즘 이해하기 - HNSW, IVF 등 원리
7. 대용량 데이터 처리 전략 - 스케일링과 최적화

[!] Windows 사용자 주의:
    Python 3.13에서 ChromaDB 사용 시 segmentation fault 발생 가능
    
    해결 방법:
    1. Python 3.11 또는 3.12로 다운그레이드 (권장)
       - pyenv 또는 conda 사용 권장
    
    2. 또는 호환 버전 강제 설치:
       pip install chromadb==0.4.22 hnswlib==0.8.0 --force-reinstall
    
    3. 그래도 안 되면 WSL2 사용 고려
"""

import os
import sys
import platform
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 공통 유틸리티 import를 위한 경로 추가
sys.path.insert(0, str(project_root))
from utils import (
    print_section_header,
    print_subsection,
    print_key_points,
    visualize_similarity_bar,
    cosine_similarity,
    interpret_cosine_similarity,
    interpret_l2_distance,
    l2_distance_to_similarity,
    get_openai_client,
    COSINE_THRESHOLDS,
    L2_DISTANCE_THRESHOLDS
)


# ============================================================================
# 해석 기준 상수 (utils.py에서 import)
# ============================================================================
# Note: COSINE_THRESHOLDS, L2_DISTANCE_THRESHOLDS,
#       interpret_cosine_similarity, interpret_l2_distance 함수는
#       utils.py에 정의되어 있습니다.

# 유사도 막대 그래프 너비 (모든 실습에서 동일하게 사용)
SIMILARITY_BAR_WIDTH = 30


def truncate_text(text: str, max_len: int = 50) -> str:
    """텍스트를 지정 길이로 자르고 ... 추가"""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# ============================================================================
# 데이터 클래스
# ============================================================================

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    distance: float
    similarity: float
    metadata: Dict[str, Any]
    rank: int


# ============================================================================
# 1. 임베딩 생성기
# ============================================================================

class EmbeddingGenerator:
    """OpenAI 임베딩 생성기"""
    
    def __init__(self, api_key: str = None):
        """
        임베딩 생성기 초기화
        
        임베딩이란?
        - 텍스트를 고정 길이의 숫자 벡터로 변환하는 것
        - 의미적으로 유사한 텍스트는 유사한 벡터를 갖음
        - 벡터 간의 거리/유사도로 의미적 유사성 측정 가능
        """
        # 공통 헬퍼 사용 (SSL 인증서 검증 우회 포함)
        self.client = get_openai_client(api_key)
        self.model = "text-embedding-3-small"
        self.dimensions = 1536  # text-embedding-3-small의 기본 차원
    
    def get_embedding(self, text: str) -> List[float]:
        """단일 텍스트의 임베딩 생성"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return response.data[0].embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트의 임베딩을 배치로 생성 (효율적)"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [data.embedding for data in response.data]


# ============================================================================
# 2. ChromaDB 관리자
# ============================================================================

class ChromaDBManager:
    """ChromaDB 관리 클래스"""
    
    def __init__(self, persist_directory: str = None):
        """
        ChromaDB 클라이언트 초기화
        
        Vector DB란?
        - 임베딩 벡터를 효율적으로 저장하고 검색하는 데이터베이스
        - 일반 DB: 정확한 값 매칭 (WHERE id = 123)
        - Vector DB: 유사도 기반 검색 (가장 비슷한 벡터 찾기)
        
        Args:
            persist_directory: 데이터를 저장할 디렉토리 (None이면 lab02/chroma_db)
        """
        if persist_directory is None:
            persist_directory = str(Path(__file__).parent / "chroma_db")
        
        self.persist_directory = persist_directory
        
        # ChromaDB 클라이언트 생성 (로컬 파일 저장)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        print(f"[OK] ChromaDB 초기화 완료")
        print(f"   저장 위치: {persist_directory}")
    
    def create_collection(self, name: str, reset: bool = False, 
                          distance_fn: str = "l2") -> chromadb.Collection:
        """
        컬렉션 생성 또는 가져오기
        
        컬렉션 = 벡터들을 그룹화하는 단위 (일반 DB의 테이블과 유사)
        
        Args:
            name: 컬렉션 이름
            reset: True면 기존 컬렉션 삭제 후 재생성
            distance_fn: 거리 함수 ("l2", "cosine", "ip" 중 하나)
                - "l2": 유클리드 거리 (기본값, 0에 가까울수록 유사)
                - "cosine": 코사인 거리 (1 - 코사인 유사도, 0에 가까울수록 유사)
                - "ip": 내적의 음수 (정규화된 벡터에서 cosine과 동일)
        
        Returns:
            컬렉션 객체
        
        Note:
            Lab01에서 배운 코사인 유사도를 사용하려면 distance_fn="cosine" 설정.
            OpenAI 임베딩은 L2 정규화되어 있어 "l2"와 "cosine" 결과가 유사합니다.
        """
        if reset:
            try:
                self.client.delete_collection(name=name)
                print(f"   기존 컬렉션 '{name}' 삭제됨")
            except:
                pass
        
        # 거리 함수에 따른 메타데이터 설정
        metadata = {
            "description": "Vector database collection",
            "hnsw:space": distance_fn  # ChromaDB의 거리 함수 설정
        }
        
        collection = self.client.get_or_create_collection(
            name=name,
            metadata=metadata
        )
        
        print(f"   컬렉션 '{name}' 준비 완료 (문서 수: {collection.count()}, 거리: {distance_fn})")
        return collection
    
    def list_collections(self) -> List[str]:
        """모든 컬렉션 목록 조회"""
        collections = self.client.list_collections()
        return [col.name for col in collections]
    
    def delete_collection(self, name: str):
        """컬렉션 삭제"""
        self.client.delete_collection(name=name)
        print(f"   컬렉션 '{name}' 삭제 완료")


# ============================================================================
# 3. 문서 관리자 (실전 예제)
# ============================================================================

class DocumentManager:
    """문서 관리 시스템 - 실전 사용 패턴"""
    
    def __init__(self, collection_name: str = "documents", reset: bool = False):
        """
        Args:
            collection_name: 컬렉션 이름
            reset: True면 기존 데이터 삭제 후 시작
        """
        self.db = ChromaDBManager()
        self.collection = self.db.create_collection(collection_name, reset=reset)
        self.embedder = EmbeddingGenerator()
    
    def add_document(self, text: str, metadata: Dict[str, Any]) -> str:
        """문서 추가"""
        doc_id = f"doc_{self.collection.count()}"
        embedding = self.embedder.get_embedding(text)
        
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[metadata]
        )
        
        return doc_id
    
    def add_documents_batch(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> List[str]:
        """여러 문서 일괄 추가 (배치 임베딩으로 효율적)"""
        start_idx = self.collection.count()
        doc_ids = [f"doc_{start_idx + i}" for i in range(len(texts))]
        embeddings = self.embedder.get_embeddings_batch(texts)
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=doc_ids,
            metadatas=metadatas
        )
        
        return doc_ids
    
    def search(self, query: str, n_results: int = 5, 
               where: Optional[Dict] = None) -> List[SearchResult]:
        """문서 검색"""
        query_embedding = self.embedder.get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
        
        search_results = []
        for i, (doc, dist, meta) in enumerate(zip(
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        )):
            # L2 거리를 유사도로 변환
            similarity = 1 / (1 + dist)
            
            search_results.append(SearchResult(
                content=doc,
                distance=dist,
                similarity=similarity,
                metadata=meta,
                rank=i + 1
            ))
        
        return search_results
    
    def get_stats(self) -> Dict:
        """컬렉션 통계"""
        return {
            'total_documents': self.collection.count(),
            'collection_name': self.collection.name
        }


# ============================================================================
# 데모 함수들
# ============================================================================

def demo_embedding_basics():
    """실습 1: 임베딩(Embedding) 이해하기"""
    print("\n" + "="*80)
    print("[1] 실습 1: 임베딩(Embedding) 이해하기")
    print("="*80)
    print("목표: 텍스트가 어떻게 숫자 벡터로 변환되는지 이해")
    print("핵심: 의미가 비슷한 텍스트 -> 비슷한 벡터 -> 가까운 거리")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    embedder = EmbeddingGenerator()
    
    # 1. 임베딩 생성 기본
    print_section_header("임베딩 생성 기본", "[INFO]")
    
    text = "Python is a programming language"
    embedding = embedder.get_embedding(text)
    
    print(f"\n입력 텍스트: '{text}'")
    print(f"\n임베딩 결과:")
    print(f"  * 벡터 차원: {len(embedding)}")
    print(f"  * 처음 5개 값: {[round(v, 4) for v in embedding[:5]]}")
    print(f"  * 마지막 5개 값: {[round(v, 4) for v in embedding[-5:]]}")
    print(f"  * 값의 범위: [{min(embedding):.4f}, {max(embedding):.4f}]")
    
    # 벡터 값 의미 설명 추가
    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] 벡터 값의 의미                                    │
  │  ─────────────────────────────────────────────────────  │
  │  * 각 차원은 텍스트의 특정 "의미적 특징"을 나타냄        │
  │  * 예: 1번 차원 = "기술 관련성", 2번 = "감정" 등         │
  │  * 실제로는 사람이 해석하기 어려운 추상적 특징           │
  │  * 중요한 것: 비슷한 의미 = 비슷한 벡터 패턴            │
  │  * 개별 값보다 전체 벡터의 "방향"이 의미를 결정          │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 2. 의미적 유사성 데모
    print_section_header("의미적 유사성 테스트", "[*]")
    
    texts = [
        "I love programming in Python",      # 프로그래밍
        "Python is great for coding",        # 프로그래밍 (유사)
        "I enjoy cooking Italian food",      # 요리 (다름)
        "The weather is sunny today",        # 날씨 (완전 다름)
    ]
    
    print("\n비교할 텍스트:")
    for i, t in enumerate(texts, 1):
        print(f"  {i}. {t}")
    
    # 임베딩 생성
    embeddings = embedder.get_embeddings_batch(texts)
    
    # 첫 번째 텍스트와 나머지의 유사도 계산 (코사인 유사도)
    # Note: cosine_similarity 함수는 utils.py에서 import
    print_subsection("첫 번째 텍스트와의 코사인 유사도")
    
    base_embedding = embeddings[0]
    print(f"\n기준: '{texts[0]}'")
    print()
    
    for i, (text, emb) in enumerate(zip(texts[1:], embeddings[1:]), 2):
        similarity = cosine_similarity(base_embedding, emb)
        bar = visualize_similarity_bar(similarity, SIMILARITY_BAR_WIDTH)
        interpretation = interpret_cosine_similarity(similarity)
        
        print(f"  vs '{text}'")
        print(f"     {bar} {similarity:.4f} {interpretation}")
        print()
    
    # 코사인 유사도 vs L2 거리 설명
    print_section_header("코사인 유사도 vs L2 거리", "[vs]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [CMP] 두 가지 유사도 측정 방법                          │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  1. 코사인 유사도 (Cosine Similarity)                   │
  │     * 벡터의 "방향"만 비교                              │
  │     * 값 범위: -1 ~ 1 (1이 가장 유사)                   │
  │     * 문서 길이에 영향 받지 않음                        │
  │     * Lab01에서 직접 계산한 방식                        │
  │                                                         │
  │  2. L2 거리 (Euclidean Distance)                       │
  │     * 벡터 간의 "직선 거리"                             │
  │     * 값 범위: 0 ~ ∞ (0이 가장 유사)                    │
  │     * ChromaDB의 기본 설정                              │
  │     * 이 Lab에서 사용할 방식                            │
  │                                                         │
  │  [!] OpenAI 임베딩은 L2 정규화되어 있어서               │
  │      두 방식 모두 비슷한 결과를 제공합니다              │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # ChromaDB에서 거리 함수 설정 방법
    print_subsection("ChromaDB 거리 함수 설정 방법")
    print("""
  [CODE] 컬렉션 생성 시 거리 함수 지정:
  ┌─────────────────────────────────────────────────────
  │ # L2 거리 (기본값)
  │ collection = client.create_collection(
  │     name="my_collection",
  │     metadata={"hnsw:space": "l2"}
  │ )
  │
  │ # 코사인 거리 (Lab01과 동일한 방식)
  │ collection = client.create_collection(
  │     name="my_collection",
  │     metadata={"hnsw:space": "cosine"}
  │ )
  │
  │ # 내적 (정규화된 벡터에서 cosine과 동일)
  │ collection = client.create_collection(
  │     name="my_collection",
  │     metadata={"hnsw:space": "ip"}
  │ )
  └─────────────────────────────────────────────────────

  [TIP] 언제 어떤 거리를 사용할까?
  * L2 (기본): 대부분의 경우 무난함
  * Cosine: 문서 길이가 다양할 때, 방향만 중요할 때
  * IP: 정규화된 벡터 + 빠른 계산이 필요할 때
  
  [!] 주의: 컬렉션 생성 후에는 거리 함수 변경 불가!
      변경하려면 컬렉션을 삭제하고 다시 생성해야 합니다.
  
  ────────────────────────────────────────────────────────────
  [!] Lab01(cosine) vs Lab02(L2) 혼동 방지:
  
  OpenAI 임베딩은 L2 정규화되어 있어서
  cosine, inner product, L2는 대부분 동일한 "순위" 결과를 줍니다.
  차이는 스케일 표현 방식일 뿐이며, 핵심은 "랭킹 순서"입니다.
  
  즉, "어떤 거리를 써도 1등은 같다!" (스코어 수치만 다름)
  ────────────────────────────────────────────────────────────
    """)
    
    # 핵심 포인트
    print_key_points([
        "- 임베딩: 텍스트 -> 고정 길이 숫자 벡터 (예: 1536차원)",
        "- OpenAI text-embedding-3-small: 성능과 비용의 균형",
        "- 의미가 비슷한 텍스트 -> 벡터 공간에서 가까이 위치",
        "- 코사인 유사도: 방향 비교 (-1 ~ 1), L2 거리: 직선 거리 (0 ~ ∞)",
        "- ChromaDB 기본: L2 거리 사용 (다음 실습에서 확인)"
    ])


def demo_basic_operations():
    """실습 2: Vector DB 기본 작업"""
    print("\n" + "="*80)
    print("[2] 실습 2: Vector DB 기본 작업")
    print("="*80)
    print("목표: ChromaDB에 벡터 저장하고 검색하는 기본 흐름 이해")
    print("핵심: 문서 추가(Add) -> 검색(Query) -> 결과 해석")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 1. DB 초기화 및 컬렉션 생성
    print_section_header("데이터베이스 초기화", "[DB]")
    
    db = ChromaDBManager()
    collection = db.create_collection("demo_basic", reset=True)
    
    # 2. 임베딩 생성기
    embedder = EmbeddingGenerator()
    
    # 3. 샘플 문서 추가
    print_section_header("문서 추가 (Add)", "[DOC]")
    
    documents = [
        "Python is a high-level programming language known for its simplicity.",
        "Machine learning is a subset of artificial intelligence that learns from data.",
        "ChromaDB is an open-source vector database for AI applications.",
        "Deep learning uses neural networks with multiple layers.",
        "JavaScript is widely used for web development.",
    ]
    
    print(f"\n추가할 문서 ({len(documents)}개):")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {truncate_text(doc, 50)}")
    
    # 배치 임베딩 생성 (효율적)
    print("\n[...] 임베딩 생성 중...")
    embeddings = embedder.get_embeddings_batch(documents)
    
    # ChromaDB에 추가
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=[{"source": "demo", "index": i} for i in range(len(documents))]
    )
    
    print(f"[OK] {len(documents)}개 문서 추가 완료")
    print(f"   컬렉션 크기: {collection.count()}개")
    
    # 4. 검색 테스트
    print_section_header("유사도 검색 (Query)", "[>>>]")
    
    query = "What is a vector database?"
    print(f"\n쿼리: '{query}'")
    
    # 쿼리 임베딩 생성
    query_embedding = embedder.get_embedding(query)
    
    # 검색 실행
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    print(f"\n상위 {len(results['documents'][0])}개 결과:")
    print(f"{'─'*60}")
    
    for i, (doc, distance, metadata) in enumerate(zip(
        results['documents'][0],
        results['distances'][0],
        results['metadatas'][0]
    ), 1):
        similarity = 1 / (1 + distance)
        bar = visualize_similarity_bar(similarity, SIMILARITY_BAR_WIDTH)
        interpretation = interpret_l2_distance(distance)
        
        print(f"\n[{i}위] 거리: {distance:.4f} | 유사도: {similarity:.4f} {interpretation}")
        print(f"     {bar}")
        print(f"     문서: {doc}")
        print(f"     메타: {metadata}")
    
    # 거리 스코어 해석 가이드 추가
    print_section_header("거리 스코어 해석 가이드", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] L2 거리 스코어 해석 (이 실습 데이터 기준 예시)     │
  │  ─────────────────────────────────────────────────────  │
  │  * OpenAI 임베딩 + L2 거리 기준 참고 범위:               │
  │                                                         │
  │    거리 0.0 ~ 0.5  ->  매우 높은 관련성 (거의 동일)      │
  │    거리 0.5 ~ 1.0  ->  높은 관련성 (주제 일치)           │
  │    거리 1.0 ~ 1.5  ->  중간 관련성 (관련 있음)           │
  │    거리 1.5 ~ 2.0  ->  낮은 관련성 (약간 관련)           │
  │    거리 2.0 이상   ->  거의 무관 (다른 주제)             │
  │                                                         │
  │  ⚠️ [!] 중요: 이 수치는 "이 실습 데이터셋" 기준 예시!    │
  │  ─────────────────────────────────────────────────────  │
  │  실무에서 L2 거리 스케일은 다음에 따라 완전히 달라집니다: │
  │  * 데이터 분포 (문서들이 얼마나 다양한지)                │
  │  * 문서 길이 (긴 문서 vs 짧은 문서)                     │
  │  * 임베딩 모델 (small vs large, 다른 제공자)            │
  │  * 정규화 여부 (L2 norm = 1인지 아닌지)                 │
  │                                                         │
  │  → 반드시 자신의 데이터셋으로 히스토그램/분포를 먼저     │
  │    분석한 뒤 임계값을 정해야 합니다!                     │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- Vector DB: 임베딩 벡터를 저장하고 유사도로 검색",
        "- Collection: 벡터들의 그룹 (일반 DB의 테이블)",
        "- Add: 문서 + 임베딩 + 메타데이터 저장",
        "- Query: 쿼리 임베딩과 가장 가까운 벡터 검색",
        "- 거리 해석: 절대 수치보다 상대 순위가 중요!",
        "- [!] L2 거리 임계값은 데이터셋마다 다름 → 직접 분석 필요"
    ])


def demo_distance_scores():
    """실습 3: 거리와 유사도 이해하기"""
    print("\n" + "="*80)
    print("[3] 실습 3: 거리와 유사도 이해하기")
    print("="*80)
    print("목표: ChromaDB의 거리 스코어가 무엇을 의미하는지 이해")
    print("핵심: 거리 DOWN = 유사도 UP = 더 관련성 높음")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 초기화
    db = ChromaDBManager()
    collection = db.create_collection("demo_scores", reset=True)
    embedder = EmbeddingGenerator()
    
    # 다양한 관련성을 가진 문서들
    print_section_header("테스트 문서 준비", "[LIST]")
    
    documents = [
        "Python programming is fun and easy to learn.",          # 프로그래밍
        "I write Python code every day for data analysis.",      # 프로그래밍 (유사)
        "Machine learning with Python is powerful.",             # 프로그래밍 + ML
        "Italian pasta is my favorite food.",                    # 요리 (다름)
        "The stock market showed gains today.",                  # 경제 (완전 다름)
    ]
    
    print("\n테스트 문서:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    
    # 임베딩 및 저장
    embeddings = embedder.get_embeddings_batch(documents)
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )
    
    # 검색 및 스코어 분석
    print_section_header("거리 스코어 분석", "[DATA]")
    
    query = "Python programming language"
    query_embedding = embedder.get_embedding(query)
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=len(documents)
    )
    
    print(f"\n쿼리: '{query}'")
    
    # 해석 기준 먼저 표시
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [INFO] 해석 기준 (⚠️ 이 실습 데이터 한정 예시)          │
  │  ─────────────────────────────────────────────────────  │
  │  * [v] 높음: L2 거리 < 1.0 (유사도 > 0.50) - 주제 일치  │
  │  * [~] 중간: L2 거리 1.0~1.8 (유사도 0.35~0.50) - 관련성│
  │  * [x] 낮음: L2 거리 > 1.8 (유사도 < 0.35) - 다른 주제  │
  │                                                         │
  │  ⚠️ 절대 일반화 금지! L2 거리 1.2가 "항상 중간"이 아님! │
  │  → 데이터 분포/문서 길이/임베딩 모델에 따라 달라집니다  │
  │  → 실무에서는 자신의 데이터로 히스토그램 분석 후 결정!  │
  └─────────────────────────────────────────────────────────┘
    """)
    
    print(f"{'─'*70}")
    print(f"{'순위':<4} {'L2 거리':<10} {'유사도':<10} {'해석':<12} 문서")
    print(f"{'─'*70}")
    
    for i, (doc, distance) in enumerate(zip(
        results['documents'][0],
        results['distances'][0]
    ), 1):
        # L2 거리를 0~1 유사도로 변환
        similarity = 1 / (1 + distance)
        
        # 해석 (전역 함수 사용으로 일관성 유지)
        interpretation = interpret_l2_distance(distance)
        
        # 시각화
        bar = visualize_similarity_bar(similarity, SIMILARITY_BAR_WIDTH)
        
        print(f"{i:<4} {distance:<10.4f} {similarity:<10.4f} {interpretation:<12} {truncate_text(doc, 35)}")
        print(f"     {bar}")
        print()
    
    # 거리 메트릭 설명
    print_section_header("거리 메트릭 이해하기", "[CALC]")
    
    print("""
  ChromaDB는 기본적으로 L2 거리(유클리드 거리)를 사용합니다.
  
  ┌─────────────────────────────────────────────────────────┐
  │  L2 거리 (Euclidean Distance)                          │
  │  ─────────────────────────────────────────────────────  │
  │  * 두 벡터 간의 직선 거리                               │
  │  * 공식: sqrt(sum((a[i] - b[i])^2))                    │
  │  * 값 범위: 0 ~ infinity                               │
  │  * 0에 가까울수록 유사                                  │
  └─────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────┐
  │  거리 -> 유사도 변환 (시각화용 휴리스틱)                 │
  │  ─────────────────────────────────────────────────────  │
  │  * 방법 1: similarity = 1 / (1 + distance)             │
  │  * 방법 2: similarity = 1 - (distance / max_distance)  │
  │  * 결과: 0 ~ 1 범위 (1이 가장 유사)                    │
  │                                                         │
  │  ⚠️ [!] 중요: 이것은 "표준 변환 공식"이 아닙니다!       │
  │  ─────────────────────────────────────────────────────  │
  │  * 위 공식들은 "상대적 점수 비교용 시각화 함수"일 뿐    │
  │  * L2 거리에는 코사인 유사도처럼 표준 정규화 변환식이   │
  │    존재하지 않습니다                                    │
  │  * 실제 검색 순위는 "거리 자체의 상대적 크기"로 판단    │
  │                                                         │
  │  [TIP] 실무에서의 사용:                                  │
  │  * L2 거리 → 그대로 랭킹용으로 사용 (작을수록 상위)     │
  │  * Cosine → score 자체를 threshold로 사용 가능         │
  │  * 중요한 것은 "절대 수치"가 아닌 "상대 순위"          │
  └─────────────────────────────────────────────────────────┘
  
  ────────────────────────────────────────────────────────────
  [!] "유사도 0.43밖에 안 되는데 괜찮은 거야?" 에 대한 답변:
  
  실무 RAG에서는 0.35~0.55 구간의 문서들도
  상위 컨텍스트로 매우 자주 사용됩니다!
  
  * 0.5 이상만 사용하면 오히려 관련 문서를 놓칠 수 있음
  * Top-3~5 결과 중 가장 좋은 것을 LLM이 판단하게 하는 것이 일반적
  * 절대 점수보다 "다른 결과 대비 상대적으로 높은가"가 핵심
  ────────────────────────────────────────────────────────────
    """)
    
    # 핵심 포인트
    print_key_points([
        "- L2 거리: 벡터 간 직선 거리, 0이 가장 가까움",
        "- 코사인 유사도: 벡터 방향의 유사성, 1이 가장 유사",
        "- 거리 DOWN = 유사도 UP = 검색 순위 UP",
        "- 실무: 유사도 0.35~0.55도 충분히 유용한 결과!",
        "- 핵심: 절대 점수보다 상대 순위가 중요",
        "- ChromaDB 기본: L2 거리 사용 (cosine으로 변경 가능)"
    ])


def demo_metadata_filtering():
    """실습 4: 메타데이터 필터링"""
    print("\n" + "="*80)
    print("[4] 실습 4: 메타데이터 필터링")
    print("="*80)
    print("목표: 벡터 검색에 조건을 추가하여 정밀한 검색 수행")
    print("핵심: 의미 검색 + 메타데이터 필터 = 더 정확한 결과")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 초기화
    db = ChromaDBManager()
    collection = db.create_collection("demo_metadata", reset=True)
    embedder = EmbeddingGenerator()
    
    # 메타데이터가 있는 문서들
    print_section_header("메타데이터가 있는 문서 추가", "[LIST]")
    
    documents = [
        {"text": "Python basics for beginners", "metadata": {"language": "python", "level": "beginner", "year": 2024}},
        {"text": "Advanced Python programming", "metadata": {"language": "python", "level": "advanced", "year": 2024}},
        {"text": "JavaScript for web developers", "metadata": {"language": "javascript", "level": "intermediate", "year": 2023}},
        {"text": "React framework tutorial", "metadata": {"language": "javascript", "level": "intermediate", "year": 2024}},
        {"text": "Python for data science", "metadata": {"language": "python", "level": "intermediate", "year": 2023}},
        {"text": "Java programming fundamentals", "metadata": {"language": "java", "level": "beginner", "year": 2023}},
    ]
    
    print("\n추가된 문서:")
    print(f"{'─'*70}")
    for i, doc in enumerate(documents, 1):
        meta = doc["metadata"]
        print(f"  {i}. {doc['text']}")
        print(f"     +-- {meta}")
    
    # 문서 추가
    texts = [d["text"] for d in documents]
    metadatas = [d["metadata"] for d in documents]
    embeddings = embedder.get_embeddings_batch(texts)
    
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=[f"doc_{i}" for i in range(len(documents))],
        metadatas=metadatas
    )
    
    # 쿼리 준비
    query = "programming tutorial"
    query_embedding = embedder.get_embedding(query)
    
    # 1. 필터 없이 검색
    print_section_header("필터 없이 검색", "[>>>]")
    print(f"\n쿼리: '{query}'")
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    print("\n결과:")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        similarity = 1 / (1 + dist)
        print(f"  [{i}] {doc}")
        print(f"      메타: {meta} | 유사도: {similarity:.4f}")
    
    # 2. 단일 필터
    print_section_header("필터: language='python'", "[FILTER]")
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"language": "python"}
    )
    
    print("\n결과 (Python만):")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        similarity = 1 / (1 + dist)
        print(f"  [{i}] {doc}")
        print(f"      메타: {meta} | 유사도: {similarity:.4f}")
    
    # 3. 복합 필터 ($and)
    print_section_header("필터: language='python' AND level='beginner'", "[*]")
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={
            "$and": [
                {"language": "python"},
                {"level": "beginner"}
            ]
        }
    )
    
    print("\n결과 (Python + 초급):")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        similarity = 1 / (1 + dist)
        print(f"  [{i}] {doc}")
        print(f"      메타: {meta} | 유사도: {similarity:.4f}")
    
    # 4. 비교 연산자
    print_section_header("필터: year >= 2024", "[DATE]")
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"year": {"$gte": 2024}}
    )
    
    print("\n결과 (2024년 이후):")
    for i, (doc, meta, dist) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    ), 1):
        similarity = 1 / (1 + dist)
        print(f"  [{i}] {doc}")
        print(f"      메타: {meta} | 유사도: {similarity:.4f}")
    
    # 필터 연산자 설명
    print_section_header("지원하는 필터 연산자", "[REF]")
    
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  비교 연산자                                            │
  │  ─────────────────────────────────────────────────────  │
  │  * $eq   : 같음          {"field": {"$eq": "value"}}   │
  │  * $ne   : 같지 않음      {"field": {"$ne": "value"}}   │
  │  * $gt   : 크다          {"field": {"$gt": 10}}        │
  │  * $gte  : 크거나 같다    {"field": {"$gte": 10}}       │
  │  * $lt   : 작다          {"field": {"$lt": 10}}        │
  │  * $lte  : 작거나 같다    {"field": {"$lte": 10}}       │
  │  * $in   : 포함          {"field": {"$in": ["a","b"]}} │
  └─────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────┐
  │  논리 연산자                                            │
  │  ─────────────────────────────────────────────────────  │
  │  * $and  : 모두 만족      {"$and": [조건1, 조건2]}      │
  │  * $or   : 하나만 만족    {"$or": [조건1, 조건2]}       │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- 메타데이터: 문서와 함께 저장되는 구조화된 정보",
        "- 벡터 검색 + 필터 = 의미 검색 + 조건 검색 결합",
        "- 실무 활용: 날짜, 카테고리, 사용자별 필터링",
        "- 주의: 필터가 너무 제한적이면 결과 없을 수 있음"
    ])


def demo_document_manager():
    """실습 5: 실전 예제 - 문서 관리 시스템"""
    print("\n" + "="*80)
    print("[5] 실습 5: 실전 예제 - 문서 관리 시스템")
    print("="*80)
    print("목표: 실제 애플리케이션에서 Vector DB를 활용하는 패턴 익히기")
    print("핵심: 클래스 설계, 배치 처리, 검색 결과 가공")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 문서 관리자 초기화
    print_section_header("DocumentManager 초기화", "[SETUP]")
    
    doc_manager = DocumentManager("knowledge_base", reset=True)
    
    # 샘플 문서 추가
    print_section_header("지식 베이스 구축", "[LIST]")
    
    knowledge_base = [
        {
            "text": "Python은 읽기 쉬운 문법과 강력한 라이브러리로 데이터 과학과 웹 개발에 널리 사용됩니다.",
            "metadata": {"topic": "programming", "language": "python", "difficulty": "beginner"}
        },
        {
            "text": "머신러닝은 데이터에서 패턴을 학습하여 예측하는 인공지능의 한 분야입니다.",
            "metadata": {"topic": "ai", "subtopic": "machine_learning", "difficulty": "intermediate"}
        },
        {
            "text": "딥러닝은 다층 신경망을 사용하여 복잡한 패턴을 학습하는 머신러닝의 하위 분야입니다.",
            "metadata": {"topic": "ai", "subtopic": "deep_learning", "difficulty": "advanced"}
        },
        {
            "text": "React는 Facebook에서 만든 사용자 인터페이스 구축을 위한 JavaScript 라이브러리입니다.",
            "metadata": {"topic": "programming", "language": "javascript", "difficulty": "intermediate"}
        },
        {
            "text": "SQL은 관계형 데이터베이스에서 데이터를 관리하고 조회하는 표준 언어입니다.",
            "metadata": {"topic": "database", "language": "sql", "difficulty": "beginner"}
        },
        {
            "text": "Vector DB는 고차원 벡터를 효율적으로 저장하고 유사도 검색을 수행하는 데이터베이스입니다.",
            "metadata": {"topic": "database", "subtopic": "vector_db", "difficulty": "intermediate"}
        },
    ]
    
    print(f"\n추가할 문서: {len(knowledge_base)}개")
    
    # 배치 추가
    texts = [d["text"] for d in knowledge_base]
    metadatas = [d["metadata"] for d in knowledge_base]
    
    doc_ids = doc_manager.add_documents_batch(texts, metadatas)
    
    print(f"[OK] 문서 추가 완료")
    print(f"   통계: {doc_manager.get_stats()}")
    
    # 다양한 검색 시나리오
    search_scenarios = [
        {
            "name": "기본 검색",
            "query": "프로그래밍을 배우고 싶어요",
            "filter": None
        },
        {
            "name": "AI 관련 검색",
            "query": "인공지능의 학습 방법은?",
            "filter": {"topic": "ai"}
        },
        {
            "name": "초급자용 콘텐츠",
            "query": "쉽게 시작할 수 있는 언어",
            "filter": {"difficulty": "beginner"}
        },
    ]
    
    for scenario in search_scenarios:
        print_section_header(f"검색: {scenario['name']}", "[>>>]")
        print(f"\n쿼리: '{scenario['query']}'")
        if scenario['filter']:
            print(f"필터: {scenario['filter']}")
        
        results = doc_manager.search(
            query=scenario['query'],
            n_results=3,
            where=scenario['filter']
        )
        
        print(f"\n결과:")
        for result in results:
            bar = visualize_similarity_bar(result.similarity, SIMILARITY_BAR_WIDTH)
            interpretation = interpret_l2_distance(result.distance)
            print(f"\n  [{result.rank}위] 유사도: {result.similarity:.4f} {interpretation}")
            print(f"       {bar}")
            print(f"       [DOC] {truncate_text(result.content, 60)}")
            print(f"       [TAG] {result.metadata}")
    
    # 한글 토큰 효율성 참고
    print_section_header("한글 쿼리와 토큰 효율성", "[TIP]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] 참고: 이 실습에서 한글 쿼리를 사용했습니다         │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  토큰 비교 (대략적인 예시):                              │
  │  * '프로그래밍을 배우고 싶어요' ≈ 14 토큰               │
  │  * 'I want to learn programming' ≈ 6 토큰              │
  │                                                         │
  │  → 한글은 영어보다 2~3배 많은 토큰 사용                 │
  │  → API 비용은 토큰 기준이므로 고려 필요                 │
  │  → 하지만 검색 품질은 동일하게 좋습니다!                │
  │                                                         │
  │  [TIP] 대용량 한글 문서 처리 시:                        │
  │  * 청킹(chunking) 크기 조정 필요                        │
  │  * 토큰 제한을 고려한 문서 분할                         │
  │  * lab01에서 배운 토큰 계산 활용                        │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- 클래스로 캡슐화하면 재사용성과 유지보수성 향상",
        "- 배치 임베딩 (get_embeddings_batch)으로 API 호출 최소화",
        "- 메타데이터를 잘 설계하면 다양한 검색 시나리오 대응 가능",
        "- 유사도 점수를 시각화하면 결과 품질을 직관적으로 파악",
        "- 한글은 영어보다 토큰 소모량이 많음 (비용 고려 필요)"
    ])


def demo_index_algorithms():
    """실습 6: ANN 인덱스 알고리즘 이해하기"""
    print("\n" + "="*80)
    print("[6] 실습 6: ANN 인덱스 알고리즘 이해하기")
    print("="*80)
    print("목표: Vector DB가 빠른 검색을 가능하게 하는 알고리즘 원리 이해")
    print("핵심: 정확도와 속도 사이의 Trade-off")
    
    # ANN이 필요한 이유
    print_section_header("왜 ANN(Approximate Nearest Neighbor)이 필요한가?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] 문제: 정확한 검색(Exact Search)은 느리다           │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  정확한 검색 (Brute-Force):                             │
  │  * 모든 벡터와 거리 계산 → O(N × D)                     │
  │  * N = 벡터 개수, D = 차원 수                           │
  │                                                         │
  │  예시 (1536차원, 다양한 데이터셋):                       │
  │  ┌──────────────┬─────────────┬───────────────────────┐ │
  │  │ 벡터 개수    │ 브루트포스  │ 실시간 서비스 가능?   │ │
  │  ├──────────────┼─────────────┼───────────────────────┤ │
  │  │ 1,000개      │ ~1ms        │ ✓ 가능               │ │
  │  │ 10,000개     │ ~10ms       │ ✓ 가능 (약간 느림)   │ │
  │  │ 100,000개    │ ~100ms      │ △ 경계선             │ │
  │  │ 1,000,000개  │ ~1초        │ ✗ 불가능             │ │
  │  │ 10,000,000개 │ ~10초       │ ✗ 완전 불가          │ │
  │  └──────────────┴─────────────┴───────────────────────┘ │
  │  (※ 실제 시간은 하드웨어/구현에 따라 다름)              │
  │                                                         │
  │  [>>>] 해결책: ANN (근사 최근접 이웃)                   │
  │  * 100% 정확하지는 않지만, 훨씬 빠름                    │
  │  * 보통 95~99% 정확도로 1000배 빠른 검색 가능          │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 주요 인덱스 알고리즘 설명
    print_section_header("주요 ANN 인덱스 알고리즘", "[ALGO]")
    
    # 1. HNSW
    print_subsection("1. HNSW (Hierarchical Navigable Small World)")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  HNSW - ChromaDB의 기본 인덱스                          │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  [원리] 계층적 그래프 구조                              │
  │                                                         │
  │     Layer 2 (상위):  ●───────────────●  (노드 적음)     │
  │                       │               │                 │
  │     Layer 1 (중간):  ●───●───●───●───●  (노드 중간)     │
  │                       │   │   │   │   │                 │
  │     Layer 0 (하위):  ●●●●●●●●●●●●●●●●●●  (모든 노드)     │
  │                                                         │
  │  [검색 과정]                                            │
  │  1. 최상위 레이어에서 시작 (노드 적음 → 빠른 탐색)      │
  │  2. 가까운 이웃으로 이동하며 하위 레이어로 내려감       │
  │  3. 최하위 레이어에서 정밀 검색                         │
  │                                                         │
  │  [장점]                                                 │
  │  * 검색 속도: O(log N) - 매우 빠름                      │
  │  * 정확도: 95~99% (파라미터 조정 가능)                  │
  │  * 메모리 효율: 적당함                                  │
  │  * 동적 추가/삭제 가능                                  │
  │                                                         │
  │  [단점]                                                 │
  │  * 인덱스 구축 시간 오래 걸림                           │
  │  * 메모리 사용량이 데이터보다 큼 (그래프 저장)          │
  │                                                         │
  │  [ChromaDB 설정 예시]                                   │
  │  collection = client.create_collection(                 │
  │      name="my_collection",                              │
  │      metadata={                                         │
  │          "hnsw:space": "cosine",                        │
  │          "hnsw:M": 16,           # 연결 수 (기본 16)    │
  │          "hnsw:ef_construction": 100  # 구축 시 탐색 폭 │
  │      }                                                  │
  │  )                                                      │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 2. IVF
    print_subsection("2. IVF (Inverted File Index)")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  IVF - 클러스터 기반 인덱스                             │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  [원리] 벡터들을 클러스터로 그룹화                      │
  │                                                         │
  │        ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐              │
  │        │ C1  │  │ C2  │  │ C3  │  │ C4  │  ← Centroids │
  │        │●●●● │  │●●●●●│  │●●●  │  │●●●●●│  ← Vectors   │
  │        └─────┘  └─────┘  └─────┘  └─────┘              │
  │                                                         │
  │  [검색 과정]                                            │
  │  1. 쿼리와 가장 가까운 클러스터 중심(Centroid) 찾기     │
  │  2. 해당 클러스터 내에서만 검색 (nprobe 개수만큼)       │
  │  3. 전체 검색 대비 훨씬 적은 벡터만 비교                │
  │                                                         │
  │  [장점]                                                 │
  │  * 메모리 효율적 (HNSW보다 적은 메모리)                 │
  │  * 대용량 데이터에 적합                                 │
  │  * 구축 속도 빠름                                       │
  │                                                         │
  │  [단점]                                                 │
  │  * HNSW보다 정확도 낮을 수 있음                         │
  │  * 클러스터 경계에 있는 벡터 놓칠 수 있음               │
  │  * nprobe 값에 따라 성능 변동                           │
  │                                                         │
  │  [Faiss 설정 예시]                                      │
  │  index = faiss.IndexIVFFlat(                            │
  │      quantizer,                                         │
  │      dimension,                                         │
  │      nlist=100    # 클러스터 개수                       │
  │  )                                                      │
  │  index.nprobe = 10  # 검색할 클러스터 수                │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 3. PQ (Product Quantization)
    print_subsection("3. PQ (Product Quantization) - 압축 기법")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  PQ - 벡터 압축으로 메모리 절약                         │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  [원리] 고차원 벡터를 작은 조각으로 나누어 양자화       │
  │                                                         │
  │  원본 벡터 (1536차원):                                  │
  │  [0.1, 0.3, ..., 0.2, 0.5, ..., 0.4, 0.1, ...]         │
  │    └── 192차원 ──┘└── 192차원 ──┘└── 192차원 ──┘       │
  │                                                         │
  │  각 조각을 코드북의 ID로 변환:                          │
  │  [code_23,    code_156,    code_89,    ...]            │
  │                                                         │
  │  [효과]                                                 │
  │  * 1536차원 × 4바이트 = 6144바이트/벡터                │
  │  * PQ 압축 후: 8~64바이트/벡터 (100배 압축!)            │
  │                                                         │
  │  [장점]                                                 │
  │  * 메모리 사용량 대폭 감소                              │
  │  * 10억 개 벡터도 메모리에 적재 가능                    │
  │                                                         │
  │  [단점]                                                 │
  │  * 정확도 손실 (압축 손실)                              │
  │  * 보통 IVF와 결합해서 사용 (IVFPQ)                     │
  │                                                         │
  │  [실무 조합] IVF + PQ = IVFPQ                           │
  │  * IVF로 후보 클러스터 선정                             │
  │  * PQ로 압축된 벡터와 거리 계산                         │
  │  * 대용량 + 제한된 메모리 환경에 최적                   │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 알고리즘 비교표
    print_section_header("인덱스 알고리즘 비교", "[vs]")
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  [CMP] 인덱스 알고리즘 비교표                                            │
  │  ─────────────────────────────────────────────────────────────────────  │
  │                                                                         │
  │  알고리즘   │ 검색속도 │ 정확도 │ 메모리 │ 구축속도 │ 동적추가 │ 용도   │
  │  ──────────┼─────────┼───────┼───────┼─────────┼─────────┼────────│
  │  Brute     │ O(N)    │ 100%  │ 낮음  │ 없음    │ ✓       │ 소량   │
  │  HNSW      │ O(logN) │ 높음  │ 높음  │ 느림    │ ✓       │ 범용   │
  │  IVF       │ O(N/k)  │ 중간  │ 중간  │ 빠름    │ △       │ 대용량 │
  │  IVFPQ     │ O(N/k)  │ 낮음  │ 낮음  │ 빠름    │ △       │ 초대용 │
  │  ──────────┴─────────┴───────┴───────┴─────────┴─────────┴────────│
  │                                                                         │
  │  [TIP] 선택 가이드:                                                     │
  │  * < 10만 건: Brute-Force 또는 HNSW (정확도 우선)                       │
  │  * 10만 ~ 100만 건: HNSW (ChromaDB 기본)                                │
  │  * 100만 ~ 1000만 건: IVF 또는 HNSW (메모리 고려)                       │
  │  * > 1000만 건: IVFPQ (압축 필수)                                       │
  │                                                                         │
  │  [!] ChromaDB는 HNSW만 지원                                             │
  │  [!] IVF, PQ가 필요하면 Faiss, Milvus 등 고려                           │
  └─────────────────────────────────────────────────────────────────────────┘
    """)
    
    # HNSW 파라미터 튜닝
    print_section_header("HNSW 파라미터 튜닝 (ChromaDB)", "[TUNE]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [CODE] ChromaDB HNSW 파라미터 설정                     │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  collection = client.create_collection(                 │
  │      name="tuned_collection",                           │
  │      metadata={                                         │
  │          "hnsw:space": "cosine",    # 거리 함수         │
  │          "hnsw:M": 16,              # 연결 수           │
  │          "hnsw:ef_construction": 100, # 구축 품질       │
  │          "hnsw:ef": 10,             # 검색 품질         │
  │      }                                                  │
  │  )                                                      │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  [파라미터 설명]                                        │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  1. M (연결 수, 기본값: 16)                             │
  │     * 각 노드가 가지는 이웃 연결 수                     │
  │     * ↑ 높을수록: 정확도 ↑, 메모리 ↑, 구축 느림        │
  │     * 권장: 12~48 (도메인에 따라 실험)                  │
  │                                                         │
  │  2. ef_construction (구축 시 탐색 폭, 기본값: 100)      │
  │     * 인덱스 구축 시 탐색하는 이웃 후보 수              │
  │     * ↑ 높을수록: 인덱스 품질 ↑, 구축 시간 ↑           │
  │     * 권장: 100~500 (한 번 구축 후 고정)                │
  │                                                         │
  │  3. ef (검색 시 탐색 폭, 기본값: 10)                    │
  │     * 검색 시 탐색하는 이웃 후보 수                     │
  │     * ↑ 높을수록: 정확도 ↑, 검색 속도 ↓                │
  │     * ef >= n_results 이어야 함                         │
  │     * 권장: 50~200 (정확도/속도 trade-off)              │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  [실험 권장]                                            │
  │  * 정확도 우선: M=32, ef_construction=200, ef=100       │
  │  * 속도 우선: M=12, ef_construction=100, ef=20          │
  │  * 균형: M=16, ef_construction=100, ef=50 (기본값)      │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- ANN: 정확도 조금 희생, 속도 대폭 향상 (95%+ 정확도, 1000배 빠름)",
        "- HNSW: 계층적 그래프, ChromaDB 기본 알고리즘, 범용적",
        "- IVF: 클러스터 기반, 대용량에 적합, Faiss에서 사용",
        "- PQ: 벡터 압축, 메모리 100배 절약, 초대용량용",
        "- 실무: 데이터 규모에 따라 알고리즘 선택이 중요",
        "- ChromaDB: HNSW 파라미터 튜닝으로 정확도/속도 조절 가능"
    ])


def demo_scaling_strategies():
    """실습 7: 대용량 데이터 처리 전략"""
    print("\n" + "="*80)
    print("[7] 실습 7: 대용량 데이터 처리 전략")
    print("="*80)
    print("목표: 100만 건 이상의 벡터를 효율적으로 처리하는 방법 학습")
    print("핵심: 메모리 관리, 배치 처리, 샤딩, 필터링 최적화")
    
    # 대용량 데이터의 도전
    print_section_header("대용량 데이터의 도전", "[!]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [WARN] 대용량 Vector DB 운영 시 고려사항               │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  1. 메모리 사용량                                       │
  │     * 1536차원 벡터 × 4바이트 = 6KB/벡터                │
  │     * 100만 벡터 = 6GB (벡터만)                         │
  │     * HNSW 인덱스 = 추가 2~4배 메모리                   │
  │     * → 100만 벡터 HNSW ≈ 18~30GB 메모리 필요          │
  │                                                         │
  │  2. 인덱스 구축 시간                                    │
  │     * 100만 벡터 HNSW 구축 ≈ 30분~2시간                 │
  │     * 1000만 벡터 ≈ 5~10시간                            │
  │                                                         │
  │  3. 검색 지연 시간                                      │
  │     * 로컬 SSD: 10~50ms/쿼리                            │
  │     * 네트워크 DB: 50~200ms/쿼리                        │
  │                                                         │
  │  4. 비용                                                │
  │     * 임베딩 생성: $0.02/1M 토큰 (text-embedding-3-small)│
  │     * 100만 문서 × 500토큰 = $10 (최초 인덱싱)          │
  │     * 매일 1만 쿼리 × 100토큰 = $0.02/일 (검색)         │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 전략 1: 배치 처리
    print_section_header("전략 1: 배치 처리 (Batch Processing)", "[BATCH]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [CODE] 대용량 임베딩 배치 처리                         │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  # ❌ 나쁜 예: 개별 API 호출                            │
  │  for doc in documents:  # 100만 번 API 호출!            │
  │      embedding = get_embedding(doc)                     │
  │      collection.add(...)                                │
  │                                                         │
  │  # ✓ 좋은 예: 배치 처리                                 │
  │  BATCH_SIZE = 100  # OpenAI 권장: 최대 2048             │
  │                                                         │
  │  for i in range(0, len(documents), BATCH_SIZE):         │
  │      batch = documents[i:i+BATCH_SIZE]                  │
  │      embeddings = get_embeddings_batch(batch)  # 1회 호출│
  │      collection.add(                                    │
  │          documents=batch,                               │
  │          embeddings=embeddings,                         │
  │          ids=[f"doc_{i+j}" for j in range(len(batch))]  │
  │      )                                                  │
  │      print(f"Progress: {i+len(batch)}/{len(documents)}")│
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  [효과]                                                 │
  │  * API 호출 횟수: 100만 → 1만 (100배 감소)              │
  │  * 네트워크 오버헤드 감소                               │
  │  * Rate Limit 회피                                      │
  │  * 처리 시간: 10시간 → 1시간 (예시)                     │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 전략 2: 샤딩
    print_section_header("전략 2: 샤딩 (Sharding)", "[SHARD]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [ARCH] 데이터를 여러 컬렉션/서버로 분산                │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  방법 1: 카테고리 기반 샤딩                             │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
  │  │ tech_docs    │  │ legal_docs   │  │ hr_docs      │  │
  │  │ (기술 문서)  │  │ (법률 문서)  │  │ (인사 문서)  │  │
  │  │ 50만 건      │  │ 30만 건      │  │ 20만 건      │  │
  │  └──────────────┘  └──────────────┘  └──────────────┘  │
  │                                                         │
  │  [CODE]                                                 │
  │  # 카테고리에 따라 컬렉션 선택                          │
  │  def get_collection(category: str):                     │
  │      return client.get_collection(f"{category}_docs")   │
  │                                                         │
  │  # 검색 시 해당 컬렉션만 조회                           │
  │  results = get_collection("tech").query(...)            │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  방법 2: 시간 기반 샤딩                                 │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
  │  │ docs_2024_Q1 │  │ docs_2024_Q2 │  │ docs_2024_Q3 │  │
  │  │ (1~3월 문서) │  │ (4~6월 문서) │  │ (7~9월 문서) │  │
  │  └──────────────┘  └──────────────┘  └──────────────┘  │
  │                                                         │
  │  [장점]                                                 │
  │  * 각 컬렉션 크기 관리 용이                             │
  │  * 오래된 데이터 아카이브/삭제 편리                     │
  │  * 병렬 검색 가능 (여러 컬렉션 동시 조회)               │
  │                                                         │
  │  [단점]                                                 │
  │  * 전체 검색 시 여러 컬렉션 조회 필요                   │
  │  * 결과 병합 로직 필요                                  │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 전략 3: 메타데이터 필터 최적화
    print_section_header("전략 3: 메타데이터 필터 최적화", "[FILTER]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] 필터링으로 검색 범위 축소                        │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  # ❌ 전체 검색 후 필터링 (느림)                        │
  │  results = collection.query(                            │
  │      query_embeddings=[embedding],                      │
  │      n_results=1000                                     │
  │  )                                                      │
  │  filtered = [r for r in results if r.year == 2024]     │
  │                                                         │
  │  # ✓ 검색 시 필터 적용 (빠름)                          │
  │  results = collection.query(                            │
  │      query_embeddings=[embedding],                      │
  │      n_results=10,                                      │
  │      where={"year": 2024}  # DB 레벨 필터링             │
  │  )                                                      │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  [메타데이터 설계 팁]                                   │
  │                                                         │
  │  1. 자주 필터링하는 필드는 반드시 메타데이터로          │
  │     * category, department, year, author 등             │
  │                                                         │
  │  2. 카디널리티(고유값 수) 고려                          │
  │     * 낮은 카디널리티: category (10개) → 효율적         │
  │     * 높은 카디널리티: user_id (100만개) → 비효율적     │
  │                                                         │
  │  3. 복합 필터 활용                                      │
  │     where={"$and": [                                    │
  │         {"category": "tech"},                           │
  │         {"year": {"$gte": 2023}}                        │
  │     ]}                                                  │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 전략 4: 캐싱
    print_section_header("전략 4: 캐싱 (Caching)", "[CACHE]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [ARCH] 자주 사용되는 쿼리/결과 캐싱                    │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  1. 임베딩 캐싱 (쿼리 임베딩)                           │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  from functools import lru_cache                        │
  │  import hashlib                                         │
  │                                                         │
  │  @lru_cache(maxsize=1000)                               │
  │  def get_cached_embedding(text_hash: str):              │
  │      # text_hash = hashlib.md5(text.encode()).hexdigest()│
  │      return get_embedding(original_text)                │
  │                                                         │
  │  # 또는 Redis 사용                                      │
  │  def get_embedding_with_redis(text: str):               │
  │      cache_key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"│
  │      cached = redis.get(cache_key)                      │
  │      if cached:                                         │
  │          return json.loads(cached)                      │
  │      embedding = get_embedding(text)                    │
  │      redis.setex(cache_key, 3600, json.dumps(embedding))│
  │      return embedding                                   │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  2. 검색 결과 캐싱                                      │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  # 동일 쿼리의 결과 캐싱 (FAQ 등)                       │
  │  def search_with_cache(query: str, ttl: int = 300):     │
  │      cache_key = f"search:{hash(query)}"                │
  │      cached = redis.get(cache_key)                      │
  │      if cached:                                         │
  │          return json.loads(cached)                      │
  │      results = collection.query(...)                    │
  │      redis.setex(cache_key, ttl, json.dumps(results))   │
  │      return results                                     │
  │                                                         │
  │  [효과]                                                 │
  │  * 반복 쿼리 응답 시간: 100ms → 1ms                     │
  │  * 임베딩 API 비용 절감                                 │
  │  * DB 부하 감소                                         │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 전략 5: 하드웨어 고려
    print_section_header("전략 5: 하드웨어 및 인프라 고려", "[INFRA]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [SPEC] 규모별 권장 사양                                │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  ┌──────────────┬────────────┬────────────┬───────────┐ │
  │  │ 벡터 수      │ RAM        │ Storage    │ 구성      │ │
  │  ├──────────────┼────────────┼────────────┼───────────┤ │
  │  │ < 10만       │ 8GB        │ SSD 50GB   │ 단일 노드 │ │
  │  │ 10만 ~ 100만 │ 32GB       │ SSD 200GB  │ 단일 노드 │ │
  │  │ 100만 ~ 500만│ 64GB       │ SSD 500GB  │ 단일 노드 │ │
  │  │ 500만 ~ 1000만│ 128GB     │ SSD 1TB    │ 샤딩 권장 │ │
  │  │ > 1000만     │ 분산 클러스터 필수                   │ │
  │  └──────────────┴────────────┴────────────┴───────────┘ │
  │  (※ HNSW 기준, 1536차원 벡터)                          │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  [클라우드 서비스 vs 자체 호스팅]                       │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  클라우드 (Pinecone, Weaviate Cloud 등):                │
  │  * ✓ 관리 부담 없음                                    │
  │  * ✓ 자동 스케일링                                     │
  │  * ✗ 비용 높음 (대용량 시)                             │
  │  * ✗ 데이터 외부 저장 (보안 고려)                      │
  │                                                         │
  │  자체 호스팅 (ChromaDB, Milvus 등):                     │
  │  * ✓ 비용 통제 가능                                    │
  │  * ✓ 데이터 내부 보관                                  │
  │  * ✗ 운영 부담                                         │
  │  * ✗ 스케일링 직접 구현                                │
  │                                                         │
  │  [TIP] 선택 가이드:                                     │
  │  * PoC/스타트업: 클라우드 (빠른 시작)                   │
  │  * 대기업/규제산업: 자체 호스팅 (데이터 주권)           │
  │  * 하이브리드: 개발은 클라우드, 프로덕션은 자체 호스팅  │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 체크리스트
    print_section_header("대용량 처리 체크리스트", "[CHECK]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [✓] 대용량 Vector DB 운영 체크리스트                   │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  □ 데이터 규모 파악                                     │
  │    - 현재 문서 수: _______                              │
  │    - 예상 증가율: 월 _______건                          │
  │    - 1년 후 예상: _______건                             │
  │                                                         │
  │  □ 메모리 계산                                          │
  │    - 벡터 메모리: 문서수 × 6KB = _______GB              │
  │    - 인덱스 메모리: 벡터 메모리 × 3 = _______GB         │
  │    - 서버 RAM >= 인덱스 메모리 × 1.5                    │
  │                                                         │
  │  □ 배치 처리 구현                                       │
  │    - 임베딩 배치 크기: 100~500                          │
  │    - DB 삽입 배치 크기: 1000~5000                       │
  │    - 진행률 로깅 구현                                   │
  │                                                         │
  │  □ 샤딩 전략 결정                                       │
  │    - 샤딩 기준: 카테고리 / 시간 / 해시                  │
  │    - 컬렉션 당 최대 크기: _______건                     │
  │                                                         │
  │  □ 캐싱 전략                                            │
  │    - 임베딩 캐시: LRU / Redis                           │
  │    - 결과 캐시 TTL: _______초                           │
  │                                                         │
  │  □ 모니터링 설정                                        │
  │    - 검색 지연 시간 추적                                │
  │    - 메모리 사용량 알림                                 │
  │    - API 비용 추적                                      │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- 배치 처리: API 호출 100배 감소, 처리 시간 10배 단축",
        "- 샤딩: 카테고리/시간 기준으로 데이터 분산, 검색 범위 축소",
        "- 메타데이터 필터: DB 레벨에서 필터링, 검색 효율 향상",
        "- 캐싱: 반복 쿼리 1ms 응답, API 비용 절감",
        "- 하드웨어: 100만 벡터 = 약 64GB RAM 필요 (HNSW)",
        "- 실무: 규모에 맞는 전략 조합이 핵심 (작게 시작, 점진적 확장)"
    ])


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """모든 데모 실행"""
    print("\n" + "="*80)
    print("[LAB] Vector Database 실습 (ChromaDB)")
    print("="*80)
    
    # Python 버전 및 OS 체크
    python_version = sys.version_info
    is_windows = platform.system() == "Windows"
    
    print(f"\n[INFO] 실행 환경:")
    print(f"   Python 버전: {python_version.major}.{python_version.minor}.{python_version.micro}")
    print(f"   운영체제: {platform.system()}")
    
    # Windows + Python 3.13 참고 안내
    if is_windows and python_version.major == 3 and python_version.minor >= 13:
        print(f"""
────────────────────────────────────────────────────────────────────────────────
[!] 참고: Python 3.13은 ChromaDB가 아직 공식 지원하지 않는 버전입니다
────────────────────────────────────────────────────────────────────────────────

현재 환경({python_version.major}.{python_version.minor})에서도 실행 가능할 수 있지만,
안정적인 실습을 위해 아래 방법을 권장합니다:

[권장] Python 3.11로 다운그레이드 (공식 지원 버전)
  - pyenv 사용: pyenv install 3.11.9 && pyenv local 3.11.9
  - conda 사용: conda create -n lab python=3.11
  
[대안] 현재 Python 3.13 환경에서 계속 진행
  - 설치 명령: pip install chromadb==0.4.22 hnswlib==0.8.0
  - 주의: 설치는 되지만 실행 중 오류 가능성 있음
  
[마지막] Windows 설치 오류 반복 시
  - WSL2 + Ubuntu 환경 사용 권장
  - 가이드: https://learn.microsoft.com/ko-kr/windows/wsl/

[진행] 계속하려면 Enter 키를 누르세요...
""")
        input()  # 사용자가 읽고 진행하도록
    
    print("\n[LIST] 실습 항목:")
    print("  1. 임베딩(Embedding) 이해하기 - 텍스트를 벡터로 변환")
    print("  2. Vector DB 기본 작업 - 저장과 검색")
    print("  3. 거리와 유사도 이해하기 - 스코어 해석")
    print("  4. 메타데이터 필터링 - 조건부 검색")
    print("  5. 실전 예제 - 문서 관리 시스템")
    print("  6. ANN 인덱스 알고리즘 이해하기 - HNSW, IVF 등 원리")
    print("  7. 대용량 데이터 처리 전략 - 스케일링과 최적화")
    
    try:
        # 1. 임베딩 기초
        demo_embedding_basics()
        
        # 2. 기본 작업
        demo_basic_operations()
        
        # 3. 스코어 이해
        demo_distance_scores()
        
        # 4. 메타데이터 필터링
        demo_metadata_filtering()
        
        # 5. 문서 관리 시스템
        demo_document_manager()
        
        # 6. ANN 인덱스 알고리즘 (이론 중심, API 호출 없음)
        demo_index_algorithms()
        
        # 7. 대용량 데이터 처리 전략 (이론 중심, API 호출 없음)
        demo_scaling_strategies()
        
        # 완료 메시지
        print("\n" + "="*80)
        print("[OK] 모든 실습 완료!")
        print("="*80)
        
        print("\n[FILE] 생성된 파일:")
        print("   - ./chroma_db/ : ChromaDB 데이터 저장소")
        print("   - 이 디렉토리를 삭제하면 모든 데이터가 초기화됩니다.")
        
        print("\n[TIP] 다음 단계:")
        print("   - lab03/rag_basic.py : RAG 시스템 구축")
        print("   - Vector DB를 활용한 검색 증강 생성 학습")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
