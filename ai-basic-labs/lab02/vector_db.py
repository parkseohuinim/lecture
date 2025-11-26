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
"""

import os
import sys
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
    visualize_similarity_bar
)


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
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
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
    
    def create_collection(self, name: str, reset: bool = False) -> chromadb.Collection:
        """
        컬렉션 생성 또는 가져오기
        
        컬렉션 = 벡터들을 그룹화하는 단위 (일반 DB의 테이블과 유사)
        
        Args:
            name: 컬렉션 이름
            reset: True면 기존 컬렉션 삭제 후 재생성
        
        Returns:
            컬렉션 객체
        """
        if reset:
            try:
                self.client.delete_collection(name=name)
                print(f"   기존 컬렉션 '{name}' 삭제됨")
            except:
                pass
        
        collection = self.client.get_or_create_collection(
            name=name,
            metadata={"description": "Vector database collection"}
        )
        
        print(f"   컬렉션 '{name}' 준비 완료 (문서 수: {collection.count()})")
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
    print_subsection("첫 번째 텍스트와의 코사인 유사도")
    
    def cosine_similarity(v1, v2):
        """코사인 유사도 계산"""
        v1, v2 = np.array(v1), np.array(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    base_embedding = embeddings[0]
    print(f"\n기준: '{texts[0]}'")
    print()
    
    for i, (text, emb) in enumerate(zip(texts[1:], embeddings[1:]), 2):
        similarity = cosine_similarity(base_embedding, emb)
        bar = visualize_similarity_bar(similarity)
        
        interpretation = ""
        if similarity >= 0.8:
            interpretation = "[v] 매우 유사"
        elif similarity >= 0.6:
            interpretation = "[~] 관련 있음"
        else:
            interpretation = "[x] 다른 주제"
        
        print(f"  vs '{text}'")
        print(f"     {bar} {similarity:.4f} {interpretation}")
        print()
    
    # 핵심 포인트
    print_key_points([
        "- 임베딩: 텍스트 -> 고정 길이 숫자 벡터 (예: 1536차원)",
        "- OpenAI text-embedding-3-small: 성능과 비용의 균형",
        "- 의미가 비슷한 텍스트 -> 벡터 공간에서 가까이 위치",
        "- 코사인 유사도: 벡터 방향의 유사성 측정 (-1 ~ 1, 1이 가장 유사)"
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
        print(f"  {i}. {doc[:50]}...")
    
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
        bar = visualize_similarity_bar(similarity, 30)
        
        print(f"\n[{i}위] 거리: {distance:.4f} | 유사도: {similarity:.4f}")
        print(f"     {bar}")
        print(f"     문서: {doc}")
        print(f"     메타: {metadata}")
    
    # 거리 스코어 해석 가이드 추가
    print_section_header("거리 스코어 해석 가이드", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [TIP] L2 거리 스코어 해석                               │
  │  ─────────────────────────────────────────────────────  │
  │  * OpenAI 임베딩 + L2 거리 기준 일반적인 범위:           │
  │                                                         │
  │    거리 0.0 ~ 0.5  ->  매우 높은 관련성 (거의 동일)      │
  │    거리 0.5 ~ 1.0  ->  높은 관련성 (주제 일치)           │
  │    거리 1.0 ~ 1.5  ->  중간 관련성 (관련 있음)           │
  │    거리 1.5 ~ 2.0  ->  낮은 관련성 (약간 관련)           │
  │    거리 2.0 이상   ->  거의 무관 (다른 주제)             │
  │                                                         │
  │  [!] 주의: 이 기준은 데이터셋에 따라 달라질 수 있음!     │
  │     실제 적용 시 자신의 데이터로 임계값 조정 필요        │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- Vector DB: 임베딩 벡터를 저장하고 유사도로 검색",
        "- Collection: 벡터들의 그룹 (일반 DB의 테이블)",
        "- Add: 문서 + 임베딩 + 메타데이터 저장",
        "- Query: 쿼리 임베딩과 가장 가까운 벡터 검색",
        "- 거리 해석: 0에 가까울수록 유사, 일반적으로 0.5~2.0 범위"
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
  │  [INFO] 해석 기준 (이 실습 데이터 기준)                  │
  │  ─────────────────────────────────────────────────────  │
  │  * [v] 높음: L2 거리 < 1.0 (유사도 > 0.50) - 주제 일치  │
  │  * [~] 중간: L2 거리 1.0~1.8 (유사도 0.35~0.50) - 관련성│
  │  * [x] 낮음: L2 거리 > 1.8 (유사도 < 0.35) - 다른 주제  │
  │                                                         │
  │  [!] 이 기준은 데이터셋마다 다름! 실무에서는 조정 필요  │
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
        
        # 해석 (거리 기준으로 변경 - 더 직관적)
        if distance < 1.0:  # 유사도 > 0.5
            interpretation = "[v] 높음"
        elif distance < 1.8:  # 유사도 > 0.35
            interpretation = "[~] 중간"
        else:
            interpretation = "[x] 낮음"
        
        # 시각화
        bar = visualize_similarity_bar(similarity, 20)
        
        print(f"{i:<4} {distance:<10.4f} {similarity:<10.4f} {interpretation:<12} {doc[:35]}...")
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
  │  거리 -> 유사도 변환                                    │
  │  ─────────────────────────────────────────────────────  │
  │  * 방법 1: similarity = 1 / (1 + distance)             │
  │  * 방법 2: similarity = 1 - (distance / max_distance)  │
  │  * 결과: 0 ~ 1 범위 (1이 가장 유사)                    │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- L2 거리: 벡터 간 직선 거리, 0이 가장 가까움",
        "- 코사인 유사도: 벡터 방향의 유사성, 1이 가장 유사",
        "- 거리 DOWN = 유사도 UP = 검색 순위 UP",
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
            bar = visualize_similarity_bar(result.similarity, 25)
            print(f"\n  [{result.rank}위] 유사도: {result.similarity:.4f}")
            print(f"       {bar}")
            print(f"       [DOC] {result.content[:60]}...")
            print(f"       [TAG] {result.metadata}")
    
    # 핵심 포인트
    print_key_points([
        "- 클래스로 캡슐화하면 재사용성과 유지보수성 향상",
        "- 배치 임베딩 (get_embeddings_batch)으로 API 호출 최소화",
        "- 메타데이터를 잘 설계하면 다양한 검색 시나리오 대응 가능",
        "- 유사도 점수를 시각화하면 결과 품질을 직관적으로 파악"
    ])


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """모든 데모 실행"""
    print("\n" + "="*80)
    print("[LAB] Vector Database 실습 (ChromaDB)")
    print("="*80)
    
    print("\n[LIST] 실습 항목:")
    print("  1. 임베딩(Embedding) 이해하기 - 텍스트를 벡터로 변환")
    print("  2. Vector DB 기본 작업 - 저장과 검색")
    print("  3. 거리와 유사도 이해하기 - 스코어 해석")
    print("  4. 메타데이터 필터링 - 조건부 검색")
    print("  5. 실전 예제 - 문서 관리 시스템")
    
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
