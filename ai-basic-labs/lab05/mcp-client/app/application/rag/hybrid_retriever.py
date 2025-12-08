"""
Hybrid Retriever - BM25 + Vector + Re-ranking

Lab03의 HybridRetriever, Reranker 참고하여 구현
"""
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    score: float
    metadata: Dict[str, Any]
    rank: int
    search_type: str  # "sparse", "dense", "hybrid", "reranked"


class HybridRetriever:
    """
    하이브리드 검색기 (BM25 + Vector + Re-ranking)
    
    특징:
    - Sparse 검색: BM25 (키워드 기반)
    - Dense 검색: ChromaDB 벡터 검색 (의미 기반)
    - Hybrid: 두 검색 결과 결합
    - Re-ranking: Cross-Encoder로 재순위화
    """
    
    # 한글 조사 패턴 (간단 버전)
    KOREAN_PARTICLES = [
        '이란', '이란?', '란', '란?', '은', '는', '이', '가', '을', '를',
        '의', '에', '에서', '으로', '로', '와', '과', '도', '만', '까지',
        '부터', '이다', '입니다', '인가', '인가?', '인지', '하는', '되는'
    ]
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "text-embedding-3-small",
        use_reranker: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Args:
            collection_name: ChromaDB 컬렉션 이름
            persist_directory: ChromaDB 저장 경로
            embedding_model: OpenAI 임베딩 모델
            use_reranker: Re-ranker 사용 여부
            reranker_model: Re-ranker 모델 이름
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.use_reranker = use_reranker
        
        # ChromaDB 클라이언트 초기화
        self.chroma_client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = None
        
        # BM25 관련
        self.corpus: List[str] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.doc_metadata: List[Dict] = []
        
        # OpenAI 클라이언트
        self.openai_client = None
        self._init_openai()
        
        # Re-ranker (lazy loading)
        self._reranker = None
        self._reranker_model = reranker_model
        
        logger.info(f"🔍 HybridRetriever 초기화: collection={collection_name}")
    
    def _init_openai(self):
        """OpenAI 클라이언트 초기화"""
        from openai import OpenAI
        import httpx
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        
        # SSL 인증서 검증 우회 (회사 방화벽 대응)
        http_client = httpx.Client(verify=False)
        self.openai_client = OpenAI(api_key=api_key, http_client=http_client)
    
    @property
    def reranker(self):
        """Re-ranker lazy loading"""
        if self._reranker is None and self.use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"📥 Re-ranker 모델 로딩 중: {self._reranker_model}")
                self._reranker = CrossEncoder(self._reranker_model)
                logger.info("✅ Re-ranker 준비 완료")
            except Exception as e:
                logger.warning(f"⚠️ Re-ranker 로딩 실패: {e}")
                self.use_reranker = False
        return self._reranker
    
    def initialize_collection(self, reset: bool = False):
        """컬렉션 초기화"""
        if reset:
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                logger.info(f"🗑️ 기존 컬렉션 삭제: {self.collection_name}")
            except:
                pass
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "l2"}  # L2 거리 사용
        )
        
        # 기존 문서 로드
        self._load_existing_documents()
        
        logger.info(f"📚 컬렉션 초기화 완료: {self.collection.count()}개 문서")
    
    def _load_existing_documents(self):
        """기존 문서를 BM25 인덱스에 로드"""
        if self.collection is None:
            return
        
        count = self.collection.count()
        if count == 0:
            return
        
        # 모든 문서 가져오기
        results = self.collection.get(include=["documents", "metadatas"])
        
        self.corpus = results["documents"] or []
        self.doc_metadata = results["metadatas"] or []
        self.tokenized_corpus = [self._tokenize_korean(doc) for doc in self.corpus]
        
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"📖 기존 {len(self.corpus)}개 문서 BM25 인덱스에 로드")
    
    def _tokenize_korean(self, text: str) -> List[str]:
        """
        한글 토큰화 (간단한 규칙 기반)
        
        ⚠️ 실무에서는 KoNLPy 형태소 분석기 권장
        """
        # 구두점을 공백으로 변환
        text = re.sub(r'[.,!?;:()"\'\[\]{}]', ' ', text)
        
        # 공백으로 분리
        tokens = text.lower().split()
        
        # 조사 제거 시도
        cleaned_tokens = []
        for token in tokens:
            cleaned = token
            for particle in sorted(self.KOREAN_PARTICLES, key=len, reverse=True):
                if cleaned.endswith(particle) and len(cleaned) > len(particle):
                    cleaned = cleaned[:-len(particle)]
                    break
            if cleaned:
                cleaned_tokens.append(cleaned)
        
        return cleaned_tokens
    
    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        """
        문서 추가 (벡터 + BM25 인덱스)
        
        Args:
            texts: 문서 텍스트 목록
            metadatas: 메타데이터 목록
            ids: 문서 ID 목록
        """
        if self.collection is None:
            self.initialize_collection()
        
        if ids is None:
            start_idx = len(self.corpus)
            ids = [f"doc_{start_idx + i}" for i in range(len(texts))]
        
        # 임베딩 생성
        logger.info(f"🔄 {len(texts)}개 문서 임베딩 생성 중...")
        embeddings = self._get_embeddings(texts)
        
        # ChromaDB에 추가
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        # BM25 인덱스 업데이트
        self.corpus.extend(texts)
        self.doc_metadata.extend(metadatas)
        new_tokenized = [self._tokenize_korean(text) for text in texts]
        self.tokenized_corpus.extend(new_tokenized)
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        logger.info(f"✅ {len(texts)}개 문서 추가 완료 (총 {len(self.corpus)}개)")
    
    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """OpenAI 임베딩 생성"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [data.embedding for data in response.data]
    
    def search(
        self,
        query: str,
        k: int = 5,
        method: str = "hybrid",
        alpha: float = 0.5,
        use_reranker: Optional[bool] = None
    ) -> List[SearchResult]:
        """
        검색 수행
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            method: 검색 방법 ("sparse", "dense", "hybrid")
            alpha: 하이브리드 검색 시 Dense 가중치 (0~1)
            use_reranker: Re-ranker 사용 여부 (None이면 기본 설정 사용)
        
        Returns:
            검색 결과 리스트
        """
        if self.collection is None or self.collection.count() == 0:
            logger.warning("⚠️ 검색할 문서가 없습니다.")
            return []
        
        # 실제 k 조정
        actual_k = min(k, len(self.corpus))
        
        # Re-ranker 사용 시 더 많은 후보 검색
        should_rerank = use_reranker if use_reranker is not None else self.use_reranker
        search_k = actual_k * 3 if should_rerank else actual_k
        
        # 검색 수행
        if method == "sparse":
            results = self._sparse_search(query, search_k)
        elif method == "dense":
            results = self._dense_search(query, search_k)
        elif method == "hybrid":
            results = self._hybrid_search(query, search_k, alpha)
        else:
            raise ValueError(f"알 수 없는 검색 방법: {method}")
        
        # Re-ranking
        if should_rerank and self.reranker and results:
            results = self._rerank(query, results, actual_k)
        else:
            results = results[:actual_k]
        
        return results
    
    def _sparse_search(self, query: str, k: int) -> List[SearchResult]:
        """BM25 Sparse 검색"""
        if self.bm25 is None:
            return []
        
        tokenized_query = self._tokenize_korean(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # 상위 k개 인덱스
        top_indices = np.argsort(scores)[::-1]
        
        results = []
        for rank, idx in enumerate(top_indices[:k], 1):
            score = float(scores[idx])
            if score > 0:
                results.append(SearchResult(
                    content=self.corpus[idx],
                    score=score,
                    metadata=self.doc_metadata[idx] if idx < len(self.doc_metadata) else {},
                    rank=rank,
                    search_type="sparse"
                ))
        
        return results
    
    def _dense_search(self, query: str, k: int) -> List[SearchResult]:
        """Vector Dense 검색"""
        query_embedding = self._get_embeddings([query])[0]
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        search_results = []
        for rank, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ), 1):
            # L2 거리를 유사도로 변환 (0~1)
            similarity = 1 / (1 + dist)
            search_results.append(SearchResult(
                content=doc,
                score=similarity,
                metadata=meta,
                rank=rank,
                search_type="dense"
            ))
        
        return search_results
    
    def _hybrid_search(self, query: str, k: int, alpha: float) -> List[SearchResult]:
        """
        하이브리드 검색 (Sparse + Dense)
        
        점수 = (1-alpha) * sparse_normalized + alpha * dense_normalized
        """
        sparse_results = self._sparse_search(query, k * 2)
        dense_results = self._dense_search(query, k * 2)
        
        # 점수 정규화
        sparse_scores = {r.content: r.score for r in sparse_results}
        dense_scores = {r.content: r.score for r in dense_results}
        
        sparse_max = max(sparse_scores.values()) if sparse_scores else 1.0
        dense_max = max(dense_scores.values()) if dense_scores else 1.0
        
        # 결합
        combined = {}
        
        for result in sparse_results:
            content = result.content
            normalized = (result.score / sparse_max) * (1 - alpha) if sparse_max > 0 else 0
            combined[content] = SearchResult(
                content=content,
                score=normalized,
                metadata=result.metadata,
                rank=0,
                search_type="hybrid"
            )
        
        for result in dense_results:
            content = result.content
            normalized = (result.score / dense_max) * alpha if dense_max > 0 else 0
            
            if content in combined:
                combined[content].score += normalized
            else:
                combined[content] = SearchResult(
                    content=content,
                    score=normalized,
                    metadata=result.metadata,
                    rank=0,
                    search_type="hybrid"
                )
        
        # 정렬 및 순위 할당
        sorted_results = sorted(combined.values(), key=lambda x: x.score, reverse=True)
        for rank, result in enumerate(sorted_results[:k], 1):
            result.rank = rank
        
        return sorted_results[:k]
    
    def _rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """Cross-Encoder Re-ranking"""
        if not self.reranker or not results:
            return results[:top_k]
        
        import math
        
        # 쿼리-문서 쌍 생성
        pairs = [[query, r.content] for r in results]
        
        # Re-ranking 점수 계산
        raw_scores = self.reranker.predict(pairs)
        
        # Sigmoid로 정규화 (0~1)
        normalized_scores = [1 / (1 + math.exp(-s)) for s in raw_scores]
        
        # 결과 업데이트
        reranked = []
        for result, score in zip(results, normalized_scores):
            reranked.append(SearchResult(
                content=result.content,
                score=score,
                metadata={**result.metadata, "original_score": result.score},
                rank=0,
                search_type="reranked"
            ))
        
        # 정렬 및 순위 할당
        reranked.sort(key=lambda x: x.score, reverse=True)
        for rank, result in enumerate(reranked[:top_k], 1):
            result.rank = rank
        
        return reranked[:top_k]
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            "collection_name": self.collection_name,
            "total_documents": len(self.corpus),
            "chroma_count": self.collection.count() if self.collection else 0,
            "reranker_enabled": self.use_reranker,
            "embedding_model": self.embedding_model
        }
    
    def delete_document(self, doc_id: str) -> bool:
        """문서 삭제"""
        if self.collection is None:
            return False
        
        try:
            self.collection.delete(ids=[doc_id])
            # BM25 인덱스 재구축 필요
            self._load_existing_documents()
            return True
        except Exception as e:
            logger.error(f"문서 삭제 실패: {e}")
            return False
    
    def clear_all(self):
        """모든 문서 삭제 (컬렉션 내 모든 데이터 삭제)"""
        try:
            if self.collection is not None:
                # 방법 1: 컬렉션 내 모든 문서 ID 가져와서 삭제
                all_ids = self.collection.get()["ids"]
                if all_ids:
                    self.collection.delete(ids=all_ids)
                    logger.info(f"🗑️ {len(all_ids)}개 문서 삭제 완료")
                
                # 방법 2: 컬렉션 자체를 삭제하고 재생성
                try:
                    self.chroma_client.delete_collection(name=self.collection_name)
                    logger.info(f"🗑️ 컬렉션 삭제: {self.collection_name}")
                except Exception as e:
                    logger.warning(f"컬렉션 삭제 중 경고: {e}")
                
                # 새 컬렉션 생성
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "l2"}
                )
                logger.info(f"📚 새 컬렉션 생성 완료")
            
        except Exception as e:
            logger.error(f"전체 삭제 중 오류: {e}")
            # 오류 발생 시 컬렉션 재초기화 시도
            try:
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "l2"}
                )
            except:
                pass
        
        # 메모리 인덱스 초기화
        self.corpus = []
        self.tokenized_corpus = []
        self.doc_metadata = []
        self.bm25 = None
        
        logger.info("🗑️ 모든 문서 삭제 완료")

