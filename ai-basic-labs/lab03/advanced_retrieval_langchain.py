"""
Advanced RAG (Retrieval-Augmented Generation) 실습
실무 RAG 팀들이 하는 최적화 기능 직접 다루기

실습 항목:
1. Sparse + Dense 하이브리드 검색
2. Re-ranking 적용 (BGE reranker)
3. Multi-hop 질의: 두 단계 검색으로 답 찾기
4. Chunk size 실험: 512 vs 1024 vs 2048
5. 컨텍스트 윈도우 관리 실험
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

# Sparse 검색 (BM25)
from rank_bm25 import BM25Okapi

# Re-ranking
from sentence_transformers import CrossEncoder

# 기본 라이브러리
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken
import pdfplumber
import numpy as np

# 공통 데이터 임포트
from shared_data import SAMPLE_TEXT, MIN_TEXT_LENGTH, get_sample_or_document_text

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    score: float
    metadata: Dict[str, Any]
    rank: int


# SAMPLE_TEXT는 shared_data.py에서 임포트됨


class DocumentProcessor:
    """문서 처리 및 청킹 (rag_basic.py의 TextChunker 방식 사용)"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: 청크 크기 (512, 1024, 2048 등)
            chunk_overlap: 청크 간 겹침 크기
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def load_pdf(self, file_path: str) -> str:
        """PDF 파일 로드"""
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def load_text(self, text: str) -> str:
        """텍스트 직접 로드 (PDF 없이 실습 가능)"""
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """텍스트를 청크로 분할 (문장 경계 고려)"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            
            # 텍스트 끝이면 그냥 추가
            if end >= text_length:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            # 문장 경계 찾기
            best_end = -1
            
            # 1순위: 단락 끝 (빈 줄)
            double_newline = text.rfind('\n\n', start, end + 50)
            if double_newline != -1:
                best_end = double_newline + 2
            
            # 2순위: 문장 끝 (마침표 + 줄바꿈)
            if best_end == -1:
                for i in range(end, max(start, end - 100), -1):
                    if i < text_length - 1 and text[i] == '.' and text[i+1] == '\n':
                        best_end = i + 2
                        break
            
            # 3순위: 마침표 + 공백
            if best_end == -1:
                period_space = text.rfind('. ', start, end + 30)
                if period_space != -1:
                    best_end = period_space + 2
            
            # 4순위: 줄바꿈
            if best_end == -1:
                newline = text.rfind('\n', start, end + 20)
                if newline != -1:
                    best_end = newline + 1
            
            # 5순위: 공백
            if best_end == -1:
                space = text.rfind(' ', start, end)
                if space != -1 and space > start + self.chunk_size // 2:
                    best_end = space + 1
            
            # 최종: 강제로 자르기
            if best_end == -1:
                best_end = end
            
            chunk = text[start:best_end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 다음 청크 시작 위치 (오버랩 적용)
            next_start = best_end - self.chunk_overlap
            
            # 진행이 없으면 강제로 앞으로 (무한 루프 방지)
            if next_start <= start:
                next_start = best_end
            
            start = next_start
        
        return chunks
    
    def create_chunks(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        """텍스트를 청크로 분할하여 Document 객체 생성"""
        chunks = self.chunk_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_metadata = metadata.copy() if metadata else {}
            doc_metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk),
                "total_chunks": len(chunks)
            })
            documents.append(Document(page_content=chunk, metadata=doc_metadata))
        
        return documents


class HybridRetriever:
    """Sparse (BM25) + Dense (Vector) 하이브리드 검색"""
    
    # 한글 조사 패턴 (간단 버전)
    KOREAN_PARTICLES = [
        '이란', '이란?', '란', '란?', '은', '는', '이', '가', '을', '를',
        '의', '에', '에서', '으로', '로', '와', '과', '도', '만', '까지',
        '부터', '이다', '입니다', '인가', '인가?', '인지', '하는', '되는'
    ]
    
    @staticmethod
    def tokenize_korean(text: str) -> List[str]:
        """
        간단한 한글 토큰화 (교육용)
        - 구두점 제거
        - 공백 분리
        - 일반적인 조사 제거
        """
        import re
        
        # 구두점을 공백으로 변환
        text = re.sub(r'[.,!?;:()"\'\[\]{}]', ' ', text)
        
        # 공백으로 분리
        tokens = text.split()
        
        # 조사 제거 시도
        cleaned_tokens = []
        for token in tokens:
            cleaned = token
            # 긴 조사부터 제거 시도
            for particle in sorted(HybridRetriever.KOREAN_PARTICLES, key=len, reverse=True):
                if cleaned.endswith(particle) and len(cleaned) > len(particle):
                    cleaned = cleaned[:-len(particle)]
                    break
            if cleaned:  # 빈 문자열이 아니면 추가
                cleaned_tokens.append(cleaned)
        
        return cleaned_tokens
    
    def __init__(
        self,
        documents: List[Document],
        embeddings: OpenAIEmbeddings,
        persist_directory: str = "./chroma_db",
        collection_name: str = "hybrid_search"
    ):
        """
        Args:
            documents: 문서 리스트
            embeddings: 임베딩 모델
            persist_directory: Chroma DB 저장 경로
            collection_name: 컬렉션 이름
        """
        self.documents = documents
        self.embeddings = embeddings
        
        # Dense 검색: Vector DB (Chroma)
        print(f"Dense 검색 준비 중... (컬렉션: {collection_name})")
        
        # 기존 컬렉션 삭제 (깨끗한 시작)
        try:
            chroma_client = chromadb.PersistentClient(path=persist_directory)
            chroma_client.delete_collection(name=collection_name)
        except:
            pass  # 컬렉션이 없으면 무시
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory,
            collection_name=collection_name
        )
        
        # Sparse 검색: BM25 (한글 토큰화 적용)
        print("Sparse 검색 준비 중... (BM25 + 한글 토큰화)")
        self.corpus = [doc.page_content for doc in documents]
        self.tokenized_corpus = [self.tokenize_korean(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"하이브리드 검색 준비 완료 (문서 수: {len(documents)})")
    
    def sparse_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """BM25 Sparse 검색 (한글 토큰화 적용)"""
        tokenized_query = self.tokenize_korean(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # 실제 문서 수만큼만 반환
        k = min(k, len(self.documents))
        
        # 상위 k개 결과 (점수 > 0인 것만)
        top_indices = np.argsort(scores)[::-1]
        
        results = []
        rank = 1
        for idx in top_indices:
            score = float(scores[idx])
            
            # 점수가 0이면 키워드 매칭 없음 - 결과에서 제외
            if score == 0:
                continue
            
            results.append(SearchResult(
                content=self.documents[idx].page_content,
                score=score,
                metadata={**self.documents[idx].metadata, "matched_tokens": tokenized_query},
                rank=rank
            ))
            rank += 1
            
            if len(results) >= k:
                break
        
        # 매칭 결과가 없으면 안내 메시지 포함
        if not results:
            # 점수가 가장 높은 문서라도 반환 (참고용)
            results.append(SearchResult(
                content=f"[키워드 '{' '.join(tokenized_query)}' 매칭 없음]",
                score=0.0,
                metadata={"no_match": True},
                rank=1
            ))
        
        return results
    
    def dense_search(self, query: str, k: int = 10) -> List[SearchResult]:
        """Vector DB Dense 검색"""
        # 실제 문서 수만큼만 반환
        k = min(k, len(self.documents))
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        
        results = []
        for rank, (doc, score) in enumerate(docs_with_scores):
            # score는 거리(distance)이므로 유사도로 변환
            # 1/(1+distance) 방식: 항상 0~1 범위, distance=0일 때 1, distance->infinity 일 때 0
            similarity = 1 / (1 + score)
            results.append(SearchResult(
                content=doc.page_content,
                score=float(similarity),
                metadata={**doc.metadata, "raw_distance": float(score)},
                rank=rank + 1
            ))
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.5
    ) -> List[SearchResult]:
        """
        하이브리드 검색 (Sparse + Dense)
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            alpha: Dense 가중치 (0~1, 1-alpha가 Sparse 가중치)
        
        점수 계산 방식:
            - Sparse: BM25 점수를 최대값으로 정규화 후 (1-alpha) 가중치
            - Dense: 유사도 점수를 최대값으로 정규화 후 alpha 가중치
            - 최종: 두 점수 합산 (0~1 범위)
        """
        # Sparse 및 Dense 검색 수행
        sparse_results = self.sparse_search(query, k=k*2)
        dense_results = self.dense_search(query, k=k*2)
        
        # 원본 점수 저장 (디버깅/분석용)
        sparse_raw_scores = {r.content: r.score for r in sparse_results}
        dense_raw_scores = {r.content: r.score for r in dense_results}
        
        # 점수 정규화
        sparse_scores = [r.score for r in sparse_results]
        dense_scores = [r.score for r in dense_results]
        
        sparse_max = max(sparse_scores) if sparse_scores and max(sparse_scores) > 0 else 1.0
        dense_max = max(dense_scores) if dense_scores and max(dense_scores) > 0 else 1.0
        
        # 결과 병합 (중복 제거)
        combined = {}
        
        for result in sparse_results:
            content = result.content
            normalized_score = (result.score / sparse_max) * (1 - alpha)
            combined[content] = SearchResult(
                content=content,
                score=normalized_score,
                metadata={
                    **result.metadata,
                    "sparse_raw": result.score,
                    "sparse_norm": normalized_score,
                },
                rank=0
            )
        
        for result in dense_results:
            content = result.content
            normalized_score = (result.score / dense_max) * alpha
            
            if content in combined:
                # 기존 Sparse 결과에 Dense 점수 추가
                combined[content].score += normalized_score
                combined[content].metadata["dense_raw"] = result.score
                combined[content].metadata["dense_norm"] = normalized_score
            else:
                combined[content] = SearchResult(
                    content=content,
                    score=normalized_score,
                    metadata={
                        **result.metadata,
                        "dense_raw": result.score,
                        "dense_norm": normalized_score,
                    },
                    rank=0
                )
        
        # 점수로 정렬
        sorted_results = sorted(combined.values(), key=lambda x: x.score, reverse=True)
        
        # 순위 재할당
        for rank, result in enumerate(sorted_results[:k]):
            result.rank = rank + 1
        
        return sorted_results[:k]


class Reranker:
    """Re-ranking 모델 (경량 모델 사용)"""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Args:
            model_name: 리랭킹 모델 이름
            기본값: cross-encoder/ms-marco-MiniLM-L-6-v2 (경량, ~80MB)
            대안: BAAI/bge-reranker-base (고성능, ~500MB)
        """
        print(f"[...] Re-ranker 모델 로딩 중... ({model_name})")
        self.model = CrossEncoder(model_name)
        print("[OK] Re-ranker 준비 완료")
    
    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        검색 결과 재순위화
        
        Args:
            query: 검색 쿼리
            results: 초기 검색 결과
            top_k: 반환할 상위 결과 수
        """
        import math
        
        if not results:
            return []
        
        # 쿼리-문서 쌍 생성
        pairs = [[query, result.content] for result in results]
        
        # Re-ranking 점수 계산 (Cross-Encoder는 로짓 점수 반환: -infinity ~ +infinity)
        raw_scores = self.model.predict(pairs)
        
        # 로짓 점수를 0~1 범위의 확률로 변환 (sigmoid 함수)
        # 이렇게 하면 Before/After 점수 스케일이 일치하여 비교 가능
        normalized_scores = [1 / (1 + math.exp(-s)) for s in raw_scores]
        
        # 점수 업데이트 및 재정렬
        reranked = []
        for result, norm_score, raw_score in zip(results, normalized_scores, raw_scores):
            reranked.append(SearchResult(
                content=result.content,
                score=float(norm_score),  # 정규화된 점수 (0~1 범위)
                metadata={**result.metadata, "raw_rerank_score": float(raw_score)},
                rank=0
            ))
        
        reranked.sort(key=lambda x: x.score, reverse=True)
        
        # 순위 재할당
        for rank, result in enumerate(reranked[:top_k]):
            result.rank = rank + 1
        
        return reranked[:top_k]


class MultiHopRetriever:
    """Multi-hop 질의: 두 단계 검색으로 답 찾기"""
    
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Optional[Reranker] = None,
        llm: Optional[ChatOpenAI] = None
    ):
        """
        Args:
            retriever: 하이브리드 검색기
            reranker: 리랭커 (선택)
            llm: LLM 모델 (쿼리 분해용)
        """
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # 쿼리 분해 프롬프트
        self.decompose_prompt = ChatPromptTemplate.from_template(
            """다음 질문을 두 개의 하위 질문으로 분해하세요.
첫 번째 질문의 답변을 바탕으로 두 번째 질문에 답할 수 있어야 합니다.

원본 질문: {question}

다음 형식으로 답변하세요:
1. [첫 번째 하위 질문]
2. [두 번째 하위 질문]

하위 질문:"""
        )
    
    def decompose_query(self, query: str) -> List[str]:
        """복잡한 쿼리를 하위 쿼리로 분해"""
        print(f"\n[>>>] 쿼리 분해 중: {query}")
        
        chain = self.decompose_prompt | self.llm
        response = chain.invoke({"question": query})
        
        # 응답 파싱
        lines = response.content.strip().split('\n')
        sub_queries = []
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # "1. " 또는 "- " 제거
                sub_query = line.split('.', 1)[-1].strip() if '.' in line else line[1:].strip()
                sub_queries.append(sub_query)
        
        print(f"  -> 하위 질문 1: {sub_queries[0] if len(sub_queries) > 0 else 'N/A'}")
        print(f"  -> 하위 질문 2: {sub_queries[1] if len(sub_queries) > 1 else 'N/A'}")
        
        return sub_queries
    
    def multi_hop_search(
        self,
        query: str,
        k_per_hop: int = 5,
        use_reranker: bool = True
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """
        Multi-hop 검색 수행
        
        Args:
            query: 원본 쿼리
            k_per_hop: 각 홉당 검색 결과 수
            use_reranker: 리랭커 사용 여부
        
        Returns:
            (최종 검색 결과, 메타데이터)
        """
        metadata = {"sub_queries": [], "hop_results": []}
        
        # 1단계: 쿼리 분해
        sub_queries = self.decompose_query(query)
        metadata["sub_queries"] = sub_queries
        
        if len(sub_queries) < 2:
            print("[!] 쿼리 분해 실패, 일반 검색으로 대체")
            results = self.retriever.hybrid_search(query, k=k_per_hop)
            if use_reranker and self.reranker:
                results = self.reranker.rerank(query, results, top_k=k_per_hop)
            return results, metadata
        
        # 2단계: 첫 번째 홉 검색
        print(f"\n[>>>] Hop 1 검색 중...")
        hop1_results = self.retriever.hybrid_search(sub_queries[0], k=k_per_hop)
        if use_reranker and self.reranker:
            hop1_results = self.reranker.rerank(sub_queries[0], hop1_results, top_k=k_per_hop)
        
        metadata["hop_results"].append({
            "query": sub_queries[0],
            "num_results": len(hop1_results)
        })
        
        # 3단계: 첫 번째 홉 결과를 컨텍스트로 두 번째 홉 검색
        print(f"[>>>] Hop 2 검색 중...")
        hop1_context = "\n\n".join([r.content[:200] + "..." for r in hop1_results[:3]])
        enhanced_query = f"{sub_queries[1]}\n\n참고 정보:\n{hop1_context}"
        
        hop2_results = self.retriever.hybrid_search(enhanced_query, k=k_per_hop)
        if use_reranker and self.reranker:
            hop2_results = self.reranker.rerank(sub_queries[1], hop2_results, top_k=k_per_hop)
        
        metadata["hop_results"].append({
            "query": sub_queries[1],
            "num_results": len(hop2_results)
        })
        
        # 4단계: 두 홉의 결과 병합 (중복 제거)
        combined = {}
        for result in hop1_results + hop2_results:
            if result.content not in combined:
                combined[result.content] = result
            else:
                # 더 높은 점수 유지
                if result.score > combined[result.content].score:
                    combined[result.content] = result
        
        final_results = sorted(combined.values(), key=lambda x: x.score, reverse=True)[:k_per_hop]
        
        # 순위 재할당
        for rank, result in enumerate(final_results):
            result.rank = rank + 1
        
        print(f"[OK] Multi-hop 검색 완료 (최종 결과: {len(final_results)}개)")
        
        return final_results, metadata


class ContextWindowManager:
    """컨텍스트 윈도우 관리"""
    
    def __init__(self, model: str = "gpt-4o-mini", max_tokens: int = 4096):
        """
        Args:
            model: 사용할 모델 이름
            max_tokens: 최대 토큰 수
        """
        self.model = model
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)
    
    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        return len(self.encoding.encode(text))
    
    def fit_context(
        self,
        query: str,
        results: List[SearchResult],
        system_prompt: str = "",
        reserve_tokens: int = 1000
    ) -> Tuple[List[SearchResult], Dict[str, int]]:
        """
        컨텍스트 윈도우에 맞게 결과 조정
        
        Args:
            query: 쿼리
            results: 검색 결과
            system_prompt: 시스템 프롬프트
            reserve_tokens: 응답 생성을 위한 예약 토큰
        
        Returns:
            (조정된 결과, 토큰 통계)
        """
        # 고정 토큰 계산
        system_tokens = self.count_tokens(system_prompt)
        query_tokens = self.count_tokens(query)
        fixed_tokens = system_tokens + query_tokens + reserve_tokens
        
        # 컨텍스트에 사용 가능한 토큰
        available_tokens = self.max_tokens - fixed_tokens
        
        print(f"\n[INFO] 컨텍스트 윈도우 관리:")
        print(f"  - 최대 토큰: {self.max_tokens}")
        print(f"  - 시스템 프롬프트: {system_tokens} 토큰")
        print(f"  - 쿼리: {query_tokens} 토큰")
        print(f"  - 예약 (응답용): {reserve_tokens} 토큰")
        print(f"  - 사용 가능: {available_tokens} 토큰")
        
        # 결과를 토큰 제한에 맞게 조정
        fitted_results = []
        used_tokens = 0
        
        for result in results:
            result_tokens = self.count_tokens(result.content)
            
            if used_tokens + result_tokens <= available_tokens:
                fitted_results.append(result)
                used_tokens += result_tokens
            else:
                # 부분적으로 포함 가능한지 확인
                remaining_tokens = available_tokens - used_tokens
                if remaining_tokens > 100:  # 최소 100 토큰은 있어야 의미 있음
                    # 텍스트 자르기
                    tokens = self.encoding.encode(result.content)
                    truncated_tokens = tokens[:remaining_tokens]
                    truncated_content = self.encoding.decode(truncated_tokens)
                    
                    fitted_results.append(SearchResult(
                        content=truncated_content + "...",
                        score=result.score,
                        metadata={**result.metadata, "truncated": True},
                        rank=result.rank
                    ))
                    used_tokens += remaining_tokens
                break
        
        stats = {
            "total_tokens": self.max_tokens,
            "system_tokens": system_tokens,
            "query_tokens": query_tokens,
            "reserve_tokens": reserve_tokens,
            "available_tokens": available_tokens,
            "used_tokens": used_tokens,
            "num_results": len(fitted_results),
            "num_truncated": sum(1 for r in fitted_results if r.metadata.get("truncated", False))
        }
        
        print(f"  - 사용된 토큰: {used_tokens}")
        print(f"  - 포함된 결과: {len(fitted_results)}개")
        print(f"  - 잘린 결과: {stats['num_truncated']}개")
        
        return fitted_results, stats


class AdvancedRAGSystem:
    """고급 RAG 시스템 통합"""
    
    def __init__(
        self,
        chunk_size: int = 1024,
        model: str = "gpt-4o-mini",
        use_reranker: bool = True
    ):
        """
        Args:
            chunk_size: 청크 크기
            model: LLM 모델
            use_reranker: 리랭커 사용 여부
        """
        self.chunk_size = chunk_size
        self.model = model
        self.use_reranker = use_reranker
        
        # 컴포넌트 초기화
        self.doc_processor = DocumentProcessor(chunk_size=chunk_size)
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.client = OpenAI()
        
        self.retriever = None
        self.reranker = None
        self.multi_hop_retriever = None
        self.context_manager = ContextWindowManager(model=model)
        
        print(f"Advanced RAG System 초기화 완료 (chunk_size={chunk_size})")
    
    def ingest_documents(
        self,
        file_path: str,
        collection_name: Optional[str] = None
    ):
        """문서 수집 및 인덱싱"""
        print(f"\n[FILE] 문서 로딩 중: {file_path}")
        
        # 문서 로드
        text = self.doc_processor.load_pdf(file_path)
        print(f"  - 원본 텍스트 길이: {len(text)} 문자")
        
        self._ingest_text_internal(text, file_path, collection_name)
    
    def ingest_text(
        self,
        text: str,
        source_name: str = "sample_text",
        collection_name: Optional[str] = None
    ):
        """텍스트 직접 수집 및 인덱싱 (PDF 없이 실습 가능)"""
        print(f"\n[DOC] 텍스트 로딩 중: {source_name}")
        print(f"  - 원본 텍스트 길이: {len(text)} 문자")
        
        self._ingest_text_internal(text, source_name, collection_name)
    
    def _ingest_text_internal(
        self,
        text: str,
        source: str,
        collection_name: Optional[str] = None
    ):
        """내부 인덱싱 로직"""
        # 청킹
        documents = self.doc_processor.create_chunks(
            text,
            metadata={"source": source, "chunk_size": self.chunk_size}
        )
        print(f"  - 생성된 청크 수: {len(documents)}개")
        
        # 하이브리드 검색기 초기화
        if collection_name is None:
            collection_name = f"advanced_rag_{self.chunk_size}"
        
        self.retriever = HybridRetriever(
            documents=documents,
            embeddings=self.embeddings,
            collection_name=collection_name
        )
        
        # 리랭커 초기화
        if self.use_reranker:
            self.reranker = Reranker()
        
        # Multi-hop 검색기 초기화
        self.multi_hop_retriever = MultiHopRetriever(
            retriever=self.retriever,
            reranker=self.reranker,
            llm=self.llm
        )
        
        print("[OK] 문서 인덱싱 완료")
    
    def search(
        self,
        query: str,
        method: str = "hybrid",
        k: int = 5,
        alpha: float = 0.5,
        use_reranker: bool = True
    ) -> List[SearchResult]:
        """
        검색 수행
        
        Args:
            query: 검색 쿼리
            method: 검색 방법 ("sparse", "dense", "hybrid")
            k: 반환할 결과 수
            alpha: 하이브리드 검색 시 Dense 가중치
            use_reranker: 리랭커 사용 여부
        """
        if not self.retriever:
            raise ValueError("문서를 먼저 인덱싱하세요 (ingest_documents 호출)")
        
        # 실제 문서 수 확인
        total_docs = len(self.retriever.documents)
        actual_k = min(k, total_docs)
        
        # 검색 수행
        if method == "sparse":
            results = self.retriever.sparse_search(query, k=actual_k*2 if use_reranker else actual_k)
        elif method == "dense":
            results = self.retriever.dense_search(query, k=actual_k*2 if use_reranker else actual_k)
        elif method == "hybrid":
            results = self.retriever.hybrid_search(query, k=actual_k*2 if use_reranker else actual_k, alpha=alpha)
        else:
            raise ValueError(f"알 수 없는 검색 방법: {method}")
        
        # 리랭킹
        if use_reranker and self.reranker:
            results = self.reranker.rerank(query, results, top_k=actual_k)
        else:
            results = results[:actual_k]
        
        return results
    
    def multi_hop_search(
        self,
        query: str,
        k: int = 5,
        use_reranker: bool = True
    ) -> Tuple[List[SearchResult], Dict[str, Any]]:
        """Multi-hop 검색"""
        if not self.multi_hop_retriever:
            raise ValueError("문서를 먼저 인덱싱하세요 (ingest_documents 호출)")
        
        return self.multi_hop_retriever.multi_hop_search(
            query,
            k_per_hop=k,
            use_reranker=use_reranker
        )
    
    def generate_answer(
        self,
        query: str,
        results: List[SearchResult],
        manage_context: bool = True
    ) -> Dict[str, Any]:
        """
        검색 결과를 바탕으로 답변 생성
        
        Args:
            query: 쿼리
            results: 검색 결과
            manage_context: 컨텍스트 윈도우 관리 여부
        """
        # 시스템 프롬프트 (상세한 답변 유도)
        system_prompt = """당신은 AI/ML 전문가이며, 제공된 문서를 바탕으로 상세하고 교육적인 답변을 제공합니다.

답변 작성 규칙:
1. 컨텍스트에서 관련 정보를 모두 활용하여 **구체적이고 상세하게** 답변
2. 핵심 개념, 예시, 응용 분야를 포함하여 **최소 3-5문장 이상** 작성
3. 리스트나 번호를 활용하여 가독성 있게 정리
4. 컨텍스트에 없는 내용은 추측하지 말고 있는 내용만 활용
5. "문서에 따르면", "컨텍스트에서" 같은 메타 언급 없이 직접 설명"""
        
        # 컨텍스트 윈도우 관리
        if manage_context:
            results, context_stats = self.context_manager.fit_context(
                query=query,
                results=results,
                system_prompt=system_prompt
            )
        else:
            context_stats = None
        
        # 컨텍스트 구성
        context = "\n\n".join([
            f"[문서 {r.rank}] (점수: {r.score:.4f})\n{r.content}"
            for r in results
        ])
        
        # 프롬프트 구성
        user_prompt = f"""컨텍스트:
{context}

질문: {query}

답변:"""
        
        # LLM 호출
        print("\n[...] 답변 생성 중...")
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,  # 약간의 창의성 허용
            max_tokens=500    # 충분한 답변 길이 보장
        )
        
        elapsed_time = time.time() - start_time
        answer = response.choices[0].message.content
        
        # 토큰 사용량 추출
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        
        print(f"[OK] 답변 생성 완료 ({elapsed_time:.2f}초, 입력: {input_tokens}토큰, 출력: {output_tokens}토큰)")
        
        return {
            "answer": answer,
            "query": query,
            "num_results": len(results),
            "context_stats": context_stats,
            "elapsed_time": elapsed_time,
            "model": self.model
        }

# ============================================================================
# 실습 공통 유틸리티
# ============================================================================

def format_chunk(content: str, indent: str = "      ") -> str:
    """청크 내용을 보기 좋게 포맷팅 (전체 내용 표시)"""
    lines = content.strip().split('\n')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line:
            formatted_lines.append(f"{indent}{line}")
    return '\n'.join(formatted_lines)


def print_search_result(result: SearchResult, index: int, show_full: bool = True):
    """검색 결과를 포맷팅하여 출력"""
    chunk_id = result.metadata.get('chunk_id', '?')
    if isinstance(chunk_id, int):
        chunk_id += 1
    
    print(f"  [{index}] 점수: {result.score:.4f} | 청크 #{chunk_id} ({len(result.content)}자)")
    
    if show_full:
        print(f"  {'─'*50}")
        print(format_chunk(result.content))
        print(f"  {'─'*50}")
    else:
        preview = result.content.replace('\n', ' ')[:100]
        print(f"      {preview}...")


# ============================================================================
# 실습 함수들
# ============================================================================

def experiment_chunk_sizes(text: str = None):
    """실습 4: Chunk size 실험"""
    print("\n" + "="*80)
    print("[4] 실습 4: Chunk Size 실험 (256 vs 512 vs 1024)")
    print("="*80)
    print("목표: 청크 크기가 검색 정확도와 답변 품질에 미치는 영향 분석")
    
    sample_text = text or SAMPLE_TEXT
    
    chunk_sizes = [256, 512, 1024]
    test_query = "강화 학습의 주요 알고리즘은?"
    
    results_comparison = []
    
    print(f"\n[*] 테스트 쿼리: '{test_query}'")
    
    for chunk_size in chunk_sizes:
        print(f"\n{'─'*60}")
        print(f"[>] Chunk Size: {chunk_size}자")
        print(f"{'─'*60}")
        
        # RAG 시스템 초기화
        rag = AdvancedRAGSystem(chunk_size=chunk_size, use_reranker=False)
        rag.doc_processor = DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_size//10)
        rag.ingest_text(sample_text, source_name="AI_가이드", collection_name=f"chunk_exp_{chunk_size}")
        
        # 검색
        search_results = rag.search(test_query, method="hybrid", k=3)
        
        # 상위 1개 결과 전체 표시 (청크 크기별 차이 확인용)
        print(f"\n상위 검색 결과:")
        print_search_result(search_results[0], 1, show_full=True)
        
        # 답변 생성
        answer_data = rag.generate_answer(test_query, search_results)
        
        results_comparison.append({
            "chunk_size": chunk_size,
            "num_chunks": len(rag.retriever.documents),
            "top_score": search_results[0].score if search_results else 0,
            "answer": answer_data["answer"],
            "elapsed_time": answer_data["elapsed_time"]
        })
    
    # 비교 결과 출력
    print("\n" + "="*80)
    print("[INFO] Chunk Size 비교 분석")
    print("="*80)
    
    print(f"\n{'청크 크기':<12} {'청크 수':<10} {'생성 시간':<12} {'답변 길이':<10}")
    print("─" * 60)
    for result in results_comparison:
        print(f"{result['chunk_size']:<12} {result['num_chunks']:<10} {result['elapsed_time']:.2f}초{'':<6} {len(result['answer'])}자")
    
    # 각 청크 크기별 답변 비교 출력
    print(f"\n{'─'*60}")
    print("[DOC] 답변 비교 (청크 크기별 차이 확인)")
    print(f"{'─'*60}")
    for result in results_comparison:
        print(f"\n[>] Chunk Size {result['chunk_size']}자 답변:")
        print(f"  {result['answer'][:300]}{'...' if len(result['answer']) > 300 else ''}")
    
    print(f"\n[TIP] Chunk Size 가이드:")
    print("  - 작은 청크 (256~512): 정밀한 검색, 더 많은 청크 처리 필요")
    print("  - 중간 청크 (512~1024): 일반적으로 균형 잡힌 선택")
    print("  - 큰 청크 (1024+): 넓은 컨텍스트, 노이즈 포함 가능성")
    print("  - 권장: 도메인과 질문 유형에 따라 실험으로 결정")


def experiment_search_methods(text: str = None):
    """실습 1: Sparse vs Dense vs Hybrid 검색 비교"""
    print("\n" + "="*80)
    print("[1] 실습 1: 검색 방법 비교 (Sparse vs Dense vs Hybrid)")
    print("="*80)
    print("목표: 키워드 기반(Sparse) vs 의미 기반(Dense) vs 결합(Hybrid) 방식 비교")
    
    # 샘플 텍스트 사용
    sample_text = text or SAMPLE_TEXT
    
    # RAG 시스템 초기화 (적절한 청크 크기로 분할)
    rag = AdvancedRAGSystem(chunk_size=400, use_reranker=False)
    rag.doc_processor = DocumentProcessor(chunk_size=400, chunk_overlap=50)
    rag.ingest_text(sample_text, source_name="AI_가이드", collection_name="search_method_exp")
    
    # 두 가지 테스트 쿼리: 키워드 매칭 vs 의미 검색
    test_queries = [
        ("강화 학습이란?", "정확한 키워드 매칭 테스트"),
        ("AI가 그림을 그리는 방법", "의미적 유사도 테스트 (키워드 불일치)"),
    ]
    
    methods = ["sparse", "dense", "hybrid"]
    
    print(f"\n총 문서 청크 수: {len(rag.retriever.documents)}개")
    
    for query, desc in test_queries:
        print(f"\n{'─'*70}")
        print(f"[*] 테스트 쿼리: '{query}'")
        # 토큰화된 쿼리 표시
        tokenized = rag.retriever.tokenize_korean(query)
        print(f"   토큰화: {tokenized}")
        print(f"   ({desc})")
        print(f"{'─'*70}")
        
        for method in methods:
            results = rag.search(query, method=method, k=3)
            
            # 점수 타입 설명 추가
            score_type = {
                "sparse": "BM25 원본",
                "dense": "벡터 유사도 (0~1)",
                "hybrid": "정규화 결합 (0~1)"
            }.get(method, "")
            
            print(f"\n[>] {method.upper()} 검색:")
            # 상위 1개 결과만 전체 표시 (가독성)
            for i, result in enumerate(results[:1], 1):
                # 키워드 매칭 없음 체크
                if result.metadata.get("no_match"):
                    print(f"  [{i}] [X] 키워드 매칭 없음 (점수: 0)")
                else:
                    chunk_id = result.metadata.get('chunk_id', '?')
                    # 원본 점수 표시 (있으면)
                    raw_distance = result.metadata.get('raw_distance', None)
                    extra_info = f" [거리: {raw_distance:.4f}]" if raw_distance is not None else ""
                    print(f"  [{i}] 점수: {result.score:.4f} ({score_type}){extra_info} | 청크 #{chunk_id+1} ({len(result.content)}자)")
                    print(f"  {'─'*50}")
                    print(format_chunk(result.content))
                    print(f"  {'─'*50}")
        
        # Hybrid 결과의 점수 분해 표시 (이미 검색한 결과 사용)
        hybrid_results = rag.search(query, method="hybrid", k=1)
        if hybrid_results:
            hr = hybrid_results[0]
            sparse_raw = hr.metadata.get('sparse_raw', None)
            dense_raw = hr.metadata.get('dense_raw', None)
            sparse_norm = hr.metadata.get('sparse_norm', None)
            dense_norm = hr.metadata.get('dense_norm', None)
            
            print(f"\n[INFO] Hybrid 점수 분해 (상위 1개):")
            print(f"  청크 #{hr.metadata.get('chunk_id', '?') + 1} 기준:")
            if sparse_raw is not None and sparse_norm is not None:
                print(f"  - Sparse: {sparse_raw:.4f} (BM25) -> {sparse_norm:.4f} (정규화x0.5)")
            else:
                print(f"  - Sparse: 매칭 없음")
            if dense_raw is not None and dense_norm is not None:
                print(f"  - Dense:  {dense_raw:.4f} (유사도) -> {dense_norm:.4f} (정규화x0.5)")
            else:
                print(f"  - Dense: 매칭 없음")
            total = (sparse_norm or 0) + (dense_norm or 0)
            print(f"  - 합계:   {total:.4f} -> 최종: {hr.score:.4f}")
        
        # 방법별 특징 설명
        print(f"\n[TIP] 분석:")
        print("  - Sparse(BM25): 쿼리 단어가 문서에 정확히 있어야 높은 점수")
        print("  - Dense(Vector): 의미가 비슷하면 단어가 달라도 높은 점수")
        print("  - Hybrid: 두 점수를 정규화 후 결합 (alpha=0.5 기본값)")
        print("  - 점수 범위: 모두 0~1로 정규화되어 비교 가능")


def experiment_reranking(text: str = None):
    """실습 2: Re-ranking 효과 비교"""
    print("\n" + "="*80)
    print("[2] 실습 2: Re-ranking 효과 비교")
    print("="*80)
    print("목표: 초기 검색 결과를 Cross-Encoder로 재순위화하여 정확도 향상")
    print("모델: cross-encoder/ms-marco-MiniLM-L-6-v2 (경량, ~80MB)")
    
    sample_text = text or SAMPLE_TEXT
    
    # RAG 시스템 초기화 (더 작은 청크로 많은 결과 생성)
    rag = AdvancedRAGSystem(chunk_size=300, use_reranker=True)
    rag.doc_processor = DocumentProcessor(chunk_size=300, chunk_overlap=30)
    rag.ingest_text(sample_text, source_name="AI_가이드", collection_name="reranking_exp")
    
    # 구체적인 질문 (일부 청크가 부분적으로만 관련있는 경우)
    test_query = "딥러닝에서 Transformer 아키텍처의 핵심 원리는?"
    
    print(f"\n[*] 테스트 쿼리: '{test_query}'")
    print(f"총 청크 수: {len(rag.retriever.documents)}개")
    
    # Re-ranking 없이 (초기 검색 결과)
    print(f"\n{'─'*60}")
    print("[>] BEFORE: Re-ranking 없이 (Hybrid 검색 결과)")
    print(f"{'─'*60}")
    results_without = rag.search(test_query, method="hybrid", k=5, use_reranker=False)
    
    # 상위 2개 결과 전체 표시
    for i, result in enumerate(results_without[:2], 1):
        print_search_result(result, i, show_full=True)
    
    # Re-ranking 적용
    print(f"\n{'─'*60}")
    print("[>] AFTER: Re-ranking 적용 (Cross-Encoder 재순위화)")
    print(f"{'─'*60}")
    results_with = rag.search(test_query, method="hybrid", k=5, use_reranker=True)
    
    # 상위 2개 결과 전체 표시
    for i, result in enumerate(results_with[:2], 1):
        print_search_result(result, i, show_full=True)
    
    # 순위 변화 분석
    print(f"\n{'─'*60}")
    print("[INFO] 분석: 순위 변화")
    print(f"{'─'*60}")
    
    # 청크 ID로 비교
    before_chunk = results_without[0].metadata.get('chunk_id', -1)
    after_chunk = results_with[0].metadata.get('chunk_id', -1)
    
    print(f"  Before 1위: 청크 #{before_chunk + 1} (점수: {results_without[0].score:.4f})")
    print(f"  After 1위:  청크 #{after_chunk + 1} (점수: {results_with[0].score:.4f})")
    
    # 원본 Cross-Encoder 로짓 점수도 표시
    raw_rerank_score = results_with[0].metadata.get('raw_rerank_score', None)
    if raw_rerank_score is not None:
        print(f"             (Cross-Encoder 로짓: {raw_rerank_score:.4f})")
    
    if before_chunk != after_chunk:
        print("\n  [OK] Re-ranking으로 순위가 변경됨!")
        print("  -> Cross-Encoder가 쿼리-문서 관련성을 더 정확히 평가")
    else:
        print("\n  -> 1위는 동일하나, 하위 순위에서 변화 발생 가능")
    
    print(f"\n[TIP] Re-ranking의 핵심:")
    print("  - 초기 검색: Hybrid 점수 (0~1 범위)")
    print("  - Re-ranking: Cross-Encoder 점수 -> sigmoid 정규화 (0~1 범위)")
    print("  - 같은 스케일로 비교 가능 (점수가 높을수록 관련성 높음)")
    print("  - 실무: 초기 20~50개 -> Re-ranking -> 상위 5~10개 사용")


def experiment_multi_hop(text: str = None):
    """실습 3: Multi-hop 검색"""
    print("\n" + "="*80)
    print("[3] 실습 3: Multi-hop 검색 (다단계 추론)")
    print("="*80)
    print("목표: 복잡한 질문을 하위 질문으로 분해하고 단계적으로 검색")
    
    sample_text = text or SAMPLE_TEXT
    
    # RAG 시스템 초기화
    rag = AdvancedRAGSystem(chunk_size=400, use_reranker=False)
    rag.doc_processor = DocumentProcessor(chunk_size=400, chunk_overlap=50)
    rag.ingest_text(sample_text, source_name="AI_가이드", collection_name="multihop_exp")
    
    # 복잡한 쿼리 (두 개념을 연결해야 답할 수 있는 질문)
    complex_query = "딥러닝 모델의 종류와 각각이 어떤 실무 분야에 적용되는지 설명해주세요"
    
    print(f"\n[*] 복잡한 쿼리: '{complex_query}'")
    print(f"총 청크 수: {len(rag.retriever.documents)}개")
    
    # 일반 검색과 Multi-hop 비교
    print(f"\n{'─'*60}")
    print("[>] 일반 Hybrid 검색 (단일 쿼리)")
    print(f"{'─'*60}")
    
    simple_results = rag.search(complex_query, method="hybrid", k=3)
    print(f"검색 결과: {len(simple_results)}개")
    # 상위 1개 전체 표시
    print_search_result(simple_results[0], 1, show_full=True)
    
    # Multi-hop 검색
    print(f"\n{'─'*60}")
    print("[>] Multi-hop 검색 (쿼리 분해 -> 단계적 검색)")
    print(f"{'─'*60}")
    
    results, metadata = rag.multi_hop_search(complex_query, k=3, use_reranker=False)
    
    print(f"\n[INFO] Multi-hop 과정:")
    print(f"  - 원본 쿼리 -> {len(metadata['sub_queries'])}개 하위 질문으로 분해")
    for i, sq in enumerate(metadata.get('sub_queries', []), 1):
        print(f"    {i}. {sq}")
    
    print(f"\n  - Hop 1: 첫 번째 질문으로 검색")
    print(f"  - Hop 2: Hop 1 결과를 참고하여 두 번째 질문 검색")
    print(f"  - 최종 결과: 두 홉의 결과 병합 ({len(results)}개)")
    
    print(f"\n상위 결과:")
    # 상위 2개 전체 표시
    for i, result in enumerate(results[:2], 1):
        print_search_result(result, i, show_full=True)
    
    # 답변 생성
    print(f"\n{'─'*60}")
    print("[>] LLM 답변 생성")
    print(f"{'─'*60}")
    answer_data = rag.generate_answer(complex_query, results)
    print(f"\n{answer_data['answer']}")
    
    print(f"\n[TIP] Multi-hop의 핵심:")
    print("  - 복잡한 질문을 단순한 하위 질문으로 분해")
    print("  - 첫 번째 검색 결과를 두 번째 검색의 컨텍스트로 활용")
    print("  - '개념 A란?' + 'A가 어디에 쓰이나?' 형태의 질문에 효과적")


def experiment_context_window(text: str = None):
    """실습 5: 컨텍스트 윈도우 관리"""
    print("\n" + "="*80)
    print("[5] 실습 5: 컨텍스트 윈도우 관리")
    print("="*80)
    print("목표: LLM의 토큰 제한을 고려한 컨텍스트 최적화")
    
    sample_text = text or SAMPLE_TEXT
    
    # RAG 시스템 초기화 (작은 청크로 많은 결과 생성)
    rag = AdvancedRAGSystem(chunk_size=300, use_reranker=False)
    rag.doc_processor = DocumentProcessor(chunk_size=300, chunk_overlap=30)
    rag.ingest_text(sample_text, source_name="AI_가이드", collection_name="context_exp")
    
    test_query = "인공지능의 모든 분야와 활용 사례를 설명해주세요"
    
    print(f"\n[*] 테스트 쿼리: '{test_query}'")
    print(f"총 청크 수: {len(rag.retriever.documents)}개")
    
    # 많은 결과 검색 (컨텍스트 윈도우 초과 가능)
    k = min(15, len(rag.retriever.documents))
    print(f"검색 결과 요청: {k}개")
    
    results = rag.search(test_query, method="hybrid", k=k)
    
    # 전체 토큰 수 계산
    total_tokens = sum(rag.context_manager.count_tokens(r.content) for r in results)
    print(f"검색된 컨텍스트 총 토큰: {total_tokens}개")
    
    # 컨텍스트 윈도우 관리 없이
    print(f"\n{'─'*60}")
    print("[>] 컨텍스트 윈도우 관리 없이 (전체 결과 사용)")
    print(f"{'─'*60}")
    
    answer_data_without = rag.generate_answer(test_query, results, manage_context=False)
    print(f"  - 사용된 결과: {len(results)}개 (전체)")
    print(f"  - 생성 시간: {answer_data_without['elapsed_time']:.2f}초")
    print(f"  - 답변 길이: {len(answer_data_without['answer'])}자")
    print(f"\n[DOC] 답변 (관리 없음):")
    print(f"  {answer_data_without['answer'][:400]}{'...' if len(answer_data_without['answer']) > 400 else ''}")
    
    # 컨텍스트 윈도우 관리 적용 (더 제한적인 설정)
    print(f"\n{'─'*60}")
    print("[>] 컨텍스트 윈도우 관리 적용 (토큰 제한 준수)")
    print(f"{'─'*60}")
    
    # 더 작은 토큰 제한으로 테스트
    rag.context_manager.max_tokens = 2048  # 더 제한적으로 설정
    
    answer_data_with = rag.generate_answer(test_query, results, manage_context=True)
    
    if answer_data_with["context_stats"]:
        stats = answer_data_with["context_stats"]
        print(f"\n[INFO] 토큰 분배:")
        print(f"  +-- 최대 토큰: {stats['total_tokens']}개")
        print(f"  +-- 시스템 프롬프트: {stats['system_tokens']}개")
        print(f"  +-- 쿼리: {stats['query_tokens']}개")
        print(f"  +-- 응답 예약: {stats['reserve_tokens']}개")
        print(f"  +-- 컨텍스트 가용: {stats['available_tokens']}개")
        print(f"  +-- 컨텍스트 사용: {stats['used_tokens']}개")
        
        print(f"\n[INFO] 결과 필터링:")
        print(f"  - 원본 결과: {len(results)}개")
        print(f"  - 포함된 결과: {stats['num_results']}개")
        print(f"  - 잘린 결과: {stats['num_truncated']}개")
        print(f"  - 제외된 결과: {len(results) - stats['num_results']}개")
    
    print(f"\n  - 생성 시간: {answer_data_with['elapsed_time']:.2f}초")
    print(f"  - 답변 길이: {len(answer_data_with['answer'])}자")
    print(f"\n[DOC] 답변 (관리 적용):")
    print(f"  {answer_data_with['answer'][:400]}{'...' if len(answer_data_with['answer']) > 400 else ''}")
    
    # 답변 비교
    print(f"\n{'─'*60}")
    print("[INFO] 답변 비교 분석")
    print(f"{'─'*60}")
    print(f"  관리 없음: {len(answer_data_without['answer'])}자, {answer_data_without['elapsed_time']:.2f}초")
    print(f"  관리 적용: {len(answer_data_with['answer'])}자, {answer_data_with['elapsed_time']:.2f}초")
    
    if len(answer_data_with['answer']) > len(answer_data_without['answer']):
        print(f"  -> 컨텍스트 제한이 오히려 더 집중된 답변 생성 (노이즈 감소 효과)")
    else:
        print(f"  -> 더 많은 컨텍스트가 더 긴 답변 생성")
    
    print(f"\n[TIP] 컨텍스트 윈도우 관리의 핵심:")
    print("  - LLM마다 최대 토큰 제한 존재 (GPT-4: 8K~128K)")
    print("  - 프롬프트 + 컨텍스트 + 응답 <= 최대 토큰")
    print("  - 관련성 높은 청크부터 포함, 초과 시 자르거나 제외")
    print("  - 실무: 토큰 비용과 품질의 균형 고려")


def run_all_experiments(text: str = None, pdf_path: str = None, force_sample: bool = False):
    """
    모든 실습 실행
    
    Args:
        text: 샘플 텍스트 (기본: 내장 SAMPLE_TEXT)
        pdf_path: PDF 파일 경로 (선택)
        force_sample: True면 PDF 무시하고 내장 텍스트 사용
    """
    print("\n[LIST] 실습 항목:")
    print("  1. 검색 방법 비교 (Sparse vs Dense vs Hybrid)")
    print("  2. Re-ranking 효과 (Cross-Encoder 활용)")
    print("  3. Multi-hop 검색 (다단계 추론)")
    print("  4. Chunk size 실험 (256 vs 512 vs 1024)")
    print("  5. 컨텍스트 윈도우 관리")
    print()
    
    # 텍스트 결정
    sample_text = None
    
    if not force_sample and pdf_path and os.path.exists(pdf_path):
        processor = DocumentProcessor()
        pdf_text = processor.load_pdf(pdf_path)
        
        if len(pdf_text) >= MIN_TEXT_LENGTH:
            print(f"[FILE] PDF 파일 사용: {pdf_path}")
            sample_text = pdf_text
        else:
            print(f"[!] PDF 파일이 너무 작음 ({len(pdf_text)}자 < {MIN_TEXT_LENGTH}자)")
            print("   -> 내장 샘플 텍스트로 대체합니다.")
            sample_text = text or SAMPLE_TEXT
    else:
        print("[DOC] 내장 샘플 텍스트 사용 (AI/ML 가이드)")
        sample_text = text or SAMPLE_TEXT
    
    print(f"   텍스트 길이: {len(sample_text)} 문자")
    
    try:
        # 실습 1: 검색 방법 비교
        experiment_search_methods(sample_text)
        
        # 실습 2: Re-ranking
        experiment_reranking(sample_text)
        
        # 실습 3: Multi-hop
        experiment_multi_hop(sample_text)
        
        # 실습 4: Chunk size
        experiment_chunk_sizes(sample_text)
        
        # 실습 5: 컨텍스트 윈도우
        experiment_context_window(sample_text)
        
        print("\n" + "="*80)
        print("[OK] 모든 실습 완료!")
        print("="*80)
        print("\n[TIP] 실습 팁:")
        print("  - 더 강력한 Re-ranker: BAAI/bge-reranker-base (~500MB)")
        print("  - 현재 사용: cross-encoder/ms-marco-MiniLM-L-6-v2 (~80MB)")
        print("  - 자신의 PDF로 실습: run_all_experiments(pdf_path='your.pdf')")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


# 메인 실행
if __name__ == "__main__":
    print("="*80)
    print("[LAB] Advanced RAG 실습 - 전체 데모")
    print("="*80)
    
    # 샘플 PDF 파일 경로 (있으면 사용, 없으면 내장 텍스트 사용)
    sample_pdf = Path(__file__).parent / "sample.pdf"
    
    if sample_pdf.exists():
        # PDF 파일이 있으면 사용 (선택)
        print(f"\n[FILE] PDF 파일 발견: {sample_pdf}")
        print("   (내장 샘플 텍스트로 실행하려면 PDF를 삭제하세요)")
        run_all_experiments(pdf_path=str(sample_pdf))
    else:
        # PDF 없이 내장 샘플 텍스트로 실행
        print("\n[DOC] PDF 없이 내장 샘플 텍스트로 실행")
        print("   (자신의 PDF로 실습하려면 sample.pdf를 lab03 폴더에 추가)")
        run_all_experiments()
