"""
RAG Service - 문서 기반 질의응답 통합 서비스

기능:
- 문서 업로드 및 인덱싱
- 하이브리드 검색 (BM25 + Vector)
- Re-ranking
- LLM 기반 답변 생성
"""
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from app.application.rag.document_processor import DocumentProcessor, ProcessedDocument, DocumentChunk
from app.application.rag.hybrid_retriever import HybridRetriever, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """RAG 응답 데이터 클래스"""
    answer: str
    sources: List[Dict[str, Any]]
    search_method: str
    total_sources: int
    query: str
    confidence: str  # "high", "medium", "low"


@dataclass
class DocumentInfo:
    """문서 정보 데이터 클래스"""
    doc_id: str
    filename: str
    file_type: str
    total_chunks: int
    uploaded_at: str
    metadata: Dict[str, Any]


class RAGService:
    """
    RAG 서비스 클래스
    
    사용법:
    ```python
    rag = RAGService()
    
    # 문서 업로드
    doc_info = await rag.upload_document(file_content, filename)
    
    # 질의응답
    response = await rag.query("이 문서의 핵심 내용은?")
    ```
    """
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_reranker: bool = True
    ):
        """
        Args:
            collection_name: ChromaDB 컬렉션 이름
            persist_directory: 데이터 저장 경로
            chunk_size: 청크 크기
            chunk_overlap: 청크 겹침
            use_reranker: Re-ranker 사용 여부
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # 컴포넌트 초기화
        self.doc_processor = DocumentProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.retriever = HybridRetriever(
            collection_name=collection_name,
            persist_directory=persist_directory,
            use_reranker=use_reranker
        )
        
        # 문서 메타데이터 저장 (메모리)
        self.documents: Dict[str, DocumentInfo] = {}
        
        # OpenAI 클라이언트
        self.openai_client = None
        self._init_openai()
        
        # 초기화
        self.retriever.initialize_collection()
        
        # 기존 문서 메타데이터 복원
        self._restore_document_metadata()
        
        logger.info(f"📚 RAG Service 초기화 완료 (복원된 문서: {len(self.documents)}개)")
    
    def _restore_document_metadata(self):
        """
        ChromaDB에서 기존 문서 메타데이터 복원
        서버 재시작 시 문서 목록을 유지하기 위함
        """
        try:
            if self.retriever.collection is None:
                return
            
            count = self.retriever.collection.count()
            if count == 0:
                return
            
            # 모든 메타데이터 가져오기
            results = self.retriever.collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            
            if not metadatas:
                return
            
            # doc_id별로 그룹화하여 문서 정보 복원
            doc_chunks: Dict[str, List[Dict]] = {}
            for meta in metadatas:
                if not meta:
                    continue
                doc_id = meta.get("doc_id")
                if doc_id:
                    if doc_id not in doc_chunks:
                        doc_chunks[doc_id] = []
                    doc_chunks[doc_id].append(meta)
            
            # DocumentInfo 복원
            for doc_id, chunks in doc_chunks.items():
                if not chunks:
                    continue
                    
                first_chunk = chunks[0]
                filename = first_chunk.get("filename", "unknown")
                file_type = first_chunk.get("file_type", "unknown")
                
                self.documents[doc_id] = DocumentInfo(
                    doc_id=doc_id,
                    filename=filename,
                    file_type=file_type,
                    total_chunks=len(chunks),
                    uploaded_at=first_chunk.get("uploaded_at", "unknown"),
                    metadata={
                        "restored": True,
                        "chunk_count": len(chunks)
                    }
                )
            
            logger.info(f"📖 {len(self.documents)}개 문서 메타데이터 복원됨")
            
        except Exception as e:
            logger.warning(f"⚠️ 문서 메타데이터 복원 실패: {e}")
    
    def _init_openai(self):
        """OpenAI 클라이언트 초기화"""
        from openai import OpenAI
        import httpx
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        
        http_client = httpx.Client(verify=False)
        self.openai_client = OpenAI(api_key=api_key, http_client=http_client)
    
    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> DocumentInfo:
        """
        문서 업로드 및 인덱싱
        
        Args:
            file_content: 파일 내용 (bytes)
            filename: 파일명
            extra_metadata: 추가 메타데이터
        
        Returns:
            DocumentInfo: 업로드된 문서 정보
        """
        try:
            logger.info(f"📤 문서 업로드 시작: {filename}")
            
            # 문서 처리
            processed = self.doc_processor.process_file(
                file_path=filename,
                file_content=file_content,
                extra_metadata=extra_metadata
            )
            
            # 청크를 검색기에 추가
            texts = [chunk.content for chunk in processed.chunks]
            metadatas = []
            ids = []
            
            uploaded_at = datetime.now().isoformat()
            
            for chunk in processed.chunks:
                meta = {
                    **chunk.metadata,
                    "doc_id": processed.doc_id,
                    "filename": processed.filename,
                    "file_type": processed.file_type,
                    "uploaded_at": uploaded_at  # 복원용 메타데이터
                }
                metadatas.append(meta)
                ids.append(f"{processed.doc_id}_{chunk.chunk_id}")
            
            self.retriever.add_documents(texts, metadatas, ids)
            
            # 문서 정보 저장
            doc_info = DocumentInfo(
                doc_id=processed.doc_id,
                filename=processed.filename,
                file_type=processed.file_type,
                total_chunks=processed.total_chunks,
                uploaded_at=datetime.now().isoformat(),
                metadata=processed.metadata
            )
            self.documents[processed.doc_id] = doc_info
            
            logger.info(f"✅ 문서 업로드 완료: {filename} ({processed.total_chunks}개 청크)")
            
            return doc_info
        
        except Exception as e:
            logger.error(f"❌ 문서 업로드 실패: {e}")
            raise
    
    async def query(
        self,
        question: str,
        k: int = 5,
        search_method: str = "hybrid",
        alpha: float = 0.5,
        use_reranker: Optional[bool] = None,
        doc_filter: Optional[str] = None
    ) -> RAGResponse:
        """
        질의응답 수행
        
        Args:
            question: 질문
            k: 검색할 문서 수
            search_method: 검색 방법 ("sparse", "dense", "hybrid")
            alpha: 하이브리드 검색 시 Dense 가중치
            use_reranker: Re-ranker 사용 여부
            doc_filter: 특정 문서만 검색 (doc_id)
        
        Returns:
            RAGResponse: 답변 및 출처
        """
        try:
            logger.info(f"🔍 질의: {question[:50]}...")
            
            # 검색
            search_results = self.retriever.search(
                query=question,
                k=k,
                method=search_method,
                alpha=alpha,
                use_reranker=use_reranker
            )
            
            # 문서 필터 적용
            if doc_filter:
                search_results = [
                    r for r in search_results 
                    if r.metadata.get("doc_id") == doc_filter
                ]
            
            if not search_results:
                return RAGResponse(
                    answer="죄송합니다. 관련 문서를 찾을 수 없습니다. 문서를 먼저 업로드해주세요.",
                    sources=[],
                    search_method=search_method,
                    total_sources=0,
                    query=question,
                    confidence="low"
                )
            
            # 컨텍스트 구성
            context = self._build_context(search_results)
            
            # LLM 답변 생성
            answer = await self._generate_answer(question, context, search_results)
            
            # 신뢰도 판단
            confidence = self._assess_confidence(search_results)
            
            # 출처 정보 구성
            sources = []
            for r in search_results:
                sources.append({
                    "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "score": round(r.score, 4),
                    "rank": r.rank,
                    "filename": r.metadata.get("filename", "unknown"),
                    "chunk_id": r.metadata.get("chunk_id", -1)
                })
            
            logger.info(f"✅ 답변 생성 완료 (출처: {len(sources)}개, 신뢰도: {confidence})")
            
            return RAGResponse(
                answer=answer,
                sources=sources,
                search_method=search_results[0].search_type if search_results else search_method,
                total_sources=len(sources),
                query=question,
                confidence=confidence
            )
        
        except Exception as e:
            logger.error(f"❌ 질의 처리 실패: {e}")
            raise
    
    def _build_context(self, results: List[SearchResult], max_tokens: int = 3000) -> str:
        """검색 결과로 컨텍스트 구성"""
        context_parts = []
        total_length = 0
        
        for r in results:
            # 대략적인 토큰 추정 (한글 기준 약 2자당 1토큰)
            estimated_tokens = len(r.content) // 2
            
            if total_length + estimated_tokens > max_tokens:
                break
            
            context_parts.append(f"[출처: {r.metadata.get('filename', 'unknown')}]\n{r.content}")
            total_length += estimated_tokens
        
        return "\n\n---\n\n".join(context_parts)
    
    async def _generate_answer(
        self,
        question: str,
        context: str,
        results: List[SearchResult]
    ) -> str:
        """LLM으로 답변 생성"""
        
        system_prompt = """당신은 문서 기반 질의응답 전문가입니다.

## 지침
1. 제공된 컨텍스트(문서)를 기반으로 질문에 답변하세요.
2. 컨텍스트에 없는 내용은 추측하지 마세요.
3. 답변에 관련 출처를 언급하세요.
4. 명확하고 구조화된 답변을 제공하세요.
5. 컨텍스트에서 답을 찾을 수 없으면 솔직히 "문서에서 관련 정보를 찾을 수 없습니다"라고 답하세요.

## 답변 형식
- 핵심 답변을 먼저 제시
- 필요시 상세 설명 추가
- 관련 출처 언급"""
        
        user_prompt = f"""## 컨텍스트 (검색된 문서)

{context}

---

## 질문

{question}

---

## 요청

위 컨텍스트를 바탕으로 질문에 상세히 답변해주세요."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"LLM 답변 생성 실패: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    async def _generate_answer_stream(
        self,
        question: str,
        context: str
    ):
        """LLM으로 스트리밍 답변 생성 (토큰 단위) - llm_service 사용"""
        from app.infrastructure.llm.llm_service import llm_service
        
        system_prompt = """당신은 문서 기반 질의응답 전문가입니다.

## 지침
1. 제공된 컨텍스트(문서)를 기반으로 질문에 답변하세요.
2. 컨텍스트에 없는 내용은 추측하지 마세요.
3. 답변에 관련 출처를 언급하세요.
4. 명확하고 구조화된 답변을 제공하세요.
5. 컨텍스트에서 답을 찾을 수 없으면 솔직히 "문서에서 관련 정보를 찾을 수 없습니다"라고 답하세요.

## 답변 형식
- 핵심 답변을 먼저 제시
- 필요시 상세 설명 추가
- 관련 출처 언급"""
        
        user_prompt = f"""## 컨텍스트 (검색된 문서)

{context}

---

## 질문

{question}

---

## 요청

위 컨텍스트를 바탕으로 질문에 상세히 답변해주세요."""
        
        try:
            # llm_service의 스트리밍 메서드 사용 (Azure/OpenAI 폴백 지원)
            async for token in llm_service.generate_response_stream(
                prompt=user_prompt,
                system_prompt=system_prompt
            ):
                yield token
                    
        except Exception as e:
            logger.error(f"LLM 스트리밍 답변 생성 실패: {e}")
            yield f"답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    async def query_stream(
        self,
        question: str,
        k: int = 5,
        search_method: str = "hybrid",
        alpha: float = 0.5,
        use_reranker: Optional[bool] = None,
        doc_filter: Optional[str] = None
    ):
        """
        스트리밍 질의응답 수행 (토큰 단위 출력)
        
        Yields:
            dict: {"type": "...", "data": ...}
            - type: "sources" | "token" | "done" | "error"
        """
        try:
            logger.info(f"🔍 스트리밍 질의: {question[:50]}...")
            
            # 검색
            search_results = self.retriever.search(
                query=question,
                k=k,
                method=search_method,
                alpha=alpha,
                use_reranker=use_reranker
            )
            
            # 문서 필터 적용
            if doc_filter:
                search_results = [
                    r for r in search_results 
                    if r.metadata.get("doc_id") == doc_filter
                ]
            
            if not search_results:
                yield {
                    "type": "error",
                    "data": "관련 문서를 찾을 수 없습니다. 문서를 먼저 업로드해주세요."
                }
                return
            
            # 출처 정보 먼저 전송
            sources = []
            for r in search_results:
                sources.append({
                    "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "score": round(r.score, 4),
                    "rank": r.rank,
                    "filename": r.metadata.get("filename", "unknown"),
                    "chunk_id": r.metadata.get("chunk_id", -1)
                })
            
            confidence = self._assess_confidence(search_results)
            search_type = search_results[0].search_type if search_results else search_method
            
            yield {
                "type": "sources",
                "data": {
                    "sources": sources,
                    "confidence": confidence,
                    "search_method": search_type,
                    "total_sources": len(sources)
                }
            }
            
            # 컨텍스트 구성
            context = self._build_context(search_results)
            
            # 토큰 단위 스트리밍 답변 생성
            async for token in self._generate_answer_stream(question, context):
                yield {
                    "type": "token",
                    "data": token
                }
            
            # 완료 신호
            yield {
                "type": "done",
                "data": None
            }
            
            logger.info(f"✅ 스트리밍 답변 완료")
        
        except Exception as e:
            logger.error(f"❌ 스트리밍 질의 실패: {e}")
            yield {
                "type": "error",
                "data": str(e)
            }
    
    def _assess_confidence(self, results: List[SearchResult]) -> str:
        """검색 결과 기반 신뢰도 평가"""
        if not results:
            return "low"
        
        top_score = results[0].score
        
        # Re-ranked 결과인 경우 (sigmoid 적용된 점수)
        if results[0].search_type == "reranked":
            if top_score >= 0.8:
                return "high"
            elif top_score >= 0.5:
                return "medium"
            else:
                return "low"
        
        # Hybrid/기타 결과
        if top_score >= 0.7:
            return "high"
        elif top_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """업로드된 문서 목록 반환"""
        return [asdict(doc) for doc in self.documents.values()]
    
    def delete_document(self, doc_id: str) -> bool:
        """문서 삭제"""
        if doc_id not in self.documents:
            return False
        
        # ChromaDB에서 해당 문서의 모든 청크 삭제
        doc_info = self.documents[doc_id]
        for i in range(doc_info.total_chunks):
            chunk_id = f"{doc_id}_{i}"
            self.retriever.delete_document(chunk_id)
        
        # 메타데이터에서 삭제
        del self.documents[doc_id]
        
        logger.info(f"🗑️ 문서 삭제 완료: {doc_id}")
        return True
    
    def clear_all_documents(self):
        """모든 문서 삭제"""
        self.retriever.clear_all()
        self.documents.clear()
        logger.info("🗑️ 모든 문서 삭제 완료")
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        retriever_stats = self.retriever.get_stats()
        return {
            **retriever_stats,
            "total_documents": len(self.documents),
            "document_list": list(self.documents.keys())
        }
    
    def update_settings(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        use_reranker: Optional[bool] = None
    ):
        """설정 업데이트"""
        if chunk_size or chunk_overlap:
            self.doc_processor.update_settings(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        if use_reranker is not None:
            self.retriever.use_reranker = use_reranker
        
        logger.info("⚙️ 설정 업데이트 완료")


# Global service instance (싱글톤)
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """RAG 서비스 싱글톤 인스턴스 반환"""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


# 편의를 위한 alias
rag_service = get_rag_service

