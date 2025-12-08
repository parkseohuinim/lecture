"""
Document Processor - 다양한 문서 형식 처리 및 청킹

지원 형식: PDF, Markdown, JSON, TXT
"""
import logging
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """문서 청크 데이터 클래스"""
    content: str
    chunk_id: int
    metadata: Dict[str, Any]
    

@dataclass
class ProcessedDocument:
    """처리된 문서 데이터 클래스"""
    doc_id: str
    filename: str
    file_type: str
    total_chunks: int
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]


class DocumentProcessor:
    """
    문서 처리 클래스
    
    - PDF, Markdown, JSON, TXT 파일 로딩
    - 지능형 청킹 (문장/단락 경계 고려)
    - 메타데이터 추출
    """
    
    SUPPORTED_EXTENSIONS = {'.pdf', '.md', '.markdown', '.json', '.txt', '.text'}
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Args:
            chunk_size: 청크 최대 크기 (문자 수)
            chunk_overlap: 청크 간 겹침 크기
            separators: 분할에 사용할 구분자 목록 (우선순위 순)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or [
            "\n\n",      # 단락 구분
            "\n",        # 줄 바꿈
            ". ",        # 문장 끝
            "? ",        # 질문 끝
            "! ",        # 느낌표 끝
            ", ",        # 쉼표
            " ",         # 공백
        ]
    
    def process_file(
        self,
        file_path: str,
        file_content: Optional[bytes] = None,
        doc_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> ProcessedDocument:
        """
        파일을 처리하여 청크로 분할
        
        Args:
            file_path: 파일 경로 (확장자 판별용)
            file_content: 파일 내용 (bytes), None이면 파일에서 읽음
            doc_id: 문서 ID (None이면 자동 생성)
            extra_metadata: 추가 메타데이터
        
        Returns:
            ProcessedDocument: 처리된 문서
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        filename = path.name
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"지원하지 않는 파일 형식: {extension}")
        
        # 파일 내용 읽기
        if file_content is None:
            with open(file_path, 'rb') as f:
                file_content = f.read()
        
        # 확장자별 텍스트 추출
        if extension == '.pdf':
            text = self._extract_pdf_text(file_content)
        elif extension == '.json':
            text = self._extract_json_text(file_content)
        else:  # .md, .markdown, .txt, .text
            text = file_content.decode('utf-8', errors='ignore')
        
        # 텍스트 정리
        text = self._clean_text(text)
        
        # 청킹
        chunks = self._chunk_text(text, filename)
        
        # 문서 ID 생성
        if doc_id is None:
            import hashlib
            doc_id = hashlib.md5(f"{filename}_{len(text)}".encode()).hexdigest()[:12]
        
        # 메타데이터 구성
        metadata = {
            "filename": filename,
            "file_type": extension[1:],  # .pdf -> pdf
            "total_length": len(text),
            "chunk_size_setting": self.chunk_size,
            "chunk_overlap_setting": self.chunk_overlap,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        
        logger.info(f"📄 문서 처리 완료: {filename} ({len(chunks)}개 청크)")
        
        return ProcessedDocument(
            doc_id=doc_id,
            filename=filename,
            file_type=extension[1:],
            total_chunks=len(chunks),
            chunks=chunks,
            metadata=metadata
        )
    
    def _extract_pdf_text(self, content: bytes) -> str:
        """PDF에서 텍스트 추출"""
        try:
            import pdfplumber
            import io
            
            text_parts = []
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"[페이지 {page_num}]\n{page_text}")
            
            return "\n\n".join(text_parts)
        except ImportError:
            logger.warning("pdfplumber가 설치되지 않음. 대체 방법 시도...")
            # 대체: pymupdf4llm 사용
            try:
                import pymupdf4llm
                import io
                return pymupdf4llm.to_markdown(io.BytesIO(content))
            except:
                raise ImportError("PDF 처리를 위해 pdfplumber 또는 pymupdf4llm이 필요합니다.")
    
    def _extract_json_text(self, content: bytes) -> str:
        """JSON에서 텍스트 추출 (재귀적으로 모든 문자열 값 추출)"""
        try:
            data = json.loads(content.decode('utf-8'))
            texts = []
            self._extract_strings_from_json(data, texts)
            return "\n\n".join(texts)
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 실패: {e}")
            # JSON 파싱 실패 시 텍스트로 처리
            return content.decode('utf-8', errors='ignore')
    
    def _extract_strings_from_json(self, obj: Any, texts: List[str], prefix: str = ""):
        """JSON에서 재귀적으로 문자열 추출"""
        if isinstance(obj, str):
            if len(obj.strip()) > 10:  # 짧은 문자열 제외
                texts.append(f"{prefix}: {obj}" if prefix else obj)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                self._extract_strings_from_json(value, texts, new_prefix)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                self._extract_strings_from_json(item, texts, new_prefix)
    
    def _clean_text(self, text: str) -> str:
        """텍스트 정리"""
        # 연속 공백 제거
        text = re.sub(r' +', ' ', text)
        # 연속 줄바꿈 정리 (3개 이상 -> 2개)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 앞뒤 공백 제거
        text = text.strip()
        return text
    
    def _chunk_text(self, text: str, source: str) -> List[DocumentChunk]:
        """
        텍스트를 청크로 분할 (문장/단락 경계 고려)
        
        Lab03의 DocumentProcessor.chunk_text 방식 참고
        """
        chunks = []
        start = 0
        text_length = len(text)
        chunk_id = 0
        
        while start < text_length:
            end = start + self.chunk_size
            
            # 텍스트 끝이면 그냥 추가
            if end >= text_length:
                chunk_text = text[start:].strip()
                if chunk_text:
                    chunks.append(DocumentChunk(
                        content=chunk_text,
                        chunk_id=chunk_id,
                        metadata={
                            "source": source,
                            "chunk_id": chunk_id,
                            "start_char": start,
                            "end_char": text_length
                        }
                    ))
                break
            
            # 적절한 분할 지점 찾기
            best_end = self._find_split_point(text, start, end)
            
            chunk_text = text[start:best_end].strip()
            if chunk_text:
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    chunk_id=chunk_id,
                    metadata={
                        "source": source,
                        "chunk_id": chunk_id,
                        "start_char": start,
                        "end_char": best_end
                    }
                ))
                chunk_id += 1
            
            # 다음 청크 시작 위치 (오버랩 적용)
            next_start = best_end - self.chunk_overlap
            
            # 진행이 없으면 강제로 앞으로 (무한 루프 방지)
            if next_start <= start:
                next_start = best_end
            
            start = next_start
        
        return chunks
    
    def _find_split_point(self, text: str, start: int, end: int) -> int:
        """적절한 분할 지점 찾기 (구분자 우선순위 적용)"""
        search_end = min(end + 50, len(text))  # 약간의 여유
        
        for separator in self.separators:
            # 현재 범위에서 구분자 찾기 (뒤에서부터)
            pos = text.rfind(separator, start, search_end)
            if pos != -1 and pos > start + self.chunk_size // 2:
                return pos + len(separator)
        
        # 구분자를 찾지 못하면 강제 분할
        return end
    
    def update_settings(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None
    ):
        """청킹 설정 업데이트"""
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
        if separators is not None:
            self.separators = separators
        
        logger.info(f"📝 청킹 설정 업데이트: size={self.chunk_size}, overlap={self.chunk_overlap}")

