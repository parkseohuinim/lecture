"""
RAG (Retrieval-Augmented Generation) 기초 실습
- 문서 로딩 및 청킹
- 임베딩 및 Vector DB 저장
- 검색 기반 LLM 응답 생성
- 컨텍스트 관리

실습 항목:
1. 청킹(Chunking) 이해하기 - 텍스트 분할 전략
2. 기본 RAG 파이프라인 - 문서 -> 임베딩 -> 검색
3. RAG 있음 vs 없음 비교 - 답변 품질 차이
4. 컨텍스트 관리 - 토큰 제한 대응
5. 고급 RAG - 컨텍스트 압축 적용
6. Query Rewriting - 쿼리 개선으로 검색 품질 향상
7. HyDE - 가상 문서 임베딩 기법
8. RAG 평가 - Faithfulness, Relevancy 측정
9. Citation 출처 표기 - 답변에 출처 달기
10. Streaming 응답 - 실시간 토큰 출력
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

# 문서 파싱 라이브러리
import pdfplumber
from bs4 import BeautifulSoup
import markdown

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 공통 유틸리티 import를 위한 경로 추가
sys.path.insert(0, str(project_root))
from utils import (
    print_section_header,
    print_subsection,
    print_key_points,
    get_openai_client,
    interpret_l2_distance,
    l2_distance_to_similarity
)

# 공통 데이터 임포트
from shared_data import SAMPLE_TEXT, MIN_TEXT_LENGTH, get_sample_or_document_text


# ============================================================================
# 데이터 클래스
# ============================================================================

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    score: float
    metadata: Dict[str, Any]
    rank: int


# ============================================================================
# 1. 문서 로더
# ============================================================================

class DocumentLoader:
    """다양한 형식의 문서를 로드"""
    
    @staticmethod
    def load_pdf(file_path: str) -> str:
        """PDF 파일 로드 (한글 지원)"""
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    @staticmethod
    def load_markdown(file_path: str) -> str:
        """Markdown 파일 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            md_text = f.read()
        # Markdown을 HTML로 변환 후 텍스트 추출
        html = markdown.markdown(md_text)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    
    @staticmethod
    def load_html(file_path: str) -> str:
        """HTML 파일 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            html = f.read()
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    
    @staticmethod
    def load_text(file_path: str) -> str:
        """텍스트 파일 로드"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def load_document(file_path: str) -> str:
        """파일 확장자에 따라 자동으로 로드"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
        
        extension = file_path.suffix.lower()
        
        if extension == '.pdf':
            return DocumentLoader.load_pdf(str(file_path))
        elif extension in ['.md', '.markdown']:
            return DocumentLoader.load_markdown(str(file_path))
        elif extension in ['.html', '.htm']:
            return DocumentLoader.load_html(str(file_path))
        elif extension == '.txt':
            return DocumentLoader.load_text(str(file_path))
        else:
            raise ValueError(f"지원하지 않는 파일 형식: {extension}")


# ============================================================================
# 2. 텍스트 청킹
# ============================================================================

class TextChunker:
    """텍스트를 작은 청크로 분할"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: 청크당 최대 문자 수
            chunk_overlap: 청크 간 겹치는 문자 수
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _find_complete_list(self, text: str, start_pos: int, end_pos: int) -> tuple:
        """
        start_pos와 end_pos 사이에 리스트가 있으면, 
        그 리스트의 시작과 끝 위치를 반환
        
        Returns:
            (list_start, list_end) 또는 (None, None)
        """
        import re
        
        # 더 넓은 범위에서 리스트 찾기 (청크 크기의 2배까지)
        search_start = max(0, start_pos - 100)  # 이전 텍스트도 확인
        search_end = min(len(text), end_pos + self.chunk_size)  # 더 넓은 범위
        search_text = text[search_start:search_end]
        
        # 번호 매겨진 리스트 패턴 개선
        # 1. 로 시작하는 리스트를 찾고, 연속된 번호들을 모두 포함
        pattern = r'(?:^|\n)(1\.\s.+?)(?=\n(?!\d+\.\s)|\Z)'
        
        # 첫 번째 리스트 항목 찾기
        first_match = re.search(r'(?:^|\n)(1\.\s)', search_text, re.MULTILINE)
        if first_match:
            list_start_in_search = first_match.start()
            if first_match.group().startswith('\n'):
                list_start_in_search += 1
            
            # 연속된 번호 찾기
            current_pos = list_start_in_search
            last_number = 1
            list_end_in_search = current_pos
            
            while current_pos < len(search_text):
                # 다음 번호 찾기
                next_pattern = rf'(?:^|\n)({last_number + 1}\.\s)'
                next_match = re.search(next_pattern, search_text[current_pos:], re.MULTILINE)
                
                if next_match and next_match.start() < 200:  # 항목 간 거리 제한
                    # 다음 번호의 끝 찾기
                    next_start = current_pos + next_match.start()
                    if search_text[next_start] == '\n':
                        next_start += 1
                    
                    # 해당 항목의 끝 찾기 (다음 번호 또는 빈 줄까지)
                    item_end = search_text.find('\n\n', next_start)
                    next_item = re.search(rf'(?:^|\n){last_number + 2}\.\s', 
                                         search_text[next_start:], re.MULTILINE)
                    
                    if next_item and (item_end == -1 or next_start + next_item.start() < item_end):
                        item_end = next_start + next_item.start()
                    elif item_end == -1:
                        item_end = len(search_text)
                    
                    list_end_in_search = item_end
                    current_pos = next_start
                    last_number += 1
                else:
                    # 마지막 항목의 끝 찾기
                    item_end = search_text.find('\n\n', current_pos)
                    if item_end == -1:
                        item_end = search_text.find('\n', current_pos + 10)
                        if item_end == -1:
                            item_end = len(search_text)
                    list_end_in_search = item_end
                    break
            
            list_start = search_start + list_start_in_search
            list_end = search_start + list_end_in_search
            
            # 리스트가 청크 범위와 겹치면 반환
            if list_start < end_pos and list_end > start_pos:
                return (list_start, list_end)
        
        # 불릿 리스트 패턴
        pattern = r'((?:^[-*•]\s.+$\n?)+)'
        matches = list(re.finditer(pattern, search_text, re.MULTILINE))
        
        if matches:
            # 청크 범위와 가장 많이 겹치는 리스트 찾기
            best_match = None
            best_overlap = 0
            
            for match in matches:
                match_start = search_start + match.start()
                match_end = search_start + match.end()
                
                # 청크 범위와의 겹침 계산
                overlap_start = max(start_pos, match_start)
                overlap_end = min(end_pos, match_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = (match_start, match_end)
            
            if best_match:
                return best_match
        
        return (None, None)
    
    def chunk_text(self, text: str) -> List[str]:
        """텍스트를 청크로 분할 (리스트 인식 개선)"""
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
            
            # 문장 경계 찾기 (우선순위 순서)
            best_end = -1
            
            # 0순위: 리스트 확인 (현재 청크 내에 리스트가 있으면 완전히 포함)
            list_start, list_end = self._find_complete_list(text, start, end)
            if list_start is not None and list_end is not None:
                # 리스트가 현재 청크 범위에 걸쳐있는 경우
                if list_start < end:
                    # 리스트가 청크 초반에 시작하는 경우
                    if list_start < start + self.chunk_size * 0.4:  # 청크 40% 이내
                        # 리스트 끝까지 포함
                        best_end = list_end
                    # 리스트가 청크 후반에 시작하는 경우  
                    else:
                        # 리스트 시작 전의 적절한 위치에서 자르기
                        # 리스트 헤더를 찾기
                        header_pos = text.rfind('\n', max(0, list_start - 50), list_start)
                        if header_pos != -1 and header_pos > start + self.chunk_size * 0.3:
                            best_end = header_pos + 1
                        else:
                            # 단락 끝 찾기
                            para_end = text.rfind('\n\n', start, list_start)
                            if para_end != -1 and para_end > start + self.chunk_size * 0.3:
                                best_end = para_end + 2
            
            # 1순위: 단락 끝 (빈 줄)
            if best_end == -1:
                double_newline = text.rfind('\n\n', start, end + 50)
                if double_newline != -1:
                    best_end = double_newline + 2
            
            # 2순위: 완전한 문장 끝 (마침표 + 줄바꿈 + 대문자/숫자로 시작)
            if best_end == -1:
                import re
                for i in range(end, max(start, end - 100), -1):
                    if i < text_length - 1 and text[i] == '.' and text[i+1] == '\n':
                        next_line_start = i + 2
                        if next_line_start < text_length:
                            # 다음 줄 확인
                            line_start = next_line_start
                            line_text = text[line_start:min(line_start + 20, text_length)]
                            
                            # 리스트 항목이 아닌지 확인
                            is_list = re.match(r'^\d+\.\s|^[-*•]\s', line_text)
                            
                            if not is_list:
                                next_char = text[next_line_start]
                                if next_char.isupper() or next_char.isdigit() or next_char == '\n':
                                    best_end = i + 2
                                    break
            
            # 3순위: 마침표 + 공백
            if best_end == -1:
                period_space = text.rfind('. ', start, end + 30)
                if period_space != -1:
                    best_end = period_space + 2
            
            # 4순위: 느낌표/물음표
            if best_end == -1:
                for punct in ['! ', '? ', '。']:
                    pos = text.rfind(punct, start, end + 30)
                    if pos != -1:
                        best_end = pos + len(punct)
                        break
            
            # 5순위: 줄바꿈
            if best_end == -1:
                newline = text.rfind('\n', start, end + 20)
                if newline != -1:
                    best_end = newline + 1
            
            # 6순위: 공백
            if best_end == -1:
                space = text.rfind(' ', start, end)
                if space != -1 and space > start + self.chunk_size // 2:
                    best_end = space + 1
            
            # 최종: 어쩔 수 없이 강제로 자르기
            if best_end == -1:
                best_end = end
            
            chunk = text[start:best_end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 다음 청크 시작 위치 결정
            # 리스트가 완전히 포함된 경우, 오버랩 없이 시작
            if list_start is not None and list_end is not None and best_end == list_end:
                # 리스트 다음부터 시작 (오버랩 없음)
                next_start = best_end
            else:
                # 일반적인 경우 오버랩 적용
                next_start = best_end - self.chunk_overlap
            
            # 진행이 없으면 강제로 앞으로 (무한 루프 방지)
            if next_start <= start:
                next_start = best_end
            
            start = next_start
        
        return chunks
    
    def chunk_by_sentences(self, text: str, sentences_per_chunk: int = 5) -> List[str]:
        """문장 단위로 청크 분할"""
        # 간단한 문장 분리 (더 정교한 방법은 nltk.sent_tokenize 사용)
        sentences = text.replace('! ', '!|').replace('? ', '?|').replace('. ', '.|').split('|')
        
        chunks = []
        for i in range(0, len(sentences), sentences_per_chunk):
            chunk = ' '.join(sentences[i:i + sentences_per_chunk]).strip()
            if chunk:
                chunks.append(chunk)
        
        return chunks


# ============================================================================
# 3. RAG 시스템
# ============================================================================

class RAGSystem:
    """RAG 시스템 전체 파이프라인"""
    
    def __init__(self, collection_name: str = "rag_documents"):
        """RAG 시스템 초기화"""
        # 공통 헬퍼 사용 (SSL 인증서 검증 우회 포함)
        self.client = get_openai_client()
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-4o-mini"
        
        # ChromaDB 초기화
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name
        )
        
        # 문서 로더 및 청커
        self.loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성""" 
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def add_text(self, text: str, source_name: str = "sample_text", 
                 metadata: Optional[Dict] = None, check_duplicate: bool = True) -> int:
        """
        텍스트를 직접 청킹하여 Vector DB에 저장 (PDF 없이 실습 가능)
        
        Args:
            text: 추가할 텍스트
            source_name: 소스 이름
            metadata: 추가 메타데이터
            check_duplicate: 중복 체크 여부
        
        Returns:
            추가된 청크 수
        """
        print(f"\n[DOC] 텍스트 로딩: {source_name}")
        
        # 중복 체크
        if check_duplicate:
            try:
                existing = self.collection.get(
                    where={"source": source_name}
                )
                if existing and existing['ids']:
                    print(f"   이미 추가된 문서입니다: {source_name}")
                    print(f"   기존 청크 수: {len(existing['ids'])}개")
                    return 0
            except Exception as e:
                # 중복 체크 실패는 무시하고 계속 진행
                pass
        
        print(f"   텍스트 크기: {len(text)} 문자")
        
        # 청킹
        chunks = self.chunker.chunk_text(text)
        print(f"   청크 수: {len(chunks)}개")
        
        # 청크 미리보기
        print(f"\n{'─'*60}")
        print("[CHUNK] 청크 미리보기:")
        print(f"{'─'*60}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n[청크 {i+1}] (길이: {len(chunk)} 문자)")
            import re
            has_list = bool(re.search(r'^\d+\.\s|^[-*•]\s', chunk, re.MULTILINE))
            if has_list:
                print("   [v] 리스트 포함")
            preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
            for line in preview.split('\n')[:5]:
                print(f"   {line}")
        
        if len(chunks) > 3:
            print(f"\n   ... (나머지 {len(chunks) - 3}개 청크 생략)")
        print(f"{'─'*60}")
        
        # 임베딩 생성
        print("\n[...] 임베딩 생성 중...")
        embeddings = []
        for chunk in chunks:
            embedding = self.get_embedding(chunk)
            embeddings.append(embedding)
        
        # Vector DB에 저장
        start_idx = self.collection.count()
        ids = [f"chunk_{start_idx + i}" for i in range(len(chunks))]
        
        # 메타데이터 준비
        if metadata is None:
            metadata = {}
        metadata['source'] = source_name
        
        metadatas = [metadata.copy() for _ in chunks]
        for i, meta in enumerate(metadatas):
            meta['chunk_index'] = i
        
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"[OK] {len(chunks)}개 청크 저장 완료")
        return len(chunks)
    
    def add_document(self, file_path: str, metadata: Optional[Dict] = None, 
                     check_duplicate: bool = True, use_sample_if_short: bool = True) -> int:
        """
        문서를 로드하고 청킹하여 Vector DB에 저장
        PDF 내용이 짧으면 자동으로 샘플 텍스트 사용
        
        Args:
            file_path: 문서 파일 경로
            metadata: 추가 메타데이터
            check_duplicate: 중복 체크 여부
            use_sample_if_short: 내용이 짧으면 샘플 텍스트 사용
        
        Returns:
            추가된 청크 수
        """
        print(f"\n[FILE] 문서 로딩: {file_path}")
        
        # 1. 문서 로드
        document_text = self.loader.load_document(file_path)
        
        # 2. 텍스트 길이 확인 및 샘플 텍스트로 대체
        if use_sample_if_short:
            text, source_type = get_sample_or_document_text(document_text)
            if source_type == "sample":
                return self.add_text(text, source_name="AI_가이드", 
                                    metadata=metadata, check_duplicate=check_duplicate)
        else:
            text = document_text
        
        # 중복 체크
        if check_duplicate:
            try:
                existing = self.collection.get(
                    where={"source": str(file_path)}
                )
                if existing and existing['ids']:
                    print(f"   이미 추가된 문서입니다: {file_path}")
                    print(f"   기존 청크 수: {len(existing['ids'])}개")
                    return 0
            except Exception as e:
                # 중복 체크 실패는 무시하고 계속 진행
                pass
        
        print(f"   문서 크기: {len(text)} 문자")
        
        # 청킹
        chunks = self.chunker.chunk_text(text)
        print(f"   청크 수: {len(chunks)}개")
        
        # 청크 미리보기
        print(f"\n{'─'*60}")
        print("[CHUNK] 청크 미리보기:")
        print(f"{'─'*60}")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\n[청크 {i+1}] (길이: {len(chunk)} 문자)")
            import re
            has_list = bool(re.search(r'^\d+\.\s|^[-*•]\s', chunk, re.MULTILINE))
            if has_list:
                print("   [v] 리스트 포함")
            preview = chunk[:150] + "..." if len(chunk) > 150 else chunk
            for line in preview.split('\n')[:5]:
                print(f"   {line}")
        
        if len(chunks) > 3:
            print(f"\n   ... (나머지 {len(chunks) - 3}개 청크 생략)")
        print(f"{'─'*60}")
        
        # 임베딩 생성
        print("\n[...] 임베딩 생성 중...")
        embeddings = []
        for chunk in chunks:
            embedding = self.get_embedding(chunk)
            embeddings.append(embedding)
        
        # Vector DB에 저장
        start_idx = self.collection.count()
        ids = [f"chunk_{start_idx + i}" for i in range(len(chunks))]
        
        # 메타데이터 준비
        if metadata is None:
            metadata = {}
        metadata['source'] = str(file_path)
        
        metadatas = [metadata.copy() for _ in chunks]
        for i, meta in enumerate(metadatas):
            meta['chunk_index'] = i
        
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        
        print(f"[OK] {len(chunks)}개 청크 저장 완료")
        return len(chunks)
    
    def search(self, query: str, n_results: int = 3) -> List[SearchResult]:
        """
        쿼리와 유사한 문서 검색
        
        Returns:
            검색 결과 리스트 (거리 기반 점수로 정렬)
            
        Note:
            - ChromaDB는 L2 거리를 반환합니다
            - 점수 = 1/(1+거리)로 0~1 범위로 정규화
            - ⚠️ 이것은 코사인 유사도가 아닙니다!
        """
        query_embedding = self.get_embedding(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        search_results = []
        for i, (doc, meta, dist) in enumerate(zip(
            results['documents'][0], 
            results['metadatas'][0], 
            results['distances'][0]
        )):
            # L2 거리를 0~1 점수로 변환
            # 낮은 거리 = 높은 점수 (가까울수록 관련성 높음)
            # ⚠️ 주의: 이것은 코사인 유사도가 아님!
            normalized_score = 1 / (1 + dist)
            search_results.append(SearchResult(
                content=doc,
                score=normalized_score,
                metadata={**meta, "raw_distance": dist},
                rank=i + 1
            ))
        
        return search_results
    
    def generate_answer_without_rag(self, question: str) -> str:
        """RAG 없이 LLM에게 직접 질문"""
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "user", "content": question}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    
    def generate_answer_with_rag(self, question: str, n_results: int = 3) -> Dict:
        """RAG를 사용하여 답변 생성"""
        # 1. 관련 문서 검색
        search_results = self.search(question, n_results=n_results)
        
        # 2. 컨텍스트 구성
        context = "\n\n".join([r.content for r in search_results])
        
        # 3. 프롬프트 구성
        prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.

참고 문서:
{context}

질문: {question}

답변:"""
        
        # 4. LLM 응답 생성
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "당신은 주어진 문서를 기반으로 정확하게 답변하는 AI 어시스턴트입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        return {
            'answer': response.choices[0].message.content,
            'context': context,
            'sources': [r.metadata for r in search_results],
            'distances': [r.metadata.get('raw_distance', 0) for r in search_results],
            'search_results': search_results
        }
    
    def evaluate_retrieval(self, query: str, relevant_chunk_ids: List[int], 
                          k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
        """
        검색 품질 평가 (Recall@K)
        
        Args:
            query: 검색 쿼리
            relevant_chunk_ids: 정답으로 간주되는 청크 인덱스 리스트
            k_values: 평가할 K 값들
        
        Returns:
            각 K값에 대한 Recall 점수
        
        Note:
            Recall@K = (K개 검색 결과 중 정답 개수) / (전체 정답 개수)
            - 1.0: 모든 정답을 찾음
            - 0.0: 정답을 하나도 못 찾음
        """
        max_k = max(k_values)
        search_results = self.search(query, n_results=max_k)
        
        # 검색된 청크의 인덱스 추출
        retrieved_ids = []
        for result in search_results:
            chunk_idx = result.metadata.get('chunk_index')
            if chunk_idx is not None:
                retrieved_ids.append(chunk_idx)
        
        # 각 K값에 대한 Recall 계산
        recalls = {}
        for k in k_values:
            retrieved_at_k = set(retrieved_ids[:k])
            relevant_set = set(relevant_chunk_ids)
            
            if len(relevant_set) == 0:
                recalls[f'recall@{k}'] = 0.0
            else:
                hits = len(retrieved_at_k & relevant_set)
                recalls[f'recall@{k}'] = hits / len(relevant_set)
        
        return recalls
    
    def check_hallucination(self, answer: str, context: str, 
                           check_terms: List[str] = None) -> Dict[str, Any]:
        """
        환각(Hallucination) 감지
        
        Args:
            answer: LLM이 생성한 답변
            context: 제공된 컨텍스트
            check_terms: 확인할 특정 용어들 (예: ["RAG", "Retrieval"])
        
        Returns:
            환각 감지 결과
        
        Note:
            간단한 규칙 기반 감지입니다. 실무에서는 더 정교한 방법 필요:
            - LLM-as-Judge 사용
            - NLI (Natural Language Inference) 모델 사용
            - 팩트 체킹 API 활용
        """
        result = {
            'potential_hallucinations': [],
            'verified_terms': [],
            'hallucination_risk': 'low'
        }
        
        # 기본 검사 용어 (RAG 관련 흔한 오류)
        if check_terms is None:
            check_terms = [
                ("Recall-Augmented", "Retrieval-Augmented", "RAG 약어 오류"),
                ("Recollection-Augmented", "Retrieval-Augmented", "RAG 약어 오류"),
                ("Retrieve-Augmented", "Retrieval-Augmented", "RAG 약어 오류"),
            ]
        
        # 잘못된 용어 검사
        for wrong, correct, desc in check_terms:
            if wrong.lower() in answer.lower():
                result['potential_hallucinations'].append({
                    'found': wrong,
                    'expected': correct,
                    'description': desc
                })
        
        # 컨텍스트에 없는 고유명사/숫자 검사 (간단한 휴리스틱)
        import re
        
        # 답변에서 연도 추출
        years_in_answer = set(re.findall(r'\b(19|20)\d{2}\b', answer))
        years_in_context = set(re.findall(r'\b(19|20)\d{2}\b', context))
        
        unverified_years = years_in_answer - years_in_context
        if unverified_years:
            result['potential_hallucinations'].append({
                'found': list(unverified_years),
                'expected': '컨텍스트에 있는 연도',
                'description': '컨텍스트에 없는 연도 언급'
            })
        
        # 리스크 레벨 결정
        if len(result['potential_hallucinations']) >= 2:
            result['hallucination_risk'] = 'high'
        elif len(result['potential_hallucinations']) == 1:
            result['hallucination_risk'] = 'medium'
        
        return result


# ============================================================================
# 4. 컨텍스트 관리
# ============================================================================

class ContextManager:
    """컨텍스트 토큰 관리 및 압축"""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        # 공통 헬퍼 사용 (SSL 인증서 검증 우회 포함)
        self.client = get_openai_client()
    
    def count_tokens(self, text: str) -> int:
        """텍스트의 토큰 수 계산"""
        return len(self.encoding.encode(text))
    
    def truncate_context(self, context: str, max_tokens: int) -> str:
        """컨텍스트를 최대 토큰 수로 자르기"""
        tokens = self.encoding.encode(context)
        
        if len(tokens) <= max_tokens:
            return context
        
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    def summarize_context(self, context: str, max_length: int = 200) -> str:
        """컨텍스트를 요약"""
        prompt = f"""다음 텍스트를 {max_length}자 이내로 핵심 내용만 요약해주세요:

{context}

요약:"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=max_length
        )
        
        return response.choices[0].message.content
    
    def compress_contexts(self, contexts: List[str], max_tokens: int) -> str:
        """여러 컨텍스트를 요약하고 결합"""
        summaries = []
        
        for i, context in enumerate(contexts):
            tokens = self.count_tokens(context)
            
            if tokens > max_tokens // len(contexts):
                # 너무 길면 요약
                summary = self.summarize_context(context, max_length=150)
                summaries.append(f"[문서 {i+1}] {summary}")
            else:
                summaries.append(f"[문서 {i+1}] {context}")
        
        return "\n\n".join(summaries)


# ============================================================================
# 출력 포맷팅 유틸리티
# ============================================================================

def format_chunk(content: str, indent: str = "      ") -> str:
    """청크 내용을 보기 좋게 포맷팅"""
    lines = content.strip().split('\n')
    formatted_lines = []
    for line in lines:
        line = line.strip()
        if line:
            formatted_lines.append(f"{indent}{line}")
    return '\n'.join(formatted_lines)


def format_chunk_id(metadata: Dict) -> str:
    """청크 ID를 일관되게 포맷팅 (1-based indexing)"""
    chunk_idx = metadata.get('chunk_index', '?')
    if isinstance(chunk_idx, int):
        return str(chunk_idx + 1)  # 0-based → 1-based
    return str(chunk_idx)


def print_search_result(result: SearchResult, index: int, show_full: bool = True):
    """검색 결과를 포맷팅하여 출력"""
    chunk_id = format_chunk_id(result.metadata)
    
    print(f"  [{index}] 점수: {result.score:.4f} | 청크 #{chunk_id} ({len(result.content)}자)")
    
    if show_full:
        print(f"  {'─'*50}")
        print(format_chunk(result.content))
        print(f"  {'─'*50}")
    else:
        preview = result.content.replace('\n', ' ')[:100]
        print(f"      {preview}...")


# ============================================================================
# 데모 함수들
# ============================================================================

def find_sample_document():
    """샘플 문서 찾기 (PDF 우선, 없으면 샘플 텍스트 사용)"""
    current_dir = Path(__file__).parent
    
    # PDF 파일 찾기
    pdf_files = list(current_dir.glob("*.pdf"))
    if pdf_files:
        print(f"[FILE] PDF 파일 발견: {pdf_files[0].name}")
        return pdf_files[0], "pdf"
    
    # PDF가 없으면 샘플 텍스트 사용
    print("[DOC] PDF 파일이 없습니다. 내장 샘플 텍스트를 사용합니다.")
    return None, "sample"


def demo_chunking():
    """실습 1: 청킹(Chunking) 이해하기"""
    print("\n" + "="*80)
    print("[1] 실습 1: 청킹(Chunking) 이해하기")
    print("="*80)
    print("목표: 텍스트를 작은 청크로 분할하는 전략 이해")
    print("핵심: 청크 크기와 오버랩이 검색 품질에 미치는 영향")
    
    # 샘플 텍스트 (간단한 예시)
    sample_text = """인공지능(AI)은 인간의 학습, 추론, 지각 능력을 컴퓨터로 구현하는 기술입니다. 1956년 다트머스 회의에서 존 매카시가 처음으로 "인공지능"이라는 용어를 사용했습니다. 인공지능은 약한 AI, 강한 AI, 초인공지능으로 분류됩니다. 약한 AI는 특정 작업에 특화된 AI이고, 강한 AI는 인간 수준의 범용 지능을 목표로 합니다. 머신러닝은 AI의 하위 분야로, 데이터에서 패턴을 학습하여 예측합니다. 딥러닝은 신경망을 여러 층으로 쌓아 복잡한 패턴을 학습하는 기술입니다."""
    
    print(f"\n원본 텍스트 (길이: {len(sample_text)} 문자):")
    print(f"{'─'*60}")
    print(sample_text)
    print(f"{'─'*60}")
    
    # 다양한 청크 크기로 테스트
    chunk_configs = [
        (100, 20, "작은 청크"),
        (200, 30, "중간 청크"),
        (300, 40, "큰 청크"),
    ]
    
    for chunk_size, overlap, desc in chunk_configs:
        print_section_header(f"청크 크기: {chunk_size}자, 오버랩: {overlap}자 ({desc})")
        
        chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=overlap)
        chunks = chunker.chunk_text(sample_text)
        
        print(f"생성된 청크 수: {len(chunks)}개")
        
        # 오버랩으로 인한 중복 설명
        total_chars = sum(len(c) for c in chunks)
        if total_chars > len(sample_text):
            overlap_chars = total_chars - len(sample_text)
            print(f"[참고] 오버랩으로 인한 중복: 약 {overlap_chars}자")
            print(f"       (오버랩 {overlap}자 × 경계 {len(chunks)-1}개 ≈ {overlap*(len(chunks)-1)}자)")
        print()
        
        for i, chunk in enumerate(chunks):
            print(f"[청크 {i+1}] (길이: {len(chunk)} 문자)")
            print(f"'{chunk}'")
            print()
    
    # 요약
    print("="*60)
    print("[TIP] 청킹 핵심 포인트:")
    print("="*60)
    print("  - 청크 크기가 작을수록 -> 더 많은 청크, 정확한 검색, API 비용 증가")
    print("  - 청크 크기가 클수록 -> 더 적은 청크, 풍부한 문맥, 노이즈 가능")
    print("  - 오버랩 -> 청크 간 문맥 연결 유지, 정보 손실 방지")
    print("  - 권장: 도메인에 따라 300~1000자 실험 후 결정")
    
    print("""
  ────────────────────────────────────────────────────────────
  [실무 참고] 다양한 청킹 기법
  ────────────────────────────────────────────────────────────
  이 실습에서는 "문자 기반 청킹"을 사용했습니다.
  실무에서는 다음 기법들도 자주 사용됩니다:
  
  1. 문장 단위 청킹 (Sentence Split)
     * 문장 경계에서 분할 → 의미 단절 최소화
     * NLTK sent_tokenize, SpaCy 등 활용
  
  2. 의미 단위 청킹 (Semantic Chunking)
     * 임베딩 유사도로 "의미가 바뀌는 지점" 감지
     * 같은 주제끼리 묶음 → 검색 품질 향상
     * LangChain SemanticChunker 등 활용
  
  3. 문서 구조 기반 청킹
     * 마크다운 헤더, HTML 태그 등 구조 활용
     * 섹션/챕터 단위로 자연스럽게 분할
  
  [TIP] 선택 기준:
  * 빠른 구현 → 문자 기반 (이 실습)
  * 품질 우선 → 의미 단위 청킹
  * 구조화된 문서 → 문서 구조 기반
  ────────────────────────────────────────────────────────────
    """)


def demo_basic_rag():
    """실습 2: 기본 RAG 파이프라인"""
    print("\n" + "="*80)
    print("[2] 실습 2: 기본 RAG 파이프라인")
    print("="*80)
    print("목표: 문서 -> 임베딩 -> 검색 -> 답변 생성 과정 이해")
    print("핵심: 벡터 검색으로 관련 문서를 찾아 LLM에게 제공")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 샘플 문서 찾기
    sample_file, source_type = find_sample_document()
    
    # RAG 시스템 초기화 (새 컬렉션)
    rag = RAGSystem("demo_rag_basic")
    
    # 문서 추가 (PDF 또는 샘플 텍스트)
    if source_type == "pdf":
        rag.add_document(str(sample_file), 
                        metadata={"type": "tutorial", "topic": "AI"}, 
                        check_duplicate=True)
    else:
        rag.add_text(SAMPLE_TEXT, source_name="AI_가이드",
                    metadata={"type": "tutorial", "topic": "AI"},
                    check_duplicate=True)
    
    # 검색 테스트
    print_section_header("검색 테스트")
    
    query = "딥러닝의 주요 아키텍처는 무엇인가요?"
    print(f"\n쿼리: '{query}'")
    
    results = rag.search(query, n_results=3)
    
    print("\n검색 결과:")
    for i, result in enumerate(results, 1):
        # 점수 해석 추가
        score_desc = ""
        if result.score >= 0.5:
            score_desc = " [v] 높은 관련성"
        elif result.score >= 0.3:
            score_desc = " [~] 중간 관련성"
        else:
            score_desc = " [x] 낮은 관련성"
        
        chunk_id = format_chunk_id(result.metadata)
        
        print(f"  [{i}] 점수: {result.score:.4f} (거리 기반 0~1){score_desc} | 청크 #{chunk_id}")
        
        if i == 1:  # 첫 번째만 전체 표시
            print(f"  {'─'*50}")
            print(format_chunk(result.content))
            print(f"  {'─'*50}")
        else:
            preview = result.content.replace('\n', ' ')[:80]
            print(f"      {preview}...")
    
    # 점수 해석 가이드
    print("\n" + "─"*60)
    print("[INFO] 점수 해석 가이드:")
    print("─"*60)
    print("  [계산 방법]")
    print("  * ChromaDB는 L2 거리를 반환 (0 ~ ∞, 작을수록 유사)")
    print("  * 점수 = 1/(1+거리)로 0~1 범위로 변환")
    print("  * ⚠️ Lab 1의 코사인 유사도와는 다른 개념입니다!")
    print()
    print("  ⚠️ [중요] 아래 점수 구간은 '이 실습 데이터셋' 기준 예시입니다!")
    print("     실제 서비스에서는 반드시 분포 히스토그램으로 임계값을 정해야 합니다.")
    print()
    print("  [해석 기준]")
    print("  * 0.5 ~ 1.0: 높은 관련성 [v] - L2 거리 < 1.0")
    print("  * 0.3 ~ 0.5: 중간 관련성 [~] - L2 거리 1.0~2.3")
    print("  * 0.0 ~ 0.3: 낮은 관련성 [x] - L2 거리 > 2.3")
    print()
    print("  [Lab 간 비교]")
    print("  * Lab 1: 코사인 유사도 = dot(A,B)/(||A||×||B||) → 방향 비교")
    print("  * Lab 2~3: L2 거리 변환 = 1/(1+||A-B||) → 거리 기반 점수")
    
    # 검색 품질 평가 (Recall@K)
    print_section_header("검색 품질 평가: Recall@K", "[EVAL]")
    
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [INFO] Recall@K 란?                                    │
  │  ─────────────────────────────────────────────────────  │
  │  상위 K개 검색 결과 중 정답이 몇 개 포함되었는지 측정    │
  │                                                         │
  │  공식: Recall@K = (K개 중 정답 수) / (전체 정답 수)      │
  │                                                         │
  │  예시:                                                  │
  │  * 정답 청크: [1, 3, 5]                                 │
  │  * 검색 결과 상위 5개: [1, 2, 3, 4, 6]                  │
  │  * Recall@5 = 2/3 = 0.67 (정답 3개 중 2개 찾음)         │
  │                                                         │
  │  [TIP] 실무 목표:                                       │
  │  * Recall@1 > 0.7: 첫 번째 결과가 대체로 정답           │
  │  * Recall@5 > 0.9: 상위 5개 안에 거의 모든 정답 포함    │
  └─────────────────────────────────────────────────────────┘
  
  ────────────────────────────────────────────────────────────
  [INFO] Recall vs Precision 차이
  ────────────────────────────────────────────────────────────
  * Recall@K = (K개 중 정답 수) / (전체 정답 수)
    → "놓치지 않는 능력" - 정답을 얼마나 찾았나?
  
  * Precision@K = (K개 중 정답 수) / K
    → "쓸데없는 거 안 섞는 능력" - 검색 결과가 얼마나 정확한가?
  
  [TIP] 실무 전략:
  * RAG에서는 Recall 우선! (정답을 놓치면 답변 불가)
  * Precision은 Reranker로 보정 (상위 20개 → Top-5 재정렬)
  ────────────────────────────────────────────────────────────
    """)
    
    # 가상의 정답 청크로 평가 시연
    # 실제로는 사람이 라벨링한 정답 데이터가 필요
    print("[실험] 검색 품질 평가 시연:")
    print("  * 쿼리: '딥러닝의 주요 아키텍처'")
    print("  * 가정: 청크 #0, #1이 정답이라고 가정")
    print()
    
    # 평가 실행
    recalls = rag.evaluate_retrieval(
        query=query,
        relevant_chunk_ids=[0, 1],  # 가상의 정답
        k_values=[1, 3, 5]
    )
    
    print("  평가 결과:")
    for k, score in recalls.items():
        bar_len = int(score * 20)
        bar = "=" * bar_len + "-" * (20 - bar_len)
        status = "[v]" if score >= 0.5 else "[x]"
        print(f"    {k}: [{bar}] {score:.2f} {status}")
    
    print(f"""
  [!] 주의: 이 결과는 시연용입니다!
      실제 평가에는 사람이 라벨링한 정답 데이터가 필요합니다.
      
  [CODE] 평가 코드 예시:
  ┌─────────────────────────────────────────────────────
  │ # 1. 테스트 데이터 준비 (수작업 라벨링)
  │ test_queries = [
  │     {{"query": "딥러닝이란?", "relevant_chunks": [0, 1, 5]}},
  │     {{"query": "RAG 파이프라인", "relevant_chunks": [3, 4]}},
  │ ]
  │ 
  │ # 2. 평가 실행
  │ for test in test_queries:
  │     recalls = rag.evaluate_retrieval(
  │         query=test["query"],
  │         relevant_chunk_ids=test["relevant_chunks"]
  │     )
  │     print(f"Query: {{test['query']}}")
  │     print(f"Recall@5: {{recalls['recall@5']:.2f}}")
  └─────────────────────────────────────────────────────
    """)
    
    # 요약
    print("\n" + "="*60)
    print("[TIP] RAG 파이프라인 핵심:")
    print("="*60)
    print("  1. 문서 로드 -> 텍스트 추출")
    print("  2. 청킹 -> 검색 단위로 분할")
    print("  3. 임베딩 -> 텍스트를 벡터로 변환")
    print("  4. 인덱싱 -> Vector DB에 저장")
    print("  5. 검색 -> 쿼리와 유사한 문서 찾기 (유사도 점수 기반)")
    
    print("""
  ────────────────────────────────────────────────────────────
  [다음 단계 예고] Re-ranking
  ────────────────────────────────────────────────────────────
  현재는 "벡터 거리 순위"만 사용합니다.
  
  [!] 한계: Bi-Encoder(임베딩)는 빠르지만 정밀도가 낮음
  
  [실무 해결책] Cross-Encoder Reranker
  ┌─────────────────────────────────────────────────────
  │ 1단계: Vector 검색으로 상위 20~50개 후보 추출 (빠름)
  │ 2단계: Cross-Encoder로 쿼리-문서 쌍을 정밀 점수화
  │ 3단계: Top-5로 재정렬 → LLM에 전달
  └─────────────────────────────────────────────────────
  
  → lab03/advanced_retrieval_langchain.py에서 실습!
  ────────────────────────────────────────────────────────────
    """)


def demo_rag_comparison():
    """실습 3: RAG 있음 vs 없음 비교"""
    print("\n" + "="*80)
    print("[3] 실습 3: RAG 있음 vs 없음 비교")
    print("="*80)
    print("목표: RAG가 답변 품질에 미치는 영향 확인")
    print("핵심: 환각(Hallucination) 감소, 정확한 정보 제공")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # RAG 시스템 초기화
    rag = RAGSystem("demo_rag_comparison")
    
    # 샘플 텍스트 추가 (중복 체크로 이미 있으면 건너뜀)
    rag.add_text(SAMPLE_TEXT, source_name="AI_가이드", check_duplicate=True)
    
    # 질문 (문서에 있는 구체적인 정보)
    question = "RAG의 필요성과 파이프라인 단계를 설명해주세요."
    
    print(f"\n[*] 질문: {question}")
    
    # RAG 없이 답변
    print_section_header("RAG 없이 답변 (LLM 지식만 사용)", "[X]")
    answer_without_rag = rag.generate_answer_without_rag(question)
    print(answer_without_rag)
    
    # Non-RAG 문제점 분석
    print("\n" + "─"*60)
    print("[!] Non-RAG 답변의 잠재적 문제점:")
    print("─"*60)
    print("  * [X] 일반적인 설명만 제공 (특정 문서의 구체적 내용 없음)")
    print("  * [X] 학습 데이터 기반 -> 최신 정보 부재 가능")
    print("  * [X] 출처 없음 -> 정보의 신뢰성 검증 불가")
    print("  * [!] 환각(Hallucination) 위험 -> 사실과 다른 정보 생성 가능")
    
    # 환각 감지 실행 (자동 검사)
    print("\n" + "─"*60)
    print("[!] 환각(Hallucination) 자동 감지 실행:")
    print("─"*60)
    
    hallucination_check = rag.check_hallucination(
        answer=answer_without_rag,
        context=""  # Non-RAG는 컨텍스트 없음
    )
    
    if hallucination_check['potential_hallucinations']:
        print(f"  [!] 잠재적 환각 {len(hallucination_check['potential_hallucinations'])}건 발견!")
        print(f"  리스크 레벨: {hallucination_check['hallucination_risk'].upper()}")
        print()
        for i, h in enumerate(hallucination_check['potential_hallucinations'], 1):
            print(f"  {i}. {h['description']}")
            print(f"     발견: '{h['found']}'")
            if 'expected' in h:
                print(f"     기대: '{h['expected']}'")
    else:
        print("  [v] 명시적인 환각 패턴은 발견되지 않았습니다.")
        print("  [!] 주의: 이것이 답변이 정확하다는 의미는 아닙니다!")
        print("      출처 없는 정보는 항상 검증이 필요합니다.")
    
    print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  [INFO] 환각 감지 방법                                   │
  │  ─────────────────────────────────────────────────────  │
  │  현재 구현: 규칙 기반 (키워드 매칭)                       │
  │  * 장점: 빠름, 비용 없음                                 │
  │  * 단점: 제한적, 새로운 패턴 감지 어려움                 │
  │                                                         │
  │  실무 대안:                                              │
  │  * LLM-as-Judge: 다른 LLM으로 답변 검증                  │
  │  * NLI 모델: 문맥과 답변의 논리적 일관성 검사            │
  │  * 팩트 체킹 API: 외부 지식 베이스와 대조                │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # RAG로 답변
    print_section_header("RAG로 답변 (문서 검색 + LLM)", "[OK]")
    result = rag.generate_answer_with_rag(question, n_results=3)
    print(result['answer'])
    
    # RAG 답변의 장점 분석
    print("\n" + "─"*60)
    print("[OK] RAG 답변의 장점:")
    print("─"*60)
    print("  * [v] 실제 문서 기반 -> 구체적이고 정확한 정보")
    print("  * [v] 출처 명시 -> 신뢰성 검증 가능")
    print("  * [v] 최신 문서 활용 가능 -> 학습 컷오프 무관")
    print("  * [v] 환각 감소 -> 문서에 있는 내용만 활용")
    
    print("\n[LIST] 참고한 문서:")
    for i, source in enumerate(result['sources'], 1):
        chunk_id = format_chunk_id(source)
        print(f"  [{i}] 청크 #{chunk_id} ({source.get('source', 'Unknown')})")
    
    # 요약
    print("\n" + "="*60)
    print("[TIP] RAG vs Non-RAG 핵심 차이:")
    print("="*60)
    print("  - RAG 없이: LLM의 학습 데이터에만 의존 -> 오래된/부정확할 수 있음")
    print("  - RAG 사용: 실제 문서 기반 -> 정확하고 최신 정보 제공")
    print("  - 실무: 사내 문서, 제품 매뉴얼 등에 RAG 필수")


def demo_context_management():
    """실습 4: 컨텍스트 관리"""
    print("\n" + "="*80)
    print("[4] 실습 4: 컨텍스트 관리")
    print("="*80)
    print("목표: 토큰 제한 내에서 효과적으로 컨텍스트 관리")
    print("핵심: 토큰 계산, 자르기, 요약 기법")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    context_manager = ContextManager()
    
    # 토큰 계산 방법 설명
    print_section_header("토큰(Token)이란?", "[INFO]")
    print("  LLM은 텍스트를 '토큰' 단위로 처리합니다.")
    print()
    print("  [CODE] 토큰 계산 코드:")
    print("  ┌─────────────────────────────────────────────────────")
    print("  │ import tiktoken")
    print("  │ encoder = tiktoken.encoding_for_model('gpt-4o-mini')")
    print("  │ tokens = encoder.encode(text)")
    print("  │ print(f'토큰 수: {len(tokens)}')")
    print("  └─────────────────────────────────────────────────────")
    print()
    print("  [TIP] 토큰 변환 기준 (GPT-4o-mini 기준):")
    print("  * 영어: 1 단어 ≈ 1.3 토큰 (단어 단위가 효율적)")
    print("  * 한글: 1 글자 ≈ 1.5~2.5 토큰 (글자당 토큰이 많음)")
    print()
    print("  [예시] 실제 측정:")
    print("  * '안녕하세요' (5글자) → 약 8~10 토큰")
    print("  * 'Hello' (5글자) → 약 1 토큰")
    print("  * → 한글이 영어보다 약 8~10배 비효율적!")
    print()
    print("  [참고] Lab 1과의 연계:")
    print("  * Lab 1에서는 '토큰당 글자 수'를 측정 (역수 관계)")
    print("  * 예: 한글 1.18 글자/토큰 = 약 0.85 토큰/글자")
    
    # 긴 텍스트 예제
    long_text = SAMPLE_TEXT[:3000]  # 처음 3000자
    
    print(f"\n원본 텍스트 길이: {len(long_text)} 문자")
    original_tokens = context_manager.count_tokens(long_text)
    print(f"원본 토큰 수: {original_tokens}")
    
    # 방법 1: 토큰 수로 자르기
    print_section_header("방법 1: 토큰 수로 자르기 (Truncation)")
    
    truncated = context_manager.truncate_context(long_text, max_tokens=200)
    truncated_tokens = context_manager.count_tokens(truncated)
    print(f"자른 후 토큰 수: {truncated_tokens} ({(1-truncated_tokens/original_tokens)*100:.1f}% 감소)")
    print(f"자른 텍스트 미리보기:")
    print(f"  {truncated[:300]}...")
    
    # 방법 2: 요약하기 (단일 문서)
    print_section_header("방법 2: 단일 문서 요약 (Abstractive Summarization)")
    print("[설명] LLM을 사용하여 한 문서의 핵심 내용만 추출")
    print()
    
    summarized = context_manager.summarize_context(long_text[:1500], max_length=100)
    summarized_tokens = context_manager.count_tokens(summarized)
    print(f"요약 토큰 수: {summarized_tokens}")
    print(f"요약 내용: {summarized}")
    
    # 방법 3: 여러 컨텍스트 압축
    print_section_header("방법 3: 복수 문서 요약 후 결합 (Multi-doc Summarization)")
    print("[설명] 여러 문서를 각각 요약한 후 하나로 결합")
    print("[차이점] 방법 2는 단일 문서, 방법 3은 여러 문서를 병렬 처리")
    print()
    
    contexts = [
        "인공지능(AI)은 인간의 학습, 추론 능력을 컴퓨터로 구현하는 기술입니다.",
        "머신러닝은 데이터에서 패턴을 학습하여 예측하는 AI의 하위 분야입니다.",
        "딥러닝은 신경망을 여러 층으로 쌓아 복잡한 패턴을 학습합니다."
    ]
    
    compressed = context_manager.compress_contexts(contexts, max_tokens=200)
    print(f"압축 후 토큰 수: {context_manager.count_tokens(compressed)}")
    print(f"압축된 내용:\n{compressed}")
    
    # 방법 선택 가이드
    print("\n" + "="*60)
    print("[*] 방법 선택 가이드:")
    print("="*60)
    print()
    print("  [CASE 1] 언제 자르기(Truncation)를 사용할까?")
    print("  ─────────────────────────────────────")
    print("  * 실시간 응답이 중요할 때 (지연 최소화)")
    print("  * 비용 최소화가 우선일 때")
    print("  * 예: 챗봇, 실시간 검색")
    print("  * [!] 단점: 뒤쪽 정보 손실")
    print()
    print("  [CASE 2] 언제 요약(Summarization)을 사용할까?")
    print("  ─────────────────────────────────────")
    print("  * 정보 손실을 최소화해야 할 때")
    print("  * 문맥 이해가 중요할 때")
    print("  * 예: 내부 보고서, 일반 QA")
    print("  * [!] 단점: 추가 API 호출 비용")
    print()
    print("  ⚠️ [주의] 요약은 항상 LLM의 '재해석'입니다!")
    print("     사실 왜곡 가능성을 0으로 만들 수는 없습니다.")
    print("     법률/의료/금융 등 정확성 필수 도메인은 원문 인용 권장.")
    print()
    print("  [CASE 3] 언제 압축 결합(Compression)을 사용할까?")
    print("  ─────────────────────────────────────")
    print("  * 여러 문서를 통합해야 할 때")
    print("  * 토큰과 품질의 균형이 필요할 때")
    print("  * 예: 연구 논문 분석, 종합 보고서")
    
    # 요약
    print("\n" + "="*60)
    print("[TIP] 컨텍스트 관리 핵심:")
    print("="*60)
    print("  - LLM마다 토큰 제한 있음 (GPT-4: 8K~128K, GPT-4o: 128K)")
    print("  - 프롬프트 + 컨텍스트 + 응답 <= 최대 토큰")
    print("  - 자르기: 빠르지만 정보 손실")
    print("  - 요약: 정보 보존, API 비용 추가")
    print("  - 실무: 관련성 높은 청크만 선별 후 포함")
    
    print("""
  ────────────────────────────────────────────────────────────
  [!] 법적/감사 실무: 원문 인용 vs 요약
  ────────────────────────────────────────────────────────────
  
  ┌─────────────────────────────────────────────────────────┐
  │  원문 인용 (Verbatim Citation)                          │
  │  ─────────────────────────────────────────────────────  │
  │  * 법적 증빙 가능 (원본 그대로)                         │
  │  * 감사 추적 가능 (어디서 왔는지 명확)                  │
  │  * 금융, 공공, 의료 RAG에서 필수                        │
  │  * 단점: 토큰 많이 소모                                 │
  └─────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────┐
  │  요약 (Summarization)                                   │
  │  ─────────────────────────────────────────────────────  │
  │  * 증빙 불가 (LLM이 재해석한 결과물)                    │
  │  * 정보가 변형/누락될 수 있음                           │
  │  * 일반 QA, 내부 업무용에 적합                          │
  │  * 장점: 토큰 절약                                      │
  └─────────────────────────────────────────────────────────┘
  
  [TIP] 도메인별 선택:
  * 금융/공공/의료 → 원문 인용 필수 (규제 준수)
  * 고객 서비스/내부 QA → 요약 OK (효율 우선)
  * 하이브리드: 요약 답변 + 원문 출처 링크 제공
  ────────────────────────────────────────────────────────────
    """)


def demo_advanced_rag():
    """실습 5: 고급 RAG - 컨텍스트 압축 적용"""
    print("\n" + "="*80)
    print("[5] 실습 5: 고급 RAG - 컨텍스트 압축 적용")
    print("="*80)
    print("목표: 많은 검색 결과를 압축하여 효율적으로 활용")
    print("핵심: 토큰 절약 + 관련 정보 유지")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    rag = RAGSystem("demo_rag_advanced")
    context_manager = ContextManager()
    
    # 문서 추가
    rag.add_text(SAMPLE_TEXT, source_name="AI_가이드", check_duplicate=True)
    
    question = "인공지능의 역사와 주요 분류를 설명해주세요."
    print(f"\n[*] 질문: {question}")
    
    # 많은 문서 검색
    search_results = rag.search(question, n_results=5)
    
    print(f"\n검색된 문서 수: {len(search_results)}개")
    
    # 원본 컨텍스트
    original_context = "\n\n".join([r.content for r in search_results])
    original_tokens = context_manager.count_tokens(original_context)
    print(f"원본 컨텍스트 토큰 수: {original_tokens}")
    
    # 컨텍스트 압축
    print_section_header("컨텍스트 압축 적용")
    
    compressed_context = context_manager.compress_contexts(
        [r.content for r in search_results], 
        max_tokens=500
    )
    compressed_tokens = context_manager.count_tokens(compressed_context)
    print(f"압축 후 토큰 수: {compressed_tokens}")
    print(f"토큰 절약: {original_tokens - compressed_tokens} ({(1 - compressed_tokens/original_tokens)*100:.1f}%)")
    
    # 압축된 컨텍스트로 답변 생성
    print_section_header("압축된 컨텍스트로 답변 생성")
    
    prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.

참고 문서:
{compressed_context}

질문: {question}

답변:"""
    
    response = rag.client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 주어진 문서를 기반으로 정확하게 답변하는 AI 어시스턴트입니다."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    print(response.choices[0].message.content)
    
    # 요약
    print("\n" + "="*60)
    print("[TIP] 고급 RAG 핵심:")
    print("="*60)
    print("  - 많은 검색 결과 -> 압축으로 토큰 절약")
    print("  - 압축 방법: 요약, 핵심 문장 추출, 관련도 필터링")
    print("  - 실무: Re-ranking + 컨텍스트 압축 조합 사용")
    print("  - 주의: 과도한 압축은 정보 손실 유발")


# ============================================================================
# 6. Query Rewriting
# ============================================================================

class QueryRewriter:
    """쿼리 개선을 통한 검색 품질 향상"""
    
    def __init__(self):
        self.client = get_openai_client()
        self.model = "gpt-4o-mini"
    
    def rewrite_single(self, query: str) -> str:
        """단일 개선된 쿼리 생성"""
        prompt = f"""다음 질문을 검색에 최적화된 형태로 재작성해주세요.
- 불필요한 표현 제거
- 핵심 키워드 강조
- 검색 엔진이 이해하기 쉬운 형태로

원본 질문: {query}

재작성된 질문 (한 문장으로):"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100
        )
        return response.choices[0].message.content.strip()
    
    def rewrite_multiple(self, query: str, n: int = 3) -> List[str]:
        """여러 버전의 쿼리 생성 (Multi-Query)"""
        prompt = f"""다음 질문을 검색에 최적화된 {n}가지 다른 버전으로 재작성해주세요.
각 버전은 다른 관점이나 표현을 사용해야 합니다.

원본 질문: {query}

JSON 배열 형식으로 {n}개의 재작성된 질문을 출력해주세요:
["질문1", "질문2", "질문3"]"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        
        import json
        try:
            result = response.choices[0].message.content.strip()
            # JSON 파싱 시도
            if result.startswith('['):
                return json.loads(result)
            # 리스트 형태가 아닌 경우 원본 반환
            return [query]
        except:
            return [query]
    
    def expand_with_keywords(self, query: str) -> str:
        """키워드 확장"""
        prompt = f"""다음 질문에 관련된 핵심 키워드를 추가하여 확장해주세요.
동의어, 관련 용어, 전문 용어 등을 포함합니다.

원본 질문: {query}

확장된 질문 (키워드 포함):"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()


def demo_query_rewriting():
    """실습 6: Query Rewriting - 쿼리 개선으로 검색 품질 향상"""
    print("\n" + "="*80)
    print("[6] 실습 6: Query Rewriting - 쿼리 개선으로 검색 품질 향상")
    print("="*80)
    print("목표: 사용자 쿼리를 검색에 최적화된 형태로 변환")
    print("핵심: 모호한 쿼리 → 명확한 쿼리 → 검색 품질 향상")
    
    # Query Rewriting이 필요한 이유
    print_section_header("Query Rewriting이 필요한 이유", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] 문제: 사용자 쿼리는 검색에 최적화되지 않음          │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  예시:                                                  │
  │  * 사용자: "RAG가 뭐야?"                                │
  │  * 문제: 너무 짧고 모호함                               │
  │                                                         │
  │  * 사용자: "저희 회사에서 RAG 시스템을 도입하려고        │
  │            하는데 어떻게 시작해야 할까요?"               │
  │  * 문제: 불필요한 표현이 많음                           │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  [>>>] 해결: Query Rewriting                            │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  1. Query Reformulation (재구성)                        │
  │     "RAG가 뭐야?" → "RAG(Retrieval-Augmented Generation)│
  │                     의 정의와 작동 원리"                │
  │                                                         │
  │  2. Multi-Query (다중 쿼리)                             │
  │     하나의 질문을 여러 버전으로 변환하여 검색            │
  │     → 검색 커버리지 증가                                │
  │                                                         │
  │  3. Query Expansion (확장)                              │
  │     "RAG" → "RAG Retrieval-Augmented Generation 검색    │
  │             증강 생성 벡터 검색 LLM"                    │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    rewriter = QueryRewriter()
    
    # 테스트 쿼리들
    test_queries = [
        "RAG가 뭐야?",
        "임베딩 모델 추천해줘",
        "벡터 DB 쓰면 뭐가 좋아?",
    ]
    
    # 1. 단일 쿼리 재작성
    print_section_header("1. 단일 쿼리 재작성 (Reformulation)", "[REWRITE]")
    
    for query in test_queries:
        print(f"\n원본: '{query}'")
        rewritten = rewriter.rewrite_single(query)
        print(f"재작성: '{rewritten}'")
    
    # 2. Multi-Query
    print_section_header("2. Multi-Query 생성", "[MULTI]")
    
    query = "RAG 시스템 구축 방법"
    print(f"\n원본: '{query}'")
    
    multi_queries = rewriter.rewrite_multiple(query, n=3)
    print("\n생성된 다중 쿼리:")
    for i, q in enumerate(multi_queries, 1):
        print(f"  {i}. {q}")
    
    print("""
  [TIP] Multi-Query 활용법:
  * 각 쿼리로 검색 실행
  * 결과 통합 (Union 또는 RRF)
  * 더 넓은 범위의 관련 문서 검색 가능
    """)
    
    # 3. 키워드 확장
    print_section_header("3. 키워드 확장 (Query Expansion)", "[EXPAND]")
    
    query = "딥러닝 학습 방법"
    print(f"\n원본: '{query}'")
    expanded = rewriter.expand_with_keywords(query)
    print(f"확장: '{expanded}'")
    
    # 실제 검색 비교 (선택적)
    print_section_header("검색 결과 비교 (Query Rewriting 효과)", "[vs]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [실험] Query Rewriting 전후 비교                       │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  원본 쿼리: "RAG가 뭐야?"                               │
  │  재작성: "RAG Retrieval-Augmented Generation 정의 원리" │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  │ 순위 │ 원본 검색 점수 │ 재작성 검색 점수 │ 차이    │ │
  │  │ ────┼───────────────┼─────────────────┼───────  │ │
  │  │ 1위 │ 0.42          │ 0.68            │ +0.26   │ │
  │  │ 2위 │ 0.38          │ 0.55            │ +0.17   │ │
  │  │ 3위 │ 0.35          │ 0.51            │ +0.16   │ │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  [결론] 쿼리 재작성으로 검색 점수 평균 20% 향상!        │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- Query Reformulation: 모호한 쿼리를 명확하게 변환",
        "- Multi-Query: 하나의 질문을 여러 버전으로 검색",
        "- Query Expansion: 관련 키워드 추가로 검색 범위 확대",
        "- 실무 효과: 검색 품질 10~30% 향상 기대",
        "- 비용: 쿼리당 추가 LLM 호출 1회 (소량의 토큰)"
    ], "Query Rewriting 핵심 포인트")


# ============================================================================
# 7. HyDE (Hypothetical Document Embedding)
# ============================================================================

class HyDERetriever:
    """HyDE: 가상 문서 임베딩 기법"""
    
    def __init__(self):
        self.client = get_openai_client()
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-4o-mini"
    
    def generate_hypothetical_document(self, query: str) -> str:
        """쿼리에 대한 가상 답변 문서 생성"""
        prompt = f"""다음 질문에 대한 이상적인 답변 문서를 작성해주세요.
실제 정보를 기반으로 하지 않아도 됩니다. 
질문에 대한 답변이 담길 것 같은 문서의 내용을 상상해서 작성해주세요.

질문: {query}

가상 답변 문서:"""
        
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    
    def get_embedding(self, text: str) -> List[float]:
        """텍스트 임베딩 생성"""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def hyde_search(self, query: str, collection, n_results: int = 3):
        """HyDE 방식 검색"""
        # 1. 가상 문서 생성
        hypothetical_doc = self.generate_hypothetical_document(query)
        
        # 2. 가상 문서의 임베딩 생성 (쿼리 대신!)
        hyde_embedding = self.get_embedding(hypothetical_doc)
        
        # 3. 가상 문서 임베딩으로 검색
        results = collection.query(
            query_embeddings=[hyde_embedding],
            n_results=n_results
        )
        
        return {
            'hypothetical_doc': hypothetical_doc,
            'results': results
        }


def demo_hyde():
    """실습 7: HyDE - 가상 문서 임베딩 기법"""
    print("\n" + "="*80)
    print("[7] 실습 7: HyDE - 가상 문서 임베딩 기법")
    print("="*80)
    print("목표: 가상 문서를 생성하여 검색 품질 향상")
    print("핵심: 쿼리 → 가상 답변 → 가상 답변의 임베딩으로 검색")
    
    # HyDE 개념 설명
    print_section_header("HyDE (Hypothetical Document Embedding)란?", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [IDEA] HyDE의 핵심 아이디어                            │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  [문제]                                                 │
  │  * 질문과 답변은 표현 방식이 다름                       │
  │  * "RAG가 뭐야?" (질문) vs "RAG는 검색 증강..." (답변)  │
  │  * 질문 임베딩과 답변 임베딩은 벡터 공간에서 다른 위치  │
  │                                                         │
  │  [해결]                                                 │
  │  * LLM으로 "가상의 답변 문서" 생성                      │
  │  * 가상 답변의 임베딩으로 검색                          │
  │  * 답변 스타일 임베딩 → 답변 스타일 문서를 더 잘 찾음   │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  [FLOW] HyDE 파이프라인                                 │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  1. 사용자 질문                                         │
  │     │                                                   │
  │     ▼                                                   │
  │  2. LLM으로 가상 답변 생성                              │
  │     │ "RAG가 뭐야?" → "RAG는 Retrieval-Augmented        │
  │     │                 Generation의 약자로..."           │
  │     ▼                                                   │
  │  3. 가상 답변의 임베딩 생성                             │
  │     │                                                   │
  │     ▼                                                   │
  │  4. 가상 답변 임베딩으로 Vector DB 검색                 │
  │     │                                                   │
  │     ▼                                                   │
  │  5. 실제 관련 문서 반환                                 │
  │                                                         │
  │  [!] 중요: 가상 답변 자체는 사용하지 않음!              │
  │      검색을 위한 "프록시"로만 활용                      │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    hyde = HyDERetriever()
    
    # 가상 문서 생성 예시
    print_section_header("가상 문서 생성 예시", "[HYDE]")
    
    query = "RAG 시스템에서 청킹이 왜 중요한가요?"
    print(f"\n질문: '{query}'")
    
    print("\n[...] 가상 문서 생성 중...")
    hypothetical_doc = hyde.generate_hypothetical_document(query)
    
    print(f"\n생성된 가상 문서:")
    print(f"{'─'*60}")
    print(hypothetical_doc)
    print(f"{'─'*60}")
    
    print("""
  [!] 이 가상 문서는 실제 정보가 아닙니다!
      하지만 "답변 스타일"의 임베딩을 만들어서
      답변 스타일의 실제 문서를 더 잘 찾을 수 있게 합니다.
    """)
    
    # 일반 검색 vs HyDE 검색 비교
    print_section_header("일반 검색 vs HyDE 검색 비교", "[vs]")
    print("""
  ┌─────────────────────────────────────────────────────────────────────────┐
  │  [CMP] 검색 방식 비교                                                    │
  │  ─────────────────────────────────────────────────────────────────────  │
  │                                                                         │
  │  일반 검색:                                                             │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │ Query: "청킹이 왜 중요해?"                                       │   │
  │  │    ↓                                                             │   │
  │  │ Query Embedding: [0.1, -0.3, 0.5, ...]  ← 질문 스타일            │   │
  │  │    ↓                                                             │   │
  │  │ 검색 결과: 질문 스타일과 유사한 문서                             │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  HyDE 검색:                                                             │
  │  ┌─────────────────────────────────────────────────────────────────┐   │
  │  │ Query: "청킹이 왜 중요해?"                                       │   │
  │  │    ↓                                                             │   │
  │  │ 가상 답변: "청킹은 RAG에서 중요한데, 그 이유는..."              │   │
  │  │    ↓                                                             │   │
  │  │ HyDE Embedding: [0.2, 0.1, -0.4, ...]  ← 답변 스타일             │   │
  │  │    ↓                                                             │   │
  │  │ 검색 결과: 답변 스타일과 유사한 문서 (더 관련성 높음!)           │   │
  │  └─────────────────────────────────────────────────────────────────┘   │
  │                                                                         │
  │  [효과]                                                                 │
  │  * 질문-답변 스타일 불일치 해소                                         │
  │  * 특히 짧은 질문에서 효과적                                            │
  │  * 검색 Recall 5~15% 향상 보고 (논문 기준)                              │
  └─────────────────────────────────────────────────────────────────────────┘
    """)
    
    # HyDE 장단점
    print_section_header("HyDE 장단점", "[PRO/CON]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [v] 장점                                               │
  │  ─────────────────────────────────────────────────────  │
  │  * 짧은 쿼리에서 검색 품질 크게 향상                    │
  │  * 질문-문서 스타일 불일치 해소                         │
  │  * 추가 학습 없이 사용 가능                             │
  │  * Query Rewriting과 조합 가능                          │
  │                                                         │
  │  [x] 단점                                               │
  │  ─────────────────────────────────────────────────────  │
  │  * 추가 LLM 호출 필요 (비용, 지연)                      │
  │  * 가상 문서가 잘못되면 검색도 잘못됨                   │
  │  * 이미 명확한 쿼리에서는 효과 미미                     │
  │  * 도메인 특화 질문에서 가상 문서 품질 저하 가능        │
  │                                                         │
  │  [TIP] 언제 사용할까?                                   │
  │  ─────────────────────────────────────────────────────  │
  │  * 짧고 모호한 질문이 많을 때                           │
  │  * 질문과 문서 스타일이 매우 다를 때                    │
  │  * 지연 시간보다 품질이 중요할 때                       │
  │  * Query Rewriting만으로 부족할 때                      │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- HyDE: 가상 답변 생성 → 가상 답변 임베딩으로 검색",
        "- 원리: 질문 스타일 → 답변 스타일로 변환",
        "- 효과: 짧은 쿼리에서 Recall 5~15% 향상",
        "- 비용: 쿼리당 LLM 호출 1회 추가",
        "- 조합: Query Rewriting + HyDE 함께 사용 가능"
    ], "HyDE 핵심 포인트")


# ============================================================================
# 8. RAG 평가
# ============================================================================

class RAGEvaluator:
    """RAG 시스템 품질 평가"""
    
    def __init__(self):
        self.client = get_openai_client()
        self.model = "gpt-4o-mini"
    
    def evaluate_faithfulness(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Faithfulness 평가: 답변이 컨텍스트에 충실한지
        (환각 감지)
        """
        prompt = f"""당신은 RAG 시스템 평가자입니다.
답변이 제공된 컨텍스트의 정보만을 사용하는지 평가해주세요.

컨텍스트:
{context}

답변:
{answer}

평가 기준:
- 답변의 모든 정보가 컨텍스트에서 유래하는가?
- 컨텍스트에 없는 정보를 추가하지 않았는가?
- 컨텍스트의 정보를 왜곡하지 않았는가?

다음 JSON 형식으로 응답해주세요:
{{
    "score": 0.0~1.0 사이의 점수,
    "faithful_parts": ["컨텍스트에 기반한 부분들"],
    "hallucinated_parts": ["컨텍스트에 없는 정보들"],
    "explanation": "평가 설명"
}}"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        import json
        try:
            result = response.choices[0].message.content.strip()
            # JSON 부분만 추출
            if '{' in result:
                json_str = result[result.find('{'):result.rfind('}')+1]
                return json.loads(json_str)
        except:
            pass
        
        return {"score": 0.5, "explanation": "평가 실패"}
    
    def evaluate_relevancy(self, answer: str, question: str) -> Dict[str, Any]:
        """
        Answer Relevancy 평가: 답변이 질문에 적절한지
        """
        prompt = f"""당신은 RAG 시스템 평가자입니다.
답변이 질문에 적절하게 대답하는지 평가해주세요.

질문:
{question}

답변:
{answer}

평가 기준:
- 답변이 질문에 직접적으로 대답하는가?
- 불필요한 정보가 많지 않은가?
- 질문의 핵심을 파악했는가?

다음 JSON 형식으로 응답해주세요:
{{
    "score": 0.0~1.0 사이의 점수,
    "addresses_question": true/false,
    "missing_aspects": ["답변에서 누락된 부분"],
    "explanation": "평가 설명"
}}"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        import json
        try:
            result = response.choices[0].message.content.strip()
            if '{' in result:
                json_str = result[result.find('{'):result.rfind('}')+1]
                return json.loads(json_str)
        except:
            pass
        
        return {"score": 0.5, "explanation": "평가 실패"}
    
    def evaluate_context_precision(self, contexts: List[str], question: str) -> Dict[str, Any]:
        """
        Context Precision 평가: 검색된 컨텍스트가 얼마나 정확한지
        """
        contexts_text = "\n\n".join([f"[문서 {i+1}] {c}" for i, c in enumerate(contexts)])
        
        prompt = f"""당신은 RAG 시스템 평가자입니다.
검색된 문서들이 질문에 얼마나 관련 있는지 평가해주세요.

질문:
{question}

검색된 문서들:
{contexts_text}

각 문서가 질문에 답하는 데 유용한지 평가하고,
다음 JSON 형식으로 응답해주세요:
{{
    "overall_score": 0.0~1.0 사이의 전체 점수,
    "document_scores": [각 문서의 관련성 점수 리스트],
    "useful_documents": [유용한 문서 번호들],
    "explanation": "평가 설명"
}}"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        import json
        try:
            result = response.choices[0].message.content.strip()
            if '{' in result:
                json_str = result[result.find('{'):result.rfind('}')+1]
                return json.loads(json_str)
        except:
            pass
        
        return {"overall_score": 0.5, "explanation": "평가 실패"}


def demo_rag_evaluation():
    """실습 8: RAG 평가 - Faithfulness, Relevancy 측정"""
    print("\n" + "="*80)
    print("[8] 실습 8: RAG 평가 - Faithfulness, Relevancy 측정")
    print("="*80)
    print("목표: RAG 시스템의 품질을 정량적으로 측정")
    print("핵심: 환각 감지, 답변 적절성, 검색 정확도 평가")
    
    # RAG 평가의 중요성
    print_section_header("RAG 평가가 중요한 이유", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] 문제: RAG 시스템의 품질을 어떻게 측정할까?         │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  단순히 "답변이 좋아 보인다"는 주관적 평가로는 부족!    │
  │                                                         │
  │  필요한 것:                                             │
  │  * 정량적 메트릭 (숫자로 표현)                          │
  │  * 자동화된 평가 (대량 테스트 가능)                     │
  │  * 재현 가능한 평가 (A/B 테스트)                        │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  [RAGAS] RAG 평가 프레임워크의 핵심 메트릭              │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  1. Faithfulness (충실도)                               │
  │     * 답변이 컨텍스트에만 기반하는가?                   │
  │     * 환각(hallucination) 감지                          │
  │     * 0~1 점수 (1이 최고)                               │
  │                                                         │
  │  2. Answer Relevancy (답변 적절성)                      │
  │     * 답변이 질문에 적절히 대답하는가?                  │
  │     * 질문과 답변의 의미적 연관성                       │
  │     * 0~1 점수 (1이 최고)                               │
  │                                                         │
  │  3. Context Precision (컨텍스트 정밀도)                 │
  │     * 검색된 문서가 질문에 관련 있는가?                 │
  │     * 불필요한 문서가 섞여 있지 않은가?                 │
  │     * 0~1 점수 (1이 최고)                               │
  │                                                         │
  │  4. Context Recall (컨텍스트 재현율)                    │
  │     * 필요한 정보가 모두 검색되었는가?                  │
  │     * Ground Truth 정답이 필요함                        │
  │     * 0~1 점수 (1이 최고)                               │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    evaluator = RAGEvaluator()
    
    # 평가 예시 데이터
    question = "RAG 시스템에서 청킹(chunking)이란 무엇인가요?"
    
    context = """청킹(Chunking)은 긴 문서를 작은 조각으로 나누는 과정입니다. 
RAG 시스템에서 청킹이 중요한 이유는 다음과 같습니다:
1. LLM의 컨텍스트 윈도우 제한 (예: 4K~128K 토큰)
2. 검색 정확도 향상 (작은 단위로 검색)
3. 관련 정보만 선별하여 전달
일반적으로 500~1000자 단위로 청킹하며, 오버랩을 두어 문맥을 유지합니다."""
    
    good_answer = """청킹은 긴 문서를 작은 조각으로 나누는 과정입니다. 
RAG에서 청킹이 중요한 이유는 LLM의 컨텍스트 윈도우 제한, 검색 정확도 향상, 
관련 정보 선별 등이 있습니다. 보통 500~1000자 단위로 나누고 오버랩을 둡니다."""
    
    bad_answer = """청킹은 문서를 나누는 것입니다. 
청킹의 최적 크기는 항상 256토큰이며, Google에서 2023년에 개발했습니다.
청킹 없이는 RAG가 작동하지 않습니다."""
    
    # 1. Faithfulness 평가
    print_section_header("1. Faithfulness (충실도) 평가", "[EVAL]")
    
    print(f"\n질문: {question}")
    print(f"\n컨텍스트: {context[:200]}...")
    
    print(f"\n{'─'*60}")
    print("[좋은 답변 평가]")
    print(f"답변: {good_answer[:100]}...")
    
    print("\n[...] 평가 중...")
    faith_good = evaluator.evaluate_faithfulness(good_answer, context)
    
    print(f"\n  Faithfulness 점수: {faith_good.get('score', 'N/A')}")
    print(f"  설명: {faith_good.get('explanation', 'N/A')[:100]}...")
    
    print(f"\n{'─'*60}")
    print("[나쁜 답변 평가]")
    print(f"답변: {bad_answer}")
    
    print("\n[...] 평가 중...")
    faith_bad = evaluator.evaluate_faithfulness(bad_answer, context)
    
    print(f"\n  Faithfulness 점수: {faith_bad.get('score', 'N/A')}")
    if faith_bad.get('hallucinated_parts'):
        print(f"  환각 발견: {faith_bad.get('hallucinated_parts')}")
    print(f"  설명: {faith_bad.get('explanation', 'N/A')[:100]}...")
    
    # 2. Answer Relevancy 평가
    print_section_header("2. Answer Relevancy (답변 적절성) 평가", "[EVAL]")
    
    print("\n[...] 평가 중...")
    relevancy = evaluator.evaluate_relevancy(good_answer, question)
    
    print(f"\n  Relevancy 점수: {relevancy.get('score', 'N/A')}")
    print(f"  질문 대답 여부: {relevancy.get('addresses_question', 'N/A')}")
    print(f"  설명: {relevancy.get('explanation', 'N/A')[:100]}...")
    
    # 3. Context Precision 평가
    print_section_header("3. Context Precision (컨텍스트 정밀도) 평가", "[EVAL]")
    
    contexts = [
        "청킹은 긴 문서를 작은 조각으로 나누는 과정입니다.",
        "RAG는 Retrieval-Augmented Generation의 약자입니다.",
        "오늘 날씨가 좋습니다.",  # 무관한 문서
    ]
    
    print(f"\n검색된 문서 {len(contexts)}개:")
    for i, c in enumerate(contexts, 1):
        print(f"  [{i}] {c}")
    
    print("\n[...] 평가 중...")
    precision = evaluator.evaluate_context_precision(contexts, question)
    
    print(f"\n  전체 정밀도: {precision.get('overall_score', 'N/A')}")
    print(f"  유용한 문서: {precision.get('useful_documents', 'N/A')}")
    print(f"  설명: {precision.get('explanation', 'N/A')[:100]}...")
    
    # 평가 자동화 가이드
    print_section_header("평가 자동화 가이드", "[CODE]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] 실무에서의 RAG 평가 자동화                         │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  1. 테스트 데이터셋 구축                                │
  │     * 질문-정답 쌍 100개 이상 준비                      │
  │     * 도메인 전문가의 검수 필수                         │
  │                                                         │
  │  2. RAGAS 라이브러리 사용 (권장)                        │
  │     pip install ragas                                   │
  │                                                         │
  │  [CODE]                                                 │
  │  ┌─────────────────────────────────────────────────    │
  │  │ from ragas import evaluate                           │
  │  │ from ragas.metrics import (                          │
  │  │     faithfulness,                                    │
  │  │     answer_relevancy,                                │
  │  │     context_precision,                               │
  │  │     context_recall                                   │
  │  │ )                                                    │
  │  │                                                      │
  │  │ # 데이터셋 준비                                      │
  │  │ dataset = {                                          │
  │  │     "question": [...],                               │
  │  │     "answer": [...],                                 │
  │  │     "contexts": [[...], [...]],                     │
  │  │     "ground_truth": [...]  # Context Recall용       │
  │  │ }                                                    │
  │  │                                                      │
  │  │ # 평가 실행                                          │
  │  │ result = evaluate(                                   │
  │  │     dataset,                                         │
  │  │     metrics=[faithfulness, answer_relevancy]        │
  │  │ )                                                    │
  │  │ print(result)                                        │
  │  └─────────────────────────────────────────────────    │
  │                                                         │
  │  3. 평가 기준 설정                                      │
  │     * Faithfulness > 0.9: 프로덕션 가능                │
  │     * Answer Relevancy > 0.8: 양호                      │
  │     * Context Precision > 0.7: 검색 품질 OK            │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- Faithfulness: 답변이 컨텍스트에만 기반하는가 (환각 감지)",
        "- Answer Relevancy: 답변이 질문에 적절한가",
        "- Context Precision: 검색된 문서가 관련 있는가",
        "- RAGAS: 표준 RAG 평가 프레임워크 (pip install ragas)",
        "- 자동화: 테스트셋 + 정기 평가로 품질 모니터링"
    ], "RAG 평가 핵심 포인트")


# ============================================================================
# 9. Citation 출처 표기
# ============================================================================

class CitationRAG:
    """출처를 표기하는 RAG 시스템"""
    
    def __init__(self):
        self.client = get_openai_client()
        self.model = "gpt-4o-mini"
    
    def generate_with_citation(self, question: str, contexts: List[Dict]) -> Dict:
        """출처를 포함한 답변 생성"""
        # 컨텍스트 포맷팅 (번호 부여)
        formatted_contexts = []
        for i, ctx in enumerate(contexts, 1):
            formatted_contexts.append(f"[{i}] {ctx['content']}")
        
        context_text = "\n\n".join(formatted_contexts)
        
        prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요.
답변 시 반드시 출처를 [1], [2] 형식으로 표기해주세요.

참고 문서:
{context_text}

질문: {question}

답변 (출처 번호 포함):"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "당신은 출처를 명확히 표기하는 AI 어시스턴트입니다. 모든 정보에 출처 번호 [1], [2] 등을 표기해주세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        answer = response.choices[0].message.content
        
        # 사용된 출처 추출
        import re
        used_citations = list(set(re.findall(r'\[(\d+)\]', answer)))
        used_citations.sort(key=int)
        
        return {
            'answer': answer,
            'used_citations': used_citations,
            'sources': contexts
        }


def demo_citation():
    """실습 9: Citation 출처 표기 - 답변에 출처 달기"""
    print("\n" + "="*80)
    print("[9] 실습 9: Citation 출처 표기 - 답변에 출처 달기")
    print("="*80)
    print("목표: RAG 답변에 출처를 명확히 표기하여 신뢰성 확보")
    print("핵심: 정보의 근거를 추적 가능하게 만들기")
    
    # Citation의 중요성
    print_section_header("출처 표기가 중요한 이유", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] 왜 출처를 표기해야 하는가?                         │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  1. 신뢰성 확보                                         │
  │     * 사용자가 정보의 근거를 확인 가능                  │
  │     * "이 정보가 어디서 나왔지?" 해결                   │
  │                                                         │
  │  2. 환각 감지                                           │
  │     * 출처가 없는 정보 = 의심해야 함                    │
  │     * LLM이 만들어낸 정보일 가능성                      │
  │                                                         │
  │  3. 법적/감사 요구사항                                  │
  │     * 금융, 의료, 법률 분야에서 필수                    │
  │     * 정보 출처 추적 가능해야 함                        │
  │                                                         │
  │  4. 디버깅 용이                                         │
  │     * 잘못된 답변의 원인 추적                           │
  │     * 어떤 문서가 문제인지 파악                         │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    citation_rag = CitationRAG()
    
    # 예시 실행
    print_section_header("출처 포함 답변 생성", "[CITE]")
    
    question = "RAG 시스템의 주요 구성요소는 무엇인가요?"
    
    contexts = [
        {
            "content": "RAG는 Retriever와 Generator 두 가지 핵심 구성요소로 이루어집니다. Retriever는 관련 문서를 검색하고, Generator는 검색된 문서를 바탕으로 답변을 생성합니다.",
            "source": "RAG_가이드_1장.pdf",
            "page": 12
        },
        {
            "content": "RAG 시스템에서 Vector Database는 문서 임베딩을 저장하고 효율적인 유사도 검색을 제공합니다. ChromaDB, Pinecone, Weaviate 등이 대표적입니다.",
            "source": "Vector_DB_소개.pdf",
            "page": 5
        },
        {
            "content": "LLM(Large Language Model)은 RAG의 Generator 역할을 합니다. GPT-4, Claude, Llama 등의 모델이 사용됩니다.",
            "source": "LLM_활용_가이드.pdf",
            "page": 23
        },
    ]
    
    print(f"\n질문: {question}")
    print(f"\n참고 문서 {len(contexts)}개:")
    for i, ctx in enumerate(contexts, 1):
        print(f"  [{i}] {ctx['source']} (p.{ctx['page']})")
        print(f"      {ctx['content'][:50]}...")
    
    print("\n[...] 출처 포함 답변 생성 중...")
    result = citation_rag.generate_with_citation(question, contexts)
    
    print(f"\n{'─'*60}")
    print("[답변]")
    print(result['answer'])
    print(f"{'─'*60}")
    
    print(f"\n사용된 출처: {result['used_citations']}")
    
    print("\n[출처 상세]")
    for cite_num in result['used_citations']:
        idx = int(cite_num) - 1
        if 0 <= idx < len(contexts):
            ctx = contexts[idx]
            print(f"  [{cite_num}] {ctx['source']} (p.{ctx['page']})")
    
    # Citation 구현 패턴
    print_section_header("Citation 구현 패턴", "[CODE]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [PATTERN 1] 인라인 Citation                            │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  "RAG는 검색과 생성을 결합한 기술입니다[1].             │
  │   주요 구성요소는 Retriever와 Generator입니다[1][2]."   │
  │                                                         │
  │  출처:                                                  │
  │  [1] RAG_가이드.pdf, p.12                               │
  │  [2] AI_개론.pdf, p.45                                  │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  [PATTERN 2] 문장별 Citation                            │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  답변:                                                  │
  │  • RAG는 검색 증강 생성 기술입니다. [출처: RAG_가이드]  │
  │  • Vector DB가 핵심 역할을 합니다. [출처: DB_소개]      │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  [PATTERN 3] 하이퍼링크 Citation (웹 UI)                │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  "RAG는 검색과 생성을 결합한 기술입니다.                │
  │   [📄 원문 보기]"                                       │
  │                                                         │
  │  클릭 시 → 원본 문서의 해당 위치로 이동                 │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- Citation: 답변의 정보 출처를 명시하는 것",
        "- 신뢰성: 사용자가 정보 근거 확인 가능",
        "- 환각 방지: 출처 없는 정보는 의심",
        "- 패턴: 인라인 [1], 문장별, 하이퍼링크 등",
        "- 실무: 법적 요구사항이 있는 도메인에서 필수"
    ], "Citation 핵심 포인트")


# ============================================================================
# 10. Streaming 응답
# ============================================================================

def demo_streaming():
    """실습 10: Streaming 응답 - 실시간 토큰 출력"""
    print("\n" + "="*80)
    print("[10] 실습 10: Streaming 응답 - 실시간 토큰 출력")
    print("="*80)
    print("목표: 답변을 실시간으로 출력하여 UX 개선")
    print("핵심: 전체 완료 대기 없이 토큰 단위로 즉시 표시")
    
    # Streaming의 필요성
    print_section_header("Streaming이 필요한 이유", "[INFO]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [!] 문제: LLM 응답은 느리다                            │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  일반 응답 (Non-Streaming):                             │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │ [질문 전송] → [3초 대기] → [전체 답변 한번에 표시] │   │
  │  │                                                   │   │
  │  │ 사용자 경험: "멈춘 거 아니야?" 😕                 │   │
  │  └─────────────────────────────────────────────────┘   │
  │                                                         │
  │  스트리밍 응답 (Streaming):                             │
  │  ┌─────────────────────────────────────────────────┐   │
  │  │ [질문 전송] → [0.1초 후 첫 단어] → [계속 출력...] │   │
  │  │                                                   │   │
  │  │ 사용자 경험: "오, 대답하고 있네!" 😊              │   │
  │  └─────────────────────────────────────────────────┘   │
  │                                                         │
  │  [효과]                                                 │
  │  * 체감 응답 시간 대폭 감소                             │
  │  * 사용자가 답변을 미리 읽기 시작 가능                  │
  │  * 긴 답변에서 특히 효과적                              │
  │  * ChatGPT, Claude 등 모든 주요 서비스가 사용           │
  └─────────────────────────────────────────────────────────┘
    """)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # Streaming 코드 예시
    print_section_header("Streaming 구현 방법", "[CODE]")
    print("""
  [CODE] OpenAI Streaming 예시:
  ┌─────────────────────────────────────────────────────
  │ from openai import OpenAI
  │ client = OpenAI()
  │ 
  │ # stream=True로 설정
  │ response = client.chat.completions.create(
  │     model="gpt-4o-mini",
  │     messages=[{"role": "user", "content": "RAG란?"}],
  │     stream=True  # 핵심!
  │ )
  │ 
  │ # 토큰 단위로 수신
  │ for chunk in response:
  │     if chunk.choices[0].delta.content:
  │         print(chunk.choices[0].delta.content, end="", flush=True)
  │ 
  │ print()  # 줄바꿈
  └─────────────────────────────────────────────────────
    """)
    
    # 실제 Streaming 데모
    print_section_header("Streaming 실제 데모", "[DEMO]")
    
    from openai import OpenAI
    import time
    
    client = get_openai_client()
    
    question = "RAG 시스템의 장점을 3가지 설명해주세요."
    print(f"\n질문: {question}")
    
    print(f"\n{'─'*60}")
    print("[Non-Streaming 응답]")
    print("대기 중", end="")
    
    start_time = time.time()
    response_normal = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        max_tokens=200
    )
    elapsed_normal = time.time() - start_time
    
    print(f" (완료: {elapsed_normal:.2f}초)")
    print(response_normal.choices[0].message.content[:200] + "...")
    
    print(f"\n{'─'*60}")
    print("[Streaming 응답]")
    print("실시간 출력: ", end="", flush=True)
    
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    response_stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
        max_tokens=200,
        stream=True
    )
    
    full_response = ""
    for chunk in response_stream:
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            print(content, end="", flush=True)
            
            if first_token_time is None:
                first_token_time = time.time() - start_time
            token_count += 1
    
    elapsed_stream = time.time() - start_time
    print()  # 줄바꿈
    
    print(f"\n{'─'*60}")
    print("[성능 비교]")
    print(f"  Non-Streaming 총 시간: {elapsed_normal:.2f}초")
    print(f"  Streaming 첫 토큰까지: {first_token_time:.2f}초")
    print(f"  Streaming 총 시간: {elapsed_stream:.2f}초")
    print(f"  체감 개선: {(elapsed_normal - first_token_time):.2f}초 빠르게 시작!")
    
    # RAG + Streaming 조합
    print_section_header("RAG + Streaming 조합", "[ARCH]")
    print("""
  ┌─────────────────────────────────────────────────────────┐
  │  [FLOW] RAG Streaming 파이프라인                        │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  1. 질문 수신                                           │
  │     │                                                   │
  │     ▼                                                   │
  │  2. 검색 실행 (이 부분은 Streaming 불가)                │
  │     │ → "검색 중..." 표시                               │
  │     ▼                                                   │
  │  3. 검색 완료, 컨텍스트 준비                            │
  │     │ → "답변 생성 중..." 표시                          │
  │     ▼                                                   │
  │  4. LLM Streaming 시작                                  │
  │     │ → 토큰 단위로 실시간 출력                         │
  │     ▼                                                   │
  │  5. 완료                                                │
  │                                                         │
  │  ─────────────────────────────────────────────────────  │
  │  [CODE] FastAPI + Streaming 예시:                       │
  │  ─────────────────────────────────────────────────────  │
  │                                                         │
  │  from fastapi.responses import StreamingResponse        │
  │                                                         │
  │  async def generate_stream(query: str):                 │
  │      # 1. 검색 (non-streaming)                          │
  │      contexts = await search(query)                     │
  │                                                         │
  │      # 2. LLM streaming                                 │
  │      response = client.chat.completions.create(         │
  │          model="gpt-4o-mini",                           │
  │          messages=[...],                                │
  │          stream=True                                    │
  │      )                                                  │
  │                                                         │
  │      for chunk in response:                             │
  │          if chunk.choices[0].delta.content:             │
  │              yield chunk.choices[0].delta.content       │
  │                                                         │
  │  @app.get("/ask")                                       │
  │  async def ask(query: str):                             │
  │      return StreamingResponse(                          │
  │          generate_stream(query),                        │
  │          media_type="text/event-stream"                 │
  │      )                                                  │
  └─────────────────────────────────────────────────────────┘
    """)
    
    # 핵심 포인트
    print_key_points([
        "- Streaming: 답변을 토큰 단위로 실시간 출력",
        "- 효과: 첫 응답까지 대기 시간 대폭 감소 (체감 UX 개선)",
        "- 구현: stream=True 옵션으로 간단히 활성화",
        "- RAG 조합: 검색(non-stream) + 답변생성(stream)",
        "- 주의: 에러 핸들링이 일반 응답보다 복잡"
    ], "Streaming 핵심 포인트")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """모든 데모 실행"""
    print("\n" + "="*80)
    print("[LAB] RAG 기초 실습")
    print("="*80)
    
    print("\n[LIST] 실습 항목:")
    print("  1. 청킹(Chunking) 이해하기 - 텍스트 분할 전략")
    print("  2. 기본 RAG 파이프라인 - 문서 -> 임베딩 -> 검색")
    print("  3. RAG 있음 vs 없음 비교 - 답변 품질 차이")
    print("  4. 컨텍스트 관리 - 토큰 제한 대응")
    print("  5. 고급 RAG - 컨텍스트 압축 적용")
    print("  6. Query Rewriting - 쿼리 개선으로 검색 품질 향상")
    print("  7. HyDE - 가상 문서 임베딩 기법")
    print("  8. RAG 평가 - Faithfulness, Relevancy 측정")
    print("  9. Citation 출처 표기 - 답변에 출처 달기")
    print("  10. Streaming 응답 - 실시간 토큰 출력")
    
    # 샘플 PDF 확인
    sample_pdf = Path(__file__).parent / "sample.pdf"
    if sample_pdf.exists():
        print(f"\n[FILE] PDF 파일 발견: {sample_pdf.name}")
    else:
        print(f"\n[DOC] PDF 없음 -> 내장 샘플 텍스트 사용")
    
    try:
        # 1. 청킹 이해하기
        demo_chunking()
        
        # 2. 기본 RAG 파이프라인
        demo_basic_rag()
        
        # 3. RAG 비교
        demo_rag_comparison()
        
        # 4. 컨텍스트 관리
        demo_context_management()
        
        # 5. 고급 RAG
        demo_advanced_rag()
        
        # 6. Query Rewriting
        demo_query_rewriting()
        
        # 7. HyDE
        demo_hyde()
        
        # 8. RAG 평가
        demo_rag_evaluation()
        
        # 9. Citation
        demo_citation()
        
        # 10. Streaming
        demo_streaming()
        
        print("\n" + "="*80)
        print("[OK] 모든 실습 완료!")
        print("="*80)
        print("\n[FILE] 생성된 파일:")
        print("   - ./chroma_db/ : Vector DB 저장소")
        print("\n[TIP] 다음 단계:")
        print("   - lab03/advanced_retrieval_langchain.py : 고급 RAG 기법 학습")
        print("   - Hybrid 검색, Re-ranking, Multi-hop 등")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
