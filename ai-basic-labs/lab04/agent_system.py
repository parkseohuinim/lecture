"""
AI Agent 시스템 실습
- 단일 에이전트에서 멀티 에이전트 오케스트레이션까지

실습 항목:
1. 단일 JSON 프롬프트 에이전트 - 의도/카테고리 분류
2. RAG Agent 통합 - 검색 + 답변 생성
3. 멀티 에이전트 오케스트레이션 - Planner -> Worker 구조
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

# OpenAI
from openai import OpenAI

# LangChain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Pydantic for structured output
from pydantic import BaseModel, Field

# 환경 변수
from dotenv import load_dotenv

# 프로젝트 루트의 .env 파일 로드
project_root = Path(__file__).parent.parent
load_dotenv(dotenv_path=project_root / '.env')

# 공통 유틸리티 import를 위한 경로 추가
sys.path.insert(0, str(project_root))
from utils import print_section_header, print_subsection, print_key_points

# 공통 데이터 임포트
from shared_data import (
    CUSTOMER_SERVICE_DOCS, 
    DEVELOPMENT_DOCS, 
    PLANNING_DOCS,
    CATEGORIES,
    SAMPLE_QUESTIONS,
    get_all_documents,
    get_document_by_category
)


# ============================================================================
# 데이터 클래스 및 Enum
# ============================================================================

class IntentCategory(str, Enum):
    """질문 카테고리"""
    CUSTOMER_SERVICE = "customer_service"
    DEVELOPMENT = "development"
    PLANNING = "planning"
    UNKNOWN = "unknown"


class IntentType(str, Enum):
    """질문 의도 유형"""
    INQUIRY = "inquiry"           # 정보 문의
    TROUBLESHOOTING = "troubleshooting"  # 문제 해결
    PROCESS = "process"           # 절차 문의
    POLICY = "policy"             # 정책 문의


@dataclass
class ClassificationResult:
    """분류 결과 데이터 클래스"""
    category: str
    intent: str
    confidence: float
    reasoning: str
    keywords: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    score: float
    metadata: Dict[str, Any]
    rank: int


@dataclass
class AgentResponse:
    """에이전트 응답 데이터 클래스"""
    agent_name: str
    output: Any
    elapsed_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Pydantic 모델 (구조화된 출력용)
# ============================================================================

class IntentClassification(BaseModel):
    """의도 분류 결과 스키마"""
    category: str = Field(description="질문의 카테고리: customer_service, development, planning, unknown 중 하나")
    intent: str = Field(description="질문의 의도: inquiry, troubleshooting, process, policy 중 하나")
    confidence: float = Field(description="분류 확신도: 0.0 ~ 1.0")
    reasoning: str = Field(description="분류 이유 설명")
    keywords: List[str] = Field(description="질문에서 추출한 핵심 키워드 리스트")


class PlanStep(BaseModel):
    """실행 계획 단계"""
    step_number: int = Field(description="단계 번호")
    agent: str = Field(description="실행할 에이전트 이름")
    action: str = Field(description="수행할 작업")
    input_from: Optional[str] = Field(default=None, description="입력을 받을 이전 단계")


class ExecutionPlan(BaseModel):
    """실행 계획 스키마"""
    question: str = Field(description="원본 질문")
    steps: List[PlanStep] = Field(description="실행 단계 리스트")
    final_agent: str = Field(description="최종 응답을 생성할 에이전트")


class SummaryResult(BaseModel):
    """요약 결과 스키마"""
    summary: str = Field(description="검색 결과 요약")
    key_points: List[str] = Field(description="핵심 포인트 리스트")
    source_count: int = Field(description="참조한 문서 수")


# ============================================================================
# 1. 단일 JSON 프롬프트 에이전트 - 의도 분류기
# ============================================================================

class IntentClassifierAgent:
    """
    의도 분류 에이전트 (Agent A)
    사용자 질문을 분석하여 카테고리와 의도를 JSON 형식으로 추론
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        self.name = "IntentClassifier"
        
        # 분류 프롬프트
        self.system_prompt = """당신은 고객 질문을 분류하는 전문가입니다.
주어진 질문을 분석하여 적절한 카테고리와 의도를 분류해주세요.

## 카테고리 정의
- customer_service: 환불, 배송, 교환, 결제, 회원 등급 등 고객 서비스 관련
- development: API, 코드, 개발, 배포, 에러, 버그 등 개발 관련
- planning: 기획, 일정, 요구사항, 스프린트, KPI 등 기획 관련
- unknown: 위 카테고리에 해당하지 않는 경우

## 의도 유형
- inquiry: 정보나 방법에 대한 문의
- troubleshooting: 문제 해결 요청
- process: 절차나 프로세스 문의
- policy: 정책이나 규정 문의

## 응답 형식
반드시 다음 JSON 형식으로만 응답하세요:
{
    "category": "카테고리명",
    "intent": "의도유형",
    "confidence": 0.0~1.0 사이의 확신도,
    "reasoning": "분류 이유",
    "keywords": ["핵심", "키워드", "리스트"]
}"""
    
    def classify(self, question: str) -> ClassificationResult:
        """
        질문을 분류하고 JSON 형식으로 결과 반환
        
        Args:
            question: 사용자 질문
        
        Returns:
            ClassificationResult 객체
        """
        start_time = time.time()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"질문: {question}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        elapsed_time = time.time() - start_time
        
        # JSON 파싱
        try:
            result = json.loads(response.choices[0].message.content)
            return ClassificationResult(
                category=result.get("category", "unknown"),
                intent=result.get("intent", "inquiry"),
                confidence=result.get("confidence", 0.0),
                reasoning=result.get("reasoning", ""),
                keywords=result.get("keywords", [])
            )
        except json.JSONDecodeError:
            return ClassificationResult(
                category="unknown",
                intent="inquiry",
                confidence=0.0,
                reasoning="JSON 파싱 실패",
                keywords=[]
            )
    
    def classify_batch(self, questions: List[str]) -> List[ClassificationResult]:
        """여러 질문을 일괄 분류"""
        return [self.classify(q) for q in questions]


# ============================================================================
# 2. RAG Agent - 검색 에이전트
# ============================================================================

class RetrievalAgent:
    """
    검색 에이전트 (Agent B)
    Vector DB에서 관련 문서를 검색
    """
    
    def __init__(self, persist_directory: str = None, collection_name: str = "agent_rag"):
        self.embeddings = OpenAIEmbeddings()
        self.client = OpenAI()
        self.name = "Retrieval"
        
        if persist_directory is None:
            persist_directory = str(Path(__file__).parent / "chroma_db")
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = None
        self.documents = []
    
    def _chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """텍스트를 청크로 분할"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            
            if end >= text_length:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break
            
            # 문장 경계 찾기
            best_end = -1
            double_newline = text.rfind('\n\n', start, end + 50)
            if double_newline != -1:
                best_end = double_newline + 2
            
            if best_end == -1:
                period = text.rfind('. ', start, end + 30)
                if period != -1:
                    best_end = period + 2
            
            if best_end == -1:
                newline = text.rfind('\n', start, end + 20)
                if newline != -1:
                    best_end = newline + 1
            
            if best_end == -1:
                best_end = end
            
            chunk = text[start:best_end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = best_end - overlap
            if start <= 0:
                start = best_end
        
        return chunks
    
    def ingest_documents(self, category_filter: str = None):
        """
        문서를 Vector DB에 저장
        
        Args:
            category_filter: 특정 카테고리만 저장 (None이면 전체)
        """
        print(f"\n[DOC] 문서 인덱싱 시작...")
        
        # 카테고리별 문서
        category_docs = {
            "customer_service": CUSTOMER_SERVICE_DOCS,
            "development": DEVELOPMENT_DOCS,
            "planning": PLANNING_DOCS
        }
        
        documents = []
        
        for category, doc_text in category_docs.items():
            if category_filter and category != category_filter:
                continue
            
            chunks = self._chunk_text(doc_text)
            
            for i, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={
                        "category": category,
                        "chunk_id": i,
                        "source": f"{CATEGORIES[category]['name']} 가이드"
                    }
                ))
        
        self.documents = documents
        print(f"   총 청크 수: {len(documents)}개")
        
        # Vector DB 생성
        import chromadb
        try:
            chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            chroma_client.delete_collection(name=self.collection_name)
        except:
            pass
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        print(f"[OK] 문서 인덱싱 완료")
    
    def search(self, query: str, k: int = 5, category_filter: str = None) -> List[SearchResult]:
        """
        쿼리와 유사한 문서 검색
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            category_filter: 특정 카테고리만 검색
        """
        if not self.vectorstore:
            raise ValueError("문서를 먼저 인덱싱하세요 (ingest_documents 호출)")
        
        # 필터 설정
        filter_dict = None
        if category_filter:
            filter_dict = {"category": category_filter}
        
        if filter_dict:
            docs_with_scores = self.vectorstore.similarity_search_with_score(
                query, k=k, filter=filter_dict
            )
        else:
            docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)
        
        results = []
        for rank, (doc, score) in enumerate(docs_with_scores):
            similarity = 1 / (1 + score)
            results.append(SearchResult(
                content=doc.page_content,
                score=similarity,
                metadata=doc.metadata,
                rank=rank + 1
            ))
        
        return results


# ============================================================================
# 3. 요약 에이전트
# ============================================================================

class SummarizationAgent:
    """
    요약 에이전트 (Agent C)
    검색된 문서를 질문에 맞게 요약
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        self.name = "Summarization"
        
        self.system_prompt = """당신은 문서 요약 전문가입니다.
주어진 검색 결과를 사용자의 질문에 맞게 요약해주세요.

## 요약 원칙
1. 질문과 관련된 정보만 추출
2. 핵심 포인트를 명확히 정리
3. 불필요한 정보 제거
4. 실행 가능한 정보 우선

## 응답 형식
반드시 다음 JSON 형식으로 응답하세요:
{
    "summary": "검색 결과 요약 (2-3문장)",
    "key_points": ["핵심 포인트 1", "핵심 포인트 2", ...],
    "source_count": 참조한 문서 수
}"""
    
    def summarize(self, question: str, search_results: List[SearchResult]) -> Dict[str, Any]:
        """
        검색 결과 요약
        
        Args:
            question: 사용자 질문
            search_results: 검색 결과 리스트
        
        Returns:
            요약 결과 딕셔너리
        """
        start_time = time.time()
        
        # 검색 결과 컨텍스트 구성
        context = "\n\n".join([
            f"[문서 {r.rank}] (관련도: {r.score:.2f})\n{r.content}"
            for r in search_results
        ])
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"질문: {question}\n\n검색 결과:\n{context}"}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        elapsed_time = time.time() - start_time
        
        try:
            result = json.loads(response.choices[0].message.content)
            return {
                "summary": result.get("summary", ""),
                "key_points": result.get("key_points", []),
                "source_count": result.get("source_count", len(search_results)),
                "elapsed_time": elapsed_time
            }
        except json.JSONDecodeError:
            return {
                "summary": "요약 생성 실패",
                "key_points": [],
                "source_count": 0,
                "elapsed_time": elapsed_time
            }


# ============================================================================
# 4. 최종 답변 에이전트
# ============================================================================

class FinalAnswerAgent:
    """
    최종 답변 에이전트 (Agent D)
    모든 정보를 종합하여 최종 답변 생성
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        self.name = "FinalAnswer"
        
        self.system_prompt = """당신은 친절하고 전문적인 AI 어시스턴트입니다.
주어진 정보를 바탕으로 사용자 질문에 답변해주세요.

## 답변 원칙
1. 제공된 정보를 기반으로 정확하게 답변
2. 친절하고 이해하기 쉬운 언어 사용
3. 필요한 경우 단계별로 설명
4. 추가 도움이 필요할 수 있는 사항 안내

## 주의사항
- 제공된 정보에 없는 내용은 추측하지 마세요
- 불확실한 경우 해당 부서에 문의하도록 안내하세요"""
    
    def generate_answer(
        self, 
        question: str,
        classification: ClassificationResult,
        summary: Dict[str, Any],
        search_results: List[SearchResult]
    ) -> str:
        """
        최종 답변 생성
        
        Args:
            question: 사용자 질문
            classification: 분류 결과
            summary: 요약 결과
            search_results: 검색 결과
        
        Returns:
            최종 답변 문자열
        """
        start_time = time.time()
        
        # 컨텍스트 구성
        category_name = CATEGORIES.get(classification.category, {}).get("name", "알 수 없음")
        
        context = f"""## 질문 분석
- 카테고리: {category_name}
- 의도: {classification.intent}
- 핵심 키워드: {', '.join(classification.keywords)}

## 검색 결과 요약
{summary.get('summary', '')}

## 핵심 포인트
{chr(10).join(['- ' + p for p in summary.get('key_points', [])])}

## 상세 정보
{chr(10).join([r.content for r in search_results[:3]])}"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"질문: {question}\n\n{context}"}
            ],
            temperature=0.5,
            max_tokens=500
        )
        
        return response.choices[0].message.content


# ============================================================================
# 5. 오케스트레이터 (Planner)
# ============================================================================

class OrchestratorAgent:
    """
    오케스트레이터 에이전트 (Planner)
    멀티 에이전트 실행을 조율
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        self.name = "Orchestrator"
        
        # 에이전트 초기화
        self.classifier = IntentClassifierAgent(model)
        self.retriever = RetrievalAgent()
        self.summarizer = SummarizationAgent(model)
        self.final_answer = FinalAnswerAgent(model)
        
        # 실행 로그
        self.execution_log = []
        
        self.plan_prompt = """당신은 멀티 에이전트 시스템의 플래너입니다.
사용자의 질문을 분석하여 어떤 에이전트들을 어떤 순서로 실행할지 계획을 세워주세요.

## 사용 가능한 에이전트
- IntentClassifier: 질문의 카테고리와 의도를 분류
- Retrieval: 관련 문서를 검색
- Summarization: 검색 결과를 요약
- FinalAnswer: 최종 답변 생성

## 일반적인 실행 흐름
1. IntentClassifier -> 질문 분류
2. Retrieval -> 분류된 카테고리에서 검색
3. Summarization -> 검색 결과 요약
4. FinalAnswer -> 최종 답변 생성

## 응답 형식
반드시 다음 JSON 형식으로 응답하세요:
{
    "question": "원본 질문",
    "steps": [
        {"step_number": 1, "agent": "에이전트명", "action": "수행할 작업", "input_from": null 또는 이전 단계 번호},
        ...
    ],
    "final_agent": "최종 응답을 생성할 에이전트"
}"""
    
    def setup(self, ingest_documents: bool = True):
        """시스템 초기화"""
        print("\n[SETUP] 오케스트레이터 초기화 중...")
        
        if ingest_documents:
            self.retriever.ingest_documents()
        
        print("[OK] 오케스트레이터 준비 완료")
    
    def create_plan(self, question: str) -> Dict[str, Any]:
        """
        실행 계획 생성
        
        Args:
            question: 사용자 질문
        
        Returns:
            실행 계획 딕셔너리
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.plan_prompt},
                {"role": "user", "content": f"질문: {question}"}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        
        try:
            plan = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            # 기본 계획 반환
            plan = {
                "question": question,
                "steps": [
                    {"step_number": 1, "agent": "IntentClassifier", "action": "질문 분류", "input_from": None},
                    {"step_number": 2, "agent": "Retrieval", "action": "문서 검색", "input_from": 1},
                    {"step_number": 3, "agent": "Summarization", "action": "결과 요약", "input_from": 2},
                    {"step_number": 4, "agent": "FinalAnswer", "action": "답변 생성", "input_from": 3}
                ],
                "final_agent": "FinalAnswer"
            }
        
        # FinalAnswer 단계가 없으면 자동으로 추가 (LLM이 누락할 수 있음)
        agent_names = [step.get("agent") for step in plan.get("steps", [])]
        if "FinalAnswer" not in agent_names:
            last_step = len(plan.get("steps", []))
            plan["steps"].append({
                "step_number": last_step + 1,
                "agent": "FinalAnswer",
                "action": "최종 답변 생성",
                "input_from": last_step
            })
            plan["final_agent"] = "FinalAnswer"
        
        return plan
    
    def execute_plan(self, question: str, verbose: bool = True) -> Dict[str, Any]:
        """
        계획을 실행하고 최종 결과 반환
        
        Args:
            question: 사용자 질문
            verbose: 상세 출력 여부
        
        Returns:
            실행 결과 딕셔너리
        """
        start_time = time.time()
        self.execution_log = []
        
        # 1. 계획 생성
        if verbose:
            print(f"\n{'='*60}")
            print(f"[PLAN] 실행 계획 생성 중...")
        
        plan = self.create_plan(question)
        
        if verbose:
            print(f"[OK] 계획 생성 완료: {len(plan['steps'])}단계")
            for step in plan['steps']:
                print(f"   {step['step_number']}. {step['agent']}: {step['action']}")
        
        # 2. 단계별 실행
        results = {}
        classification = None
        search_results = []
        summary = {}
        final_answer = ""
        
        for step in plan['steps']:
            step_start = time.time()
            agent_name = step['agent']
            
            if verbose:
                print(f"\n{'─'*40}")
                print(f"[>] Step {step['step_number']}: {agent_name}")
            
            if agent_name == "IntentClassifier":
                classification = self.classifier.classify(question)
                results['classification'] = classification
                
                if verbose:
                    print(f"   카테고리: {classification.category}")
                    print(f"   의도: {classification.intent}")
                    print(f"   확신도: {classification.confidence:.2f}")
                    print(f"   키워드: {classification.keywords}")
            
            elif agent_name == "Retrieval":
                category = classification.category if classification else None
                search_results = self.retriever.search(question, k=5, category_filter=category)
                results['search_results'] = search_results
                
                if verbose:
                    print(f"   검색 결과: {len(search_results)}개")
                    if search_results:
                        print(f"   상위 결과 점수: {search_results[0].score:.4f}")
            
            elif agent_name == "Summarization":
                summary = self.summarizer.summarize(question, search_results)
                results['summary'] = summary
                
                if verbose:
                    print(f"   핵심 포인트: {len(summary.get('key_points', []))}개")
            
            elif agent_name == "FinalAnswer":
                final_answer = self.final_answer.generate_answer(
                    question, classification, summary, search_results
                )
                results['final_answer'] = final_answer
            
            step_time = time.time() - step_start
            self.execution_log.append({
                "step": step['step_number'],
                "agent": agent_name,
                "elapsed_time": step_time
            })
            
            if verbose:
                print(f"   소요 시간: {step_time:.2f}초")
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"[OK] 전체 실행 완료 (총 {total_time:.2f}초)")
        
        return {
            "question": question,
            "plan": plan,
            "classification": classification,
            "search_results": search_results,
            "summary": summary,
            "final_answer": final_answer,
            "total_time": total_time,
            "execution_log": self.execution_log
        }
    
    def process_question(self, question: str, verbose: bool = True) -> str:
        """
        질문 처리 및 답변 반환 (간단한 인터페이스)
        
        Args:
            question: 사용자 질문
            verbose: 상세 출력 여부
        
        Returns:
            최종 답변 문자열
        """
        result = self.execute_plan(question, verbose=verbose)
        return result['final_answer']


# ============================================================================
# 6. 단순화된 RAG Agent (실습 2용)
# ============================================================================

class SimpleRAGAgent:
    """
    단순화된 RAG 에이전트
    질문 -> 분류 -> 검색 -> 재정리 -> 답변의 단일 파이프라인
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        self.retriever = RetrievalAgent()
        self.classifier = IntentClassifierAgent(model)  # 분류기 추가
        self.name = "SimpleRAG"
    
    def setup(self):
        """초기화"""
        self.retriever.ingest_documents()
    
    def answer(self, question: str, k: int = 3, use_classification: bool = True) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성
        
        Args:
            question: 사용자 질문
            k: 검색할 문서 수
            use_classification: 분류 후 해당 카테고리에서만 검색할지 여부
        
        Returns:
            답변 결과 딕셔너리
        """
        start_time = time.time()
        
        # 1. 의도 분류 (카테고리 결정)
        classification = None
        category_filter = None
        
        if use_classification:
            classification = self.classifier.classify(question)
            # unknown이 아니면 해당 카테고리에서만 검색
            if classification.category != "unknown":
                category_filter = classification.category
        
        # 2. 검색 (분류된 카테고리 필터 적용)
        search_results = self.retriever.search(question, k=k, category_filter=category_filter)
        
        # 3. 컨텍스트 구성
        context = "\n\n".join([
            f"[참고 {i+1}] {r.content}"
            for i, r in enumerate(search_results)
        ])
        
        # 4. 답변 생성
        system_prompt = """당신은 친절한 AI 어시스턴트입니다.
제공된 문서를 참고하여 질문에 정확하게 답변해주세요.
문서에 없는 내용은 추측하지 말고, 해당 부서에 문의하도록 안내하세요."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"질문: {question}\n\n참고 문서:\n{context}"}
            ],
            temperature=0.5,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        elapsed_time = time.time() - start_time
        
        return {
            "question": question,
            "answer": answer,
            "search_results": search_results,
            "classification": classification,
            "category_filter": category_filter,
            "elapsed_time": elapsed_time
        }


# ============================================================================
# 데모 함수들
# ============================================================================

def demo_single_agent():
    """실습 1: 단일 JSON 프롬프트 에이전트"""
    print("\n" + "="*80)
    print("[1] 실습 1: 단일 JSON 프롬프트 에이전트")
    print("="*80)
    print("목표: 사용자 질문을 JSON 형식으로 의도/카테고리 추론")
    print("핵심: 구조화된 출력으로 다음 단계 처리 용이")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 분류기 초기화
    classifier = IntentClassifierAgent()
    
    # 테스트 질문들
    test_questions = [
        "구매한 지 10일 됐는데 환불 가능한가요?",
        "API 인증은 어떤 방식을 사용하나요?",
        "새 프로젝트 기획서에 뭘 포함해야 하나요?",
        "오늘 날씨 어때요?"  # unknown 테스트
    ]
    
    print_section_header("질문 분류 테스트", "[>>>]")
    
    for question in test_questions:
        print(f"\n{'─'*60}")
        print(f"[*] 질문: {question}")
        print(f"{'─'*60}")
        
        result = classifier.classify(question)
        
        # 카테고리 이름
        category_name = CATEGORIES.get(result.category, {}).get("name", "알 수 없음")
        
        print(f"\n[JSON] 분류 결과:")
        print(f"  {{")
        print(f'    "category": "{result.category}" ({category_name}),')
        print(f'    "intent": "{result.intent}",')
        print(f'    "confidence": {result.confidence:.2f},')
        print(f'    "reasoning": "{result.reasoning}",')
        print(f'    "keywords": {result.keywords}')
        print(f"  }}")
        
        # 확신도 시각화
        bar_len = int(result.confidence * 20)
        bar = "=" * bar_len + "-" * (20 - bar_len)
        print(f"\n  확신도: [{bar}] {result.confidence:.0%}")
    
    # 핵심 포인트
    print_key_points([
        "- JSON 출력: 구조화된 형식으로 후처리 용이",
        "- response_format: {'type': 'json_object'}로 JSON 강제",
        "- 확신도: 분류 결과의 신뢰성 판단에 활용",
        "- 키워드: 후속 검색 쿼리 확장에 활용 가능"
    ], "JSON 프롬프트 에이전트 핵심")


def demo_rag_agent():
    """실습 2: RAG Agent 통합"""
    print("\n" + "="*80)
    print("[2] 실습 2: RAG Agent 통합")
    print("="*80)
    print("목표: 질문 -> 분류 -> 검색 -> 재정리 -> 답변 파이프라인 구현")
    print("핵심: 분류 후 해당 카테고리에서만 검색하여 정확도 향상")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # RAG 에이전트 초기화
    print_section_header("RAG Agent 초기화", "[SETUP]")
    rag_agent = SimpleRAGAgent()
    rag_agent.setup()
    
    # 테스트 질문들
    test_questions = [
        "환불 절차가 어떻게 되나요?",
        "API 인증 토큰의 유효 기간은 얼마인가요?",
        "스프린트 회고는 어떻게 진행하나요?"
    ]
    
    print_section_header("RAG 질의응답 테스트", "[>>>]")
    
    for question in test_questions:
        print(f"\n{'='*60}")
        print(f"[*] 질문: {question}")
        print(f"{'='*60}")
        
        result = rag_agent.answer(question)
        
        # 분류 결과
        if result.get('classification'):
            cls = result['classification']
            category_name = CATEGORIES.get(cls.category, {}).get("name", "알 수 없음")
            print(f"\n[CLASSIFY] 분류 결과: {cls.category} ({category_name})")
            print(f"   확신도: {cls.confidence:.0%} | 키워드: {cls.keywords}")
        
        # 검색 결과
        print(f"\n[SEARCH] 검색 결과 ({len(result['search_results'])}개):")
        if result.get('category_filter'):
            print(f"   ('{result['category_filter']}' 카테고리에서 검색)")
        for sr in result['search_results'][:2]:
            print(f"  [{sr.rank}] 점수: {sr.score:.4f} | {sr.metadata.get('category', '')}")
            preview = sr.content[:80].replace('\n', ' ')
            print(f"      {preview}...")
        
        # 답변
        print(f"\n[ANSWER] 답변:")
        print(f"{'─'*50}")
        for line in result['answer'].split('\n'):
            print(f"  {line}")
        print(f"{'─'*50}")
        
        print(f"\n[INFO] 처리 시간: {result['elapsed_time']:.2f}초")
    
    # 핵심 포인트
    print_key_points([
        "- 파이프라인: 분류 -> 검색 -> 컨텍스트 구성 -> 답변 생성",
        "- 분류 우선: 카테고리 분류 후 해당 영역에서만 검색",
        "- Vector DB: 의미 기반 검색으로 관련 문서 찾기",
        "- 카테고리 필터: 잘못된 문서 검색 방지, 정확도 향상"
    ], "RAG Agent 핵심")


def demo_multi_agent():
    """실습 3: 멀티 에이전트 오케스트레이션"""
    print("\n" + "="*80)
    print("[3] 실습 3: 멀티 에이전트 오케스트레이션")
    print("="*80)
    print("목표: Planner -> Worker 구조로 복잡한 질문 처리")
    print("핵심: 각 에이전트가 전문 영역 담당, 오케스트레이터가 조율")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 오케스트레이터 아키텍처 설명
    print_section_header("멀티 에이전트 아키텍처", "[ARCH]")
    print("""
  +-----------------------------------------------------------+
  |                    Orchestrator (Planner)                  |
  |  - 실행 계획 수립                                          |
  |  - 에이전트 간 데이터 전달                                 |
  |  - 전체 흐름 조율                                          |
  +-----------------------------------------------------------+
                              |
       +----------------------+----------------------+
       |                      |                      |
       v                      v                      v
  +-----------+      +-------------+      +-------------+
  | Agent A   |      | Agent B     |      | Agent C     |
  | Intent    | ---> | Retrieval   | ---> | Summarize   |
  | Classify  |      | Search      |      |             |
  +-----------+      +-------------+      +-------------+
                                                 |
                                                 v
                                          +-------------+
                                          | Agent D     |
                                          | Final       |
                                          | Answer      |
                                          +-------------+
    """)
    
    # 오케스트레이터 초기화
    print_section_header("오케스트레이터 초기화", "[SETUP]")
    orchestrator = OrchestratorAgent()
    orchestrator.setup()
    
    # 테스트 질문
    test_questions = [
        "VIP 등급이 되려면 얼마나 구매해야 하고, 어떤 혜택이 있나요?",
        "개발 환경 설정과 코딩 컨벤션에 대해 알려주세요.",
    ]
    
    print_section_header("멀티 에이전트 질의응답", "[>>>]")
    
    for question in test_questions:
        print(f"\n{'='*70}")
        print(f"[*] 질문: {question}")
        print(f"{'='*70}")
        
        result = orchestrator.execute_plan(question, verbose=True)
        
        # 최종 답변
        print(f"\n{'='*70}")
        print(f"[FINAL] 최종 답변:")
        print(f"{'─'*60}")
        for line in result['final_answer'].split('\n'):
            print(f"  {line}")
        print(f"{'─'*60}")
        
        # 실행 로그 요약
        print(f"\n[LOG] 실행 로그:")
        for log in result['execution_log']:
            print(f"  Step {log['step']}: {log['agent']} ({log['elapsed_time']:.2f}초)")
        print(f"\n  총 소요 시간: {result['total_time']:.2f}초")
    
    # 핵심 포인트
    print_key_points([
        "- Planner: 질문 분석 후 실행 계획 수립",
        "- Worker Agents: 각자 전문 영역 담당",
        "  - IntentClassifier: 의도/카테고리 분류",
        "  - Retrieval: 관련 문서 검색",
        "  - Summarization: 검색 결과 요약",
        "  - FinalAnswer: 최종 답변 생성",
        "- 장점: 모듈화, 재사용성, 디버깅 용이"
    ], "멀티 에이전트 핵심")


def demo_full_pipeline():
    """전체 파이프라인 데모 - 실습 미션"""
    print("\n" + "="*80)
    print("[MISSION] 실습 미션: 고객센터/개발/기획 질의 자동 분류 + RAG 응답")
    print("="*80)
    print("시나리오: 다양한 부서의 질문을 자동으로 분류하고 적절한 답변 제공")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\n[!] OPENAI_API_KEY 환경변수를 설정해주세요!")
        return
    
    # 오케스트레이터 초기화
    print_section_header("시스템 초기화", "[SETUP]")
    orchestrator = OrchestratorAgent()
    orchestrator.setup()
    
    # 샘플 질문으로 테스트
    print_section_header("자동 분류 + RAG 응답 테스트", "[TEST]")
    
    # SAMPLE_QUESTIONS에서 각 카테고리별 1개씩
    test_questions = [
        SAMPLE_QUESTIONS[0],  # 고객센터
        SAMPLE_QUESTIONS[3],  # 개발
        SAMPLE_QUESTIONS[6],  # 기획
    ]
    
    results = []
    
    for sample in test_questions:
        question = sample["question"]
        expected = sample["expected_category"]
        
        print(f"\n{'='*70}")
        print(f"[*] 질문: {question}")
        print(f"   예상 카테고리: {expected}")
        print(f"{'='*70}")
        
        result = orchestrator.execute_plan(question, verbose=False)
        
        # 분류 결과 확인
        actual = result['classification'].category
        match = "[OK]" if actual == expected else "[X]"
        
        print(f"\n[CLASSIFY] 분류 결과: {actual} {match}")
        print(f"   확신도: {result['classification'].confidence:.0%}")
        
        # 답변
        print(f"\n[ANSWER] 답변:")
        print(f"{'─'*60}")
        answer_lines = result['final_answer'].split('\n')
        for line in answer_lines[:10]:  # 최대 10줄
            print(f"  {line}")
        if len(answer_lines) > 10:
            print(f"  ... (이하 생략)")
        print(f"{'─'*60}")
        
        results.append({
            "question": question,
            "expected": expected,
            "actual": actual,
            "match": actual == expected
        })
    
    # 결과 요약
    print_section_header("테스트 결과 요약", "[RESULT]")
    
    correct = sum(1 for r in results if r['match'])
    total = len(results)
    accuracy = correct / total * 100
    
    print(f"\n정확도: {correct}/{total} ({accuracy:.0f}%)")
    print(f"\n상세 결과:")
    for r in results:
        status = "[OK]" if r['match'] else "[X]"
        print(f"  {status} {r['question'][:30]}...")
        print(f"      예상: {r['expected']} | 실제: {r['actual']}")
    
    # 핵심 포인트
    print_key_points([
        "- 자동 분류: 질문을 적절한 부서로 라우팅",
        "- RAG 응답: 해당 부서 문서에서 검색 후 답변",
        "- 실무 적용: 챗봇, 헬프데스크, 내부 지원 시스템",
        "- 확장: 새 카테고리 추가 시 문서만 추가하면 됨"
    ], "실습 미션 핵심")


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """모든 실습 실행"""
    print("\n" + "="*80)
    print("[LAB] AI Agent 시스템 실습")
    print("="*80)
    
    print("\n[LIST] 실습 항목:")
    print("  1. 단일 JSON 프롬프트 에이전트 - 의도/카테고리 분류")
    print("  2. RAG Agent 통합 - 검색 + 답변 생성")
    print("  3. 멀티 에이전트 오케스트레이션 - Planner -> Worker")
    print("  4. [MISSION] 고객센터/개발/기획 질의 자동 분류 + RAG 응답")
    
    try:
        # 1. 단일 에이전트
        demo_single_agent()
        
        # 2. RAG 에이전트
        demo_rag_agent()
        
        # 3. 멀티 에이전트
        demo_multi_agent()
        
        # 4. 실습 미션
        demo_full_pipeline()
        
        # 완료 메시지
        print("\n" + "="*80)
        print("[OK] 모든 실습 완료!")
        print("="*80)
        
        print("\n[INFO] 오늘 배운 내용:")
        print("  1. JSON 프롬프트 에이전트: 구조화된 출력으로 의도 분류")
        print("  2. RAG Agent: 검색 증강 생성으로 정확한 답변")
        print("  3. 멀티 에이전트: 복잡한 작업을 전문 에이전트로 분할")
        print("  4. 오케스트레이션: Planner가 Worker들을 조율")
        
        print("\n[TIP] 실무 적용:")
        print("  - 챗봇: 의도 분류 -> 적절한 핸들러로 라우팅")
        print("  - 지원 시스템: 부서별 지식 베이스 + RAG")
        print("  - 복잡한 업무: 멀티 에이전트로 단계별 처리")
        
        print("\n[FILE] 생성된 파일:")
        print("   - ./chroma_db/ : Vector DB 저장소")
        
    except Exception as e:
        print(f"\n[X] 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

