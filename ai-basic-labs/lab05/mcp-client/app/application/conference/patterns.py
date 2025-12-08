"""
LangGraph Multi-Agent Patterns
- Sequential (파이프라인)
- Planner-Executor (계획-실행)
- Role-based Collaboration (역할 분담)
- Hierarchical (상하 구조)
- Debate / Critic (토론·검증)
- Swarm / Market-based (군집·경쟁)
"""
from typing import TypedDict, Annotated, Sequence, List, Dict, Any, Optional, Callable, Awaitable
from typing_extensions import TypedDict as TypedDictExt
import operator
import json
import logging
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger(__name__)


# ============================================================================
# 스트리밍 콜백 타입 정의
# ============================================================================
StreamCallback = Callable[[str, str], Awaitable[None]]  # (node_name, token) -> None


# ============================================================================
# State Definitions
# ============================================================================

class AgentState(TypedDict):
    """기본 에이전트 상태"""
    topic: str
    messages: Annotated[List[BaseMessage], operator.add]
    current_step: str
    results: Dict[str, Any]


class RoleBasedState(TypedDict):
    """Role-based 패턴 전용 상태 (병렬 실행 지원)"""
    topic: str
    # 각 역할별 독립적인 필드 (병렬 업데이트 가능)
    pm_opinion: Optional[str]
    dev_opinion: Optional[str]
    design_opinion: Optional[str]
    qa_opinion: Optional[str]
    final_decision: Optional[str]
    # 메시지는 자동 합쳐짐
    messages: Annotated[List[BaseMessage], operator.add]


class HierarchicalState(TypedDict):
    """Hierarchical 패턴 전용 상태 (병렬 실행 지원)"""
    topic: str
    assignments: Optional[Dict[str, str]]
    # 각 워커별 독립적인 필드
    worker1_result: Optional[str]
    worker2_result: Optional[str]
    worker3_result: Optional[str]
    final_report: Optional[str]
    messages: Annotated[List[BaseMessage], operator.add]


class SwarmAgentState(TypedDict):
    """Swarm 패턴 전용 상태 (Market-based 경쟁 지원)"""
    task: str
    # 선택 기준 (목적 함수)
    selection_criteria: Optional[Dict[str, Any]]
    # 각 에이전트별 독립적인 필드 (입찰 정보 포함)
    agent1_proposal: Optional[Dict[str, Any]]
    agent2_proposal: Optional[Dict[str, Any]]
    agent3_proposal: Optional[Dict[str, Any]]
    agent4_proposal: Optional[Dict[str, Any]]
    agent5_proposal: Optional[Dict[str, Any]]
    winner: Optional[Dict[str, Any]]
    selection_reasoning: Optional[str]  # 선택 근거
    messages: Annotated[List[BaseMessage], operator.add]


class PlannerState(TypedDict):
    """Planner-Executor 상태"""
    task: str
    plan: List[Dict[str, str]]
    current_step: int
    executions: List[Dict[str, Any]]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    final_result: str


class DebateState(TypedDict):
    """Debate 상태"""
    topic: str
    proposal: str
    critique: str
    round_num: int
    max_rounds: int
    conversation: List[Dict[str, str]]
    messages: Annotated[Sequence[BaseMessage], operator.add]
    final_decision: str


class SwarmState(TypedDict):
    """Swarm 상태"""
    task: str
    agent_proposals: List[Dict[str, Any]]
    num_agents: int
    messages: Annotated[Sequence[BaseMessage], operator.add]
    winner: Dict[str, Any]


class ReflectionState(TypedDict):
    """Reflection / Self-Refinement 패턴 상태"""
    task: str
    current_draft: Optional[str]
    reflection: Optional[str]
    revision_history: List[Dict[str, str]]
    iteration: int
    max_iterations: int
    quality_score: Optional[float]
    previous_score: Optional[float]  # 이전 점수 (개선폭 계산용)
    quality_threshold: float
    improvement_threshold: float  # 최소 개선폭 (이하면 종료)
    termination_reason: Optional[str]  # 종료 사유
    messages: Annotated[List[BaseMessage], operator.add]
    final_output: Optional[str]


class RoutingState(TypedDict):
    """Routing / Orchestration 패턴 상태 (다중 후보 경쟁 지원)"""
    user_request: str
    routing_decision: Optional[Dict[str, Any]]
    selected_agent: Optional[str]
    confidence_score: Optional[float]
    # ✨ 다중 후보 경쟁을 위한 필드
    candidate_scores: Optional[Dict[str, Dict[str, Any]]]  # 모든 후보의 점수 및 평가
    elimination_reasons: Optional[Dict[str, str]]  # 탈락 사유
    agent_result: Optional[str]
    routing_log: List[Dict[str, Any]]
    messages: Annotated[List[BaseMessage], operator.add]
    final_response: Optional[str]


class HITLState(TypedDict):
    """Human-in-the-Loop 패턴 상태 (실제 사람 개입 지원)"""
    task: str
    agent_proposal: Optional[str]
    # ✨ 워크플로우 상태 (실제 사람 개입 위한 상태 확장)
    workflow_status: str  # "processing", "awaiting_human_input", "approved", "rejected", "revision_requested", "completed"
    awaiting_input: bool  # 사람 입력 대기 중 여부
    human_feedback: Optional[str]
    human_decision: Optional[str]  # "approve", "reject", "revision"
    revision_count: int
    max_revisions: int
    revision_history: List[Dict[str, Any]]  # 수정 이력
    messages: Annotated[List[BaseMessage], operator.add]
    final_output: Optional[str]


# ============================================================================
# 1️⃣ Sequential (파이프라인) Pattern
# ============================================================================

class SequentialPattern:
    """
    Sequential Pattern: A → B → C → D
    각 에이전트가 순차적으로 실행되며, 이전 결과를 다음 에이전트가 받아 처리
    """
    
    def __init__(self, llm_service, stream_callback: Optional[StreamCallback] = None):
        self.llm_service = llm_service
        self.stream_callback = stream_callback
    
    def set_stream_callback(self, callback: Optional[StreamCallback]):
        """스트리밍 콜백 설정"""
        self.stream_callback = callback
    
    async def _generate_with_streaming(self, prompt: str, node_name: str) -> str:
        """스트리밍 콜백이 있으면 토큰 단위로 스트리밍, 없으면 일반 생성"""
        if self.stream_callback:
            # 토큰 스트리밍 모드
            content = ""
            async for token in self.llm_service.generate_response_stream(prompt):
                content += token
                await self.stream_callback(node_name, token)
            return content
        else:
            # 일반 모드
            return await self.llm_service.generate_response(prompt)
    
    async def agent_summarizer(self, state: AgentState) -> AgentState:
        """Agent 1: 요약 전문가"""
        logger.info("🤖 [Agent 1: 요약 전문가] 시작")
        
        summary = await self._generate_with_streaming(
            f"""[역할: 요약 전문가]
            다음 주제를 간단명료하게 요약해주세요:
            
            주제: {state['topic']}
            
            요약은 3-5문장으로 핵심만 추출하세요.""",
            "summarizer"
        )
        
        state['results']['summary'] = summary
        state['messages'].append(AIMessage(content=f"[요약 전문가]\n{summary}"))
        state['current_step'] = 'analyzer'
        
        logger.info(f"✅ [Agent 1] 완료: {len(summary)} characters")
        return state
    
    async def agent_analyzer(self, state: AgentState) -> AgentState:
        """Agent 2: 분석 전문가"""
        logger.info("🤖 [Agent 2: 분석 전문가] 시작")
        
        summary = state['results'].get('summary', '')
        
        analysis = await self._generate_with_streaming(
            f"""[역할: 분석 전문가]
            다음 요약을 바탕으로 심층 분석을 수행해주세요:
            
            요약:
            {summary}
            
            분석 항목:
            1. 핵심 개념
            2. 장단점
            3. 실무 적용 가능성""",
            "analyzer"
        )
        
        state['results']['analysis'] = analysis
        state['messages'].append(AIMessage(content=f"[분석 전문가]\n{analysis}"))
        state['current_step'] = 'validator'
        
        logger.info(f"✅ [Agent 2] 완료: {len(analysis)} characters")
        return state
    
    async def agent_validator(self, state: AgentState) -> AgentState:
        """Agent 3: 검증 전문가"""
        logger.info("🤖 [Agent 3: 검증 전문가] 시작")
        
        analysis = state['results'].get('analysis', '')
        
        validation = await self._generate_with_streaming(
            f"""[역할: 검증 전문가]
            다음 분석 내용을 검증하고 최종 의견을 제시해주세요:
            
            분석 내용:
            {analysis}
            
            검증 항목:
            1. 논리적 일관성
            2. 누락된 중요 사항
            3. 최종 추천 사항""",
            "validator"
        )
        
        state['results']['validation'] = validation
        state['messages'].append(AIMessage(content=f"[검증 전문가]\n{validation}"))
        state['current_step'] = 'end'
        
        logger.info(f"✅ [Agent 3] 완료: {len(validation)} characters")
        return state
    
    def create_graph(self) -> StateGraph:
        """Sequential 워크플로우 그래프 생성"""
        workflow = StateGraph(AgentState)
        
        # 노드 추가
        workflow.add_node("summarizer", self.agent_summarizer)
        workflow.add_node("analyzer", self.agent_analyzer)
        workflow.add_node("validator", self.agent_validator)
        
        # 엣지 추가 (순차 실행)
        workflow.set_entry_point("summarizer")
        workflow.add_edge("summarizer", "analyzer")
        workflow.add_edge("analyzer", "validator")
        workflow.add_edge("validator", END)
        
        return workflow.compile()


# ============================================================================
# 2️⃣ Planner-Executor Pattern
# ============================================================================

class PlannerExecutorPattern:
    """
    Planner-Executor Pattern: 계획 수립 → 단계별 실행
    Planner가 작업을 여러 단계로 나누고, Executor가 각 단계를 실행
    """
    
    def __init__(self, llm_service, stream_callback: Optional[StreamCallback] = None):
        self.llm_service = llm_service
        self.stream_callback = stream_callback
    
    def set_stream_callback(self, callback: Optional[StreamCallback]):
        """스트리밍 콜백 설정"""
        self.stream_callback = callback
    
    async def _generate_with_streaming(self, prompt: str, node_name: str) -> str:
        """스트리밍 콜백이 있으면 토큰 단위로 스트리밍, 없으면 일반 생성"""
        if self.stream_callback:
            content = ""
            async for token in self.llm_service.generate_response_stream(prompt):
                content += token
                await self.stream_callback(node_name, token)
            return content
        else:
            return await self.llm_service.generate_response(prompt)
    
    async def planner_node(self, state: PlannerState) -> PlannerState:
        """Planner: 작업을 단계별로 분해"""
        logger.info("📋 [Planner] 작업 분석 및 계획 수립 중...")
        
        plan_response = await self._generate_with_streaming(
            f"""[역할: Planner]
            다음 작업을 3-5개의 구체적인 단계로 나누어 계획을 수립해주세요.
            
            작업: {state['task']}
            
            출력 형식 (JSON):
            {{
              "steps": [
                {{"step": 1, "action": "구체적인 작업 내용"}},
                {{"step": 2, "action": "구체적인 작업 내용"}},
                {{"step": 3, "action": "구체적인 작업 내용"}}
              ]
            }}
            
            JSON 형식으로만 응답하세요.""",
            "planner"
        )
        
        # JSON 파싱
        try:
            # JSON 추출 (```json ... ``` 제거)
            if '```json' in plan_response:
                plan_response = plan_response.split('```json')[1].split('```')[0].strip()
            elif '```' in plan_response:
                plan_response = plan_response.split('```')[1].split('```')[0].strip()
            
            plan_data = json.loads(plan_response)
            state['plan'] = plan_data.get('steps', [])
            state['current_step'] = 0
            
            logger.info(f"✅ [Planner] 계획 수립 완료: {len(state['plan'])}개 단계")
            state['messages'].append(AIMessage(content=f"[Planner]\n계획 수립 완료: {len(state['plan'])}개 단계"))
            
        except Exception as e:
            logger.error(f"❌ [Planner] JSON 파싱 실패: {e}")
            # 폴백: 기본 계획
            state['plan'] = [
                {"step": 1, "action": "작업 분석"},
                {"step": 2, "action": "실행"},
                {"step": 3, "action": "검증"}
            ]
            state['messages'].append(AIMessage(content=f"[Planner]\n기본 계획 사용"))
        
        return state
    
    async def executor_node(self, state: PlannerState) -> PlannerState:
        """Executor: 현재 단계 실행"""
        current_idx = state['current_step']
        
        if current_idx >= len(state['plan']):
            state['current_step'] = -1  # 완료 표시
            return state
        
        step = state['plan'][current_idx]
        logger.info(f"⚙️ [Executor] Step {step['step']} 실행 중...")
        
        execution_result = await self._generate_with_streaming(
            f"""[역할: Executor]
            다음 작업을 수행해주세요:
            
            작업: {step['action']}
            전체 컨텍스트: {state['task']}
            
            구체적이고 실행 가능한 결과를 제시하세요.""",
            "executor"
        )
        
        state['executions'].append({
            "step": step['step'],
            "action": step['action'],
            "result": execution_result
        })
        
        state['messages'].append(AIMessage(content=f"[Executor - Step {step['step']}]\n{execution_result}"))
        state['current_step'] += 1
        
        logger.info(f"✅ [Executor] Step {step['step']} 완료")
        return state
    
    def should_continue(self, state: PlannerState) -> str:
        """다음 단계 결정"""
        if state['current_step'] < 0 or state['current_step'] >= len(state['plan']):
            return "summarize"
        return "execute"
    
    async def summarizer_node(self, state: PlannerState) -> PlannerState:
        """최종 요약"""
        logger.info("📊 [Summarizer] 최종 요약 중...")
        
        all_executions = "\n\n".join([
            f"Step {ex['step']}: {ex['action']}\n결과: {ex['result']}"
            for ex in state['executions']
        ])
        
        final_summary = await self._generate_with_streaming(
            f"""[역할: Summarizer]
            다음 모든 실행 결과를 종합하여 최종 보고서를 작성해주세요:
            
            원래 작업: {state['task']}
            
            실행 결과:
            {all_executions}
            
            최종 보고서에는 다음을 포함하세요:
            1. 주요 달성 사항
            2. 핵심 인사이트
            3. 다음 단계 제안""",
            "summarizer_node"
        )
        
        state['final_result'] = final_summary
        state['messages'].append(AIMessage(content=f"[최종 보고서]\n{final_summary}"))
        
        logger.info("✅ [Summarizer] 완료")
        return state
    
    def create_graph(self) -> StateGraph:
        """Planner-Executor 워크플로우 그래프 생성"""
        workflow = StateGraph(PlannerState)
        
        # 노드 추가
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("executor", self.executor_node)
        workflow.add_node("summarizer", self.summarizer_node)
        
        # 엣지 추가
        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "executor")
        
        # 조건부 엣지: executor 후 계속 실행할지 결정
        workflow.add_conditional_edges(
            "executor",
            self.should_continue,
            {
                "execute": "executor",  # 다음 단계 실행
                "summarize": "summarizer"  # 완료 후 요약
            }
        )
        
        workflow.add_edge("summarizer", END)
        
        return workflow.compile()


# ============================================================================
# 3️⃣ Role-based Collaboration Pattern
# ============================================================================

class RoleBasedPattern:
    """
    Role-based Collaboration Pattern: 여러 역할의 에이전트가 **병렬**로 의견 제시
    PM, Developer, Designer, QA가 각자 관점에서 **동시에** 의견 → Leader가 통합
    
    ✨ 핵심: 병렬 실행으로 시간 절약, 독립적인 관점 보장
    """
    
    def __init__(self, llm_service, stream_callback: Optional[StreamCallback] = None):
        self.llm_service = llm_service
        self.stream_callback = stream_callback
        self.roles = ["PM", "Developer", "Designer", "QA"]
    
    def set_stream_callback(self, callback: Optional[StreamCallback]):
        """스트리밍 콜백 설정"""
        self.stream_callback = callback
    
    async def _generate_with_streaming(self, prompt: str, node_name: str) -> str:
        """스트리밍 콜백이 있으면 토큰 단위로 스트리밍, 없으면 일반 생성"""
        if self.stream_callback:
            content = ""
            async for token in self.llm_service.generate_response_stream(prompt):
                content += token
                await self.stream_callback(node_name, token)
            return content
        else:
            return await self.llm_service.generate_response(prompt)
    
    async def agent_pm(self, state: RoleBasedState) -> dict:
        """PM Agent - 독립적인 필드 업데이트"""
        logger.info("👔 [PM] 의견 제시 중...")
        opinion = await self._generate_with_streaming(
            f"""[역할: Product Manager]
            다음 주제에 대해 PM 관점에서 의견을 제시해주세요:
            
            주제: {state['topic']}
            
            다음 사항을 포함하세요:
            1. 비즈니스 가치
            2. 우선순위
            3. 리스크""",
            "pm"
        )
        logger.info("✅ [PM] 완료")
        return {
            'pm_opinion': opinion,
            'messages': [AIMessage(content=f"[PM]\n{opinion}")]
        }
    
    async def agent_developer(self, state: RoleBasedState) -> dict:
        """Developer Agent - 독립적인 필드 업데이트"""
        logger.info("💻 [Developer] 의견 제시 중...")
        opinion = await self._generate_with_streaming(
            f"""[역할: Developer]
            다음 주제에 대해 개발자 관점에서 의견을 제시해주세요:
            
            주제: {state['topic']}
            
            다음 사항을 포함하세요:
            1. 기술적 타당성
            2. 구현 복잡도
            3. 유지보수성""",
            "developer"
        )
        logger.info("✅ [Developer] 완료")
        return {
            'dev_opinion': opinion,
            'messages': [AIMessage(content=f"[Developer]\n{opinion}")]
        }
    
    async def agent_designer(self, state: RoleBasedState) -> dict:
        """Designer Agent - 독립적인 필드 업데이트"""
        logger.info("🎨 [Designer] 의견 제시 중...")
        opinion = await self._generate_with_streaming(
            f"""[역할: UX Designer]
            다음 주제에 대해 디자이너 관점에서 의견을 제시해주세요:
            
            주제: {state['topic']}
            
            다음 사항을 포함하세요:
            1. 사용자 경험
            2. 접근성
            3. 디자인 일관성""",
            "designer"
        )
        logger.info("✅ [Designer] 완료")
        return {
            'design_opinion': opinion,
            'messages': [AIMessage(content=f"[Designer]\n{opinion}")]
        }
    
    async def agent_qa(self, state: RoleBasedState) -> dict:
        """QA Agent - 독립적인 필드 업데이트"""
        logger.info("🔍 [QA] 의견 제시 중...")
        opinion = await self._generate_with_streaming(
            f"""[역할: QA Engineer]
            다음 주제에 대해 QA 관점에서 의견을 제시해주세요:
            
            주제: {state['topic']}
            
            다음 사항을 포함하세요:
            1. 테스트 가능성
            2. 품질 리스크
            3. 검증 전략""",
            "qa"
        )
        logger.info("✅ [QA] 완료")
        return {
            'qa_opinion': opinion,
            'messages': [AIMessage(content=f"[QA]\n{opinion}")]
        }
    
    async def agent_leader(self, state: RoleBasedState) -> dict:
        """Leader Agent: 모든 의견 통합"""
        logger.info("👨‍💼 [Team Leader] 의견 통합 중...")
        
        all_opinions = "\n\n".join([
            f"[PM의 의견]\n{state.get('pm_opinion', '(의견 없음)')}",
            f"[Developer의 의견]\n{state.get('dev_opinion', '(의견 없음)')}",
            f"[Designer의 의견]\n{state.get('design_opinion', '(의견 없음)')}",
            f"[QA의 의견]\n{state.get('qa_opinion', '(의견 없음)')}"
        ])
        
        final_decision = await self._generate_with_streaming(
            f"""[역할: Team Leader]
            팀원들의 의견을 종합하여 최종 결정을 내려주세요:
            
            주제: {state['topic']}
            
            {all_opinions}
            
            최종 결정에는 다음을 포함하세요:
            1. 핵심 합의 사항
            2. 트레이드오프 분석
            3. 실행 계획""",
            "leader"
        )
        
        logger.info("✅ [Team Leader] 완료")
        return {
            'final_decision': final_decision,
            'messages': [AIMessage(content=f"[Team Leader - 최종 결정]\n{final_decision}")]
        }
    
    def create_graph(self) -> StateGraph:
        """Role-based 워크플로우 그래프 생성 - 병렬 실행!"""
        workflow = StateGraph(RoleBasedState)
        
        # 각 역할 노드 추가
        workflow.add_node("pm", self.agent_pm)
        workflow.add_node("developer", self.agent_developer)
        workflow.add_node("designer", self.agent_designer)
        workflow.add_node("qa", self.agent_qa)
        workflow.add_node("leader", self.agent_leader)
        
        # 병렬 실행: 4개 역할이 동시에 시작
        workflow.set_entry_point("pm")
        workflow.set_entry_point("developer")
        workflow.set_entry_point("designer")
        workflow.set_entry_point("qa")
        
        # 모든 역할 → Leader (모두 완료되면 Leader 실행)
        workflow.add_edge("pm", "leader")
        workflow.add_edge("developer", "leader")
        workflow.add_edge("designer", "leader")
        workflow.add_edge("qa", "leader")
        
        workflow.add_edge("leader", END)
        
        logger.info("✅ [Role-based] 그래프 생성 완료 - 4개 노드 병렬 실행")
        return workflow.compile()


# ============================================================================
# 4️⃣ Hierarchical Pattern
# ============================================================================

class HierarchicalPattern:
    """
    Hierarchical Pattern: Manager → Workers (병렬) → Manager
    Manager가 작업을 분배하고, Worker들이 **병렬** 실행, Manager가 결과 통합
    
    ✨ 핵심: 상하 구조, 병렬 작업 분산
    """
    
    def __init__(self, llm_service, stream_callback: Optional[StreamCallback] = None):
        self.llm_service = llm_service
        self.stream_callback = stream_callback
        self.num_workers = 3
    
    def set_stream_callback(self, callback: Optional[StreamCallback]):
        """스트리밍 콜백 설정"""
        self.stream_callback = callback
    
    async def _generate_with_streaming(self, prompt: str, node_name: str) -> str:
        """스트리밍 콜백이 있으면 토큰 단위로 스트리밍, 없으면 일반 생성"""
        if self.stream_callback:
            content = ""
            async for token in self.llm_service.generate_response_stream(prompt):
                content += token
                await self.stream_callback(node_name, token)
            return content
        else:
            return await self.llm_service.generate_response(prompt)
    
    async def manager_delegate(self, state: HierarchicalState) -> dict:
        """Manager: 작업 분배"""
        logger.info("👨‍💼 [Manager] 작업 분배 중...")
        
        delegation_response = await self._generate_with_streaming(
            f"""[역할: Manager]
            다음 작업을 {self.num_workers}명의 워커에게 분배해주세요.
            각 워커에게 구체적이고 독립적인 작업을 할당하세요.
            
            작업: {state['topic']}
            
            출력 형식 (JSON):
            {{
              "worker1": "구체적인 작업 설명",
              "worker2": "구체적인 작업 설명",
              "worker3": "구체적인 작업 설명"
            }}
            
            JSON 형식으로만 응답하세요.""",
            "manager_delegate"
        )
        
        # JSON 파싱
        try:
            if '```json' in delegation_response:
                delegation_response = delegation_response.split('```json')[1].split('```')[0].strip()
            elif '```' in delegation_response:
                delegation_response = delegation_response.split('```')[1].split('```')[0].strip()
            
            assignments = json.loads(delegation_response)
            logger.info(f"✅ [Manager] 작업 분배 완료: {len(assignments)}개 작업")
            
        except Exception as e:
            logger.error(f"❌ [Manager] JSON 파싱 실패: {e}")
            # 폴백
            assignments = {
                f"worker{i+1}": f"{state['topic']}의 일부 작업 {i+1}"
                for i in range(self.num_workers)
            }
        
        return {
            'assignments': assignments,
            'messages': [AIMessage(content=f"[Manager - 작업 분배]\n{json.dumps(assignments, ensure_ascii=False, indent=2)}")]
        }
    
    async def worker1_node(self, state: HierarchicalState) -> dict:
        """Worker 1 - 독립적인 필드 업데이트"""
        assignment = state.get('assignments', {}).get('worker1', "작업 없음")
        logger.info(f"👷 [Worker 1] 작업 수행 중...")
        
        result = await self._generate_with_streaming(
            f"""[역할: Worker 1]
            다음 작업을 수행해주세요:
            
            작업: {assignment}
            
            구체적이고 실행 가능한 결과를 제시하세요.""",
            "worker1"
        )
        
        logger.info(f"✅ [Worker 1] 완료")
        return {
            'worker1_result': result,
            'messages': [AIMessage(content=f"[Worker 1]\n{result}")]
        }
    
    async def worker2_node(self, state: HierarchicalState) -> dict:
        """Worker 2 - 독립적인 필드 업데이트"""
        assignment = state.get('assignments', {}).get('worker2', "작업 없음")
        logger.info(f"👷 [Worker 2] 작업 수행 중...")
        
        result = await self._generate_with_streaming(
            f"""[역할: Worker 2]
            다음 작업을 수행해주세요:
            
            작업: {assignment}
            
            구체적이고 실행 가능한 결과를 제시하세요.""",
            "worker2"
        )
        
        logger.info(f"✅ [Worker 2] 완료")
        return {
            'worker2_result': result,
            'messages': [AIMessage(content=f"[Worker 2]\n{result}")]
        }
    
    async def worker3_node(self, state: HierarchicalState) -> dict:
        """Worker 3 - 독립적인 필드 업데이트"""
        assignment = state.get('assignments', {}).get('worker3', "작업 없음")
        logger.info(f"👷 [Worker 3] 작업 수행 중...")
        
        result = await self._generate_with_streaming(
            f"""[역할: Worker 3]
            다음 작업을 수행해주세요:
            
            작업: {assignment}
            
            구체적이고 실행 가능한 결과를 제시하세요.""",
            "worker3"
        )
        
        logger.info(f"✅ [Worker 3] 완료")
        return {
            'worker3_result': result,
            'messages': [AIMessage(content=f"[Worker 3]\n{result}")]
        }
    
    async def manager_integrate(self, state: HierarchicalState) -> dict:
        """Manager: 결과 통합"""
        logger.info("👨‍💼 [Manager] 결과 통합 중...")
        
        assignments = state.get('assignments', {})
        all_results = "\n\n".join([
            f"Worker 1:\n작업: {assignments.get('worker1', '')}\n결과: {state.get('worker1_result', '')}",
            f"Worker 2:\n작업: {assignments.get('worker2', '')}\n결과: {state.get('worker2_result', '')}",
            f"Worker 3:\n작업: {assignments.get('worker3', '')}\n결과: {state.get('worker3_result', '')}"
        ])
        
        final_report = await self._generate_with_streaming(
            f"""[역할: Manager]
            워커들의 결과를 통합하여 최종 보고서를 작성해주세요:
            
            원래 작업: {state['topic']}
            
            워커 결과:
            {all_results}
            
            최종 보고서에는 다음을 포함하세요:
            1. 전체 요약
            2. 주요 성과
            3. 개선 제안""",
            "manager_integrate"
        )
        
        logger.info("✅ [Manager] 결과 통합 완료")
        return {
            'final_report': final_report,
            'messages': [AIMessage(content=f"[Manager - 최종 보고서]\n{final_report}")]
        }
    
    def create_graph(self) -> StateGraph:
        """Hierarchical 워크플로우 그래프 생성 - 병렬 실행!"""
        workflow = StateGraph(HierarchicalState)
        
        # 노드 추가
        workflow.add_node("manager_delegate", self.manager_delegate)
        workflow.add_node("worker1", self.worker1_node)
        workflow.add_node("worker2", self.worker2_node)
        workflow.add_node("worker3", self.worker3_node)
        workflow.add_node("manager_integrate", self.manager_integrate)
        
        # Manager → Workers (병렬)
        workflow.set_entry_point("manager_delegate")
        workflow.add_edge("manager_delegate", "worker1")
        workflow.add_edge("manager_delegate", "worker2")
        workflow.add_edge("manager_delegate", "worker3")
        
        # Workers → Manager (모두 완료되면 Manager 실행)
        workflow.add_edge("worker1", "manager_integrate")
        workflow.add_edge("worker2", "manager_integrate")
        workflow.add_edge("worker3", "manager_integrate")
        
        workflow.add_edge("manager_integrate", END)
        
        logger.info("✅ [Hierarchical] 그래프 생성 완료 - 3개 워커 병렬 실행")
        return workflow.compile()


# ============================================================================
# 5️⃣ Debate / Critic Pattern
# ============================================================================

class DebatePattern:
    """
    Debate Pattern: Proposer ↔ Critic 반복 → Judge
    제안자와 비평가가 여러 라운드 토론 후 심판이 최종 결정
    """
    
    def __init__(self, llm_service, max_rounds=3, stream_callback: Optional[StreamCallback] = None):
        self.llm_service = llm_service
        self.max_rounds = max_rounds
        self.stream_callback = stream_callback
    
    def set_stream_callback(self, callback: Optional[StreamCallback]):
        """스트리밍 콜백 설정"""
        self.stream_callback = callback
    
    async def _generate_with_streaming(self, prompt: str, node_name: str) -> str:
        """스트리밍 콜백이 있으면 토큰 단위로 스트리밍, 없으면 일반 생성"""
        if self.stream_callback:
            content = ""
            async for token in self.llm_service.generate_response_stream(prompt):
                content += token
                await self.stream_callback(node_name, token)
            return content
        else:
            return await self.llm_service.generate_response(prompt)
    
    async def proposer_initial(self, state: DebateState) -> DebateState:
        """Proposer: 초기 제안"""
        logger.info("💡 [Proposer] 초기 제안 작성 중...")
        
        proposal = await self._generate_with_streaming(
            f"""[역할: Proposer]
            다음 주제에 대한 제안을 작성해주세요:
            
            주제: {state['topic']}
            
            제안서에 포함할 내용:
            1. 핵심 아이디어
            2. 기대 효과
            3. 실행 방안""",
            "proposer_initial"
        )
        
        state['proposal'] = proposal
        state['conversation'].append({"role": "proposer", "content": proposal})
        state['messages'].append(AIMessage(content=f"[Proposer - 초기 제안]\n{proposal}"))
        state['round_num'] = 1
        
        logger.info("✅ [Proposer] 초기 제안 완료")
        return state
    
    async def critic_node(self, state: DebateState) -> DebateState:
        """Critic: 비판"""
        logger.info(f"🔍 [Critic] Round {state['round_num']} 비판 중...")
        
        critique = await self._generate_with_streaming(
            f"""[역할: Critic]
            다음 제안의 문제점을 날카롭게 지적해주세요:
            
            제안:
            {state['proposal']}
            
            비판 항목:
            1. 논리적 오류
            2. 실현 가능성 문제
            3. 누락된 중요 사항""",
            "critic"
        )
        
        state['critique'] = critique
        state['conversation'].append({"role": "critic", "content": critique})
        state['messages'].append(AIMessage(content=f"[Critic - Round {state['round_num']}]\n{critique}"))
        
        logger.info(f"✅ [Critic] Round {state['round_num']} 완료")
        return state
    
    async def proposer_refine(self, state: DebateState) -> DebateState:
        """Proposer: 제안 개선"""
        logger.info(f"💡 [Proposer] Round {state['round_num']} 제안 개선 중...")
        
        refined_proposal = await self._generate_with_streaming(
            f"""[역할: Proposer]
            비판을 반영하여 제안을 개선해주세요:
            
            원래 제안:
            {state['proposal']}
            
            받은 비판:
            {state['critique']}
            
            개선된 제안을 작성하세요.""",
            "proposer_refine"
        )
        
        state['proposal'] = refined_proposal
        state['conversation'].append({"role": "proposer_refined", "content": refined_proposal})
        state['messages'].append(AIMessage(content=f"[Proposer - 개선안 Round {state['round_num']}]\n{refined_proposal}"))
        state['round_num'] += 1
        
        logger.info(f"✅ [Proposer] Round {state['round_num']-1} 개선 완료")
        return state
    
    def should_continue_debate(self, state: DebateState) -> str:
        """토론 계속 여부 결정"""
        if state['round_num'] > state['max_rounds']:
            return "judge"
        return "critic"
    
    async def judge_node(self, state: DebateState) -> DebateState:
        """Judge: 최종 판결"""
        logger.info("⚖️ [Judge] 최종 판결 중...")
        
        conversation_text = "\n\n".join([
            f"[{conv['role']}]\n{conv['content']}"
            for conv in state['conversation']
        ])
        
        final_decision = await self._generate_with_streaming(
            f"""[역할: Judge]
            다음 토론 내용을 바탕으로 최종 판결을 내려주세요:
            
            주제: {state['topic']}
            
            토론 내용:
            {conversation_text}
            
            최종 판결에 포함할 내용:
            1. 토론의 주요 쟁점
            2. 각 측의 강점과 약점
            3. 최종 결정 및 근거""",
            "judge"
        )
        
        state['final_decision'] = final_decision
        state['messages'].append(AIMessage(content=f"[Judge - 최종 판결]\n{final_decision}"))
        
        logger.info("✅ [Judge] 최종 판결 완료")
        return state
    
    def create_graph(self) -> StateGraph:
        """Debate 워크플로우 그래프 생성"""
        workflow = StateGraph(DebateState)
        
        # 노드 추가
        workflow.add_node("proposer_initial", self.proposer_initial)
        workflow.add_node("critic", self.critic_node)
        workflow.add_node("proposer_refine", self.proposer_refine)
        workflow.add_node("judge", self.judge_node)
        
        # 엣지 추가
        workflow.set_entry_point("proposer_initial")
        workflow.add_edge("proposer_initial", "critic")
        workflow.add_edge("critic", "proposer_refine")
        
        # 조건부 엣지: 계속 토론할지 판결할지
        workflow.add_conditional_edges(
            "proposer_refine",
            self.should_continue_debate,
            {
                "critic": "critic",  # 다음 라운드
                "judge": "judge"     # 종료
            }
        )
        
        workflow.add_edge("judge", END)
        
        return workflow.compile()


# ============================================================================
# 6️⃣ Swarm / Market-based Pattern (진정한 시장 경쟁 구현)
# ============================================================================

class SwarmPattern:
    """
    Swarm / Market-based Pattern: 이질적 전략을 가진 에이전트들의 경쟁 입찰
    
    ✨ 핵심 3요소:
    1. 이질적 전략: 각 에이전트가 완전히 다른 관점/전략으로 접근
    2. 경쟁 입찰: 자기평가가 아닌 비용/시간/위험도 등의 입찰 정보 제출
    3. 시장 메커니즘: 목적 함수 기반의 합리적 선택 (조건 기반 자동 선택)
    
    🏆 선택 기준 예시:
    - "최소 비용": cost 최소화
    - "최단 시간": duration 최소화
    - "균형": (cost * 0.3) + (duration * 0.3) + (risk * 0.2) + (1/performance * 0.2)
    """
    
    # 5개 에이전트의 고유 전략 정의
    AGENT_STRATEGIES = {
        1: {
            "name": "비용 최적화 전문가",
            "strategy": "cost_optimizer",
            "focus": "최소 비용으로 문제 해결",
            "approach": "오픈소스, 자체 구축, 비용 효율적인 솔루션 선호"
        },
        2: {
            "name": "성능 최우선 전문가",
            "strategy": "performance_first",
            "focus": "최고 성능과 확장성 확보",
            "approach": "프리미엄 솔루션, 엔터프라이즈급 도구, 성능 최적화"
        },
        3: {
            "name": "보안 최우선 전문가",
            "strategy": "security_first",
            "focus": "보안과 컴플라이언스 최우선",
            "approach": "보안 인증 솔루션, 감사 추적, 암호화, 접근 제어"
        },
        4: {
            "name": "속도 최우선 전문가",
            "strategy": "speed_first",
            "focus": "최단 시간 내 구현 완료",
            "approach": "SaaS 솔루션, 턴키 서비스, 빠른 배포, 관리형 서비스"
        },
        5: {
            "name": "자동화 최우선 전문가",
            "strategy": "automation_first",
            "focus": "운영 자동화와 장기 유지보수 최소화",
            "approach": "IaC, GitOps, 자동 스케일링, 셀프힐링 시스템"
        }
    }
    
    # 기본 선택 기준 (목적 함수)
    DEFAULT_CRITERIA = {
        "priority": "balanced",  # balanced, cost, speed, performance, security
        "max_cost": 100,  # 최대 허용 비용 (단위: 만원/월)
        "max_duration_weeks": 4,  # 최대 허용 구축 기간 (주)
        "min_performance": 7,  # 최소 성능 점수 (1-10)
        "weights": {
            "cost": 0.25,
            "duration": 0.25,
            "risk": 0.25,
            "performance": 0.25
        }
    }
    
    def __init__(self, llm_service, num_agents=5, stream_callback: Optional[StreamCallback] = None):
        self.llm_service = llm_service
        self.num_agents = min(num_agents, 5)
        self.stream_callback = stream_callback
    
    def set_stream_callback(self, callback: Optional[StreamCallback]):
        """스트리밍 콜백 설정"""
        self.stream_callback = callback
    
    async def _generate_with_streaming(self, prompt: str, node_name: str) -> str:
        """스트리밍 콜백이 있으면 토큰 단위로 스트리밍, 없으면 일반 생성"""
        if self.stream_callback:
            content = ""
            async for token in self.llm_service.generate_response_stream(prompt):
                content += token
                await self.stream_callback(node_name, token)
            return content
        else:
            return await self.llm_service.generate_response(prompt)
    
    async def agent1_node(self, state: SwarmAgentState) -> dict:
        """Agent 1: 비용 최적화 전문가"""
        return await self._agent_logic(1, state)
    
    async def agent2_node(self, state: SwarmAgentState) -> dict:
        """Agent 2: 성능 최우선 전문가"""
        return await self._agent_logic(2, state)
    
    async def agent3_node(self, state: SwarmAgentState) -> dict:
        """Agent 3: 보안 최우선 전문가"""
        return await self._agent_logic(3, state)
    
    async def agent4_node(self, state: SwarmAgentState) -> dict:
        """Agent 4: 속도 최우선 전문가"""
        return await self._agent_logic(4, state)
    
    async def agent5_node(self, state: SwarmAgentState) -> dict:
        """Agent 5: 자동화 최우선 전문가"""
        return await self._agent_logic(5, state)
    
    async def _agent_logic(self, agent_id: int, state: SwarmAgentState) -> dict:
        """에이전트별 고유 전략 기반 솔루션 제안 및 입찰"""
        strategy = self.AGENT_STRATEGIES[agent_id]
        logger.info(f"🐝 [{strategy['name']}] 입찰 준비 중...")
        
        prompt = f"""[역할: {strategy['name']}]
[전략: {strategy['focus']}]
[접근 방식: {strategy['approach']}]

당신은 **{strategy['strategy']}** 전략을 가진 전문가입니다.
다음 문제에 대해 당신의 전략에 맞는 솔루션을 제안하고 입찰 정보를 제출하세요.

**문제:** {state['task']}

**중요:** 당신의 전략({strategy['focus']})에 충실한 솔루션을 제안하세요.
다른 전략과 차별화된 고유한 접근법을 사용해야 합니다.

**출력 형식 (JSON):**
{{
    "solution_name": "솔루션 이름 (예: Prometheus 기반 모니터링)",
    "solution_description": "솔루션에 대한 상세 설명 (3-5문장)",
    "key_components": ["핵심 구성요소 1", "핵심 구성요소 2", "핵심 구성요소 3"],
    "bid": {{
        "cost_monthly": 50,
        "duration_weeks": 2,
        "risk_level": 5,
        "performance_score": 8,
        "maintenance_effort": 6
    }},
    "trade_offs": {{
        "strengths": ["장점 1", "장점 2"],
        "weaknesses": ["약점 1", "약점 2"]
    }},
    "why_choose_me": "이 솔루션을 선택해야 하는 이유 (1-2문장)"
}}

**입찰 정보 가이드:**
- cost_monthly: 월 운영 비용 (만원 단위, 10~500 범위)
- duration_weeks: 구축 기간 (주 단위, 1~12 범위)
- risk_level: 위험도 (1=매우 낮음 ~ 10=매우 높음)
- performance_score: 성능 점수 (1=낮음 ~ 10=최고)
- maintenance_effort: 유지보수 노력 (1=거의 없음 ~ 10=매우 많음)
            
            JSON 형식으로만 응답하세요."""

        proposal_response = await self._generate_with_streaming(prompt, f"agent{agent_id}")
        
        # JSON 파싱
        try:
            if '```json' in proposal_response:
                proposal_response = proposal_response.split('```json')[1].split('```')[0].strip()
            elif '```' in proposal_response:
                proposal_response = proposal_response.split('```')[1].split('```')[0].strip()
            
            proposal_data = json.loads(proposal_response)
            
            # 입찰 정보 추출
            bid = proposal_data.get("bid", {})
            
            agent_proposal = {
                "agent_id": agent_id,
                "strategy": strategy,
                "solution_name": proposal_data.get("solution_name", "제안 없음"),
                "solution_description": proposal_data.get("solution_description", ""),
                "key_components": proposal_data.get("key_components", []),
                "bid": {
                    "cost_monthly": bid.get("cost_monthly", 100),
                    "duration_weeks": bid.get("duration_weeks", 4),
                    "risk_level": bid.get("risk_level", 5),
                    "performance_score": bid.get("performance_score", 5),
                    "maintenance_effort": bid.get("maintenance_effort", 5)
                },
                "trade_offs": proposal_data.get("trade_offs", {"strengths": [], "weaknesses": []}),
                "why_choose_me": proposal_data.get("why_choose_me", "")
            }
            
        except Exception as e:
            logger.error(f"❌ [{strategy['name']}] JSON 파싱 실패: {e}")
            agent_proposal = {
                "agent_id": agent_id,
                "strategy": strategy,
                "solution_name": "파싱 실패",
                "solution_description": str(e),
                "key_components": [],
                "bid": {
                    "cost_monthly": 999,
                    "duration_weeks": 99,
                    "risk_level": 10,
                    "performance_score": 1,
                    "maintenance_effort": 10
                },
                "trade_offs": {"strengths": [], "weaknesses": ["파싱 실패"]},
                "why_choose_me": ""
            }
        
        bid_info = agent_proposal["bid"]
        logger.info(f"✅ [{strategy['name']}] 입찰 완료 - 비용: {bid_info['cost_monthly']}만원/월, 기간: {bid_info['duration_weeks']}주")
        
        # 메시지 포맷팅
        message_content = f"""**{strategy['name']}의 입찰**

**솔루션: {agent_proposal['solution_name']}**
{agent_proposal['solution_description']}

**핵심 구성요소:**
{chr(10).join(['• ' + comp for comp in agent_proposal['key_components']])}

**입찰 정보:**
| 항목 | 값 |
|------|-----|
| 월 비용 | {bid_info['cost_monthly']}만원 |
| 구축 기간 | {bid_info['duration_weeks']}주 |
| 위험도 | {bid_info['risk_level']}/10 |
| 성능 점수 | {bid_info['performance_score']}/10 |
| 유지보수 노력 | {bid_info['maintenance_effort']}/10 |

**장점:** {', '.join(agent_proposal['trade_offs'].get('strengths', []))}
**약점:** {', '.join(agent_proposal['trade_offs'].get('weaknesses', []))}

**선택 이유:** {agent_proposal['why_choose_me']}"""
        
        return {
            f'agent{agent_id}_proposal': agent_proposal,
            'messages': [AIMessage(content=message_content)]
        }
    
    async def selector_node(self, state: SwarmAgentState) -> dict:
        """Selector: 목적 함수 기반 합리적 선택 (Market-based)"""
        logger.info("🏆 [Market Selector] 입찰 평가 및 선정 중...")
        
        # 선택 기준 가져오기
        criteria = state.get('selection_criteria') or self.DEFAULT_CRITERIA
        priority = criteria.get('priority', 'balanced')
        weights = criteria.get('weights', self.DEFAULT_CRITERIA['weights'])
        
        # 모든 입찰 수집
        all_proposals = []
        for i in range(1, 6):
            proposal = state.get(f'agent{i}_proposal')
            if proposal:
                all_proposals.append(proposal)
        
        if not all_proposals:
            logger.error("❌ [Selector] 입찰 없음")
            return {
                'winner': None,
                'selection_reasoning': "입찰이 없습니다.",
                'messages': [AIMessage(content="[Market Selector] 입찰이 없습니다.")]
            }
        
        # 입찰 평가 테이블 생성
        evaluation_table = []
        for p in all_proposals:
            bid = p['bid']
            strategy = p['strategy']
            
            # 정규화 점수 계산 (0-1 범위, 낮을수록 좋음으로 통일)
            # cost: 낮을수록 좋음 → 그대로
            # duration: 낮을수록 좋음 → 그대로
            # risk: 낮을수록 좋음 → 그대로
            # performance: 높을수록 좋음 → 역수 사용
            
            cost_score = bid['cost_monthly'] / 500  # 0-1 정규화 (500 기준)
            duration_score = bid['duration_weeks'] / 12  # 0-1 정규화 (12주 기준)
            risk_score = bid['risk_level'] / 10
            # 성능은 높을수록 좋으므로 역산
            perf_score = 1 - (bid['performance_score'] / 10)
            
            # 가중 합계 점수 (낮을수록 좋음)
            total_score = (
                weights['cost'] * cost_score +
                weights['duration'] * duration_score +
                weights['risk'] * risk_score +
                weights['performance'] * perf_score
            )
            
            evaluation_table.append({
                'proposal': p,
                'scores': {
                    'cost': cost_score,
                    'duration': duration_score,
                    'risk': risk_score,
                    'performance': perf_score
                },
                'total_score': total_score
            })
        
        # 우선순위에 따른 정렬
        if priority == 'cost':
            evaluation_table.sort(key=lambda x: x['scores']['cost'])
        elif priority == 'speed':
            evaluation_table.sort(key=lambda x: x['scores']['duration'])
        elif priority == 'performance':
            evaluation_table.sort(key=lambda x: x['scores']['performance'])
        elif priority == 'security':
            evaluation_table.sort(key=lambda x: x['scores']['risk'])
        else:  # balanced
            evaluation_table.sort(key=lambda x: x['total_score'])
        
        # 최적 제안 선택
        winner_eval = evaluation_table[0]
        winner = winner_eval['proposal']
        
        # 선택 근거 생성
        priority_labels = {
            'balanced': '균형 잡힌 종합 점수',
            'cost': '최소 비용',
            'speed': '최단 구축 시간',
            'performance': '최고 성능',
            'security': '최저 위험도'
        }
        
        selection_reasoning = f"""**목적 함수:** {priority_labels.get(priority, priority)}
**가중치:** 비용 {weights['cost']*100:.0f}%, 시간 {weights['duration']*100:.0f}%, 위험도 {weights['risk']*100:.0f}%, 성능 {weights['performance']*100:.0f}%

선정 근거: {winner['strategy']['name']}의 솔루션이 목적 함수 기준 최적 점수({winner_eval['total_score']:.3f})를 기록했습니다."""
        
        # 입찰 비교 테이블 메시지 생성
        comparison_rows = []
        for i, ev in enumerate(evaluation_table):
            p = ev['proposal']
            bid = p['bid']
            rank = f"#{i+1}" if i < 3 else f" {i+1}"
            selected = "[선정]" if i == 0 else ""
            comparison_rows.append(
                f"| {rank} {p['strategy']['name'][:12]} | {p['solution_name'][:15]} | {bid['cost_monthly']}만원 | {bid['duration_weeks']}주 | {bid['risk_level']}/10 | {bid['performance_score']}/10 | {ev['total_score']:.3f} | {selected} |"
            )
        
        comparison_table = "\n".join(comparison_rows)
        
        message_content = f"""**Market Selector - 입찰 평가 결과**

**선택 기준 (목적 함수)**
- 우선순위: **{priority_labels.get(priority, priority)}**
- 가중치: 비용 {weights['cost']*100:.0f}% | 시간 {weights['duration']*100:.0f}% | 위험도 {weights['risk']*100:.0f}% | 성능 {weights['performance']*100:.0f}%

**입찰 비교표** (점수가 낮을수록 유리)
| 순위 | 솔루션 | 비용 | 기간 | 위험도 | 성능 | 종합점수 | 선정 |
|------|--------|------|------|--------|------|----------|------|
{comparison_table}

---

**최종 선정: {winner['strategy']['name']}**

**솔루션:** {winner['solution_name']}
{winner['solution_description']}

**입찰 정보:**
- 월 비용: {winner['bid']['cost_monthly']}만원
- 구축 기간: {winner['bid']['duration_weeks']}주
- 위험도: {winner['bid']['risk_level']}/10
- 성능: {winner['bid']['performance_score']}/10

**선정 사유:**
{selection_reasoning}

**차점자 대안:** {evaluation_table[1]['proposal']['strategy']['name']} - {evaluation_table[1]['proposal']['solution_name']} (점수: {evaluation_table[1]['total_score']:.3f})"""
        
        logger.info(f"✅ [Selector] {winner['strategy']['name']} 선정됨 (점수: {winner_eval['total_score']:.3f})")
        
        return {
            'winner': winner,
            'selection_reasoning': selection_reasoning,
            'messages': [AIMessage(content=message_content)]
        }
    
    def create_graph(self) -> StateGraph:
        """Swarm 워크플로우 그래프 생성 - 병렬 경쟁 입찰!"""
        workflow = StateGraph(SwarmAgentState)
        
        # 에이전트 노드 추가 (각자 다른 전략)
        workflow.add_node("agent1", self.agent1_node)  # 비용 최적화
        workflow.add_node("agent2", self.agent2_node)  # 성능 최우선
        workflow.add_node("agent3", self.agent3_node)  # 보안 최우선
        workflow.add_node("agent4", self.agent4_node)  # 속도 최우선
        workflow.add_node("agent5", self.agent5_node)  # 자동화 최우선
        workflow.add_node("selector", self.selector_node)
        
        # 병렬 실행: 모든 에이전트 동시 시작 (경쟁 입찰)
        workflow.set_entry_point("agent1")
        workflow.set_entry_point("agent2")
        workflow.set_entry_point("agent3")
        workflow.set_entry_point("agent4")
        workflow.set_entry_point("agent5")
        
        # 모든 에이전트 → Market Selector
        workflow.add_edge("agent1", "selector")
        workflow.add_edge("agent2", "selector")
        workflow.add_edge("agent3", "selector")
        workflow.add_edge("agent4", "selector")
        workflow.add_edge("agent5", "selector")
        
        workflow.add_edge("selector", END)
        
        logger.info("✅ [Swarm/Market] 그래프 생성 완료 - 5개 전략 에이전트 경쟁 입찰")
        return workflow.compile()


# ============================================================================
# 7️⃣ Reflection / Self-Refinement Pattern
# ============================================================================

class ReflectionPattern:
    """
    Reflection / Self-Refinement Pattern: 생성 → 평가 → 개선의 반복 루프
    
    ✨ 핵심 3단계:
    1. Generator: 초안 생성 / 피드백 반영 수정
    2. Reflector: 품질 평가 및 개선점 도출
    3. Finalizer: 최종 결과 출력
    
    ✨ 종료 조건 (3가지):
    1. 품질 기준 충족: score >= threshold
    2. 최대 반복 횟수 도달: iteration >= max_iterations
    3. 개선 정체/퇴화 감지: improvement < min_threshold 또는 score 하락
    
    ✨ Finalizer 점수 정책:
    - 최종 점수 = 마지막 Reflector 점수 (논리적 일관성 보장)
    - 또는 전체 Reflector 점수의 평균
    """
    
    def __init__(self, llm_service, max_iterations=3, quality_threshold=8.0, improvement_threshold=0.3, stream_callback: Optional[StreamCallback] = None):
        self.llm_service = llm_service
        self.max_iterations = max_iterations
        self.quality_threshold = quality_threshold
        self.improvement_threshold = improvement_threshold  # 최소 개선폭
        self.stream_callback = stream_callback
    
    def set_stream_callback(self, callback: Optional[StreamCallback]):
        """스트리밍 콜백 설정"""
        self.stream_callback = callback
    
    async def _generate_with_streaming(self, prompt: str, node_name: str) -> str:
        """스트리밍 콜백이 있으면 토큰 단위로 스트리밍, 없으면 일반 생성"""
        if self.stream_callback:
            content = ""
            async for token in self.llm_service.generate_response_stream(prompt):
                content += token
                await self.stream_callback(node_name, token)
            return content
        else:
            return await self.llm_service.generate_response(prompt)
    
    async def generator_node(self, state: ReflectionState) -> dict:
        """Generator: 초안 생성 또는 피드백 반영 수정"""
        iteration = state.get('iteration', 0)
        
        if iteration == 0:
            logger.info("📝 [Generator] 초안 생성 중...")
            # ✨ 의도적으로 "개선 여지가 있는" 기본 초안 생성
            # → 명확한 개선 곡선을 보여주기 위함
            prompt = f"""다음 작업에 대한 **기본 초안**을 작성해주세요.

작업: {state['task']}

**초안 작성 가이드라인 (의도적으로 개선 여지를 남김):**
- 핵심 개념과 구조를 잡는 데 집중
- 구체적인 도구명이나 수치는 아직 포함하지 않음
- 추상적인 수준의 설명으로 작성
- 실제 사례나 정량 데이터는 다음 개선에서 추가 예정

이 초안은 이후 평가와 개선 과정을 통해 점진적으로 발전합니다.
먼저 기본 뼈대를 잡는 초안을 작성해주세요."""
        else:
            logger.info(f"📝 [Generator] 피드백 반영 수정 중 (Iteration {iteration + 1})...")
            # ✨ 개선 시 구체적인 요소 추가 유도
            prompt = f"""이전 초안과 피드백을 바탕으로 **대폭 개선된 버전**을 작성해주세요.

원래 작업: {state['task']}

이전 초안:
{state.get('current_draft', '')}

Reflector 피드백:
{state.get('reflection', '')}

**🎯 반드시 다음 개선 요소를 추가하세요 (점수 상승을 위한 핵심):**

1. **정량 데이터 추가**: 구체적인 수치, 통계, KPI 포함
   예: "장애 복구 시간 42% 감소", "비용 28% 절감"

2. **구체적 도구/기술명**: 실제 도구나 기술 이름 명시
   예: "Prometheus + Grafana", "Datadog APM", "ELK Stack"

3. **실제 사례**: 기업명을 포함한 구체적 사례
   예: "Netflix는 Chaos Engineering으로...", "Google SRE 사례에서..."

4. **운영/조직 관점**: SRE, 온콜, SLA/SLO 등 운영 관점 추가

5. **최신 트렌드**: AI/ML 기반 예측, 자동화 등 최신 기술 언급

**반드시 피드백의 모든 개선 제안을 충실히 반영**하세요.
이전보다 품질 점수가 확실히 올라갈 수 있도록 구체적인 개선을 적용하세요."""
        
        draft = await self._generate_with_streaming(prompt, "generator")
        
        logger.info(f"✅ [Generator] 완료 (Iteration {iteration + 1})")
        
        return {
            'current_draft': draft,
            'iteration': iteration + 1,
            'previous_score': state.get('quality_score'),  # 이전 점수 저장
            'messages': [AIMessage(content=f"**[Generator - Iteration {iteration + 1}]**\n\n{draft}")]
        }
    
    async def reflector_node(self, state: ReflectionState) -> dict:
        """Reflector: 품질 평가 및 개선점 도출 (엄격한 평가 기준)"""
        iteration = state.get('iteration', 1)
        previous_score = state.get('previous_score')
        
        logger.info(f"🔍 [Reflector] 품질 평가 중 (Iteration {iteration})...")
        
        # 이전 점수가 있으면 참조하도록 프롬프트 구성
        comparison_note = ""
        if previous_score is not None:
            comparison_note = f"""

📊 **이전 버전 점수:** {previous_score}점
- 개선이 **실제로 반영**되었다면 점수를 올려주세요.
- 개선이 부족하면 동일하거나 낮은 점수를 부여하세요."""
        
        # ✨ 엄격한 평가 기준으로 초기 점수가 너무 높지 않도록
        prompt = f"""다음 초안을 **엄격한 기준**으로 평가하고 개선점을 도출해주세요.

원래 작업: {state['task']}

현재 초안 (Iteration {iteration}):
{state['current_draft']}{comparison_note}

---

## 🎯 평가 기준 (각 항목당 최대 2점, 총 10점)

| 항목 | 평가 기준 | 배점 |
|------|----------|------|
| 1. 구조/논리 | 명확한 구조와 논리적 흐름 | 0-2점 |
| 2. 정량 데이터 | 구체적인 수치, 통계, KPI 포함 여부 | 0-2점 |
| 3. 구체적 도구/기술 | 실제 도구명, 기술 스택 언급 여부 | 0-2점 |
| 4. 실제 사례 | 기업명 포함 실제 적용 사례 여부 | 0-2점 |
| 5. 전문성/깊이 | 운영 관점, 최신 트렌드, 전문 용어 활용 | 0-2점 |

**⚠️ 점수 가이드라인:**
- **3-4점**: 기본 구조만 있고 구체성 부족 (초안 수준)
- **5-6점**: 구조 + 일부 구체성 있음
- **7점**: 절반 이상의 기준 충족
- **8점**: 대부분의 기준 충족
- **9점**: 모든 기준 충족 + 높은 완성도
- **10점**: 실무 문서 수준의 완벽한 품질

---

다음 형식으로 **정확하게** 평가해주세요:

**항목별 평가:**
- 구조/논리: ?/2점
- 정량 데이터: ?/2점
- 구체적 도구/기술: ?/2점
- 실제 사례: ?/2점
- 전문성/깊이: ?/2점

**⚠️ 중요: 품질 점수 = 위 5개 항목 점수의 합계입니다!**
**품질 점수:** [항목별 점수를 모두 더한 값, 예: 2+0+1+0+1=4점이면 4]

**잘된 점:**
- (구체적으로 나열)

**개선이 필요한 점:**
- (구체적으로 나열)

**구체적인 개선 제안:**
- (실행 가능한 구체적 제안 - 이것이 추가되면 점수가 올라갈 것)

**반드시** 항목별 점수의 합계를 품질 점수로 기입하세요. 불일치하면 안 됩니다!"""
        
        reflection = await self._generate_with_streaming(prompt, "reflector")
        
        import re
        
        # 디버깅을 위해 응답 앞부분 로깅
        logger.info(f"📝 [Reflector] 응답 앞부분: {reflection[:200]}...")
        
        # ✨ 1단계: 항목별 점수 파싱 및 합산 (가장 정확한 방법)
        # 패턴: "구조/논리: 2/2점" 또는 "- 구조/논리: 2/2점"
        item_patterns = [
            r'구조\s*/?\s*논리\s*:\s*(\d+(?:\.\d+)?)\s*/\s*2',      # 구조/논리: 2/2
            r'정량\s*데이터\s*:\s*(\d+(?:\.\d+)?)\s*/\s*2',         # 정량 데이터: 1/2
            r'구체적\s*도구\s*/?\s*기술\s*:\s*(\d+(?:\.\d+)?)\s*/\s*2',  # 구체적 도구/기술: 2/2
            r'실제\s*사례\s*:\s*(\d+(?:\.\d+)?)\s*/\s*2',           # 실제 사례: 0/2
            r'전문성\s*/?\s*깊이\s*:\s*(\d+(?:\.\d+)?)\s*/\s*2',    # 전문성/깊이: 1/2
        ]
        
        item_scores = []
        for pattern in item_patterns:
            match = re.search(pattern, reflection, re.IGNORECASE)
            if match:
                item_scores.append(float(match.group(1)))
        
        # 항목별 점수 합산 (5개 항목 모두 파싱된 경우)
        calculated_score = None
        if len(item_scores) == 5:
            calculated_score = sum(item_scores)
            logger.info(f"📊 [Reflector] 항목별 점수 합산: {item_scores} = {calculated_score}")
        
        # ✨ 2단계: LLM이 명시한 품질 점수 파싱 (백업)
        llm_stated_score = None
        score_patterns = [
            r'\*\*품질\s*점수:\*\*\s*(\d+(?:\.\d+)?)',     # **품질 점수:** 8
            r'\*\*품질\s*점수\*\*\s*:\s*(\d+(?:\.\d+)?)',  # **품질 점수**: 8
            r'품질\s*점수\s*:\s*(\d+(?:\.\d+)?)',          # 품질 점수: 5
            r'품질\s*점수\s*:\s*\*\*(\d+(?:\.\d+)?)\*\*',  # 품질 점수: **8**
        ]
        
        for pattern in score_patterns:
            try:
                match = re.search(pattern, reflection, re.IGNORECASE | re.MULTILINE)
                if match:
                    score = float(match.group(1))
                    if 1 <= score <= 10:
                        llm_stated_score = score
                        logger.info(f"📊 [Reflector] LLM 명시 점수: {score}")
                        break
            except Exception as e:
                continue
        
        # ✨ 3단계: 최종 점수 결정 (항목별 합산 우선)
        # 규칙: 항목별 합산 점수가 있으면 그것을 사용, 없으면 LLM 명시 점수 사용
        quality_score = None
        
        if calculated_score is not None:
            quality_score = calculated_score
            if llm_stated_score is not None and abs(calculated_score - llm_stated_score) > 0.5:
                logger.warning(f"⚠️ [Reflector] 점수 불일치! 합산: {calculated_score}, LLM 명시: {llm_stated_score} → 합산 점수 사용")
        elif llm_stated_score is not None:
            quality_score = llm_stated_score
            logger.info(f"📊 [Reflector] LLM 명시 점수 사용: {llm_stated_score}")
        
        # 파싱 실패 시 기본값 사용
        if quality_score is None:
            logger.warning(f"⚠️ [Reflector] 점수 파싱 실패!")
            if previous_score is not None:
                quality_score = previous_score
                logger.warning(f"⚠️ [Reflector] 이전 점수 유지: {quality_score}")
            else:
                quality_score = 5.0
                logger.warning(f"⚠️ [Reflector] 기본값 사용: {quality_score}")
        
        # 개선폭 계산
        improvement = 0.0
        if previous_score is not None:
            improvement = quality_score - previous_score
        
        logger.info(f"✅ [Reflector] 완료 - 점수: {quality_score}/10" + 
                   (f" (개선폭: {improvement:+.1f})" if previous_score else ""))
        
        # 히스토리에 추가
        revision_history = state.get('revision_history', [])
        revision_history.append({
            'iteration': iteration,
            'draft': state['current_draft'],
            'reflection': reflection,
            'quality_score': quality_score,
            'previous_score': previous_score,
            'improvement': improvement
        })
        
        # 메시지에 개선폭 정보 추가
        improvement_info = ""
        if previous_score is not None:
            if improvement > 0:
                improvement_info = f"\n\n📈 **개선폭:** +{improvement:.1f}점 (이전: {previous_score} → 현재: {quality_score})"
            elif improvement < 0:
                improvement_info = f"\n\n📉 **퇴화 감지:** {improvement:.1f}점 (이전: {previous_score} → 현재: {quality_score})"
            else:
                improvement_info = f"\n\n➡️ **점수 유지:** {quality_score}점 (변화 없음)"
        
        return {
            'reflection': reflection,
            'quality_score': quality_score,
            'revision_history': revision_history,
            'messages': [AIMessage(content=f"**[Reflector - 품질 평가 (Iteration {iteration})]**\n\n{reflection}{improvement_info}")]
        }
    
    def should_continue(self, state: ReflectionState) -> str:
        """반복 계속 여부 결정 (3가지 종료 조건)"""
        iteration = state.get('iteration', 0)
        quality_score = state.get('quality_score', 0)
        previous_score = state.get('previous_score')
        max_iterations = state.get('max_iterations', self.max_iterations)
        quality_threshold = state.get('quality_threshold', self.quality_threshold)
        improvement_threshold = state.get('improvement_threshold', self.improvement_threshold)
        
        # 종료 조건 1: 품질 기준 충족
        if quality_score >= quality_threshold:
            logger.info(f"✅ [종료 조건 1] 품질 기준 충족 ({quality_score} >= {quality_threshold})")
            return "finalize"
        
        # 종료 조건 2: 최대 반복 횟수 도달
        if iteration >= max_iterations:
            logger.info(f"⚠️ [종료 조건 2] 최대 반복 횟수 도달 ({iteration}/{max_iterations})")
            return "finalize"
        
        # 종료 조건 3: 개선 정체 또는 퇴화 감지 (2회차 이후)
        if previous_score is not None:
            improvement = quality_score - previous_score
            
            # 퇴화 감지: 점수가 하락한 경우
            if improvement < 0:
                logger.info(f"⚠️ [종료 조건 3] 퇴화 감지 ({previous_score} → {quality_score}, 개선폭: {improvement:.1f})")
                return "finalize"
            
            # 개선 정체: 개선폭이 임계값 미만
            if improvement < improvement_threshold and iteration >= 2:
                logger.info(f"⚠️ [종료 조건 3] 개선 정체 (개선폭 {improvement:.2f} < {improvement_threshold})")
                return "finalize"
        
        logger.info(f"🔄 개선 계속 ({quality_score} < {quality_threshold}, iteration {iteration}/{max_iterations})")
        return "revise"
    
    async def finalizer_node(self, state: ReflectionState) -> dict:
        """Finalizer: 최종 출력 생성 (Reflector 점수 일관성 보장)"""
        logger.info("🎯 [Finalizer] 최종 결과 생성 중...")
        
        iteration = state.get('iteration', 0)
        revision_history = state.get('revision_history', [])
        
        # ✅ 점수 히스토리 추출 (revision_history에서 가져옴)
        score_history = []
        for h in revision_history:
            if 'quality_score' in h and h['quality_score'] is not None:
                score_history.append(h['quality_score'])
        
        # ✅ 핵심 규칙: Final Score = Last Reflector Score
        # 1순위: revision_history의 마지막 점수
        # 2순위: state의 quality_score
        # 3순위: 기본값 (이 경우는 오류 상황)
        if score_history:
            final_score = score_history[-1]  # 마지막 Reflector 점수 사용
            logger.info(f"📊 [Finalizer] 점수 히스토리에서 최종 점수 추출: {final_score}")
        elif state.get('quality_score') is not None:
            final_score = state.get('quality_score')
            logger.info(f"📊 [Finalizer] state에서 최종 점수 추출: {final_score}")
        else:
            final_score = 7.0  # 파싱 완전 실패 시 기본값
            logger.warning(f"⚠️ [Finalizer] 점수 추출 실패 - 기본값 사용: {final_score}")
        
        # 평균 점수 계산
        avg_score = sum(score_history) / len(score_history) if score_history else final_score
        
        # 종료 사유 분석
        termination_reason = "알 수 없음"
        quality_threshold = state.get('quality_threshold', self.quality_threshold)
        max_iterations = state.get('max_iterations', self.max_iterations)
        
        if final_score >= quality_threshold:
            termination_reason = f"✅ 품질 기준 충족 ({final_score:.1f} >= {quality_threshold})"
        elif iteration >= max_iterations:
            termination_reason = f"⏱️ 최대 반복 횟수 도달 ({iteration}/{max_iterations})"
        elif len(score_history) >= 2:
            last_improvement = score_history[-1] - score_history[-2]
            if last_improvement < 0:
                termination_reason = f"📉 퇴화 감지 (개선폭: {last_improvement:+.1f})"
            elif last_improvement < self.improvement_threshold:
                termination_reason = f"📊 개선 정체 (개선폭: {last_improvement:.2f} < {self.improvement_threshold})"
        
        # 점수 변화 추이 문자열
        score_trend = " → ".join([f"{s:.1f}" for s in score_history]) if score_history else f"{final_score:.1f}"
        
        # 로깅: 점수 일관성 검증
        logger.info(f"📊 [Finalizer] 점수 검증 - 히스토리: {score_history}, 최종: {final_score}")
        
        summary = f"""**Reflection 완료**

**종료 사유:** {termination_reason}

**반복 통계:**
- 총 반복 횟수: {iteration}회
- 점수 변화: {score_trend}
- 평균 점수: {avg_score:.1f}/10
- **최종 점수: {final_score:.1f}/10** (마지막 Reflector 평가 기준)

---

**최종 결과:**
{state['current_draft']}"""
        
        logger.info(f"✅ [Finalizer] 완료 - {iteration}회 반복, 최종 점수: {final_score:.1f}/10")
        
        return {
            'final_output': state['current_draft'],
            'termination_reason': termination_reason,
            'messages': [AIMessage(content=summary)]
        }
    
    def create_graph(self) -> StateGraph:
        """Reflection 워크플로우 그래프 생성"""
        workflow = StateGraph(ReflectionState)
        
        # 노드 추가
        workflow.add_node("generator", self.generator_node)
        workflow.add_node("reflector", self.reflector_node)
        workflow.add_node("finalizer", self.finalizer_node)
        
        # 엣지 추가
        workflow.set_entry_point("generator")
        workflow.add_edge("generator", "reflector")
        
        # 조건부 엣지: 계속 개선할지 종료할지
        workflow.add_conditional_edges(
            "reflector",
            self.should_continue,
            {
                "revise": "generator",  # 개선 필요 → Generator로 돌아감
                "finalize": "finalizer"  # 품질 충족/정체/퇴화 → 종료
            }
        )
        
        workflow.add_edge("finalizer", END)
        
        logger.info("✅ [Reflection] 그래프 생성 완료 - 생성→평가→개선 루프 (개선폭/퇴화 감지 포함)")
        return workflow.compile()


# ============================================================================
# 8️⃣ Routing / Dynamic Orchestration Pattern (다중 후보 경쟁)
# ============================================================================

class RoutingPattern:
    """
    Dynamic Routing / Orchestration Pattern: 다중 후보 경쟁 기반 동적 라우팅
    
    ✨ 핵심 3요소 (업그레이드):
    1. Router: 모든 후보 전문가에 대한 적합도 점수 산출 → 최고 점수 선택
    2. Specialist Agents: 각 분야 전문 에이전트 (병렬 평가 대상)
    3. Aggregator: 점수표 + 탈락 사유 + 최종 결과 통합
    
    ✨ 라우팅 흐름:
    [User Request]
         ↓
    [Router] ─┬─ 기술 전문가: 0.92 ← 선택!
              ├─ 보안 전문가: 0.65
              ├─ 비즈니스 전문가: 0.40
              └─ 일반 어시스턴트: 0.25
         ↓
    [Selected Expert Agent]
         ↓
    [Aggregator] → 점수표 + 탈락 사유 + 응답
    
    ✨ vs 기본형:
    - 기본형: if "시스템" in query → 기술전문가 (정적)
    - 업그레이드: 모든 후보 점수 계산 → 최고 점수 선택 (동적 경쟁)
    """
    
    # 전문 에이전트 정의 (보안 전문가 추가)
    SPECIALIST_AGENTS = {
        "technical": {
            "name": "기술 전문가",
            "emoji": "💻",
            "description": "기술적인 질문, 코드, 아키텍처, 시스템 설계, 인프라 관련",
            "strengths": ["시스템 아키텍처", "코드 리뷰", "기술 스택 선택", "성능 최적화"],
            "keywords": ["코드", "개발", "기술", "시스템", "아키텍처", "API", "서버", "데이터베이스", "인프라"]
        },
        "security": {
            "name": "보안 전문가",
            "emoji": "🔒",
            "description": "보안 정책, 취약점 분석, 컴플라이언스, 인증/인가 관련",
            "strengths": ["취약점 분석", "보안 아키텍처", "컴플라이언스", "침해 대응"],
            "keywords": ["보안", "취약점", "인증", "암호화", "해킹", "방화벽", "접근제어", "감사"]
        },
        "business": {
            "name": "비즈니스 전문가",
            "emoji": "📊",
            "description": "비즈니스 전략, 시장 분석, ROI, 수익 모델 관련",
            "strengths": ["시장 분석", "ROI 계산", "비즈니스 모델", "경쟁 분석"],
            "keywords": ["비즈니스", "수익", "시장", "전략", "ROI", "고객", "매출", "마케팅", "투자"]
        },
        "creative": {
            "name": "크리에이티브 전문가",
            "emoji": "🎨",
            "description": "창작, 콘텐츠, 디자인, UX/UI, 브랜딩 관련",
            "strengths": ["UX 설계", "디자인 시스템", "콘텐츠 전략", "브랜딩"],
            "keywords": ["디자인", "콘텐츠", "창작", "브랜딩", "UX", "UI", "스토리", "경험"]
        },
        "general": {
            "name": "일반 어시스턴트",
            "emoji": "🤖",
            "description": "일반적인 질문, 분류가 불분명한 요청, 기타",
            "strengths": ["일반 지식", "요약", "정리", "기타"],
            "keywords": []
        }
    }
    
    # 선택 임계값 (이 이상이면 해당 전문가 선택 가능)
    SELECTION_THRESHOLD = 0.3
    
    def __init__(self, llm_service, stream_callback: Optional[StreamCallback] = None):
        self.llm_service = llm_service
        self.stream_callback = stream_callback
    
    def set_stream_callback(self, callback: Optional[StreamCallback]):
        """스트리밍 콜백 설정"""
        self.stream_callback = callback
    
    async def _generate_with_streaming(self, prompt: str, node_name: str) -> str:
        """스트리밍 콜백이 있으면 토큰 단위로 스트리밍, 없으면 일반 생성"""
        if self.stream_callback:
            content = ""
            async for token in self.llm_service.generate_response_stream(prompt):
                content += token
                await self.stream_callback(node_name, token)
            return content
        else:
            return await self.llm_service.generate_response(prompt)
    
    async def router_node(self, state: RoutingState) -> dict:
        """
        Router: 다중 후보 경쟁 기반 동적 라우팅
        
        모든 전문가에 대해 적합도 점수를 계산하고, 최고 점수 전문가를 선택합니다.
        """
        logger.info("🔀 [Router] 다중 후보 경쟁 평가 시작...")
        
        user_request = state['user_request']
        
        # 모든 전문가 정보 구성
        agents_detailed = []
        for key, info in self.SPECIALIST_AGENTS.items():
            agents_detailed.append(f"""- **{key}** ({info['emoji']} {info['name']}):
  - 전문 분야: {info['description']}
  - 강점: {', '.join(info['strengths'])}""")
        
        agents_desc = "\n".join(agents_detailed)
        
        # ✨ 다중 후보 경쟁 평가 프롬프트
        prompt = f"""당신은 사용자 요청을 분석하여 가장 적합한 전문가를 선택하는 라우터입니다.

**사용자 요청:**
{user_request}

**가용 전문가 목록:**
{agents_desc}

---

## 🎯 평가 지침

각 전문가에 대해 **적합도 점수(0.0~1.0)**를 평가하세요.

**점수 기준:**
- 0.9~1.0: 완벽한 매칭 (해당 전문가의 핵심 영역)
- 0.7~0.8: 높은 적합도 (관련성 높음)
- 0.5~0.6: 중간 적합도 (부분적 관련)
- 0.3~0.4: 낮은 적합도 (약간의 관련)
- 0.0~0.2: 거의 무관함

---

다음 **JSON 형식으로만** 응답하세요:

```json
{{
    "analysis": "요청 분석 내용 (1-2문장)",
    "scores": {{
        "technical": {{"score": 0.0, "reason": "점수 사유"}},
        "security": {{"score": 0.0, "reason": "점수 사유"}},
        "business": {{"score": 0.0, "reason": "점수 사유"}},
        "creative": {{"score": 0.0, "reason": "점수 사유"}},
        "general": {{"score": 0.0, "reason": "점수 사유"}}
    }},
    "selected": "최고 점수 전문가 키",
    "selection_reason": "선택 근거 (구체적으로)"
}}
```

JSON 형식으로만 응답하세요."""
        
        response = await self._generate_with_streaming(prompt, "router")
        
        # JSON 파싱
        import re
        candidate_scores = {}
        elimination_reasons = {}
        selected_agent = "general"
        confidence = 0.5
        analysis = ""
        selection_reason = ""
        
        try:
            # JSON 추출
            json_str = response
            if '```json' in response:
                json_str = response.split('```json')[1].split('```')[0].strip()
            elif '```' in response:
                json_str = response.split('```')[1].split('```')[0].strip()
            
            data = json.loads(json_str)
            
            analysis = data.get("analysis", "")
            scores_data = data.get("scores", {})
            selected_agent = data.get("selected", "general")
            selection_reason = data.get("selection_reason", "")
            
            # 점수 정보 추출
            max_score = 0.0
            for agent_key, score_info in scores_data.items():
                if agent_key in self.SPECIALIST_AGENTS:
                    score = float(score_info.get("score", 0.0))
                    reason = score_info.get("reason", "")
                    
                    candidate_scores[agent_key] = {
                        "score": score,
                        "reason": reason,
                        "name": self.SPECIALIST_AGENTS[agent_key]["name"],
                        "emoji": self.SPECIALIST_AGENTS[agent_key]["emoji"]
                    }
                    
                    if score > max_score:
                        max_score = score
                        selected_agent = agent_key
                    
                    # 탈락 사유 생성 (선택되지 않은 경우)
                    if score < max_score:
                        elimination_reasons[agent_key] = reason
            
            confidence = max_score
            
            # 최종 선택된 에이전트 확인
            if selected_agent not in self.SPECIALIST_AGENTS:
                selected_agent = "general"
            
            # 탈락 사유 재구성 (선택된 에이전트 제외)
            elimination_reasons = {}
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1]["score"], reverse=True)
            for agent_key, info in sorted_candidates[1:]:  # 1등 제외
                elimination_reasons[agent_key] = info["reason"]
            
            logger.info(f"✅ [Router] 점수 파싱 완료 - {len(candidate_scores)}개 후보 평가")
            
        except Exception as e:
            logger.error(f"❌ [Router] JSON 파싱 실패: {e}")
            # 폴백: 기본 점수
            for agent_key in self.SPECIALIST_AGENTS.keys():
                candidate_scores[agent_key] = {
                    "score": 0.2 if agent_key != "general" else 0.5,
                    "reason": "파싱 실패로 기본값 적용",
                    "name": self.SPECIALIST_AGENTS[agent_key]["name"],
                    "emoji": self.SPECIALIST_AGENTS[agent_key]["emoji"]
                }
            selected_agent = "general"
            confidence = 0.5
        
        agent_info = self.SPECIALIST_AGENTS[selected_agent]
        
        routing_decision = {
            "selected_agent": selected_agent,
            "agent_name": agent_info["name"],
            "agent_emoji": agent_info["emoji"],
            "confidence": confidence,
            "analysis": analysis,
            "selection_reason": selection_reason
        }
        
        # 라우팅 로그 업데이트
        routing_log = state.get('routing_log', [])
        routing_log.append({
            "request": user_request,
            "decision": routing_decision,
            "all_scores": candidate_scores
        })
        
        logger.info(f"✅ [Router] {agent_info['emoji']} {agent_info['name']} 선택 (적합도: {confidence:.0%})")
        
        # ✨ 점수표 메시지 생성
        sorted_scores = sorted(candidate_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        score_table_rows = []
        for i, (agent_key, info) in enumerate(sorted_scores):
            rank = f"🥇" if i == 0 else f"🥈" if i == 1 else f"🥉" if i == 2 else f" {i+1}"
            selected_mark = "✅ **선택**" if agent_key == selected_agent else ""
            bar_length = int(info["score"] * 20)
            score_bar = "█" * bar_length + "░" * (20 - bar_length)
            score_table_rows.append(
                f"| {rank} | {info['emoji']} {info['name']} | {score_bar} | **{info['score']:.0%}** | {selected_mark} |"
            )
        
        score_table = "\n".join(score_table_rows)
        
        message_content = f"""**[Router - 다중 후보 경쟁 평가]**

**📝 요청 분석:**
{analysis if analysis else user_request}

---

**📊 전문가별 적합도 점수표:**

| 순위 | 전문가 | 적합도 | 점수 | 선택 |
|------|--------|--------|------|------|
{score_table}

---

**🎯 라우팅 결정:**
- **선택된 전문가:** {agent_info['emoji']} **{agent_info['name']}**
- **적합도 점수:** {confidence:.0%}
- **선택 근거:** {selection_reason if selection_reason else "최고 점수 전문가"}

**📋 평가 세부 내용:**
{chr(10).join([f"- {info['emoji']} {info['name']}: {info['reason']}" for _, info in sorted_scores])}"""
        
        return {
            'routing_decision': routing_decision,
            'selected_agent': selected_agent,
            'confidence_score': confidence,
            'candidate_scores': candidate_scores,
            'elimination_reasons': elimination_reasons,
            'routing_log': routing_log,
            'messages': [AIMessage(content=message_content)]
        }
    
    async def technical_agent(self, state: RoutingState) -> dict:
        """기술 전문가 에이전트"""
        return await self._specialist_logic("technical", state)
    
    async def security_agent(self, state: RoutingState) -> dict:
        """보안 전문가 에이전트"""
        return await self._specialist_logic("security", state)
    
    async def business_agent(self, state: RoutingState) -> dict:
        """비즈니스 전문가 에이전트"""
        return await self._specialist_logic("business", state)
    
    async def creative_agent(self, state: RoutingState) -> dict:
        """크리에이티브 전문가 에이전트"""
        return await self._specialist_logic("creative", state)
    
    async def general_agent(self, state: RoutingState) -> dict:
        """일반 어시스턴트 에이전트"""
        return await self._specialist_logic("general", state)
    
    async def _specialist_logic(self, agent_type: str, state: RoutingState) -> dict:
        """전문 에이전트 공통 로직"""
        agent_info = self.SPECIALIST_AGENTS[agent_type]
        logger.info(f"🎯 [{agent_info['emoji']} {agent_info['name']}] 응답 생성 중...")
        
        prompt = f"""당신은 {agent_info['emoji']} **{agent_info['name']}**입니다.

**전문 분야:** {agent_info['description']}
**핵심 강점:** {', '.join(agent_info['strengths'])}

다음 요청에 대해 당신의 전문 분야 관점에서 상세하고 실용적인 응답을 제공하세요.

**사용자 요청:**
{state['user_request']}

---

전문가답게 다음 내용을 포함하여 응답하세요:
1. 핵심 분석/답변
2. 구체적인 권장 사항
3. 주의 사항 또는 고려 사항

전문 용어는 설명을 덧붙이고, 실무에서 바로 활용 가능한 답변을 제공하세요."""
        
        response = await self._generate_with_streaming(prompt, agent_type)
        
        logger.info(f"✅ [{agent_info['name']}] 응답 완료")
        
        return {
            'agent_result': response,
            'messages': [AIMessage(content=f"**[{agent_info['emoji']} {agent_info['name']} 응답]**\n\n{response}")]
        }
    
    def route_to_agent(self, state: RoutingState) -> str:
        """선택된 에이전트로 라우팅"""
        selected = state.get('selected_agent', 'general')
        return selected
    
    async def aggregator_node(self, state: RoutingState) -> dict:
        """
        Aggregator: 결과 통합 + 점수표 + 탈락 사유
        
        ✨ 실무급 오케스트레이션 출력:
        - 전체 점수표
        - 선택된 전문가 정보
        - 탈락 전문가 사유
        - 최종 응답
        """
        logger.info("📊 [Aggregator] 결과 통합 중...")
        
        routing_decision = state.get('routing_decision', {})
        candidate_scores = state.get('candidate_scores', {})
        elimination_reasons = state.get('elimination_reasons', {})
        agent_result = state.get('agent_result', '')
        
        # 점수표 생성
        sorted_scores = sorted(candidate_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        
        score_summary = []
        for i, (agent_key, info) in enumerate(sorted_scores):
            rank_emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"#{i+1}"
            status = "✅ 선택됨" if agent_key == routing_decision.get('selected_agent') else "❌ 탈락"
            score_summary.append(f"- {rank_emoji} {info['emoji']} {info['name']}: **{info['score']:.0%}** ({status})")
        
        score_summary_text = "\n".join(score_summary)
        
        # 탈락 사유 생성
        elimination_text = ""
        if elimination_reasons:
            elimination_items = []
            for agent_key, reason in elimination_reasons.items():
                if agent_key in candidate_scores:
                    info = candidate_scores[agent_key]
                    elimination_items.append(f"- {info['emoji']} {info['name']}: {reason}")
            if elimination_items:
                elimination_text = f"""
---

**❌ 탈락 사유:**
{chr(10).join(elimination_items)}"""
        
        summary = f"""**[최종 응답 - Dynamic Orchestration]**

---

**📊 라우팅 점수표:**
{score_summary_text}

---

**🎯 선택된 전문가:**
- {routing_decision.get('agent_emoji', '🤖')} **{routing_decision.get('agent_name', 'N/A')}**
- 적합도: **{routing_decision.get('confidence', 0):.0%}**
- 선택 근거: {routing_decision.get('selection_reason', 'N/A')}
{elimination_text}

---

**📝 전문가 응답:**

{agent_result}"""
        
        logger.info("✅ [Aggregator] 완료 - 점수표 및 탈락 사유 포함")
        
        return {
            'final_response': agent_result,
            'messages': [AIMessage(content=summary)]
        }
    
    def create_graph(self) -> StateGraph:
        """Dynamic Routing 워크플로우 그래프 생성"""
        workflow = StateGraph(RoutingState)
        
        # 노드 추가 (보안 전문가 추가)
        workflow.add_node("router", self.router_node)
        workflow.add_node("technical", self.technical_agent)
        workflow.add_node("security", self.security_agent)  # ✨ 추가
        workflow.add_node("business", self.business_agent)
        workflow.add_node("creative", self.creative_agent)
        workflow.add_node("general", self.general_agent)
        workflow.add_node("aggregator", self.aggregator_node)
        
        # 엣지 추가
        workflow.set_entry_point("router")
        
        # 조건부 엣지: Router 결과에 따라 분기
        workflow.add_conditional_edges(
            "router",
            self.route_to_agent,
            {
                "technical": "technical",
                "security": "security",  # ✨ 추가
                "business": "business",
                "creative": "creative",
                "general": "general"
            }
        )
        
        # 모든 전문 에이전트 → Aggregator
        workflow.add_edge("technical", "aggregator")
        workflow.add_edge("security", "aggregator")  # ✨ 추가
        workflow.add_edge("business", "aggregator")
        workflow.add_edge("creative", "aggregator")
        workflow.add_edge("general", "aggregator")
        
        workflow.add_edge("aggregator", END)
        
        logger.info("✅ [Dynamic Routing] 그래프 생성 완료 - 다중 후보 경쟁 오케스트레이션")
        return workflow.compile()


# ============================================================================
# 9️⃣ Human-in-the-Loop (HITL) Pattern - 실제 사람 개입
# ============================================================================

class HITLPattern:
    """
    Human-in-the-Loop Pattern: 실제 사람의 검토/승인/수정 지원
    
    ✨ 핵심 특징:
    - 실제 사람이 WebSocket을 통해 결정을 내림
    - LLM 시뮬레이션이 아닌 진정한 Human-in-the-Loop
    
    ✨ 3단 분기 결정 구조:
    - ✅ APPROVE: 제안 승인 → Finalizer로 종료
    - 🟡 REVISION: 수정 요청 → Generator 재호출 (피드백 반영)
    - ⛔ REJECT: 제안 거부 → Finalizer로 종료
    
    ✨ 워크플로우:
    [Agent: 제안 생성]
           ↓
    [Human Gate: 대기] ← 실제 사람 결정 대기!
           ↓
    ┌──────┼──────┐
    ✅     🟡     ⛔
    승인   수정   거부
    ↓      ↓      ↓
    종료   재생성  종료
    """
    
    def __init__(self, llm_service, max_revisions=3, stream_callback: Optional[StreamCallback] = None):
        self.llm_service = llm_service
        self.max_revisions = max_revisions
        self.stream_callback = stream_callback
    
    def set_stream_callback(self, callback: Optional[StreamCallback]):
        """스트리밍 콜백 설정"""
        self.stream_callback = callback
    
    async def _generate_with_streaming(self, prompt: str, node_name: str) -> str:
        """스트리밍 콜백이 있으면 토큰 단위로 스트리밍, 없으면 일반 생성"""
        if self.stream_callback:
            content = ""
            async for token in self.llm_service.generate_response_stream(prompt):
                content += token
                await self.stream_callback(node_name, token)
            return content
        else:
            return await self.llm_service.generate_response(prompt)
    
    async def proposal_generator(self, state: HITLState) -> dict:
        """Agent: 제안 생성 (피드백 반영 포함)"""
        revision_count = state.get('revision_count', 0)
        human_feedback = state.get('human_feedback')
        revision_history = state.get('revision_history', [])
        
        if revision_count == 0:
            logger.info("📝 [Agent] 초기 제안 생성 중...")
            prompt = f"""다음 작업에 대한 상세 제안서를 작성해주세요.

**작업:** {state['task']}

---

## 제안서 구조 (반드시 포함):

### 1. 📋 목표 및 범위
- 핵심 목표를 명확히 정의
- 범위와 제약 조건 명시

### 2. 🛠️ 접근 방법
- 단계별 실행 계획
- 필요한 리소스 및 기술 스택

### 3. 📈 예상 결과
- 정량적 성과 지표 (KPI)
- 기대 효과

### 4. ⚠️ 리스크 및 대응 방안
- 잠재적 위험 요소
- 각 리스크별 완화 전략

### 5. 📅 일정 및 마일스톤
- 주요 마일스톤
- 예상 소요 기간

---

구체적이고 실행 가능한 제안서를 작성하세요."""
        else:
            logger.info(f"📝 [Agent] 피드백 반영 수정 중 (수정 {revision_count}회)...")
            
            # 이전 수정 이력 요약
            history_summary = ""
            if revision_history:
                history_items = []
                for i, h in enumerate(revision_history):
                    history_items.append(f"- 수정 {i+1}회: {h.get('summary', 'N/A')}")
                history_summary = f"\n**이전 수정 이력:**\n{chr(10).join(history_items)}\n"
            
            prompt = f"""사람의 피드백을 **충실히 반영**하여 제안서를 수정해주세요.

**원래 작업:** {state['task']}
{history_summary}
**현재 제안서:**
{state.get('agent_proposal', '')}

---

**🔴 반드시 반영해야 할 피드백:**
{human_feedback}

---

**수정 가이드라인:**
1. 피드백에서 지적한 모든 사항을 구체적으로 개선하세요
2. 수정된 부분을 명확히 표시하세요 (예: "🔄 수정됨")
3. 기존의 장점은 유지하면서 개선하세요
4. 피드백에 없는 부분도 전반적인 품질 향상을 위해 개선하세요

**개선된 제안서를 작성하세요:**"""
        
        proposal = await self._generate_with_streaming(prompt, "proposal_generator")
        
        logger.info(f"✅ [Agent] 제안 완료 (수정 {revision_count}회)")
        
        # 메시지 구성
        revision_badge = f" (🔄 {revision_count}차 수정본)" if revision_count > 0 else ""
        header = f"**[Agent - 제안서 생성{revision_badge}]**"
        
        if revision_count > 0:
            header += f"\n\n📌 **반영된 피드백:** {human_feedback[:100]}..." if len(human_feedback or '') > 100 else f"\n\n📌 **반영된 피드백:** {human_feedback}"
        
        return {
            'agent_proposal': proposal,
            'workflow_status': 'awaiting_human_input',
            'awaiting_input': True,  # ✨ 사람 입력 대기 신호!
            'messages': [AIMessage(content=f"{header}\n\n{proposal}")]
        }
    
    async def human_gate(self, state: HITLState) -> dict:
        """
        Human Gate: 실제 사람의 결정을 적용
        
        ⚠️ 이 노드는 외부에서 human_decision, human_feedback, revision_count가
        state에 주입된 후에 실행됩니다.
        (revision_count 증가는 service.py에서 처리)
        """
        logger.info("👤 [Human Gate] 사람의 결정 처리 중...")
        
        # service.py에서 이미 증가된 revision_count 사용
        revision_count = state.get('revision_count', 0)
        decision = state.get('human_decision', 'approve')
        feedback = state.get('human_feedback', '')
        revision_history = state.get('revision_history', [])
        max_revisions = state.get('max_revisions', self.max_revisions)
        
        # 결정에 따른 상태 매핑
        status_map = {
            "approve": "approved",
            "reject": "rejected",
            "revision": "revision_requested"
        }
        
        # 결정 이모지
        decision_emoji = {
            "approve": "✅",
            "reject": "⛔",
            "revision": "🟡"
        }
        
        emoji = decision_emoji.get(decision, "❓")
        new_status = status_map.get(decision, 'approved')
        
        # 최대 수정 횟수 체크
        if decision == "revision" and revision_count >= max_revisions:
            logger.warning(f"⚠️ 최대 수정 횟수({max_revisions}회) 도달")
        
        logger.info(f"✅ [Human Gate] 결정: {decision.upper()} (수정 {revision_count}회)")
        
        # 메시지 구성
        message_content = f"""**[Human Review - 검토 결과]**

---

**{emoji} 결정:** {decision.upper()}

**📝 피드백:**
{feedback if feedback else "(피드백 없음)"}

---

**📊 통계:**
- 현재 수정 횟수: {revision_count}회 / 최대 {max_revisions}회
- 워크플로우 상태: {new_status}"""

        if decision == "revision" and revision_count >= max_revisions:
            message_content += f"\n\n⚠️ **주의:** 최대 수정 횟수에 도달했습니다. 다음 결정에서 승인 또는 거부를 선택해주세요."
        
        return {
            'human_decision': decision,
            'human_feedback': feedback,
            'workflow_status': new_status,
            'awaiting_input': False,
            # revision_count는 service.py에서 이미 설정됨, 그대로 유지
            'messages': [AIMessage(content=message_content)]
        }
    
    def should_continue(self, state: HITLState) -> str:
        """다음 단계 결정 (3단 분기)"""
        decision = state.get('human_decision', '')
        revision_count = state.get('revision_count', 0)
        max_revisions = state.get('max_revisions', self.max_revisions)
        
        # 최대 수정 횟수 도달 시 강제 종료
        if revision_count >= max_revisions and decision == "revision":
            logger.info(f"⚠️ [종료] 최대 수정 횟수({max_revisions}회) 도달 - 자동 승인 처리")
            return "finalize"
        
        if decision == "approve":
            logger.info("✅ [분기] 승인 → Finalizer")
            return "finalize"
        elif decision == "reject":
            logger.info("⛔ [분기] 거부 → Finalizer")
            return "finalize"
        elif decision == "revision":
            logger.info(f"🟡 [분기] 수정 요청 → Generator (수정 {revision_count}회)")
            return "revise"
        else:
            logger.warning(f"❓ [분기] 알 수 없는 결정: {decision} → Finalizer")
            return "finalize"
    
    async def finalizer_node(self, state: HITLState) -> dict:
        """Finalizer: 최종 결과 생성"""
        logger.info("🎯 [Finalizer] 최종 결과 생성 중...")
        
        decision = state.get('human_decision', '')
        revision_count = state.get('revision_count', 0)
        revision_history = state.get('revision_history', [])
        
        # 결정에 따른 상태 텍스트
        if decision == "approve":
            status_emoji = "✅"
            status_text = "승인됨"
            final_output = state.get('agent_proposal', '')
        elif decision == "reject":
            status_emoji = "⛔"
            status_text = "거부됨"
            final_output = f"제안이 거부되었습니다.\n\n**거부 사유:**\n{state.get('human_feedback', '사유 없음')}"
        else:
            # 최대 수정 횟수 도달로 인한 자동 종료
            status_emoji = "⚠️"
            status_text = "자동 승인 (최대 수정 횟수 도달)"
            final_output = state.get('agent_proposal', '')
        
        # 수정 이력 요약
        history_summary = ""
        if revision_history:
            history_items = []
            for h in revision_history:
                history_items.append(f"- **수정 {h['iteration']}회:** {h['summary']}")
            history_summary = f"""
---

**📜 수정 이력:**
{chr(10).join(history_items)}"""
        
        summary = f"""**[HITL 워크플로우 완료]**

---

**{status_emoji} 최종 상태:** {status_text}
**📊 총 수정 횟수:** {revision_count}회 / {self.max_revisions}회
{history_summary}

---

**📄 최종 제안서:**

{final_output}"""
        
        logger.info(f"✅ [Finalizer] 완료 - {status_text}")
        
        return {
            'workflow_status': 'completed',
            'awaiting_input': False,
            'final_output': final_output,
            'messages': [AIMessage(content=summary)]
        }
    
    def create_graph(self) -> StateGraph:
        """HITL 워크플로우 그래프 생성 (실제 사람 개입)"""
        workflow = StateGraph(HITLState)
        
        # 노드 추가
        workflow.add_node("proposal_generator", self.proposal_generator)
        workflow.add_node("human_gate", self.human_gate)  # ✨ 실제 사람 결정 처리
        workflow.add_node("finalizer", self.finalizer_node)
        
        # 엣지 추가
        workflow.set_entry_point("proposal_generator")
        workflow.add_edge("proposal_generator", "human_gate")
        
        # 조건부 엣지: Human 결정에 따라 3단 분기
        workflow.add_conditional_edges(
            "human_gate",
            self.should_continue,
            {
                "revise": "proposal_generator",  # 🟡 수정 요청 → Generator 재호출
                "finalize": "finalizer"  # ✅⛔ 승인/거부 → 종료
            }
        )
        
        workflow.add_edge("finalizer", END)
        
        logger.info("✅ [HITL] 그래프 생성 완료 - 실제 사람 개입 + 3단 분기 (approve/revision/reject)")
        return workflow.compile()


# ============================================================================
# Pattern Registry
# ============================================================================

def get_pattern(pattern_name: str, llm_service, stream_callback: Optional[StreamCallback] = None, **kwargs):
    """패턴 팩토리 - 토큰 스트리밍 콜백 지원"""
    patterns = {
        "sequential": SequentialPattern,
        "planner_executor": PlannerExecutorPattern,
        "role_based": RoleBasedPattern,
        "hierarchical": HierarchicalPattern,
        "debate": DebatePattern,
        "swarm": SwarmPattern,
        "reflection": ReflectionPattern,
        "routing": RoutingPattern,
        "hitl": HITLPattern,
    }
    
    pattern_class = patterns.get(pattern_name)
    if not pattern_class:
        raise ValueError(f"Unknown pattern: {pattern_name}. Available: {list(patterns.keys())}")
    
    # 패턴별 추가 인자 전달 + 스트리밍 콜백
    if pattern_name == "debate":
        return pattern_class(llm_service, max_rounds=kwargs.get("max_rounds", 3), stream_callback=stream_callback)
    elif pattern_name == "swarm":
        return pattern_class(llm_service, num_agents=kwargs.get("num_agents", 5), stream_callback=stream_callback)
    elif pattern_name == "reflection":
        return pattern_class(
            llm_service, 
            max_iterations=kwargs.get("max_iterations", 3),
            quality_threshold=kwargs.get("quality_threshold", 8.0),
            improvement_threshold=kwargs.get("improvement_threshold", 0.3),
            stream_callback=stream_callback
        )
    elif pattern_name == "hitl":
        return pattern_class(llm_service, max_revisions=kwargs.get("max_revisions", 3), stream_callback=stream_callback)
    else:
        return pattern_class(llm_service, stream_callback=stream_callback)

