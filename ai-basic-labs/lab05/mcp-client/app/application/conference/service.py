"""Conference Service - Multi-Agent Orchestration Service"""
import logging
import asyncio
from typing import Dict, Any, Optional, AsyncGenerator
from fastapi import WebSocket
import json

from app.infrastructure.llm.llm_service import llm_service
from app.application.conference.patterns import get_pattern

logger = logging.getLogger(__name__)


class ConferenceService:
    """멀티 에이전트 회의 orchestration 서비스"""
    
    # 패턴별 병렬 노드 그룹 정의
    PARALLEL_GROUPS = {
        "role_based": {
            "parallel_nodes": ["pm", "developer", "designer", "qa"],
            "final_node": "leader",
            "group_title": "병렬 분석",
            "group_description": "각 역할별 동시 분석"
        },
        "hierarchical": {
            "parallel_nodes": ["worker1", "worker2", "worker3"],
            "final_node": "manager_integrate",
            "pre_node": "manager_delegate",
            "group_title": "병렬 작업 수행",
            "group_description": "Worker들의 동시 작업 수행"
        },
        "swarm": {
            "parallel_nodes": ["agent1", "agent2", "agent3", "agent4", "agent5"],
            "final_node": "selector",
            "group_title": "Market-based 경쟁 입찰",
            "group_description": "5개 전략 전문가의 동시 입찰 (비용/성능/보안/속도/자동화)"
        }
    }
    
    def __init__(self):
        self.llm_service = llm_service
        self.active_sessions = {}
    
    async def run_conference(
        self,
        pattern: str,
        topic: str,
        websocket: Optional[WebSocket] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        회의 실행 (WebSocket 스트리밍 지원)
        
        Args:
            pattern: 패턴 이름 (sequential, planner_executor, role_based, hierarchical, debate, swarm)
            topic: 회의 주제
            websocket: WebSocket 연결 (선택사항, 실시간 스트리밍용)
            **kwargs: 패턴별 추가 옵션 (max_rounds, num_agents 등)
        
        Returns:
            회의 결과
        """
        try:
            logger.info(f"🎯 회의 시작: pattern={pattern}, topic={topic}")
            
            # 스트리밍 콜백 생성 (WebSocket이 있을 경우)
            # 병렬 패턴은 병렬 노드에서 스트리밍 비활성화
            stream_callback = None
            if websocket:
                stream_callback = self._create_stream_callback(websocket, pattern)
            
            # 패턴 인스턴스 생성 (스트리밍 콜백 전달)
            pattern_instance = get_pattern(pattern, self.llm_service, stream_callback=stream_callback, **kwargs)
            
            # LangGraph 워크플로우 생성
            workflow = pattern_instance.create_graph()
            
            # 초기 상태 준비
            initial_state = self._prepare_initial_state(pattern, topic, **kwargs)
            
            # WebSocket 연결이 있으면 스트리밍
            if websocket:
                return await self._run_with_streaming(workflow, initial_state, pattern, websocket, pattern_instance)
            else:
                # 일반 실행
                return await self._run_without_streaming(workflow, initial_state, pattern)
        
        except Exception as e:
            logger.error(f"❌ 회의 실행 실패: {e}", exc_info=True)
            raise
    
    def _create_stream_callback(self, websocket: WebSocket, pattern: str = None):
        """
        WebSocket을 통해 토큰을 전송하는 스트리밍 콜백 생성
        
        병렬 노드(role_based, hierarchical, swarm의 동시 실행 노드)는 
        스트리밍을 비활성화하여 뒤섞임 방지
        """
        # 병렬 노드 목록 수집
        parallel_nodes = set()
        if pattern and pattern in self.PARALLEL_GROUPS:
            parallel_nodes = set(self.PARALLEL_GROUPS[pattern]["parallel_nodes"])
        
        async def stream_callback(node_name: str, token: str):
            # 병렬 노드는 스트리밍 비활성화 (뒤섞임 방지)
            if node_name in parallel_nodes:
                return  # 토큰 전송하지 않음 - 완성된 메시지만 표시
            
            try:
                await websocket.send_json({
                    "type": "agent_token",
                    "node": node_name,
                    "token": token,
                    "status": "streaming"
                })
            except Exception as e:
                logger.error(f"❌ 토큰 스트리밍 전송 실패: {e}")
        return stream_callback
    
    def _prepare_initial_state(self, pattern: str, topic: str, **kwargs) -> Dict[str, Any]:
        """패턴별 초기 상태 준비"""
        
        if pattern == "sequential":
            return {
                "topic": topic,
                "messages": [],
                "current_step": "summarizer",
                "results": {}
            }
        
        elif pattern == "planner_executor":
            return {
                "task": topic,
                "plan": [],
                "current_step": 0,
                "executions": [],
                "messages": [],
                "final_result": ""
            }
        
        elif pattern == "role_based":
            # Role-based 전용 State
            return {
                "topic": topic,
                "pm_opinion": None,
                "dev_opinion": None,
                "design_opinion": None,
                "qa_opinion": None,
                "final_decision": None,
                "messages": []
            }
        
        elif pattern == "hierarchical":
            # Hierarchical 전용 State
            return {
                "topic": topic,
                "assignments": None,
                "worker1_result": None,
                "worker2_result": None,
                "worker3_result": None,
                "final_report": None,
                "messages": []
            }
        
        elif pattern == "debate":
            return {
                "topic": topic,
                "proposal": "",
                "critique": "",
                "round_num": 0,
                "max_rounds": kwargs.get("max_rounds", 3),
                "conversation": [],
                "messages": [],
                "final_decision": ""
            }
        
        elif pattern == "swarm":
            # Swarm 전용 State (Market-based 경쟁)
            return {
                "task": topic,
                # 선택 기준 (목적 함수) - kwargs에서 가져오거나 기본값 사용
                "selection_criteria": kwargs.get("selection_criteria", {
                    "priority": "balanced",  # balanced, cost, speed, performance, security
                    "weights": {
                        "cost": 0.25,
                        "duration": 0.25,
                        "risk": 0.25,
                        "performance": 0.25
                    }
                }),
                "agent1_proposal": None,
                "agent2_proposal": None,
                "agent3_proposal": None,
                "agent4_proposal": None,
                "agent5_proposal": None,
                "winner": None,
                "selection_reasoning": None,
                "messages": []
            }
        
        elif pattern == "reflection":
            # Reflection / Self-Refinement 전용 State
            return {
                "task": topic,
                "current_draft": None,
                "reflection": None,
                "revision_history": [],
                "iteration": 0,
                "max_iterations": kwargs.get("max_iterations", 3),
                "quality_score": None,
                "previous_score": None,  # 개선폭 계산용
                "quality_threshold": kwargs.get("quality_threshold", 8.0),
                "improvement_threshold": kwargs.get("improvement_threshold", 0.3),  # 최소 개선폭
                "termination_reason": None,
                "messages": [],
                "final_output": None
            }
        
        elif pattern == "routing":
            # Routing / Dynamic Orchestration 전용 State (다중 후보 경쟁)
            return {
                "user_request": topic,
                "routing_decision": None,
                "selected_agent": None,
                "confidence_score": None,
                "candidate_scores": None,  # 모든 후보의 점수
                "elimination_reasons": None,  # 탈락 사유
                "agent_result": None,
                "routing_log": [],
                "messages": [],
                "final_response": None
            }
        
        elif pattern == "hitl":
            # Human-in-the-Loop 전용 State (실제 사람 개입 지원)
            return {
                "task": topic,
                "agent_proposal": None,
                "workflow_status": "processing",
                "awaiting_input": False,  # 사람 입력 대기 여부
                "human_feedback": None,
                "human_decision": None,
                "revision_count": 0,
                "max_revisions": kwargs.get("max_revisions", 3),
                "revision_history": [],  # 수정 이력
                "messages": [],
                "final_output": None
            }
        
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    async def _run_with_streaming(
        self,
        workflow,
        initial_state: Dict[str, Any],
        pattern: str,
        websocket: WebSocket,
        pattern_instance=None
    ) -> Dict[str, Any]:
        """WebSocket 스트리밍과 함께 실행 (토큰 단위 스트리밍 지원)"""
        
        try:
            # 병렬 그룹 정보 가져오기
            parallel_info = self.PARALLEL_GROUPS.get(pattern)
            parallel_nodes = parallel_info["parallel_nodes"] if parallel_info else []
            final_node = parallel_info["final_node"] if parallel_info else None
            pre_node = parallel_info.get("pre_node") if parallel_info else None
            
            # 병렬 노드 완료 추적
            parallel_started = False
            completed_parallel_nodes = set()
            
            # 현재 스트리밍 중인 노드 추적
            streaming_nodes = set()
            
            # 시작 알림
            await websocket.send_json({
                "type": "conference_start",
                "pattern": pattern,
                "status": "started",
                "token_streaming_enabled": True  # 토큰 스트리밍 활성화 알림
            })
            
            final_state = None
            
            # LangGraph 스트리밍 실행
            async for event in workflow.astream(initial_state):
                # 이벤트 처리
                for node_name, node_output in event.items():
                    logger.info(f"📡 노드 완료: {node_name}")
                    
                    # 스트리밍이 완료된 노드임을 표시
                    if node_name in streaming_nodes:
                        # 스트리밍 완료 이벤트 전송
                        await websocket.send_json({
                            "type": "agent_stream_end",
                            "node": node_name,
                            "status": "stream_completed"
                        })
                        streaming_nodes.discard(node_name)
                    
                    # 병렬 노드인 경우 처리
                    if parallel_info and node_name in parallel_nodes:
                        # 첫 번째 병렬 노드일 때 parallel_start 전송
                        if not parallel_started:
                            parallel_started = True
                            await websocket.send_json({
                                "type": "parallel_start",
                                "pattern": pattern,
                                "parallel_nodes": parallel_nodes,
                                "group_title": parallel_info["group_title"],
                                "group_description": parallel_info["group_description"],
                                "topic": initial_state.get("topic") or initial_state.get("task", ""),
                                "status": "parallel_running"
                            })
                        
                        # 완료된 노드 추적
                        completed_parallel_nodes.add(node_name)
                    
                    # 메시지 추출
                    messages = node_output.get('messages', [])
                    if messages:
                        latest_message = messages[-1]
                        
                        # 병렬 노드인지 여부 표시
                        is_parallel = node_name in parallel_nodes if parallel_info else False
                        
                        # WebSocket으로 전송 (최종 메시지 - 스트리밍이 이미 완료된 상태)
                        await websocket.send_json({
                            "type": "agent_message",
                            "node": node_name,
                            "content": latest_message.content if hasattr(latest_message, 'content') else str(latest_message),
                            "status": "completed",
                            "is_parallel": is_parallel,
                            "parallel_index": parallel_nodes.index(node_name) if is_parallel else None,
                            "parallel_total": len(parallel_nodes) if is_parallel else None
                        })
                    
                    # 마지막 병렬 노드가 완료되면 parallel_end 전송
                    if parallel_info and parallel_started and len(completed_parallel_nodes) == len(parallel_nodes):
                        if node_name in parallel_nodes:
                            await websocket.send_json({
                                "type": "parallel_end",
                                "pattern": pattern,
                                "completed_nodes": list(completed_parallel_nodes),
                                "next_node": final_node,
                                "status": "parallel_completed"
                            })
                    
                    final_state = node_output
            
            # 완료 알림
            await websocket.send_json({
                "type": "conference_complete",
                "pattern": pattern,
                "status": "completed"
            })
            
            return self._format_result(pattern, final_state)
        
        except Exception as e:
            logger.error(f"❌ 스트리밍 실행 실패: {e}")
            
            # 에러 전송
            if websocket:
                try:
                    await websocket.send_json({
                        "type": "conference_error",
                        "error": str(e),
                        "status": "error"
                    })
                except:
                    pass
            
            raise
    
    async def _run_without_streaming(
        self,
        workflow,
        initial_state: Dict[str, Any],
        pattern: str
    ) -> Dict[str, Any]:
        """일반 실행 (스트리밍 없음)"""
        
        try:
            # LangGraph 실행
            final_state = await workflow.ainvoke(initial_state)
            
            return self._format_result(pattern, final_state)
        
        except Exception as e:
            logger.error(f"❌ 실행 실패: {e}")
            raise
    
    async def run_hitl_step(
        self,
        session_id: str,
        human_decision: str = None,
        human_feedback: str = None,
        websocket: WebSocket = None
    ) -> Dict[str, Any]:
        """
        HITL 패턴 단계별 실행 (사람 입력 처리)
        
        Args:
            session_id: HITL 세션 ID
            human_decision: 사람의 결정 ("approve", "revision", "reject")
            human_feedback: 사람의 피드백
            websocket: WebSocket 연결
        
        Returns:
            현재 상태 및 다음 단계 정보
        """
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"세션을 찾을 수 없습니다: {session_id}")
        
        workflow = session["workflow"]
        state = session["state"]
        
        # 사람 결정 적용
        if human_decision:
            state["human_decision"] = human_decision
            state["human_feedback"] = human_feedback or ""
            state["awaiting_input"] = False
            
            # ✨ APPROVE/REJECT인 경우 바로 종료 처리
            if human_decision in ["approve", "reject"]:
                logger.info(f"👤 [HITL] 사람 결정 적용: {human_decision} → 워크플로우 종료")
                
                # 최종 결과 생성
                revision_count = state.get("revision_count", 0)
                revision_history = state.get("revision_history", [])
                max_revisions = state.get("max_revisions", 3)
                
                if human_decision == "approve":
                    status_emoji = "✅"
                    status_text = "승인됨"
                    final_output = state.get("agent_proposal", "")
                else:  # reject
                    status_emoji = "⛔"
                    status_text = "거부됨"
                    final_output = f"제안이 거부되었습니다.\n\n**거부 사유:**\n{human_feedback or '사유 없음'}"
                
                # 수정 이력 요약
                history_summary = ""
                if revision_history:
                    history_items = [f"- **수정 {h['iteration']}회:** {h['summary']}" for h in revision_history]
                    history_summary = f"\n\n---\n\n**📜 수정 이력:**\n{chr(10).join(history_items)}"
                
                summary = f"""**[HITL 워크플로우 완료]**

---

**{status_emoji} 최종 상태:** {status_text}
**📊 총 수정 횟수:** {revision_count}회 / {max_revisions}회
{history_summary}

---

**📄 최종 제안서:**

{final_output}"""
                
                # 메시지 전송
                if websocket:
                    await websocket.send_json({
                        "type": "agent_message",
                        "node": "finalizer",
                        "content": summary,
                        "status": "completed"
                    })
                    
                    await websocket.send_json({
                        "type": "conference_complete",
                        "pattern": "hitl",
                        "status": "completed"
                    })
                
                # 세션 정리
                del self.active_sessions[session_id]
                
                state["workflow_status"] = "completed"
                state["final_output"] = final_output
                
                return self._format_result("hitl", state)
            
            # ✨ REVISION인 경우 revision_count 증가 및 이력 저장
            elif human_decision == "revision":
                current_count = state.get("revision_count", 0)
                max_revisions = state.get("max_revisions", 3)
                
                # 최대 수정 횟수 체크
                if current_count >= max_revisions:
                    logger.warning(f"⚠️ [HITL] 최대 수정 횟수({max_revisions}회) 도달 - 자동 승인 처리")
                    # 자동 승인으로 전환
                    state["human_decision"] = "approve"
                    return await self.run_hitl_step(session_id, "approve", "최대 수정 횟수 도달로 자동 승인", websocket)
                
                state["revision_count"] = current_count + 1
                
                # revision_history 업데이트
                revision_history = state.get("revision_history", [])
                revision_history.append({
                    "iteration": current_count + 1,
                    "feedback": human_feedback or "",
                    "summary": (human_feedback or "")[:50] + "..." if len(human_feedback or "") > 50 else (human_feedback or "")
                })
                state["revision_history"] = revision_history
                
                logger.info(f"👤 [HITL] 사람 결정 적용: {human_decision} (수정 {state['revision_count']}회)")
        
        try:
            # 다음 단계 실행
            result_state = None
            async for event in workflow.astream(state):
                for node_name, node_output in event.items():
                    logger.info(f"📡 [HITL] 노드 완료: {node_name}")
                    
                    # 메시지 전송
                    messages = node_output.get('messages', [])
                    if messages and websocket:
                        latest_message = messages[-1]
                        await websocket.send_json({
                            "type": "agent_message",
                            "node": node_name,
                            "content": latest_message.content if hasattr(latest_message, 'content') else str(latest_message),
                            "status": "completed"
                        })
                    
                    # 상태 업데이트
                    for key, value in node_output.items():
                        if key != 'messages':
                            state[key] = value
                        elif key == 'messages':
                            state['messages'] = state.get('messages', []) + value
                    
                    result_state = state.copy()
                    
                    # awaiting_input이 True면 중단하고 사용자 입력 대기
                    if state.get('awaiting_input', False):
                        logger.info("⏸️ [HITL] 사람 입력 대기 중...")
                        
                        if websocket:
                            await websocket.send_json({
                                "type": "hitl_awaiting_input",
                                "proposal": state.get('agent_proposal', ''),
                                "revision_count": state.get('revision_count', 0),
                                "max_revisions": state.get('max_revisions', 3),
                                "status": "awaiting_human_input"
                            })
                        
                        # 세션 상태 저장
                        session["state"] = state
                        
                        return {
                            "status": "awaiting_human_input",
                            "proposal": state.get('agent_proposal', ''),
                            "revision_count": state.get('revision_count', 0),
                            "max_revisions": state.get('max_revisions', 3)
                        }
            
            # 워크플로우 완료
            if result_state and result_state.get('workflow_status') == 'completed':
                # 세션 정리
                del self.active_sessions[session_id]
                
                if websocket:
                    await websocket.send_json({
                        "type": "conference_complete",
                        "pattern": "hitl",
                        "status": "completed"
                    })
                
                return self._format_result("hitl", result_state)
            
            # 세션 상태 저장
            session["state"] = state
            
            return {
                "status": "processing",
                "state": state
            }
        
        except Exception as e:
            logger.error(f"❌ [HITL] 실행 오류: {e}")
            raise
    
    async def start_hitl_session(
        self,
        topic: str,
        websocket: WebSocket = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        HITL 세션 시작
        
        Args:
            topic: 작업 주제
            websocket: WebSocket 연결
            **kwargs: 추가 옵션
        
        Returns:
            세션 ID 및 초기 상태
        """
        import uuid
        session_id = str(uuid.uuid4())[:8]
        
        logger.info(f"🚀 [HITL] 세션 시작: {session_id}")
        
        # 스트리밍 콜백 생성 (WebSocket이 있을 경우)
        stream_callback = None
        if websocket:
            stream_callback = self._create_stream_callback(websocket, "hitl")
        
        # 패턴 인스턴스 생성 (스트리밍 콜백 전달)
        pattern_instance = get_pattern("hitl", self.llm_service, stream_callback=stream_callback, **kwargs)
        workflow = pattern_instance.create_graph()
        
        # 초기 상태 준비
        initial_state = self._prepare_initial_state("hitl", topic, **kwargs)
        
        # 세션 저장
        self.active_sessions[session_id] = {
            "workflow": workflow,
            "state": initial_state,
            "topic": topic,
            "websocket": websocket
        }
        
        # 시작 알림
        if websocket:
            await websocket.send_json({
                "type": "hitl_session_start",
                "session_id": session_id,
                "topic": topic,
                "max_revisions": initial_state.get('max_revisions', 3),
                "status": "started"
            })
        
        # 첫 번째 단계 실행 (proposal_generator)
        result = await self.run_hitl_step(session_id, websocket=websocket)
        result["session_id"] = session_id
        return result
    
    def _format_result(self, pattern: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """결과 포맷팅"""
        
        # 메시지 변환
        messages = []
        for msg in state.get('messages', []):
            if hasattr(msg, 'content'):
                messages.append({
                    "role": msg.__class__.__name__.replace('Message', '').lower(),
                    "content": msg.content
                })
            else:
                messages.append({
                    "role": "unknown",
                    "content": str(msg)
                })
        
        result = {
            "pattern": pattern,
            "status": "completed",
            "messages": messages,
            "results": state.get('results', {})
        }
        
        # 패턴별 추가 정보
        if pattern == "planner_executor":
            result["plan"] = state.get('plan', [])
            result["executions"] = state.get('executions', [])
            result["final_result"] = state.get('final_result', '')
        
        elif pattern == "role_based":
            result["opinions"] = {
                "pm": state.get('pm_opinion'),
                "developer": state.get('dev_opinion'),
                "designer": state.get('design_opinion'),
                "qa": state.get('qa_opinion')
            }
            result["final_decision"] = state.get('final_decision', '')
        
        elif pattern == "hierarchical":
            result["assignments"] = state.get('assignments', {})
            result["worker_results"] = {
                "worker1": state.get('worker1_result'),
                "worker2": state.get('worker2_result'),
                "worker3": state.get('worker3_result')
            }
            result["final_report"] = state.get('final_report', '')
        
        elif pattern == "debate":
            result["conversation"] = state.get('conversation', [])
            result["final_decision"] = state.get('final_decision', '')
            result["rounds"] = state.get('round_num', 0)
        
        elif pattern == "swarm":
            proposals = []
            for i in range(1, 6):
                prop = state.get(f'agent{i}_proposal')
                if prop:
                    proposals.append(prop)
            result["proposals"] = proposals
            result["winner"] = state.get('winner', {})
            result["selection_reasoning"] = state.get('selection_reasoning', '')
            result["selection_criteria"] = state.get('selection_criteria', {})
        
        elif pattern == "reflection":
            result["revision_history"] = state.get('revision_history', [])
            result["final_output"] = state.get('final_output', '')
            result["iterations"] = state.get('iteration', 0)
            result["quality_score"] = state.get('quality_score', 0)
        
        elif pattern == "routing":
            result["routing_decision"] = state.get('routing_decision', {})
            result["candidate_scores"] = state.get('candidate_scores', {})  # 다중 후보 점수
            result["elimination_reasons"] = state.get('elimination_reasons', {})  # 탈락 사유
            result["routing_log"] = state.get('routing_log', [])
            result["final_response"] = state.get('final_response', '')
        
        elif pattern == "hitl":
            result["final_output"] = state.get('final_output', '')
            result["human_decision"] = state.get('human_decision', '')
            result["revision_count"] = state.get('revision_count', 0)
            result["revision_history"] = state.get('revision_history', [])  # 수정 이력
            result["workflow_status"] = state.get('workflow_status', '')
        
        return result
    
    def get_available_patterns(self) -> list[Dict[str, str]]:
        """사용 가능한 패턴 목록"""
        return [
            {
                "id": "sequential",
                "name": "Sequential (파이프라인)",
                "description": "에이전트들이 순차적으로 실행되며, 이전 결과를 다음 에이전트가 받아 처리",
                "icon": "→",
                "difficulty": "easy"
            },
            {
                "id": "planner_executor",
                "name": "Planner-Executor (계획-실행)",
                "description": "Planner가 작업을 단계별로 분해하고, Executor가 각 단계를 순차 실행",
                "icon": "📋",
                "difficulty": "easy"
            },
            {
                "id": "role_based",
                "name": "Role-based Collaboration (역할 분담)",
                "description": "PM, 개발자, 디자이너, QA가 동시에 병렬로 의견 제시 후 리더가 통합",
                "icon": "👥",
                "difficulty": "medium"
            },
            {
                "id": "hierarchical",
                "name": "Hierarchical (상하 구조)",
                "description": "Manager가 작업을 분배하고 Worker들이 병렬 실행 후 Manager가 결과 통합",
                "icon": "🏢",
                "difficulty": "medium"
            },
            {
                "id": "debate",
                "name": "Debate / Critic (토론·검증)",
                "description": "제안자와 비평가가 여러 라운드 토론하며 개선하고, 심판이 최종 결정",
                "icon": "⚖️",
                "difficulty": "hard"
            },
            {
                "id": "swarm",
                "name": "Swarm / Market-based (시장 경쟁)",
                "description": "5개 전략 전문가(비용/성능/보안/속도/자동화)가 경쟁 입찰, 목적 함수 기반 자동 선정",
                "icon": "🐝",
                "difficulty": "hard"
            },
            {
                "id": "reflection",
                "name": "Reflection / Self-Refinement (자기 개선)",
                "description": "생성→평가→개선 반복 루프, 품질 기준 충족까지 자동 개선",
                "icon": "🔄",
                "difficulty": "medium"
            },
            {
                "id": "routing",
                "name": "Routing / Orchestration (동적 라우팅)",
                "description": "요청 분석 후 적절한 전문 에이전트(기술/비즈니스/창작)로 자동 라우팅",
                "icon": "🔀",
                "difficulty": "medium"
            },
            {
                "id": "hitl",
                "name": "Human-in-the-Loop (사람 참여)",
                "description": "AI 제안 → 사람 검토/승인 → 피드백 반영의 협업 워크플로우",
                "icon": "👤",
                "difficulty": "hard"
            }
        ]


# Global service instance
conference_service = ConferenceService()

