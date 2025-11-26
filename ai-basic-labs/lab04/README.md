# Lab 04: AI Agent 시스템 실습

단일 에이전트에서 멀티 에이전트 오케스트레이션까지 경험하는 실습입니다.

## 학습 목표

1. JSON 프롬프트 에이전트로 구조화된 출력 생성
2. RAG Agent로 검색 증강 생성 파이프라인 구축
3. 멀티 에이전트 오케스트레이션 패턴 이해
4. 실무 적용 가능한 질의 분류 + RAG 응답 시스템 구현

## 실습 항목

### 실습 1: 단일 JSON 프롬프트 에이전트

- 사용자 질문을 분석하여 의도(intent)와 카테고리(category) 추론
- JSON 형식의 구조화된 출력 생성
- `response_format={"type": "json_object"}` 활용

```python
# 분류 결과 예시
{
    "category": "customer_service",
    "intent": "refund_inquiry",
    "confidence": 0.95,
    "reasoning": "환불 관련 키워드 포함",
    "keywords": ["환불", "10일", "가능"]
}
```

### 실습 2: RAG Agent 통합

- 질문 -> 검색 -> 재정리 -> 답변 단일 파이프라인
- Vector DB (ChromaDB) 기반 의미 검색
- 검색 결과를 컨텍스트로 활용한 답변 생성

```
[질문] --> [Vector DB 검색] --> [컨텍스트 구성] --> [LLM 답변 생성]
```

### 실습 3: 멀티 에이전트 오케스트레이션

Planner -> Worker 구조로 복잡한 질문 처리:

```
+-----------------------------------------------------------+
|                    Orchestrator (Planner)                  |
|  - 실행 계획 수립                                          |
|  - 에이전트 간 데이터 전달                                 |
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
```

**에이전트 역할:**
- **Agent A (IntentClassifier)**: 질문 의도/카테고리 분류
- **Agent B (Retrieval)**: 관련 문서 검색
- **Agent C (Summarization)**: 검색 결과 요약
- **Agent D (FinalAnswer)**: 최종 답변 생성

### 실습 미션: 고객센터/개발/기획 질의 자동 분류 + RAG 응답

실제 비즈니스 시나리오:
- 고객센터: 환불, 배송, 회원 등급 문의
- 개발팀: API, 코드, 배포, 에러 문의
- 기획팀: 프로젝트, 일정, 스프린트 문의

자동으로 질문을 분류하고 해당 부서의 지식 베이스에서 답변 생성

## 파일 구조

```
lab04/
├── README.md           # 이 파일
├── agent_system.py     # 메인 실습 코드
├── shared_data.py      # 부서별 지식 베이스 데이터
└── chroma_db/          # Vector DB 저장소 (실행 시 생성)
```

## 실행 방법

```bash
# 프로젝트 루트에서 실행
cd ai-basic-labs

# 환경 변수 설정 (.env 파일에 OPENAI_API_KEY 추가)
echo "OPENAI_API_KEY=sk-your-api-key" > .env

# 실습 실행
python lab04/agent_system.py
```

## 핵심 개념

### 1. JSON 프롬프트 에이전트

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    response_format={"type": "json_object"}  # JSON 출력 강제
)
```

### 2. 에이전트 패턴

| 패턴 | 설명 | 적합한 경우 |
|------|------|------------|
| 단일 에이전트 | 하나의 LLM이 모든 처리 | 단순한 작업 |
| 파이프라인 | 순차적 처리 | 단계가 명확한 작업 |
| 오케스트레이션 | Planner가 Worker 조율 | 복잡한 멀티스텝 작업 |

### 3. 오케스트레이터 역할

1. **계획 수립**: 질문 분석 후 실행 단계 결정
2. **에이전트 호출**: 각 단계에 맞는 에이전트 실행
3. **데이터 전달**: 이전 단계 결과를 다음 단계로 전달
4. **결과 통합**: 모든 단계 결과를 종합하여 최종 응답

## 확장 아이디어

1. **에이전트 추가**: 번역 에이전트, 감정 분석 에이전트 등
2. **도구 연동**: 외부 API 호출, 데이터베이스 조회 등
3. **병렬 처리**: 독립적인 에이전트 동시 실행
4. **피드백 루프**: 결과 검증 후 재실행

## 참고 자료

- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [Multi-Agent Systems](https://www.anthropic.com/research/building-effective-agents)

## 다음 단계

- 더 복잡한 에이전트 패턴 (ReAct, Plan-and-Execute)
- 도구 사용 에이전트 (Tool-using Agents)
- 자율 에이전트 (Autonomous Agents)

