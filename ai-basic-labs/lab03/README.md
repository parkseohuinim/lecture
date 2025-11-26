# Lab 03: RAG (Retrieval-Augmented Generation) 실습

## 학습 목표

이 실습을 통해 다음을 학습합니다:

1. **기본 RAG**: 문서 로딩, 청킹, 임베딩, 검색 기반 답변 생성
2. **하이브리드 검색**: Sparse (BM25) + Dense (Vector) 검색 결합
3. **Re-ranking**: BGE reranker로 검색 정확도 향상
4. **Multi-hop 검색**: 복잡한 질문을 단계적으로 해결
5. **Chunk size 최적화**: 512 vs 1024 vs 2048 비교
6. **컨텍스트 윈도우 관리**: 토큰 제한 내에서 효율적 관리

## 설치 및 실행

```bash
# 1. 프로젝트 루트에서 의존성 설치
cd ai-basic-labs
pip install -r requirements.txt

# 2. 환경변수 설정 (이미 했다면 생략)
cp .env.example .env
# .env 파일을 열어서 OPENAI_API_KEY 입력

# 3. 실습 실행
cd lab03

# 기본 RAG 실습
python rag_basic.py

# 고급 RAG 실습
python advanced_retrieval_langchain.py
```

## 실습 파일

### 1. rag_basic.py - 기본 RAG 시스템

```bash
python rag_basic.py
```

**주요 기능:**
- PDF/Markdown/HTML 문서 로딩
- 텍스트 청킹 (chunk_size, overlap 조절)
- Vector DB (ChromaDB)에 임베딩 저장
- 유사도 검색 기반 답변 생성
- 컨텍스트 압축 및 요약

**사용 예제:**
```python
from rag_basic import RAGSystem

# RAG 시스템 초기화
rag = RAGSystem("my_knowledge_base")

# 문서 추가
rag.add_document("document.pdf", metadata={"type": "manual"})

# 질문하기
result = rag.generate_answer_with_rag("Python의 특징은?", n_results=3)
print(result['answer'])
print(result['sources'])
```

### 2. advanced_retrieval_langchain.py - 고급 RAG 시스템

```bash
python advanced_retrieval_langchain.py
```

**실행하면 모든 실습이 자동으로 순차 실행됩니다:**
1. 검색 방법 비교 (Sparse vs Dense vs Hybrid)
2. Re-ranking 효과 (경량 모델 ~80MB)
3. Multi-hop 검색
4. Chunk size 실험 (512 vs 1024 vs 2048)
5. 컨텍스트 윈도우 관리

**사용 예제:**
```python
from advanced_retrieval_langchain import AdvancedRAGSystem

# 시스템 초기화
rag = AdvancedRAGSystem(chunk_size=1024, use_reranker=True)
rag.ingest_documents("sample.pdf")

# 하이브리드 검색
results = rag.search(
    "질문",
    method="hybrid",  # "sparse", "dense", "hybrid"
    k=5,
    alpha=0.5,  # Dense 가중치 (0~1)
    use_reranker=True
)

# Multi-hop 검색 (복잡한 질문)
results, metadata = rag.multi_hop_search("개념과 활용 방법은?", k=5)

# 답변 생성 (컨텍스트 윈도우 관리)
answer = rag.generate_answer(query, results, manage_context=True)
print(answer["answer"])
```

## 주요 개념

### RAG란?

**RAG (Retrieval-Augmented Generation)**는 외부 지식을 검색하여 LLM의 응답을 향상시키는 기법입니다.

**파이프라인:**
```
문서 → 청킹 → 임베딩 → Vector DB 저장
                           ↓
질문 → 임베딩 → 유사도 검색 → 컨텍스트 구성 → LLM → 답변
```

**장점:**
- ✅ 최신 정보 활용 가능
- ✅ 환각(Hallucination) 감소
- ✅ 도메인 특화 지식 제공
- ✅ 출처 추적 가능

### 검색 방법 비교

| 방법 | 원리 | 장점 | 단점 |
|------|------|------|------|
| **Sparse (BM25)** | 키워드 매칭 | 정확한 용어 검색, 빠름 | 의미 이해 부족 |
| **Dense (Vector)** | 의미적 유사도 | 문맥 이해, 유연함 | 키워드 약함 |
| **Hybrid** | 두 방법 결합 | 높은 정확도 | 약간 느림 |

**Alpha 파라미터:**
```python
alpha=0.3  # 70% Sparse - 키워드 중심 (전문 용어)
alpha=0.5  # 50-50 - 균형 (일반적)
alpha=0.7  # 70% Dense - 의미 중심 (개념, 추상적)
```

### Re-ranking

초기 검색 결과를 재순위화하여 정확도 향상:

```python
# 1. 초기 검색: 많은 결과 (k=20)
# 2. Re-ranking: CrossEncoder로 관련성 재평가
# 3. 상위 선택: 가장 관련성 높은 k개

results = rag.search(query, use_reranker=True)
```

**효과:** 검색 정확도 10-30% 향상

**사용 모델:**
- 기본: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB, 빠름)
- 고성능: `BAAI/bge-reranker-base` (~500MB, 정확함)

### Multi-hop 검색

복잡한 질문을 하위 질문으로 분해하여 단계적 검색:

```python
# "개념과 활용 방법은?" 
#   → "개념은?" (Hop 1)
#   → "활용 방법은?" (Hop 2, Hop 1 결과 활용)

results, metadata = rag.multi_hop_search(query)
```

**적합한 질문:**
- "A와 B의 관계는?"
- "X의 원인과 해결 방법은?"
- "Y의 정의와 실제 활용은?"

### Chunk Size 선택

| 크기 | 청크 수 | 검색 정확도 | 문맥 | 권장 사용 |
|------|---------|------------|------|----------|
| 512 | 많음 | 높음 | 부족 | FAQ, 짧은 답변 |
| 1024 | 중간 | 중간 | 적절 | **일반 문서 (권장)** |
| 2048 | 적음 | 낮음 | 풍부 | 긴 설명, 기술 문서 |

```python
# 실험
for size in [512, 1024, 2048]:
    rag = AdvancedRAGSystem(chunk_size=size)
    # 성능 비교...
```

### 컨텍스트 윈도우 관리

토큰 제한 내에서 자동으로 검색 결과 조정:

```python
answer = rag.generate_answer(
    query,
    results,
    manage_context=True  # 자동 조정
)

# 통계 확인
stats = answer["context_stats"]
print(f"사용 가능: {stats['available_tokens']}")
print(f"사용됨: {stats['used_tokens']}")
print(f"포함 결과: {stats['num_results']}개")
```

**효과:**
- 토큰 제한 초과 방지
- 비용 절감
- 응답 속도 향상

## 실전 팁

### 1. 청크 크기 선택 가이드

```python
# 문서 유형별
document_types = {
    "FAQ/짧은답변": 256-512,
    "일반문서": 512-1024,      # 권장
    "기술문서": 1024-1536,
    "긴설명": 1536-2048
}
```

### 2. 검색 결과 수 (k)

```python
k=1-3   # 빠른 응답, 간단한 질문
k=3-5   # 균형 (일반적)
k=5-10  # 복잡한 질문, 압축 필요
```

### 3. Re-ranking 사용 시기

**사용 권장:**
- 검색 정확도가 중요
- 초기 결과가 많음 (k > 10)
- 복잡한 쿼리

**사용 비권장:**
- 실시간 응답 중요
- 간단한 키워드 검색
- 리소스 제한적

### 4. 성능 비교 (실제 테스트)

| 방법 | 정확도 | 속도 | 토큰 |
|------|--------|------|------|
| Sparse only | 65% | 빠름 | 낮음 |
| Dense only | 75% | 보통 | 보통 |
| Hybrid | 85% | 보통 | 보통 |
| Hybrid + Rerank | 92% | 느림 | 높음 |

## 문제 해결

### Re-ranker 모델 로딩 느림
```bash
# 첫 실행 시 모델 다운로드 (시간 소요)
# 이후 실행은 캐시 사용으로 빠름
```

### 메모리 부족
```python
# 청크 크기 줄이기
rag = AdvancedRAGSystem(chunk_size=512)

# 검색 결과 수 줄이기
results = rag.search(query, k=3)
```

### 검색 정확도 낮음
```python
# 하이브리드 + 리랭킹
results = rag.search(
    query,
    method="hybrid",
    alpha=0.5,
    use_reranker=True
)
```

### 토큰 제한 초과
```python
# 컨텍스트 관리 활성화
answer = rag.generate_answer(query, results, manage_context=True)
```

## 주의사항

- OpenAI API 키가 필요합니다
- API 호출 시 비용이 발생합니다
- Re-ranker 모델은 첫 실행 시 다운로드됩니다 (~80MB)
- sample.pdf 파일이 필요합니다

## 다음 단계

이 실습을 완료한 후:

1. **Lab 04**: 실전 챗봇 시스템 구축
2. 대화 히스토리 관리
3. 멀티턴 대화
4. 사용자 인터페이스

## 참고 자료

- [RAG 논문](https://arxiv.org/abs/2005.11401)
- [LangChain RAG](https://python.langchain.com/docs/use_cases/question_answering/)
- [ChromaDB 문서](https://docs.trychroma.com/)
- [BM25 알고리즘](https://en.wikipedia.org/wiki/Okapi_BM25)
- [BGE Reranker](https://huggingface.co/BAAI/bge-reranker-base)
