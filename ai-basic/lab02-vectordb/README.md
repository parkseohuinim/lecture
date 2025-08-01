# Lab 2: 벡터 데이터베이스 구축

## 연계 이론
- **2.3**: 벡터 데이터베이스의 원리와 활용
- **2.4**: 효율적인 유사도 검색 알고리즘

## 학습 목표
1. ChromaDB를 활용한 로컬 벡터 데이터베이스 구축
2. 대용량 문서의 효율적인 인덱싱 및 저장
3. 메타데이터 기반 필터링 검색
4. 벡터 데이터베이스의 영속성 관리
5. 검색 성능 최적화 기법

## 실습 환경 설정

### 1. 필수 패키지 설치
```bash
# 프로젝트 루트에서 모든 패키지 한 번에 설치 (권장)
pip install -r requirements.txt

# 또는 Lab 2 필수 패키지만 설치
pip install chromadb numpy scikit-learn matplotlib python-dotenv tiktoken
```

### 2. API 키 설정
`.env` 파일에 OpenAI API 키가 설정되어 있는지 확인:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

## 실습 단계

### Step 1: ChromaDB 기초 (30분)
**파일: `basic_chroma.py`**
- ChromaDB 클라이언트 설정 및 컬렉션 생성
- 기본적인 문서 추가, 검색, 삭제 작업
- 임베딩 자동 생성 vs 수동 제공 비교

### Step 2: 문서 인덱싱 시스템 (45분) 
**파일: `document_indexing.py`**
- 대용량 문서 컬렉션의 배치 처리
- 청킹 전략을 통한 긴 문서 분할
- 진행률 추적 및 에러 처리

### Step 3: 메타데이터 활용 (30분)
**파일: `metadata_filtering.py`**
- 복합 메타데이터를 활용한 필터링 검색
- 날짜, 카테고리, 작성자별 검색
- 하이브리드 검색 (의미 + 필터) 구현

### Step 4: 성능 최적화 (30분)
**파일: `performance_comparison.py`**
- 메모리 vs 디스크 영속성 비교
- 검색 성능 벤치마킹
- 인덱스 크기 및 검색 속도 분석

## 실습 결과물
1. **로컬 벡터 데이터베이스**: ChromaDB 기반 지식베이스 시스템
2. **문서 인덱싱 파이프라인**: 대용량 문서 자동 처리 시스템
3. **고급 검색 엔진**: 메타데이터 필터링이 포함된 검색 시스템
4. **성능 분석 보고서**: 다양한 설정에 따른 성능 비교

## 실습 데이터
Lab 1에서 사용한 문서 컬렉션을 확장:
- **기술 문서**: AI, 머신러닝, 소프트웨어 개발 관련 문서
- **뉴스 기사**: 최신 기술 뉴스 및 동향
- **학술 논문**: 컴퓨터 과학 관련 논문 초록
- **기업 문서**: 제품 설명서, FAQ, 매뉴얼

## 체크포인트

### 기본 이해도 확인
- [ ] ChromaDB의 컬렉션과 문서 개념을 이해했는가?
- [ ] 벡터 데이터베이스의 장점을 설명할 수 있는가?
- [ ] 메타데이터 필터링의 필요성을 알고 있는가?

### 실습 완료도 확인
- [ ] ChromaDB에 문서를 추가하고 검색할 수 있는가?
- [ ] 대용량 문서를 효율적으로 인덱싱할 수 있는가?
- [ ] 복합 조건으로 문서를 필터링할 수 있는가?
- [ ] 영속성을 활용해 데이터를 저장/로드할 수 있는가?

## 도전 과제

### 기본 도전
1. **문서 분류 시스템**: 자동 태깅 및 카테고리 분류
2. **중복 문서 탐지**: 유사도 기반 중복 제거
3. **검색 랭킹 개선**: 메타데이터 가중치 조정

### 고급 도전  
1. **분산 인덱싱**: 여러 컬렉션을 활용한 분산 저장
2. **실시간 업데이트**: 문서 변경사항 실시간 반영
3. **검색 최적화**: 캐싱 및 인덱스 최적화 구현

## 예상 결과

실습 완료 후 다음과 같은 시스템을 구축할 수 있습니다:

```
검색 쿼리: "머신러닝 성능 최적화"
필터: category="기술", date_range="2024"

검색 결과:
1. [0.912] GPU를 활용한 딥러닝 모델 최적화 기법
   카테고리: 기술 | 작성일: 2024-01-15 | 저자: 김AI
   
2. [0.887] 머신러닝 파이프라인 성능 튜닝 가이드  
   카테고리: 기술 | 작성일: 2024-02-03 | 저자: 박데이터

검색 시간: 0.03초 | 검색된 문서: 847개 중 상위 10개
```

## ChromaDB 주요 특징

### 장점
- **로컬 실행**: 외부 서비스 의존성 없음
- **자동 임베딩**: OpenAI API 자동 연동
- **영속성**: 데이터 자동 저장 및 복구
- **메타데이터**: 풍부한 필터링 옵션
- **Python 친화적**: 간단한 API

### 주의사항
- 대용량 데이터 시 메모리 사용량 증가
- 임베딩 모델 변경 시 재인덱싱 필요
- 동시 접근 시 락 메커니즘 고려

## 다음 단계
Lab 2를 완료하면 [Lab 3 - 기본 RAG 시스템](../lab03-rag/README.md)로 진행합니다.

이번 실습에서 구축한 벡터 데이터베이스를 활용하여 질문-답변이 가능한 RAG(Retrieval-Augmented Generation) 시스템을 구현해보겠습니다. 