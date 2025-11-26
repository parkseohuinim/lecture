# Lab 01: NLP 기초 실습

## 학습 목표

이 실습을 통해 다음을 학습합니다:

1. **토큰화 (Tokenization)**: tiktoken을 사용한 토큰 수 계산
2. **텍스트 전처리**: NLTK를 사용한 불용어 제거, lemmatization
3. **임베딩 생성**: OpenAI API를 사용한 텍스트 임베딩
4. **유사도 계산**: 코사인 유사도를 통한 1:N, N:M 비교
5. **검색 엔진**: 간단한 의미 기반 문장 검색기 (RAG의 기초)

## 설치 및 실행

```bash
# 1. 프로젝트 루트에서 의존성 설치
cd ai-basic-labs
pip install -r requirements.txt

# 2. 환경변수 설정
cp .env.example .env
# .env 파일을 열어서 OPENAI_API_KEY 입력

# 3. 실습 실행
cd lab01
python nlp_basics.py
```

## 실행 방법

### 전체 데모 실행

```bash
python nlp_basics.py
```

### 개별 함수 실행

Python 인터프리터에서:

```python
from nlp_basics import *

# NLTK 데이터 다운로드 (최초 1회)
download_nltk_data()

# 1. 토큰 수 세기
demo_tiktoken()

# 2. 전처리 파이프라인
demo_preprocessing()

# 3. 임베딩 생성
demo_embeddings()

# 4. 유사도 계산
demo_similarity()

# 5. 검색 엔진
demo_search_engine()
```

## 주요 기능

### 1. tiktoken으로 토큰 수 세기

```python
from nlp_basics import count_tokens_with_tiktoken

text = "Hello, how are you?"
token_count = count_tokens_with_tiktoken(text)
print(f"토큰 수: {token_count}")
```

### 2. 텍스트 전처리

```python
from nlp_basics import TextPreprocessor

preprocessor = TextPreprocessor()
text = "The cats are running quickly through the gardens."
tokens = preprocessor.preprocess(text)
print(tokens)
# 출력: ['cat', 'running', 'quickly', 'garden']
```

### 3. 임베딩 생성

```python
from nlp_basics import EmbeddingGenerator

generator = EmbeddingGenerator()
embedding = generator.get_embedding("Hello world")
print(f"임베딩 차원: {len(embedding)}")
```

### 4. 코사인 유사도 계산

```python
from nlp_basics import cosine_similarity, one_to_many_similarity

# 두 벡터 간 유사도
similarity = cosine_similarity(vec1, vec2)

# 1:N 유사도
similarities = one_to_many_similarity(query_embedding, doc_embeddings)
```

### 5. 간단한 검색 엔진

```python
from nlp_basics import SimpleSearchEngine

# 검색 엔진 초기화
search_engine = SimpleSearchEngine()

# 문서 추가
documents = [
    "Python is a programming language.",
    "Machine learning is fascinating.",
    "I love cooking pasta."
]
search_engine.add_documents(documents)

# 검색
results = search_engine.search("Tell me about Python", top_k=2)
for doc, score in results:
    print(f"{score:.4f}: {doc}")
```

## 실습 내용 상세

### 1. tiktoken으로 토큰 수 세기

- OpenAI 모델이 사용하는 토큰화 방식 이해
- 다양한 언어(영어, 한글)의 토큰 수 비교
- 실제 토큰으로 어떻게 분리되는지 확인

### 2. NLTK 전처리 파이프라인

- **토큰화**: 문장을 단어로 분리
- **소문자 변환**: 대소문자 통일
- **불용어 제거**: 'the', 'is', 'are' 등 의미 없는 단어 제거
- **표제어 추출**: 'running' → 'run', 'cats' → 'cat'

### 3. OpenAI 임베딩 API

- `text-embedding-3-small` 모델 사용
- 단일 텍스트 임베딩 생성
- 배치 처리로 여러 텍스트 한 번에 처리
- 임베딩 벡터의 차원과 값 확인

### 4. 코사인 유사도

- **1:N 유사도**: 하나의 쿼리와 여러 문서 비교
- **N:M 유사도**: 여러 쿼리와 여러 문서의 유사도 행렬
- 가장 유사한 문서 찾기

### 5. 문장 검색 엔진 (RAG의 기초)

- 문서를 임베딩으로 변환하여 인덱싱
- 쿼리를 임베딩으로 변환
- 코사인 유사도로 가장 관련 있는 문서 검색
- Top-K 결과 반환

## 학습 포인트

1. **토큰화의 중요성**: API 비용과 모델 입력 크기 제한 이해
2. **전처리의 효과**: 노이즈 제거로 더 나은 분석 가능
3. **임베딩의 의미**: 텍스트를 벡터로 표현하여 수학적 연산 가능
4. **유사도 계산**: 의미적으로 유사한 텍스트 찾기
5. **RAG의 기초**: 검색 기반 생성 시스템의 핵심 원리

## 주의사항

- OpenAI API 키가 필요합니다
- API 호출 시 비용이 발생할 수 있습니다
- NLTK 데이터는 최초 1회 다운로드가 필요합니다
- 인터넷 연결이 필요합니다

## 다음 단계

이 실습을 완료한 후:

1. **Lab 02**: Vector Database (ChromaDB) 사용
2. **Lab 03**: RAG (Retrieval-Augmented Generation) 시스템 구축
3. **Lab 04**: 지능형 챗봇 개발

## 참고 자료

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [tiktoken Documentation](https://github.com/openai/tiktoken)
- [NLTK Documentation](https://www.nltk.org/)

