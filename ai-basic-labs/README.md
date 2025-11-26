# AI Basic Labs

AI 기초 실습 프로젝트 모음

## 🚀 빠른 시작

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경변수 설정
cp .env.example .env
# .env 파일을 열어서 OPENAI_API_KEY 입력

# 3. 실습 시작
cd lab01
python nlp_basics.py
```

## 📁 구조

```
ai-basic-labs/
├── .env.example       # 환경변수 예제
├── requirements.txt   # 공통 패키지
├── lab01/            # NLP 기초 실습
├── lab02/            # Vector Database 실습
└── lab03/            # RAG 시스템 실습
    ├── rag_basic.py                    # 기본 RAG
    ├── advanced_retrieval_langchain.py # 고급 RAG (하이브리드 검색, 리랭킹 등)
    └── test_advanced_rag.py           # 테스트 스크립트
```

## 📚 실습 내용

### Lab 01: NLP 기초
- 토큰화 및 텍스트 처리
- OpenAI API 사용법
- 프롬프트 엔지니어링

### Lab 02: Vector Database
- 임베딩 생성 및 저장
- 유사도 검색
- ChromaDB 활용

### Lab 03: RAG 시스템
**기본 RAG (rag_basic.py)**
- 문서 로딩 및 청킹
- Vector DB 인덱싱
- 검색 기반 답변 생성
- 컨텍스트 관리

**고급 RAG (advanced_retrieval_langchain.py)**
- ✨ Sparse + Dense 하이브리드 검색
- ✨ Re-ranking (BGE reranker)
- ✨ Multi-hop 질의 (두 단계 검색)
- ✨ Chunk size 실험 (512/1024/2048)
- ✨ 컨텍스트 윈도우 관리

## 🔑 API 키 설정

1. [OpenAI Platform](https://platform.openai.com/)에서 API 키 발급
2. `.env.example`을 `.env`로 복사
3. `OPENAI_API_KEY=your-key` 입력

