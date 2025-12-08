# Lab05: MCP (Model Context Protocol) 실습

## 📚 학습 목표

이 실습에서는 MCP(Model Context Protocol)를 활용한 웹 크롤링 및 HTML 처리 시스템을 구축합니다.

### 핵심 학습 내용
1. **MCP 프로토콜**: AI 도구 간 표준화된 통신 방법
2. **마이크로서비스 아키텍처**: 클라이언트-서버 분리 구조
3. **지능형 HTML 처리**: HTML → Markdown → 구조화된 JSON 변환
4. **LLM 통합**: OpenAI API와 MCP 도구 연동

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────┐
│   Web UI (NextJS) - Port 3000      │
│   - 채팅 인터페이스                   │
│   - 파일 업로드 (선택)                │
│   - 실시간 대화                       │
└─────────────────────────────────────┘
              ↓ HTTP
┌─────────────────────────────────────┐
│   MCP Client (FastAPI) - Port 8000 │
│                                     │
│  ┌─────────────┬─────────────────┐  │
│  │ Chat API    │   LLM Service   │  │
│  │ (의도 분석)  │   (OpenAI API)  │  │
│  └─────────────┴─────────────────┘  │
│          MCP Service                │
│      (HTTP 통신 관리)                │
└─────────────────────────────────────┘
              ↓ HTTP
┌─────────────────────────────────────┐
│  MCP Server (FastMCP) - Port 4200  │
│                                     │
│  ┌─────────────┬─────────────────┐  │
│  │ HTML Parser │ Navigation      │  │
│  │(trafilatura)│   Extractor     │  │
│  └─────────────┴─────────────────┘  │
│     HTML → Markdown 변환 도구        │
└─────────────────────────────────────┘
```

## 📁 디렉토리 구조

```
lab05/
├── README.md                    # 이 파일
├── requirements.txt             # 패키지 의존성
├── sample1.html                # 테스트용 HTML 파일
├── sample2.html                # 테스트용 HTML 파일
├── sample3.html                # 테스트용 HTML 파일
│
├── mcp-server/                 # MCP 서버 (FastMCP)
│   └── server.py              # MCP 도구 서버
│
└── mcp-client/                 # MCP 클라이언트 (FastAPI)
    ├── main.py                # FastAPI 애플리케이션
    └── app/
        ├── config.py          # 설정 관리
        ├── models.py          # 데이터 모델
        ├── routers/
        │   └── api.py         # REST API 엔드포인트
        ├── application/
        │   └── ari/
        │       └── ari_service.py  # HTML 처리 서비스
        ├── infrastructure/
        │   ├── mcp/
        │   │   └── mcp_service.py  # MCP 클라이언트 서비스
        │   └── llm/
        │       └── llm_service.py  # OpenAI LLM 서비스
        ├── core/
        │   └── logging.py     # 로깅 설정
        ├── exceptions/
        │   └── base.py        # 예외 클래스
        └── utils/
            └── schema_converter.py  # 스키마 변환 도구
```

## 🎨 새로운 기능: 웹 채팅 UI

**NextJS + TypeScript + Tailwind CSS로 만든 세련된 채팅 인터페이스**

```bash
# Web UI 실행 (터미널 3)
cd web-ui
npm install
npm run dev
# → http://localhost:3000
```

**채팅 UI 특징:**
- ✅ 실시간 대화형 인터페이스
- ✅ HTML 파일 드래그 앤 드롭 업로드
- ✅ LLM 의도 분석 자동화
  - 일반 대화: "안녕하세요" → 도구 호출 없이 직접 답변
  - HTML 처리: "이 파일 내용 추출해줘" → MCP 도구 자동 호출
- ✅ 다크 모드 지원
- ✅ 반응형 디자인

## 🚀 실습 단계

### Step 1: 환경 설정 (10분)

#### 1.1 환경 변수 설정

프로젝트 루트(ai-basic-labs/)에 `.env` 파일이 있는지 확인하고, 없다면 생성:

```bash
# ai-basic-labs/.env
OPENAI_API_KEY=your_actual_openai_api_key_here
OPENAI_MODEL=gpt-4o

# MCP 서버 설정 (기본값 사용 가능)
MCP_SERVER_URL=http://127.0.0.1:4200/my-custom-path/
MCP_CONNECTION_TIMEOUT=30
MCP_RETRY_ATTEMPTS=3

# 서버 설정 (기본값 사용 가능)
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
```

#### 1.2 패키지 설치

```bash
cd ai-basic-labs/lab05

# 가상환경이 활성화되어 있는지 확인
# 없다면: python -m venv .venv && source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### Step 2: MCP 서버 실행 (5분)

MCP 서버는 HTML 처리 도구들을 제공합니다.

```bash
# 터미널 1: MCP 서버 시작
cd ai-basic-labs/lab05/mcp-server
python server.py

# 출력 예시:
# 🚀 ARI Processing MCP Server 시작 중...
# 📍 서버 주소: http://0.0.0.0:4200/my-custom-path
# 🔧 사용 가능한 도구: health_check, ari_parse_html, ari_html_to_markdown, ari_markdown_to_json
```

**MCP 서버가 제공하는 도구들:**
- `health_check`: 서버 상태 확인
- `ari_parse_html`: HTML 파싱 및 메타데이터 추출
- `ari_html_to_markdown`: HTML을 마크다운으로 변환
- `ari_markdown_to_json`: 마크다운을 구조화된 JSON으로 변환

### Step 3: MCP 클라이언트 실행 (5분)

MCP 클라이언트는 FastAPI 기반 REST API를 제공합니다.

```bash
# 터미널 2: MCP 클라이언트 시작
cd ai-basic-labs/lab05/mcp-client
python main.py

# 출력 예시:
# INFO:     Started server process
# INFO:     Uvicorn running on http://0.0.0.0:8000
# MCP Client connected to http://127.0.0.1:4200/my-custom-path/
```

### Step 4: 웹 UI 실행 (권장) (10분)

```bash
# 터미널 3: NextJS 웹 UI 실행
cd web-ui
npm install  # 처음 한 번만
npm run dev

# 브라우저에서 접속
open http://localhost:3000
```

**웹 UI 테스트:**
1. **일반 대화 테스트**:
   - 메시지: "안녕하세요"
   - 예상: AI가 인사 답변 (도구 호출 없음)

2. **HTML 파일 분석 테스트**:
   - 파일 업로드 버튼 클릭 → sample1.html 선택
   - 메시지: "이 HTML 파일의 내용을 추출해줘"
   - 예상: Markdown 변환 결과 (ari_html_to_markdown 호출됨)
   - 로그에서 "사용된 도구: ari_html_to_markdown" 확인

3. **일반 질문 (파일 있음)**:
   - 파일 업로드: sample1.html
   - 메시지: "HTML이란 무엇인가요?"
   - 예상: AI가 직접 설명 (도구 호출 없음)

### Step 5: Swagger API 테스트 (선택사항)

#### 4.1 Swagger UI로 테스트 (권장)

브라우저에서 Swagger UI 열기:
```
http://localhost:8000/docs
```

**제공하는 API:**
1. **GET `/health`**: 서버 상태 확인
2. **POST `/api/llm/query-with-html`**: HTML 파일 처리 후 Frontmatter Markdown 다운로드 (핵심 실습 API)
   - **개선사항**: LLM이 자연어 질문을 분석하여 자동으로 도구 선택
   - **출력**: Frontmatter 방식 Markdown 파일
     - RAG 설정 자동 추천 (chunk_size, separators 등)
     - 문서 구조 분석 결과 포함
     - `<!-- RAG_CONTENT_START -->` 마커로 실제 콘텐츠 구분

#### 4.2 실습 예제

**예제 1: 건강 체크**
```bash
curl http://localhost:8000/health
```

**예제 2: HTML 파일 처리 (Frontmatter 방식)**
```bash
# 추출 요청 - 도구 자동 호출 + RAG 설정 포함
curl -X POST "http://localhost:8000/api/llm/query-with-html" \
  -H "Content-Type: multipart/form-data" \
  -F "question=이 HTML 파일의 내용을 추출해줘" \
  -F "files=@sample1.html" \
  --output "content_frontmatter.md"

# 다운로드된 파일 구조:
# ---json
# {
#   "rag_config": {
#     "separators": ["\n\n", "\n", " "],
#     "chunk_size": 2000,
#     "chunk_overlap": 400
#   },
#   "metadata": {
#     "processed_at": "2025-12-08T...",
#     "document_analysis": {...}
#   }
# }
# ---
# 
# <!-- RAG_CONTENT_START -->
# 
# # 실제 마크다운 내용
# ...
```

## 💡 핵심 개념

### 1. MCP (Model Context Protocol)란?

AI 애플리케이션과 도구 간의 **표준화된 통신 프로토콜**입니다.

**MCP의 작동 흐름:**
```
1. MCP 서버 → 도구(tool) 목록 제공
2. MCP 클라이언트 → 도구 목록 조회
3. LLM → 사용자 질문 분석하여 적절한 도구 선택
4. MCP 클라이언트 → 선택된 도구 실행 요청
5. MCP 서버 → 도구 실행 후 결과 반환
```

**왜 MCP를 사용하나요?**
- ✅ 도구를 한 번 만들면 여러 AI 앱에서 재사용 가능
- ✅ 표준화된 방식으로 도구 통신
- ✅ 서비스를 독립적으로 확장 가능

### 2. 이 실습의 처리 흐름 (Frontmatter 방식)

```
사용자가 HTML 업로드 + 자연어 질문
         ↓
LLM이 질문 분석 (OpenAI API)
├─ "추출해줘", "요약해줘" 등 → HTML 처리 의도
│    ↓
│    MCP 도구 호출 (ari_html_to_markdown)
│    ↓
│    MCP 서버가 HTML → Markdown 변환
│    ↓
│    Markdown 구조 분석 (자동)
│    - 헤더, 테이블, 리스트 존재 여부 확인
│    - 평균 단락 길이 계산
│    - 최적의 RAG 설정 자동 추천
│    ↓
│    Frontmatter 생성
│    - RAG 설정 (chunk_size, separators)
│    - 문서 분석 결과
│    - 처리 메타데이터
│    ↓
│    Frontmatter + Markdown 결합
│    ↓
│    Markdown 파일 다운로드
│
└─ "HTML이란?", 일반 질문 등 → 일반 질문
     ↓
     LLM이 직접 답변 (도구 호출 없음)
     ↓
     텍스트 응답
```

**✨ 핵심 개선:**
1. ❌ 기존: 모든 요청에 강제로 도구 호출
   ✅ 개선: LLM이 의도를 분석하여 필요할 때만 도구 호출

2. ✅ **Frontmatter 방식**: RAG 설정이 파일 상단에 포함
   - 파일 하나로 모든 정보 완결
   - RAG 시스템에서 쉽게 파싱 가능
   - 사람이 읽기 쉬운 형태

### 3. 마이크로서비스 구조

**MCP Server (Port 4200)** - HTML 처리 전문
- `ari_html_to_markdown`: HTML을 깔끔한 Markdown으로 변환
- `ari_markdown_to_json`: Markdown을 구조화된 JSON으로 변환
- 독립적으로 실행 가능

**MCP Client (Port 8000)** - REST API 제공
- FastAPI 기반 웹 서버
- LLM이 자동으로 MCP 도구 선택
- 사용자는 Swagger UI로 쉽게 테스트

## 📝 실습 과제

### 기본 실습: LLM 의도 분석 테스트

1. **MCP 서버 실행**: `mcp-server/server.py` 실행
2. **MCP 클라이언트 실행**: `mcp-client/main.py` 실행
3. **Swagger UI 접속**: `http://localhost:8000/docs`
4. **API 테스트 1 - 추출 요청** (도구 호출됨):
   - `/api/llm/query-with-html` 선택
   - question: "이 HTML 파일의 내용을 추출해줘"
   - files: sample1.html 업로드
   - Execute 버튼 클릭
   - **Frontmatter Markdown 파일 다운로드 확인**
   - 로그에서 "ari_html_to_markdown 호출" 확인
   - 로그에서 "마크다운 구조 분석 중..." 확인
   - 로그에서 "RAG 설정 자동 추천 완료" 확인

5. **다운로드된 파일 구조 확인**:
   ```markdown
   ---json
   {
     "rag_config": {
       "separators": ["\n\n", "\n", " "],
       "chunk_size": 2000,
       "chunk_overlap": 400,
       "document_type": "confluence_page"
     },
     "metadata": {
       "processed_at": "2025-12-08T10:30:00",
       "html_size": 123456,
       "markdown_size": 45678,
       "tools_used": ["ari_html_to_markdown"],
       "document_analysis": {
         "total_length": 45678,
         "has_headers": true,
         "has_tables": true,
         ...
       }
     }
   }
   ---
   
   <!-- RAG_CONTENT_START -->
   
   # 실제 마크다운 내용
   ...
   ```

6. **API 테스트 2 - 일반 질문** (도구 호출 안됨):
   - question: "HTML이란 무엇인가요?"
   - files: sample1.html 업로드
   - Execute 버튼 클릭
   - LLM이 직접 답변하는지 확인
   - 로그에서 도구 호출 없음 확인

### 추가 실습: 의도 분석 다양하게 테스트

**도구가 호출되는 질문들:**
- "이 HTML 파일을 요약해줘"
- "내용을 정리해줘"
- "HTML을 마크다운으로 변환해줘"
- "파일의 핵심 내용만 추출해줘"

**도구가 호출되지 않는 질문들:**
- "HTML이란 무엇인가요?"
- "이 파일의 용도는 뭐야?"
- "오늘 날씨 어때?"

## 🔧 트러블슈팅

### 1. MCP 연결 실패
```bash
# MCP 서버 상태 확인
curl http://localhost:4200/my-custom-path/health

# 포트 사용 여부 확인
lsof -ti:4200
lsof -ti:8000
```

### 2. 패키지 의존성 오류
```bash
# 가상환경 재생성
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. OpenAI API 오류
```bash
# .env 파일 확인
cat ../../.env | grep OPENAI_API_KEY

# API 키가 유효한지 확인
```

## 📚 추가 학습 자료

- [MCP 공식 문서](https://modelcontextprotocol.io/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [FastMCP GitHub](https://github.com/jlowin/fastmcp)
- [BeautifulSoup 문서](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

## 🎯 학습 체크리스트

- [ ] MCP 프로토콜이 무엇인지 설명할 수 있다
- [ ] MCP 서버와 클라이언트를 각각 실행할 수 있다
- [ ] **LLM의 의도 분석 방식을 이해한다** ✨
  - "추출/요약" 키워드 → 도구 호출
  - 일반 질문 → 직접 답변
- [ ] HTML → Markdown 변환 과정을 안다
- [ ] **Frontmatter 방식을 이해한다** ✨
  - RAG 설정이 파일 상단에 포함
  - `<!-- RAG_CONTENT_START -->` 마커로 콘텐츠 구분
  - 파일 하나로 모든 정보 완결
- [ ] **Markdown 구조 자동 분석을 이해한다** ✨
  - 헤더, 테이블, 리스트 존재 여부 확인
  - 평균 단락 길이 계산
  - 최적의 chunk_size, separators 자동 추천
- [ ] Swagger UI로 API를 테스트할 수 있다
- [ ] Frontmatter Markdown 다운로드 결과를 확인할 수 있다
- [ ] **강제 tool calling과 의도 분석의 차이를 설명할 수 있다** ✨

## 💬 자주 묻는 질문

**Q: MCP를 왜 사용하나요?**  
A: AI 도구를 표준화하여 재사용성을 높이기 위함입니다. 한 번 만든 HTML 처리 도구를 다른 AI 프로젝트에서도 사용할 수 있습니다.

**Q: 강제 tool calling과 의도 분석의 차이는?**  
A:
- **강제 tool calling** (기존): 모든 요청에 무조건 도구 호출 → 비효율적
- **의도 분석** (개선): LLM이 질문을 분석하여 필요할 때만 도구 호출 → 효율적

**Q: LLM은 어떻게 의도를 판단하나요?**  
A: system prompt에 정의된 키워드("추출", "요약", "변환" 등)를 기반으로 사용자 질문을 분석합니다. 이런 키워드가 있으면 HTML 처리 의도로 판단하고, 없으면 일반 질문으로 처리합니다.

**Q: Frontmatter 방식이 뭔가요?**  
A: 파일 상단에 메타데이터를 포함하는 방식입니다. `---json ... ---` 사이에 RAG 설정과 문서 분석 결과가 JSON 형식으로 들어가고, `<!-- RAG_CONTENT_START -->` 마커 이후부터 실제 콘텐츠가 시작됩니다.

**Q: RAG 시스템에서 어떻게 사용하나요?**  
A: 
```python
# 1. 파일 읽기
with open('content_frontmatter.md', 'r') as f:
    content = f.read()

# 2. Frontmatter 파싱
_, frontmatter, rest = content.split('---', 2)
metadata = json.loads(frontmatter.replace('json', '', 1))
rag_config = metadata['rag_config']

# 3. RAG 콘텐츠만 추출
rag_content = rest.split('<!-- RAG_CONTENT_START -->', 1)[1].strip()

# 4. 추천된 설정으로 chunk 분할
chunk_size = rag_config['chunk_size']
separators = rag_config['separators']
# ... 임베딩 및 인덱싱
```

**Q: 실무에서 어떻게 활용하나요?**  
A: Confluence/SharePoint 문서 수집, 자동 문서 변환, RAG 시스템용 데이터 전처리 등에 사용할 수 있습니다.

---

**Happy Learning! 🚀**

