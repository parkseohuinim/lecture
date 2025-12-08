# MCP Lab05 Web UI

NextJS + TypeScript + Tailwind CSS로 만든 채팅 인터페이스

## 실행 방법

```bash
# 1. 의존성 설치
npm install

# 2. 개발 서버 실행
npm run dev

# 3. 브라우저에서 접속
# http://localhost:3000
```

## 기능

- ✅ 실시간 채팅 인터페이스
- ✅ HTML 파일 업로드 (선택사항)
- ✅ LLM 의도 분석 자동화
- ✅ 다크 모드 지원
- ✅ 반응형 디자인

## 사용 예시

### 일반 대화
```
사용자: 안녕하세요
AI: 안녕하세요! 무엇을 도와드릴까요?
(도구 호출 없음)
```

### HTML 파일 분석
```
사용자: (sample1.html 업로드) 이 HTML 파일의 내용을 추출해줘
AI: [HTML → Markdown 변환 결과]
(ari_html_to_markdown 도구 호출됨)
```
