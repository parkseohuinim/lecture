# Cross-Encoder 파인튜닝 예제

## 배포된 모델
허깅페이스에 배포된 파인튜닝 모델: https://huggingface.co/your-username/your-model-name

## 개요
Vue Todo 앱의 코드베이스에 대한 질문-답변 쌍으로 Cross-Encoder 리랭커 모델을 파인튜닝한 예제입니다.

## 학습 데이터
Vue.js 프로젝트의 다양한 컴포넌트와 관련된 질문-답변 쌍으로 구성되어 있습니다.

### 데이터 구조
```json
[
  {
    "fragment_type": "component",
    "fragment_path": "src/components/todo/TodoItem.vue",
    "fragment_summary": "할일 항목을 표시하는 컴포넌트로, 체크박스, 우선순위 표시, 카테고리, 마감일 정보를 포함",
    "questions": [
      "Vue Todo 앱에서 각 할일 항목(TodoItem)의 우선순위를 시각적으로 어떻게 구분하나요?",
      "할일 항목의 기한이 지났을 때 어떻게 표시되나요?",
      "체크박스를 클릭해서 완료 처리한 할일은 UI에서 어떻게 변화가 있나요?"
    ]
  },
  {
    "fragment_type": "template",
    "fragment_path": "src/components/todo/TodoForm.vue",
    "fragment_summary": "할일 추가/수정을 위한 입력 폼으로 제목, 설명, 우선순위, 카테고리, 마감일 필드 제공",
    "questions": [
      "할일 추가 폼에서 사용자가 입력해야 하는 필수 필드는 무엇인가요?",
      "할일 수정 모드와 생성 모드에서 폼의 제출 버튼 텍스트가 어떻게 다른가요?",
      "TodoForm 컴포넌트에서 사용자 입력 값의 유효성을 어떻게 검사하나요?"
    ]
  },
  {
    "fragment_type": "script",
    "fragment_path": "src/store/modules/todos.js",
    "fragment_summary": "할일 관리를 위한 Vuex 스토어 모듈로 할일 추가, 수정, 삭제, 상태 관리 등의 기능 제공",
    "questions": [
      "할일 목록을 어디에 저장하고 있으며, 앱 재시작 시 데이터가 유지되는 방식은 무엇인가요?",
      "완료된 할일만 필터링하는 기능은 어떤 함수를 통해 구현되었나요?",
      "새 할일을 생성할 때 자동으로 부여되는 기본값(default values)에는 어떤 것들이 있나요?"
    ]
  },
  {
    "fragment_type": "style",
    "fragment_path": "src/components/todo/TodoItem.vue",
    "fragment_summary": "할일 항목의 스타일로 우선순위별 색상 구분, 완료 항목의 시각적 표현 등 정의",
    "questions": [
      "할일 항목의 우선순위(높음, 중간, 낮음)는 CSS에서 어떻게 시각적으로 구분하나요?",
      "완료된 할일 항목의 배경색은 어떻게 스타일링되어 있나요?",
      "TodoItem 컴포넌트에서 호버 효과는 어떻게 구현되어 있나요?"
    ]
  },
  {
    "fragment_type": "script",
    "fragment_path": "src/components/home/RecentTodosCard.vue",
    "fragment_summary": "홈 화면에 표시되는 최근 할일 목록 카드 컴포넌트의 로직",
    "questions": [
      "최근 할일 카드에서 날짜 포맷은 어떤 방식으로 처리되나요?",
      "기한이 지난 할일은 RecentTodosCard에서 어떻게 감지하고 표시하나요?",
      "할일의 우선순위별 색상 클래스는 어떻게 관리되고 있나요?"
    ]
  },
  {
    "fragment_type": "script",
    "fragment_path": "src/views/TodoStats.vue",
    "fragment_summary": "할일 통계 페이지로 Chart.js를 사용한 다양한 통계 차트 제공",
    "questions": [
      "완료율을 계산하는 공식은 무엇이며 어떤 Vue 계산된 속성에 정의되어 있나요?",
      "Chart.js로 구현된 차트 종류는 무엇이며 각각 어떤 통계를 보여주나요?",
      "주간 완료 추이 차트의 데이터는 어떻게 생성하고 가공하나요?"
    ]
  },
  {
    "fragment_type": "template",
    "fragment_path": "src/views/Settings.vue",
    "fragment_summary": "앱 설정 화면으로 다크모드 전환, 카테고리 관리, 데이터 가져오기/내보내기 기능 제공",
    "questions": [
      "다크 모드 토글 스위치는 어떤 HTML/CSS 요소로 구현되어 있나요?",
      "카테고리 추가 및 관리 인터페이스는 어떻게 구성되어 있나요?",
      "데이터 내보내기 기능은 어떤 파일 형식을 지원하며 어떻게 구현되어 있나요?"
    ]
  },
  {
    "fragment_type": "script",
    "fragment_path": "src/store/modules/categories.js",
    "fragment_summary": "카테고리 관리를 위한 Vuex 스토어 모듈",
    "questions": [
      "기본 카테고리('일반')는 어떤 특별한 제약 조건이 있나요?",
      "카테고리를 삭제할 때 해당 카테고리를 사용하고 있는 할일은 어떻게 처리되나요?",
      "새로운 카테고리 추가 시 중복 검사는 어떻게 이루어지나요?"
    ]
  },
  {
    "fragment_type": "component",
    "fragment_path": "src/components/icons/BarChartIcon.vue",
    "fragment_summary": "차트 아이콘 SVG 컴포넌트",
    "questions": [
      "SVG 요소의 viewBox 속성값은 무엇인가요?",
      "아이콘의 stroke 속성은 어떤 값으로 설정되어 있나요?",
      "SVG 패스의 stroke-linecap 속성값은 무엇인가요?",
      "SVG 패스의 stroke-width 값은 얼마인가요?",
      "이 컴포넌트의 name 속성은 정확히 어떻게 정의되어 있나요?"
    ]
  },
  {
    "fragment_type": "script",
    "fragment_path": "src/views/Home.vue",
    "fragment_summary": "홈 화면의 데이터 처리 및 컴포넌트 로직",
    "questions": [
      "HomeStatsCard 컴포넌트에 전달되는 props의 이름은 무엇인가요?",
      "최근 할일 목록을 생성하는 정확한 계산식은 어떻게 구현되어 있나요?",
      "홈 화면에서 created 라이프사이클 훅에서 호출하는 Vuex 액션은 무엇인가요?",
      "features 배열에 정의된 아이콘 이름 세 가지는 무엇인가요?",
      "stats 계산 속성에서 overdue 값을 가져오는 방식은 무엇인가요?"
    ]
  }
]
```

## 학습 방법

### 사용 스크립트
`train_cross_encoder.py` 스크립트를 사용하여 파인튜닝을 진행했습니다.

### 베이스 모델
- `dragonkue/bge-reranker-v2-m3-ko` (한국어 최적화 리랭커 모델)

### 학습 설정
![Fine-tuning Hyperparameters](./Fine-tuning%20Hyperparameters.png)


### 데이터 생성 방식
1. 각 질문에 대해 정답 fragment를 긍정 샘플로 사용 (레이블: 1)
2. 무작위로 선택된 다른 fragment를 부정 샘플로 사용 (레이블: 0)
3. 긍정:부정 비율을 1:3으로 설정하여 불균형 방지

## 사용 예시

학습된 모델은 질문과 코드 fragment 간의 관련성을 점수화하여, RAG 시스템에서 더 정확한 문서 검색을 가능하게 합니다.

```python
from app.embedding.cross_encoder import CrossEncoder

# 모델 로드
cross_encoder = CrossEncoder(model_name="./trained_model")

# 재랭킹
ranked_results = cross_encoder.rerank(
    query="할일 항목의 우선순위를 어떻게 표시하나요?",
    documents=all_fragments,
    top_k=3
)
```