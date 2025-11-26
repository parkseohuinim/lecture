"""
Lab04 공통 데이터 및 유틸리티
- 고객센터/개발/기획 부서별 지식 베이스
- 샘플 질문 데이터
"""

# ============================================================================
# 부서별 지식 베이스 (RAG용 문서)
# ============================================================================

CUSTOMER_SERVICE_DOCS = """
# 고객센터 업무 가이드

## 1. 환불 정책

### 일반 환불 규정
- 구매 후 7일 이내: 전액 환불 가능
- 구매 후 7~14일: 제품 상태 확인 후 90% 환불
- 구매 후 14일 이후: 환불 불가, 교환만 가능
- 디지털 상품: 다운로드 전에만 환불 가능

### 환불 절차
1. 고객이 환불 요청서 작성
2. 상담원이 구매 이력 확인
3. 제품 상태 검수 (실물 제품의 경우)
4. 환불 승인 후 3~5 영업일 내 처리
5. 환불 완료 알림 발송

### 예외 사항
- 맞춤 제작 상품은 환불 불가
- 개봉된 소프트웨어는 환불 불가
- 파손된 상품은 책임 소재 확인 후 처리

## 2. 배송 안내

### 배송 기간
- 일반 배송: 주문 후 2~3일
- 빠른 배송: 주문 후 1일 (추가 요금 3,000원)
- 새벽 배송: 전날 오후 6시까지 주문 시 익일 오전 7시 전 도착

### 배송 조회
- 마이페이지 > 주문내역에서 실시간 조회 가능
- 송장번호 발급 후 배송업체 홈페이지에서도 조회 가능

### 배송 문제 해결
- 배송 지연: 3일 이상 지연 시 보상 쿠폰 지급
- 오배송: 무료 수거 후 재배송, 사과 쿠폰 지급
- 파손 배송: 사진 증빙 후 즉시 재발송

## 3. 회원 등급 및 혜택

### 등급 기준
- 브론즈: 가입 즉시
- 실버: 최근 6개월 구매금액 10만원 이상
- 골드: 최근 6개월 구매금액 30만원 이상
- VIP: 최근 6개월 구매금액 100만원 이상

### 등급별 혜택
- 브론즈: 기본 적립 1%
- 실버: 적립 2% + 무료배송 쿠폰 월 1회
- 골드: 적립 3% + 무료배송 + 생일 쿠폰
- VIP: 적립 5% + 전 혜택 + 전용 상담원 연결

## 4. 자주 묻는 질문 (FAQ)

### 결제 관련
Q: 결제가 실패했는데 어떻게 해야 하나요?
A: 카드 한도, 유효기간, 비밀번호를 확인해 주세요. 문제가 지속되면 카드사에 문의하시거나 다른 결제 수단을 이용해 주세요.

Q: 현금영수증 발급은 어떻게 하나요?
A: 결제 시 현금영수증 발급 옵션을 선택하시면 됩니다. 미발급 시 마이페이지에서 7일 이내 신청 가능합니다.

### 상품 관련
Q: 품절 상품은 언제 입고되나요?
A: 상품 페이지에서 재입고 알림 신청을 하시면 입고 시 알림을 드립니다. 통상 2~4주 소요됩니다.

Q: 상품 교환은 어떻게 하나요?
A: 마이페이지 > 주문내역 > 교환 신청 버튼을 클릭하세요. 무료 수거 후 새 상품을 보내드립니다.
"""

DEVELOPMENT_DOCS = """
# 개발팀 기술 가이드

## 1. 개발 환경 설정

### Python 환경
- Python 3.9 이상 권장
- 가상환경 필수 사용: venv 또는 conda
- 의존성 관리: requirements.txt 또는 pyproject.toml

### 필수 도구
- Git: 버전 관리
- Docker: 컨테이너화
- VS Code 또는 PyCharm: IDE
- Postman: API 테스트

### 코드 품질 도구
- Black: 코드 포맷터
- Flake8: 린터
- MyPy: 타입 체커
- Pytest: 테스트 프레임워크

## 2. API 개발 가이드

### RESTful API 설계 원칙
- 명사 사용: /users (O), /getUsers (X)
- HTTP 메서드 활용: GET(조회), POST(생성), PUT(수정), DELETE(삭제)
- 상태 코드: 200(성공), 201(생성), 400(잘못된 요청), 404(없음), 500(서버 오류)

### API 문서화
- OpenAPI(Swagger) 스펙 사용
- 요청/응답 예제 필수 포함
- 에러 응답 형식 통일

### 인증 방식
- JWT (JSON Web Token) 사용
- Access Token: 1시간 유효
- Refresh Token: 7일 유효
- 토큰은 Authorization 헤더에 Bearer 형식으로 전달

## 3. 데이터베이스

### PostgreSQL 사용 가이드
- 네이밍 컨벤션: snake_case 사용
- 기본키: id (auto increment 또는 UUID)
- 생성/수정 시간: created_at, updated_at 필수

### 인덱스 전략
- 자주 검색되는 컬럼에 인덱스 추가
- 복합 인덱스는 조회 순서 고려
- EXPLAIN ANALYZE로 쿼리 성능 분석

### 마이그레이션
- Alembic 사용 권장
- 마이그레이션 파일은 Git에 커밋
- 롤백 스크립트 필수 작성

## 4. 배포 프로세스

### CI/CD 파이프라인
1. 코드 푸시 > 자동 테스트 실행
2. 테스트 통과 > Docker 이미지 빌드
3. 이미지 빌드 > 스테이징 환경 배포
4. QA 테스트 > 프로덕션 배포 승인
5. 프로덕션 배포 > 모니터링

### 환경 분리
- development: 개발자 로컬 환경
- staging: 테스트 및 QA 환경
- production: 실제 서비스 환경

### 롤백 절차
1. 문제 발견 시 즉시 이전 버전으로 롤백
2. 롤백 후 원인 분석
3. 핫픽스 개발 및 테스트
4. 핫픽스 배포

## 5. 코딩 컨벤션

### Python 스타일
- PEP 8 준수
- 함수/변수: snake_case
- 클래스: PascalCase
- 상수: UPPER_SNAKE_CASE

### 문서화
- 모든 공개 함수/클래스에 docstring 작성
- 복잡한 로직에 주석 추가
- README.md 작성 필수

### Git 커밋 메시지
- feat: 새로운 기능
- fix: 버그 수정
- docs: 문서 수정
- refactor: 리팩토링
- test: 테스트 추가

## 6. 트러블슈팅

### 자주 발생하는 에러
Q: ModuleNotFoundError가 발생해요
A: 가상환경 활성화 확인, pip install -r requirements.txt 실행

Q: 데이터베이스 연결이 안 돼요
A: 환경변수 DATABASE_URL 확인, PostgreSQL 서비스 상태 확인

Q: API 응답이 느려요
A: N+1 쿼리 확인, 캐시 적용 검토, 인덱스 추가 고려

Q: Docker 빌드가 실패해요
A: Dockerfile 문법 확인, 베이스 이미지 버전 확인, 캐시 초기화 후 재시도
"""

PLANNING_DOCS = """
# 기획팀 업무 가이드

## 1. 프로젝트 기획 프로세스

### 기획 단계
1. **아이디어 수집**: 브레인스토밍, 시장 조사, 고객 피드백
2. **타당성 검토**: 기술적 가능성, 비용 대비 효과, 시장성
3. **기획서 작성**: 목표, 범위, 일정, 리소스 정의
4. **이해관계자 리뷰**: 관련 부서 피드백 수렴
5. **최종 승인**: 경영진 승인 및 킥오프

### 기획서 구성 요소
- 프로젝트 개요 및 목표
- 문제 정의 및 해결 방안
- 주요 기능 및 범위
- 일정 및 마일스톤
- 필요 리소스 (인력, 예산)
- 리스크 및 대응 방안
- 성공 지표 (KPI)

## 2. 요구사항 정의

### 요구사항 수집 방법
- 사용자 인터뷰
- 설문 조사
- 경쟁사 분석
- 데이터 분석
- 워크숍

### 요구사항 문서화
- User Story 형식: "As a [사용자], I want to [행동] so that [가치]"
- 기능 요구사항: 시스템이 해야 하는 것
- 비기능 요구사항: 성능, 보안, 사용성

### 우선순위 결정
- MoSCoW 방법론
  - Must have: 필수 기능
  - Should have: 중요하지만 필수는 아님
  - Could have: 있으면 좋은 기능
  - Won't have: 이번에는 제외

## 3. 일정 관리

### 프로젝트 일정 수립
- WBS (Work Breakdown Structure) 작성
- 각 작업의 소요 시간 추정
- 의존성 파악 및 크리티컬 패스 식별
- 버퍼 시간 포함

### 일정 도구
- Jira: 애자일 프로젝트 관리
- Asana: 태스크 관리
- Notion: 문서 및 일정 통합 관리
- Gantt 차트: 전체 일정 시각화

### 일정 지연 대응
1. 지연 원인 파악
2. 영향도 분석
3. 일정 조정 또는 범위 축소 검토
4. 이해관계자 커뮤니케이션
5. 재발 방지책 수립

## 4. 스프린트 운영

### 스프린트 계획
- 스프린트 기간: 2주 권장
- 스프린트 목표 설정
- 백로그에서 작업 선정
- 스토리 포인트 추정

### 데일리 스탠드업
- 매일 15분 내외
- 어제 한 일, 오늘 할 일, 블로커 공유

### 스프린트 회고
- 잘된 점 (Keep)
- 개선할 점 (Problem)
- 시도할 것 (Try)

## 5. 출시 관리

### 출시 체크리스트
- 기능 테스트 완료
- 성능 테스트 통과
- 보안 검토 완료
- 문서화 완료
- 마케팅 자료 준비
- CS 팀 교육 완료

### 출시 후 모니터링
- 실시간 에러 모니터링
- 사용자 피드백 수집
- 핵심 지표 추적
- 긴급 대응 체계 가동

## 6. KPI 및 성과 측정

### 주요 KPI
- DAU/MAU: 일간/월간 활성 사용자
- Retention Rate: 재방문율
- Conversion Rate: 전환율
- Customer Satisfaction: 고객 만족도
- NPS: 순추천지수

### 성과 보고
- 주간 보고: 진행 상황, 이슈, 다음 주 계획
- 월간 보고: KPI 달성 현황, 인사이트
- 분기 보고: 목표 대비 성과, 전략 조정
"""

# ============================================================================
# 카테고리 정의
# ============================================================================

CATEGORIES = {
    "customer_service": {
        "name": "고객센터",
        "keywords": ["환불", "배송", "교환", "결제", "회원", "등급", "쿠폰", "주문", "취소", "반품", "AS", "고객", "상담"],
        "description": "환불, 배송, 회원 등급 등 고객 서비스 관련 문의"
    },
    "development": {
        "name": "개발팀",
        "keywords": ["API", "코드", "개발", "배포", "데이터베이스", "서버", "에러", "버그", "테스트", "Git", "Docker", "Python"],
        "description": "API, 코드, 배포, 에러 등 개발 관련 문의"
    },
    "planning": {
        "name": "기획팀",
        "keywords": ["기획", "일정", "요구사항", "스프린트", "출시", "KPI", "프로젝트", "마일스톤", "백로그", "회의", "보고"],
        "description": "프로젝트 기획, 일정, 요구사항 등 기획 관련 문의"
    }
}

# ============================================================================
# 테스트용 샘플 질문
# ============================================================================

SAMPLE_QUESTIONS = [
    # 고객센터 질문
    {
        "question": "구매한 지 10일 됐는데 환불 가능한가요?",
        "expected_category": "customer_service",
        "expected_intent": "refund_inquiry"
    },
    {
        "question": "VIP 등급이 되려면 얼마나 구매해야 하나요?",
        "expected_category": "customer_service",
        "expected_intent": "membership_inquiry"
    },
    {
        "question": "배송이 3일째 안 오는데 어떻게 확인하나요?",
        "expected_category": "customer_service", 
        "expected_intent": "delivery_inquiry"
    },
    
    # 개발팀 질문
    {
        "question": "API 인증은 어떤 방식을 사용하나요?",
        "expected_category": "development",
        "expected_intent": "technical_inquiry"
    },
    {
        "question": "데이터베이스 마이그레이션은 어떻게 하나요?",
        "expected_category": "development",
        "expected_intent": "technical_inquiry"
    },
    {
        "question": "ModuleNotFoundError 에러가 계속 발생해요",
        "expected_category": "development",
        "expected_intent": "troubleshooting"
    },
    
    # 기획팀 질문
    {
        "question": "새 프로젝트 기획서에 뭘 포함해야 하나요?",
        "expected_category": "planning",
        "expected_intent": "process_inquiry"
    },
    {
        "question": "스프린트 회고는 어떻게 진행하나요?",
        "expected_category": "planning",
        "expected_intent": "process_inquiry"
    },
    {
        "question": "프로젝트 KPI는 어떤 것들을 추적해야 하나요?",
        "expected_category": "planning",
        "expected_intent": "kpi_inquiry"
    }
]

# ============================================================================
# 문서 합치기 함수
# ============================================================================

def get_all_documents() -> str:
    """모든 부서 문서를 하나로 합침"""
    return f"{CUSTOMER_SERVICE_DOCS}\n\n{DEVELOPMENT_DOCS}\n\n{PLANNING_DOCS}"


def get_document_by_category(category: str) -> str:
    """카테고리별 문서 반환"""
    docs = {
        "customer_service": CUSTOMER_SERVICE_DOCS,
        "development": DEVELOPMENT_DOCS,
        "planning": PLANNING_DOCS
    }
    return docs.get(category, "")


def get_category_info(category: str) -> dict:
    """카테고리 정보 반환"""
    return CATEGORIES.get(category, {})

