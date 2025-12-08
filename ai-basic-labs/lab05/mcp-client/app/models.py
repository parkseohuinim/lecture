"""Pydantic models for API requests and responses"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class HealthResponse(BaseModel):
    """Response model for health check endpoint"""
    status: str = Field(..., description="Health status")
    mcp_connected: bool = Field(..., description="Whether MCP client is connected")
    tools_available: int = Field(..., description="Number of available tools")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")


class RagConfig(BaseModel):
    """RAG 시스템 설정 모델"""
    separators: List[str] = Field(
        default=["\n\n", "\n", " ", ""],
        description="텍스트 분할을 위한 구분자 리스트"
    )
    chunk_size: int = Field(
        default=1000,
        description="청크 크기 (문자 수)"
    )
    chunk_overlap: int = Field(
        default=200,
        description="청크 간 중복 크기 (문자 수)"
    )
    document_type: str = Field(
        default="confluence_page",
        description="문서 타입"
    )
    strategy: Optional[str] = Field(
        default=None,
        description="RAG 전략 (table_aware, sentence_aware, balanced)"
    )


class DocumentAnalysis(BaseModel):
    """문서 구조 분석 결과 모델"""
    total_length: int = Field(..., description="총 문자 수")
    total_lines: int = Field(..., description="총 라인 수")
    has_headers: bool = Field(..., description="헤더 존재 여부")
    has_horizontal_rules: bool = Field(..., description="수평선 존재 여부")
    has_lists: bool = Field(..., description="리스트 존재 여부")
    has_tables: bool = Field(..., description="테이블 존재 여부")
    paragraph_count: int = Field(..., description="단락 개수")
    avg_paragraph_length: int = Field(..., description="평균 단락 길이")
    empty_line_count: int = Field(..., description="빈 줄 개수")
    double_newline_count: int = Field(..., description="이중 개행 개수")


class ProcessingMetadata(BaseModel):
    """처리 메타데이터 모델"""
    processed_at: str = Field(..., description="처리 시간 (ISO 8601)")
    html_size: int = Field(..., description="원본 HTML 크기 (bytes)")
    markdown_size: int = Field(..., description="변환된 Markdown 크기 (characters)")
    tools_used: List[str] = Field(default=[], description="사용된 MCP 도구 목록")
    document_analysis: Optional[DocumentAnalysis] = Field(None, description="문서 구조 분석 결과")
    rag_content_marker: str = Field(
        default="<!-- RAG_CONTENT_START -->",
        description="RAG 콘텐츠 시작 지점을 표시하는 마커 (이 마커 이후부터 실제 RAG 인덱싱 대상)"
    )


class NavigationItem(BaseModel):
    """네비게이션 메뉴 아이템 모델"""
    page_id: str = Field(..., description="페이지 ID")
    title: str = Field(..., description="페이지 제목")
    url: Optional[str] = Field(None, description="페이지 URL")
    level: int = Field(..., description="계층 깊이 (0=root, 1=child, 2=grandchild...)")
    has_children: bool = Field(default=False, description="하위 페이지 존재 여부")
    is_expanded: bool = Field(default=False, description="펼쳐진 상태 여부")


class NavigationMenu(BaseModel):
    """네비게이션 메뉴 구조 모델"""
    current_page_id: Optional[str] = Field(None, description="현재 페이지 ID")
    parent_page_id: Optional[str] = Field(None, description="부모 페이지 ID")
    root_pages: List[NavigationItem] = Field(default=[], description="최상위 페이지 목록")
    all_pages: List[NavigationItem] = Field(default=[], description="전체 페이지 목록 (계층 구조 포함)")
