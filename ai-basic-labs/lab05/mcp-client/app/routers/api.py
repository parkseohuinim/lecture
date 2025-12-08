"""API routes for ARI Processing"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, Response
from fastapi.responses import FileResponse
from typing import List, Optional
import tempfile
import os
import logging
from datetime import datetime
import json
import yaml

from app.models import HealthResponse, RagConfig, ProcessingMetadata, DocumentAnalysis, NavigationMenu, NavigationItem
from app.infrastructure.mcp.mcp_service import mcp_service
from app.infrastructure.llm.llm_service import llm_service

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()

@router.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {"message": "ARI Processing Server is running"}

@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint"""
    try:
        health_data = await mcp_service.health_check()
        
        return HealthResponse(
            status="healthy" if health_data["connected"] else "unhealthy",
            mcp_connected=health_data["connected"],
            tools_available=health_data["tools_available"],
            details=health_data
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="error",
            mcp_connected=False,
            tools_available=0,
            details={"error": str(e)}
        )



def analyze_markdown_structure(markdown_content: str) -> dict:
    """
    마크다운 내용을 분석하여 최적의 RAG 설정을 추천
    
    Returns:
        dict: {
            'separators': List[str],
            'chunk_size': int,
            'chunk_overlap': int,
            'analysis': dict  # 분석 상세 정보
        }
    """
    lines = markdown_content.split('\n')
    total_length = len(markdown_content)
    
    # 패턴 분석
    has_headers = any(line.strip().startswith('#') for line in lines)
    has_horizontal_rules = any(line.strip() == '---' for line in lines)
    has_lists = any(line.strip().startswith(('-', '*', '+')) for line in lines)
    
    # 표 감지: 마크다운 테이블 형식 또는 "[표" 패턴
    has_markdown_tables = '|' in markdown_content and any('---' in line and '|' in line for line in lines)
    has_list_tables = any('[표' in line or '[Table' in line.lower() for line in lines)
    has_tables = has_markdown_tables or has_list_tables
    
    # 단락 구분 분석
    empty_line_count = sum(1 for line in lines if not line.strip())
    double_newline_count = markdown_content.count('\n\n')
    
    # 평균 단락 길이 계산
    paragraphs = [p.strip() for p in markdown_content.split('\n\n') if p.strip()]
    avg_paragraph_length = sum(len(p) for p in paragraphs) / len(paragraphs) if paragraphs else 0
    
    # Separator 우선순위 결정
    separators = []
    
    # 1. 헤더 기반 분할 (가장 큰 단위)
    if has_headers:
        # 헤더 레벨별로 분할
        separators.extend(["\n### ", "\n## ", "\n# "])
    
    # 2. 수평선 기반 분할
    if has_horizontal_rules:
        separators.append("\n---\n")
    
    # 3. 이중 개행 (단락 구분) - 항상 포함
    separators.append("\n\n")
    
    # 4. 단일 개행 - 항상 포함
    separators.append("\n")
    
    # 5. 문장 단위 분할 (표 데이터 등 긴 줄 대응)
    # 마침표, 물음표, 느낌표 뒤 공백으로 문장 구분
    separators.extend([". ", "? ", "! "])
    
    # 6. 쉼표/세미콜론 단위 분할 (더 세밀한 분할)
    separators.extend([", ", "; ", ": "])
    
    # 7. 공백은 조건부 추가 (표 데이터가 있을 때만)
    if has_tables:
        separators.append(" ")
    
    # 8. 빈 문자열은 제거 (과도한 분할 방지)
    # separators.append("")
    
    # Chunk Size 결정
    # 표 데이터가 있으면 더 큰 청크 사용
    if has_tables:
        # 표가 있으면 큰 청크 사용 (표가 잘리지 않도록)
        chunk_size = 3000
    elif avg_paragraph_length > 0:
        # 단락이 짧으면 여러 단락을 하나의 청크로
        if avg_paragraph_length < 300:
            chunk_size = 2000
        elif avg_paragraph_length < 600:
            chunk_size = 1500
        else:
            chunk_size = 1200
    else:
        chunk_size = 1500  # 기본값
    
    # Chunk Overlap 결정 (chunk_size의 15-20%, 최대 500)
    chunk_overlap = min(500, int(chunk_size * 0.2))
    
    analysis = {
        'total_length': total_length,
        'total_lines': len(lines),
        'has_headers': has_headers,
        'has_horizontal_rules': has_horizontal_rules,
        'has_lists': has_lists,
        'has_tables': has_tables,
        'paragraph_count': len(paragraphs),
        'avg_paragraph_length': int(avg_paragraph_length),
        'empty_line_count': empty_line_count,
        'double_newline_count': double_newline_count
    }
    
    return {
        'separators': separators,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'analysis': analysis
    }


@router.post("/chat", tags=["chat"])
async def chat_endpoint(
    background_tasks: BackgroundTasks,
    question: str = Form(..., description="사용자 메시지 또는 질문"),
    files: List[UploadFile] = File(default=[], description="HTML 파일들 (선택사항)")
):
    """
    통합 채팅 API - 일반 대화 + HTML 파일 분석
    
    **동작 방식:**
    
    1. **일반 대화** (파일 없음):
       - 질문: "안녕하세요"
       - 응답: AI 인사 (도구 호출 없음)
    
    2. **HTML 파일 분석** (파일 있음 + 처리 키워드):
       - 질문: "이 HTML 내용을 추출해줘"
       - 응답: Markdown 결과 + **Frontmatter 파일 자동 생성**
       - download_url 필드에 다운로드 링크 포함
    
    3. **일반 질문** (파일 있음 + 일반 키워드):
       - 질문: "HTML이란?"
       - 응답: AI 설명 (도구 호출 없음)
    
    **응답 형식:**
    ```markdown
    ---json
    {
      "rag_config": {
        "separators": ["\n\n", "\n", " ", ""],
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "document_type": "confluence_page"
      },
      "metadata": {
        "processed_at": "2025-10-29T...",
        "html_size": 123456,
        "markdown_size": 45678,
        "tools_used": ["ari_html_to_markdown"]
      },
      "navigation_menu": {
        "current_page_id": "180192188",
        "parent_page_id": "180192092",
        ...
      }
    }
    ---
    
    <!-- RAG_CONTENT_START -->
    
    # 실제 마크다운 내용
    ...
    ```
    
    **구조 설명:**
    - `---json ... ---`: 메타데이터 영역 (RAG 설정, 처리 정보, 네비게이션 등)
    - `<!-- RAG_CONTENT_START -->`: RAG 인덱싱 시작 지점 마커
    - 마커 이후: 실제 RAG 인덱싱 대상 콘텐츠
    
    **RAG 파이프라인 처리 방법:**
    ```python
    # 파일 읽기
    with open('content.md', 'r') as f:
        content = f.read()
    
    # 메타데이터 추출
    if '---json' in content:
        _, frontmatter, rest = content.split('---', 2)
        metadata = json.loads(frontmatter.replace('json', '', 1))
    
    # RAG 콘텐츠만 추출
    if '<!-- RAG_CONTENT_START -->' in rest:
        rag_content = rest.split('<!-- RAG_CONTENT_START -->', 1)[1].strip()
    else:
        rag_content = rest.strip()
    
    # rag_content만 임베딩 및 인덱싱
    ```
    """
    try:
        if not question.strip():
            raise HTTPException(status_code=400, detail="메시지를 입력해주세요")
        
        # 1. HTML 파일 읽기 (선택사항)
        html_content = None
        content_length = 0
        file_info = []
        
        if files:
            html_files = [f for f in files if f.filename and f.filename.endswith('.html')]
            
            if html_files:
                logger.info(f"📁 {len(html_files)}개 HTML 파일 감지")
                
                # 첫 번째 HTML 파일만 처리
                first_file = html_files[0]
                try:
                    content = await first_file.read()
                    html_content = content.decode('utf-8', errors='ignore')
                    content_length = len(html_content)
                    file_info.append({
                        "filename": first_file.filename,
                        "size": content_length
                    })
                    logger.info(f"📄 HTML 파일: {first_file.filename} ({content_length:,} bytes)")
                except Exception as e:
                    logger.error(f"파일 읽기 실패: {e}")
                    raise HTTPException(status_code=400, detail=f"파일 읽기 실패: {str(e)}")
        
        # 2. MCP 도구 목록 가져오기
        available_tools = mcp_service.available_tools
        
        if not available_tools:
            raise HTTPException(status_code=503, detail="사용 가능한 MCP 도구가 없습니다")
        
        # 3. LLM 의도 분석 및 처리
        if html_content:
            # HTML 파일이 있는 경우
            logger.info("🤖 HTML 파일과 함께 질문 처리")
            try:
                answer, tools_used = await llm_service.query_with_raw_result_and_html(
                    question=question,
                    available_tools=available_tools,
                    html_content=html_content
                )
            except Exception as llm_error:
                logger.error(f"❌ LLM 처리 실패: {llm_error}")
                raise HTTPException(
                    status_code=503,
                    detail=f"LLM 서비스 오류: {str(llm_error)}. API 키가 유효한지 확인해주세요."
                )
        else:
            # 일반 대화 (파일 없음)
            logger.info("💬 일반 대화 처리")
            answer = await llm_service.generate_response(question)
            tools_used = []
        
        logger.info(f"✅ LLM 처리 완료: {len(answer):,} characters")
        logger.info(f"🔧 사용된 도구: {', '.join(tools_used) if tools_used else '없음'}")
        
        # HTML 처리 의도가 아니면 (일반 질문이면) JSON 응답만
        if not tools_used or 'ari_html_to_markdown' not in tools_used:
            logger.info("💬 일반 질문 - JSON 응답만")
            return {
                "success": True,
                "answer": answer,
                "tools_used": [],
                "has_markdown": False,
                "file_info": file_info
            }
        
        # HTML 처리 의도일 경우 - Frontmatter 파일 생성 + JSON 응답
        # 4. 마크다운 구조 분석 및 최적 RAG 설정 자동 추천
        logger.info("🔍 마크다운 구조 분석 중...")
        rag_analysis = analyze_markdown_structure(answer)
        
        logger.info(f"📊 [Frontmatter] 분석 결과:")
        logger.info(f"   - 총 길이: {rag_analysis['analysis']['total_length']:,} characters")
        logger.info(f"   - 단락 수: {rag_analysis['analysis']['paragraph_count']}")
        logger.info(f"   - 평균 단락 길이: {rag_analysis['analysis']['avg_paragraph_length']} characters")
        logger.info(f"   - 헤더 존재: {rag_analysis['analysis']['has_headers']}")
        logger.info(f"   - 테이블 존재: {rag_analysis['analysis']['has_tables']}")
        logger.info(f"   - 권장 청크 크기: {rag_analysis['chunk_size']}")
        logger.info(f"   - 권장 청크 중복: {rag_analysis['chunk_overlap']}")
        logger.info(f"   - 권장 Separators: {rag_analysis['separators'][:3]}...")
        
        # 5. 네비게이션 메뉴 추출 (HTML 파일이 있을 때만)
        navigation_menu = None
        
        if html_content:
            logger.info("🗂️ 네비게이션 메뉴 추출 중...")
            
            try:
                nav_result = await mcp_service.call_tool(
                    tool_name="ari_extract_navigation",
                    arguments={"html_content": html_content}
                )
                
                # MCP 결과 파싱
                if hasattr(nav_result, 'content') and nav_result.content:
                    nav_json = json.loads(nav_result.content[0].text)
                    if nav_json.get('success') and 'result' in nav_json:
                        nav_data = nav_json['result']
                        
                        # NavigationItem 리스트 생성
                        root_pages = [NavigationItem(**page) for page in nav_data.get('root_pages', [])]
                        all_pages = [NavigationItem(**page) for page in nav_data.get('all_pages', [])]
                        
                        navigation_menu = NavigationMenu(
                            current_page_id=nav_data.get('current_page_id'),
                            parent_page_id=nav_data.get('parent_page_id'),
                            root_pages=root_pages,
                            all_pages=all_pages
                        )
                        
                        logger.info(f"   - 현재 페이지 ID: {navigation_menu.current_page_id}")
                        logger.info(f"   - 부모 페이지 ID: {navigation_menu.parent_page_id}")
                        logger.info(f"   - 최상위 페이지 수: {len(navigation_menu.root_pages)}")
                        logger.info(f"   - 전체 페이지 수: {len(navigation_menu.all_pages)}")
            except Exception as e:
                logger.warning(f"   - 네비게이션 메뉴 추출 실패: {e}")
        
        # 6. RAG 설정 생성 (서버에서 추천한 설정 사용)
        # MCP 서버가 최적화된 설정을 제공하는지 확인
        server_rag_config = None
        if tools_used and 'ari_html_to_markdown' in tools_used:
            # MCP 결과에서 RAG 설정 추출 시도
            try:
                # answer가 JSON 형식인 경우 파싱
                if '{"success":' in answer:
                    import re
                    json_match = re.search(r'\{"success":.*\}', answer, re.DOTALL)
                    if json_match:
                        result_json = json.loads(json_match.group())
                        if result_json.get('success') and 'result' in result_json:
                            server_rag_config = result_json['result'].get('rag_config')
            except:
                pass
        
        # 서버 추천 설정이 있으면 사용, 없으면 분석 기반 설정 사용
        if server_rag_config:
            # 서버가 추천한 단일 separator 사용
            primary_sep = server_rag_config.get('primary_separator', '\n- ')
            fallback_sep = server_rag_config.get('fallback_separator', '\n\n')
            
            # 단일 separator만 지원하는 시스템을 위한 설정
            rag_config = RagConfig(
                separators=[primary_sep],  # 단일 separator
                chunk_size=server_rag_config.get('chunk_size', 2000),
                chunk_overlap=server_rag_config.get('chunk_overlap', 400),
                document_type="confluence_page",
                strategy=server_rag_config.get('strategy', 'balanced')
            )
            
            logger.info(f"🎯 [Frontmatter] 서버 추천 RAG 설정 사용:")
            logger.info(f"   - Primary Separator: {repr(primary_sep)}")
            logger.info(f"   - Chunk Size: {rag_config.chunk_size}")
            logger.info(f"   - Chunk Overlap: {rag_config.chunk_overlap}")
            logger.info(f"   - Strategy: {rag_config.strategy}")
        else:
            # 기본 분석 기반 설정 (단순화)
            # 리스트 존재 여부 확인
            has_lists = rag_analysis['analysis'].get('has_lists', False)
            
            rag_config = RagConfig(
                separators=['\n- '] if has_lists else ['\n\n'],  # 단일 separator
                chunk_size=rag_analysis['chunk_size'],
                chunk_overlap=rag_analysis['chunk_overlap'],
            document_type="confluence_page"
        )
        
        metadata = ProcessingMetadata(
            processed_at=datetime.now().isoformat(),
            html_size=content_length,
            markdown_size=len(answer),
            tools_used=tools_used if tools_used else [],
            document_analysis=DocumentAnalysis(**rag_analysis['analysis'])
        )
        
        # 7. 프론트매터 생성 (JSON 형식 - YAML보다 안정적)
        frontmatter_data = {
            "rag_config": rag_config.model_dump(),
            "metadata": metadata.model_dump()
        }
        
        # 네비게이션 메뉴가 있으면 추가
        if navigation_menu:
            frontmatter_data["navigation_menu"] = navigation_menu.model_dump()
        
        # JSON으로 직렬화 (개행 문자 등이 이스케이프됨)
        frontmatter_json = json.dumps(
            frontmatter_data,
            ensure_ascii=False,
            indent=2
        )
        
        # 7. 프론트매터 + 마크다운 결합 (RAG 콘텐츠 시작 마커 추가)
        final_content = f"""---json
{frontmatter_json}
---

<!-- RAG_CONTENT_START -->

{answer}
"""
        
        # 8. Markdown 파일 저장 (정적 파일로 저장)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"content_frontmatter_{timestamp}.md"
        
        # 정적 파일 디렉토리 생성
        static_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'downloads')
        os.makedirs(static_dir, exist_ok=True)
        
        # 파일 저장
        file_path = os.path.join(static_dir, filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        logger.info(f"💾 Frontmatter 파일 저장 완료: {filename} ({len(final_content):,} bytes)")
        logger.info(f"🎯 RAG 콘텐츠 구분 마커 추가됨: <!-- RAG_CONTENT_START -->")
        
        # 9. 다운로드 URL 생성
        download_url = f"/downloads/{filename}"
        
        # 10. 파일 자동 삭제 설정 (1시간 후)
        def cleanup_file():
            try:
                import time
                time.sleep(3600)  # 1시간 후
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.info(f"🗑️ 파일 자동 삭제 완료: {filename}")
            except Exception as e:
                logger.warning(f"파일 삭제 실패: {e}")
        
        background_tasks.add_task(cleanup_file)
        
        # 11. JSON 응답 (채팅 + 다운로드 URL)
        return {
            "success": True,
            "answer": answer,
            "tools_used": tools_used,
            "has_markdown": True,
            "file_info": file_info,
            "frontmatter_file": {
                "filename": filename,
                "download_url": download_url,
                "size": len(final_content),
                "rag_config": rag_config.model_dump()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"처리 중 오류: {e}")
        return {
            "success": False,
            "answer": f"오류가 발생했습니다: {str(e)}",
            "tools_used": [],
            "has_markdown": False
        }