from fastmcp import FastMCP
import asyncio
import logging
import re
from typing import Dict, Any
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 불필요한 디버그 로그 숨기기
logging.getLogger("mcp.server").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("sse_starlette").setLevel(logging.WARNING)
logging.getLogger("mcp.server.lowlevel").setLevel(logging.WARNING)

mcp = FastMCP(name="AriProcessingServer")

# Health check endpoint (MCP tool)
@mcp.tool
def health_check() -> Dict[str, Any]:
    """
    ARI Processing Server 헬스체크
    - BeautifulSoup 임포트 가능 여부 확인
    """
    logger.info("[MCP] health_check called")
    soup_ok = False

    # BeautifulSoup import 확인
    try:
        import importlib
        importlib.import_module("bs4")
        soup_ok = True
    except Exception as e:
        logger.warning(f"BeautifulSoup import failed: {e}")

    status = "healthy" if soup_ok else "unhealthy"
    return {
        "success": soup_ok,
        "status": status,
        "service": "ari-processing-server",
        "dependencies": {
            "beautifulsoup": soup_ok,
        }
    }


# ============================================================================
# ARI CONTENT PROCESSING TOOLS (HTML 구조화 및 전용 파싱)
# ============================================================================

def _process_nested_table(table, depth=0) -> str:
    """
    중첩된 표를 재귀적으로 처리
    
    Args:
        table: BeautifulSoup table 객체
        depth: 중첩 깊이 (들여쓰기용)
        
    Returns:
        표를 마크다운으로 변환한 문자열
    """
    from bs4 import BeautifulSoup
    import logging
    import copy
    logger = logging.getLogger(__name__)
    
    indent = "  " * depth  # 중첩 수준에 따른 들여쓰기
    result_lines = []
    
    try:
        # 모든 행 가져오기
        all_rows = table.find_all('tr')
        
        if not all_rows:
            return ""
        
        # 각 행을 순차적으로 처리 (헤더/데이터 구분 없이)
        for row_idx, row in enumerate(all_rows):
            cells = row.find_all(['td', 'th'])
            
            if not cells:
                continue
                
            row_items = []
            
            for cell in cells:
                # 중첩된 표 확인
                nested_tables = cell.find_all('table')
                
                if nested_tables:
                    # 중첩된 표가 있는 경우
                    # 먼저 표를 제외한 텍스트 추출
                    cell_copy = copy.copy(cell)
                    for nt in cell_copy.find_all('table'):
                        nt.decompose()
                    
                    cell_text = cell_copy.get_text(strip=True)
                    if cell_text:
                        row_items.append(cell_text)
                    
                    # 중첩된 표를 재귀적으로 처리
                    for nt in nested_tables:
                        nested_result = _process_nested_table(nt, depth + 1)
                        if nested_result:
                            row_items.append(f"\n{indent}  [중첩된 표]\n" + nested_result)
                else:
                    # 일반 텍스트만 있는 경우
                    cell_text = cell.get_text(strip=True)
                    if cell_text:
                        row_items.append(cell_text)
            
            # 행에 데이터가 있으면 추가
            if row_items:
                # 각 항목을 개별 라인으로 추가
                for item in row_items:
                    if item and item.strip():
                        result_lines.append(f"{indent}- {item}")
    
    except Exception as e:
        logger.warning(f"표 처리 중 오류: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    return "\n".join(result_lines)

def _optimize_markdown_for_rag(markdown_text: str) -> tuple:
    """
    RAG 시스템을 위한 마크다운 최적화
    
    Args:
        markdown_text: 원본 마크다운 텍스트
        
    Returns:
        (최적화된 마크다운, 추천 RAG 설정) 튜플
    """
    import re
    
    lines = markdown_text.split('\n')
    enhanced_lines = []
    
    # 표 섹션을 명확하게 구분
    in_table = False
    table_content = []
    
    for i, line in enumerate(lines):
        # 새로운 표 시작
        if line.startswith('[표 '):
            # 이전 표 내용이 있으면 처리
            if table_content:
                enhanced_lines.extend(table_content)
                enhanced_lines.append('')  # 빈 줄로 구분
                table_content = []
            
            enhanced_lines.append('\n---\n')  # 표 구분자
            enhanced_lines.append(line)
            in_table = True
            
        # 중첩된 표 시작
        elif '[중첩된 표]' in line:
            enhanced_lines.append('\n~~~ 중첩 시작 ~~~')
            enhanced_lines.append(line)
            
        # 긴 텍스트 항목 분할 (200자 이상)
        elif line.startswith('- ') and len(line) > 200:
            # 문장 단위로 분할
            text = line[2:]  # '- ' 제거
            
            # 마침표, 쉼표 등으로 분할
            sentences = re.split(r'(?<=[.!?]) (?=[A-Z가-힣])', text)
            
            if len(sentences) > 1:
                enhanced_lines.append('- ' + sentences[0])
                for sent in sentences[1:]:
                    if sent.strip():
                        enhanced_lines.append('  ' + sent.strip())
            else:
                # 쉼표로도 분할 시도
                parts = text.split(', ')
                if len(parts) > 3:  # 충분히 긴 경우만
                    enhanced_lines.append('- ' + parts[0] + ',')
                    for part in parts[1:]:
                        if part.strip():
                            enhanced_lines.append('  ' + part.strip() + ',')
                else:
                    enhanced_lines.append(line)
        else:
            if in_table and line.strip() == '':
                in_table = False
            enhanced_lines.append(line)
    
    # 마지막 표 내용 처리
    if table_content:
        enhanced_lines.extend(table_content)
    
    # 최적화된 마크다운
    optimized_markdown = '\n'.join(enhanced_lines)
    
    # RAG 설정 추천
    # 표가 많은 경우와 일반 텍스트가 많은 경우를 구분
    table_count = optimized_markdown.count('[표 ')
    avg_line_length = sum(len(line) for line in enhanced_lines) / max(len(enhanced_lines), 1)
    
    if table_count > 5:  # 표가 많은 문서
        recommended_config = {
            "separators": ["\n---\n", "\n~~~ 중첩", "\n\n", "\n- ", "\n", ". ", ", "],
            "chunk_size": 2500,  # 표를 위해 더 큰 크기
            "chunk_overlap": min(500, int(2500 * 0.2)),  # 최대 500, chunk_size의 20%
            "strategy": "table_aware"
        }
    elif avg_line_length > 100:  # 긴 텍스트가 많은 문서
        recommended_config = {
            "separators": ["\n\n", "\n- ", ". ", ", ", "\n"],
            "chunk_size": 2000,
            "chunk_overlap": min(500, int(2000 * 0.2)),  # 최대 500, chunk_size의 20%
            "strategy": "sentence_aware"
        }
    else:  # 일반적인 경우
        recommended_config = {
            "separators": ["\n- ", "\n\n", "\n", ". ", ", "],
            "chunk_size": 2000,
            "chunk_overlap": min(500, int(2000 * 0.2)),  # 최대 500, chunk_size의 20%
            "strategy": "balanced"
        }
    
    return optimized_markdown, recommended_config

def _extract_cell_parts_by_html_structure(cell_obj) -> list:
    """
    HTML 셀 내부의 구조(p, ul, li, br 등)를 기반으로 의미 단위로 분할
    
    Args:
        cell_obj: BeautifulSoup 셀 객체
        
    Returns:
        분할된 텍스트 리스트
    """
    parts = []
    
    # 전체 텍스트 먼저 추출
    full_text = cell_obj.get_text(separator=' ', strip=True)
    
    # 전체 텍스트가 짧으면 그대로 반환 (분할 불필요)
    if len(full_text) <= 500:
        return [full_text] if full_text else []
    
    # 1. <ul>/<ol> 리스트로 분할 시도 (가장 명확한 구조)
    list_tags = cell_obj.find_all(['ul', 'ol'], recursive=False)
    if list_tags:
        # 리스트 항목 추출
        for list_tag in list_tags:
            li_tags = list_tag.find_all('li')
            for li in li_tags:
                text = li.get_text(separator=' ', strip=True)
                # 의미 있는 길이만 (20자 이상)
                if text and len(text) > 20:
                    parts.append(text)
        
        # 리스트 외 텍스트 추가 (중복 방지: 이미 추출된 텍스트 제외)
        # 리스트를 임시로 제거한 복사본에서 텍스트 추출
        from bs4 import BeautifulSoup
        temp_cell = BeautifulSoup(str(cell_obj), 'html.parser')
        for list_tag in temp_cell.find_all(['ul', 'ol']):
            list_tag.decompose()  # 리스트 제거
        
        remaining_text = temp_cell.get_text(separator=' ', strip=True)
        if remaining_text and len(remaining_text) > 20:
            # 이미 추출된 내용과 중복되지 않는지 확인
            is_duplicate = any(remaining_text in part or part in remaining_text for part in parts)
            if not is_duplicate:
                parts.insert(0, remaining_text)  # 리스트 전에 나온 텍스트이므로 앞에 추가
        
        if parts:
            return parts
    
    # 2. <br> 태그로 분할 시도
    html_str = str(cell_obj)
    if '<br' in html_str.lower():
        from bs4 import BeautifulSoup
        temp_soup = BeautifulSoup(html_str, 'html.parser')
        for br in temp_soup.find_all('br'):
            br.replace_with('\n')
        text = temp_soup.get_text()
        parts = [line.strip() for line in text.split('\n') if line.strip() and len(line.strip()) > 20]
        if len(parts) > 1:
            return parts
    
    # 3. <p> 태그로 분할 시도 (단, 의미 있는 크기만)
    p_tags = cell_obj.find_all('p', recursive=False)
    if p_tags and len(p_tags) > 1:  # 2개 이상일 때만
        for p in p_tags:
            text = p.get_text(separator=' ', strip=True)
            # 의미 있는 길이만 (50자 이상)
            if text and len(text) > 50:
                parts.append(text)
        if parts:
            return parts
    
    # 4. 구조가 없거나 너무 작은 구조면 600자 단위로 분할
    if len(full_text) > 600:
        chunk_size = 600
        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i + chunk_size]
            
            # 단어 경계 고려
            if i + chunk_size < len(full_text):
                last_space = chunk.rfind(' ')
                if last_space > chunk_size * 0.7:
                    chunk = chunk[:last_space]
            
            if chunk.strip():
                parts.append(chunk.strip())
        return parts
    
    # 5. 그대로 반환
    return [full_text] if full_text else []


@mcp.tool
def ari_html_to_markdown(html_content: str, extract_tables: bool = True, use_trafilatura: bool = True) -> Dict[str, Any]:
    """
    HTML을 RAG 친화적인 마크다운으로 변환하는 도구 (trafilatura + BeautifulSoup + markdownify)
    
    - HTML에서 순수 컨텐츠만 추출 (불필요한 HTML 코드 제거)
    - trafilatura로 노이즈 제거 (광고, 네비게이션 등)
    - BeautifulSoup으로 표(table) 구조 파싱 및 구조화된 텍스트로 변환
    - RAG 시스템에 바로 사용 가능한 깔끔한 Markdown 출력
    
    Args:
        html_content: HTML 본문 문자열
        extract_tables: 표를 별도로 추출할지 여부 (기본값: True)
        use_trafilatura: trafilatura 사용 여부 (기본값: False)
    
    Returns:
        변환 결과 딕셔너리 (success, result 포함)
    """
    logger.info(f"[MCP] ari_html_to_markdown 호출됨 - HTML 크기: {len(html_content)} chars")
    logger.info(f"[MCP] 옵션: extract_tables={extract_tables}, use_trafilatura={use_trafilatura}")
    
    try:
        from bs4 import BeautifulSoup, Tag
        import trafilatura
        from markdownify import markdownify as md
        from datetime import datetime
        
        # 1. BeautifulSoup으로 HTML 파싱
        soup = BeautifulSoup(html_content, 'lxml')
        
        # 1-1. 불필요한 요소 제거 (Confluence 특화)
        # 스크립트, 스타일, 네비게이션, 광고 등 제거
        for selector in ['script', 'style', 'nav', 'header', 'footer', 'aside', 
                        '.aui-page-header-actions', '.page-actions', '.aui-toolbar2',
                        '.comment-container', '.like-button-container', '.page-labels',
                        'svg', '.aui-icon']:
            for element in soup.select(selector):
                element.decompose()
        
        logger.info("[MCP] 불필요한 HTML 요소 제거 완료")
        
        # 1-2. 페이지 메타데이터 추출 (작성자, 날짜 등)
        metadata_parts = []
        
        # 제목 추출
        title = soup.find('h1', {'id': 'title-text'}) or soup.find('title')
        if title:
            metadata_parts.append(f"# {title.get_text(strip=True)}\n")
        
        # 작성자/수정자 정보 추출
        page_metadata = soup.find('div', class_='page-metadata') or soup.find('div', id='page-metadata')
        if page_metadata:
            metadata_text = page_metadata.get_text(separator=' ', strip=True)
            if metadata_text:
                metadata_parts.append(f"**메타데이터**: {metadata_text}\n")
        
        # 브레드크럼 추출
        breadcrumbs = soup.find('ol', {'id': 'breadcrumbs'}) or soup.find('div', class_='breadcrumbs')
        if breadcrumbs:
            breadcrumb_text = breadcrumbs.get_text(separator=' > ', strip=True)
            if breadcrumb_text:
                metadata_parts.append(f"**경로**: {breadcrumb_text}\n")
        
        metadata_content = "\n".join(metadata_parts) if metadata_parts else ""
        logger.info(f"[MCP] 메타데이터 추출 완료: {len(metadata_content)} characters")
        
        # 2. 표 추출 (trafilatura 전에 먼저 추출)
        tables_text = ""
        tables_count = 0
        
        if extract_tables:
            tables_text_list = []
            # 모든 표를 처리하되, 중첩된 표는 부모 표 처리 시에만 처리
            all_tables = soup.find_all('table')
            logger.info(f"[MCP] 발견된 전체 표 개수: {len(all_tables)}")
            
            # 이미 처리된 표를 추적
            processed_tables = set()
            
            table_idx = 0
            for table in all_tables:
                # 이미 처리된 표는 건너뛰기
                if id(table) in processed_tables:
                    continue
                    
                # 중첩된 표인지 확인 (부모가 table이면 중첩된 표)
                if table.find_parent('table'):
                    # 중첩된 표는 부모 표 처리 시 처리되므로 건너뛰기
                    continue
                
                table_idx += 1
                
                try:
                    # 이 표는 최상위 표
                    # 내부에 중첩된 표가 있는지 확인
                    nested_tables = table.find_all('table')
                    has_nested_tables = len(nested_tables) > 0
                    if has_nested_tables:
                        logger.info(f"[MCP] 표 {table_idx}: 내부에 {len(nested_tables)}개의 중첩된 표 포함")
                        # 중첩된 표들을 processed 목록에 추가
                        for nt in nested_tables:
                            processed_tables.add(id(nt))
                    
                    # 표 제목 찾기
                    table_title = None
                    caption = table.find('caption')
                    if caption:
                        table_title = caption.get_text(strip=True)
                    
                    # 헤더 추출
                    headers = []
                    has_thead = False
                    used_first_row_as_header = False
                    
                    thead = table.find('thead')
                    if thead:
                        has_thead = True
                        header_row = thead.find('tr')
                        if header_row:
                            # 모든 th와 td 태그 추출 (병합된 셀 고려)
                            header_cells = header_row.find_all(['th', 'td'])
                            for cell in header_cells:
                                # colspan 체크
                                colspan = int(cell.get('colspan', 1))
                                cell_text = cell.get_text(strip=True)
                                
                                # colspan이 있으면 해당 수만큼 헤더 추가 (빈 헤더로)
                                if colspan > 1:
                                    # 첫 번째는 실제 텍스트, 나머지는 빈 문자열
                                    headers.append(cell_text)
                                    for _ in range(colspan - 1):
                                        headers.append('')  # 병합된 셀의 나머지 부분
                                else:
                                    headers.append(cell_text)
                            
                            logger.info(f"[MCP] 표 {table_idx} 헤더 추출 (thead): {len(headers)}개 - {headers}")
                    
                    # thead가 없거나 헤더가 비어있으면 첫 번째 행 사용
                    if not headers:
                        first_row = table.find('tr')
                        if first_row:
                            header_cells = first_row.find_all(['th', 'td'])
                            for cell in header_cells:
                                colspan = int(cell.get('colspan', 1))
                                cell_text = cell.get_text(strip=True)
                                
                                if colspan > 1:
                                    headers.append(cell_text)
                                    for _ in range(colspan - 1):
                                        headers.append('')
                                else:
                                    headers.append(cell_text)
                            
                            used_first_row_as_header = True
                            logger.info(f"[MCP] 표 {table_idx} 헤더 추출 (첫 행): {len(headers)}개 - {headers}")
                    
                    # 데이터 행 추출
                    rows = []
                    tbody = table.find('tbody')
                    if tbody:
                        data_rows = tbody.find_all('tr')
                    else:
                        # tbody가 없는 경우
                        all_rows = table.find_all('tr')
                        # 첫 번째 행을 헤더로 사용했다면 두 번째 행부터, 아니면 첫 번째 행부터
                        start_idx = 1 if used_first_row_as_header else 0
                        data_rows = all_rows[start_idx:]
                    
                    for row in data_rows:
                        # 셀 객체 자체를 저장 (나중에 HTML 구조 기반 분할 위해)
                        cell_objects = row.find_all(['td', 'th'])
                        if cell_objects:
                            rows.append(cell_objects)
                    
                    # 구조화된 텍스트 생성
                    if rows:  # 헤더가 없어도 행이 있으면 처리
                        # 헤더가 없으면 첫 번째 행을 헤더로 사용하거나 기본값 생성
                        if not headers:
                            # 첫 번째 행의 셀 수를 확인
                            first_row_cell_count = len(rows[0]) if rows else 0
                            # 기본 헤더 생성
                            headers = [f"열{i+1}" for i in range(first_row_cell_count)]
                            logger.info(f"[MCP] 표 {table_idx}: 헤더 없음, 기본 헤더 생성 - {headers}")
                        
                        # 빈 헤더 제거 및 실제 헤더만 사용
                        actual_headers = [h for h in headers if h.strip()]
                        if not actual_headers:
                            # 모든 헤더가 비어있으면 기본값 사용
                            actual_headers = [f"열{i+1}" for i in range(len(headers))]
                        
                        logger.info(f"[MCP] 표 {table_idx} 실제 헤더: {len(actual_headers)}개 - {actual_headers}")
                        
                        table_lines = []
                        if table_title:
                            table_lines.append(f"\n[표: {table_title}]")
                        else:
                            table_lines.append(f"\n[표 {table_idx}]")
                        
                        for row_idx, row in enumerate(rows):
                            # 디버깅: 원본 셀 개수 확인
                            original_cell_count = len(row)
                            if original_cell_count != len(actual_headers):
                                logger.warning(f"[MCP] 표 {table_idx} 행 {row_idx+1}: 셀 개수({original_cell_count}) != 실제 헤더 개수({len(actual_headers)})")
                            
                            # 실제 헤더 수에 맞춰 셀 수 조정
                            # 셀이 부족하면 None 추가
                            while len(row) < len(actual_headers):
                                row.append(None)
                            
                            row_items = []
                            # 실제 헤더와 셀 매핑
                            for col_idx, (header, cell_obj) in enumerate(zip(actual_headers, row)):
                                if cell_obj is None:
                                    logger.debug(f"[MCP] 표 {table_idx} 행 {row_idx+1} 열 {col_idx+1}({header}): None 셀")
                                    continue
                                    
                                # 셀 내부에 중첩된 표가 있는지 확인
                                nested_tables_in_cell = cell_obj.find_all('table')
                                if nested_tables_in_cell:
                                    # 중첩된 표가 있는 경우
                                    logger.debug(f"[MCP] 표 {table_idx} 행 {row_idx+1} 열 {col_idx+1}({header}): {len(nested_tables_in_cell)}개의 중첩된 표 포함")
                                    
                                    # 중첩된 표를 제외한 텍스트 추출
                                    from bs4 import BeautifulSoup
                                    temp_cell = BeautifulSoup(str(cell_obj), 'html.parser')
                                    
                                    # 중첩된 표들을 처리하고 제거
                                    nested_results = []
                                    for nested_table in temp_cell.find_all('table'):
                                        # 중첩된 표를 재귀적으로 처리
                                        nested_result = _process_nested_table(nested_table, depth=1)
                                        if nested_result:
                                            nested_results.append(nested_result)
                                        nested_table.decompose()  # temp_cell에서 표 제거
                                    
                                    # 원본 cell_obj에서도 중첩된 표 제거 (중복 방지)
                                    for nested_table in cell_obj.find_all('table'):
                                        nested_table.decompose()
                                    
                                    # 표를 제거한 후 남은 텍스트
                                    cell_text = temp_cell.get_text(separator=' ', strip=True)
                                    
                                    # 셀 텍스트와 중첩된 표 결과 결합
                                    cell_parts = []
                                    if cell_text:
                                        cell_parts.append(cell_text)
                                    for nested_result in nested_results:
                                        cell_parts.append("\n    [중첩된 표]\n" + "\n".join("    " + line for line in nested_result.split("\n")))
                                    
                                    if not cell_parts:
                                        cell_parts = []
                                else:
                                    # 중첩된 표가 없는 경우 기존 로직 사용
                                    # 먼저 전체 텍스트 추출 (디버깅 및 폴백)
                                    full_text = cell_obj.get_text(separator=' ', strip=True)
                                    
                                    # HTML 구조 기반 분할 시도
                                    try:
                                        cell_parts = _extract_cell_parts_by_html_structure(cell_obj)
                                    except Exception as e:
                                        logger.debug(f"[MCP] 셀 분할 실패: {e}")
                                        cell_parts = []
                                    
                                    # 분할이 실패하거나 비어있으면 전체 텍스트 사용
                                    if not cell_parts and full_text:
                                        cell_parts = [full_text]
                                        logger.debug(f"[MCP] 표 {table_idx} 행 {row_idx+1} 열 {col_idx+1}({header}): 전체 텍스트 사용")
                                
                                # 셀 내용 추가
                                for part in cell_parts:
                                    if part.strip():
                                        # 중첩된 표는 들여쓰기로 구분
                                        if "[중첩된 표]" in part:
                                            row_items.append(part)  # 중첩된 표는 그대로 추가
                                        else:
                                            row_items.append(f"{header}: {part.strip()}")
                            
                            if row_items:
                                # 각 항목을 개별 줄로 분리 (청킹 개선)
                                for item in row_items:
                                    table_lines.append(f"- {item}")
                        
                        tables_text_list.append("\n".join(table_lines))
                        logger.info(f"[MCP] 표 {table_idx} 변환 완료: {len(headers)}개 열, {len(rows)}개 행")
                    
                    # 현재 표를 처리된 목록에 추가
                    processed_tables.add(id(table))
                
                except Exception as e:
                    logger.warning(f"[MCP] 표 {table_idx} 처리 중 오류: {e}")
                    continue
            
            tables_text = "\n\n".join(tables_text_list)
            tables_count = table_idx  # 실제 처리된 최상위 표 개수
            
            # 표를 HTML에서 제거 (중복 방지)
            for table in soup.find_all('table'):
                table.decompose()
            logger.info(f"[MCP] 표 {tables_count}개 추출 후 HTML에서 제거")
        
        # 3. trafilatura로 주요 텍스트 추출 (표 제거된 HTML 사용)
        main_text = ""
        if use_trafilatura:
            try:
                logger.info("[MCP] trafilatura로 주요 텍스트 추출 중...")
                html_without_tables = str(soup)
                main_text = trafilatura.extract(
                    html_without_tables,
                    include_tables=False,
                    include_comments=False,
                    include_formatting=True,
                    include_links=False,  # 링크 제외
                    no_fallback=False,
                    favor_precision=False,
                    favor_recall=True,  # 더 많은 텍스트 추출
                    output_format='txt'
                )
                if main_text:
                    logger.info(f"[MCP] trafilatura 추출 완료: {len(main_text):,} characters")
                else:
                    logger.warning("[MCP] trafilatura 추출 실패, BeautifulSoup으로 폴백")
            except Exception as e:
                logger.warning(f"[MCP] trafilatura 오류: {e}, BeautifulSoup으로 폴백")
        
        # 4. 최종 Markdown 생성
        final_markdown = ""
        method = ""
        
        # 메타데이터 먼저 추가
        if metadata_content:
            final_markdown = metadata_content + "\n---\n\n"
        
        # 본문 콘텐츠 추가
        if main_text:
            final_markdown += main_text
            method = "trafilatura + BeautifulSoup"
        else:
            logger.info("[MCP] BeautifulSoup으로 전체 HTML 처리 중...")
            
            # main-content 또는 wiki-content div 찾기
            main_content = soup.find('div', {'id': 'main-content'}) or soup.find('div', class_='wiki-content')
            if main_content:
                content_html = str(main_content)
            else:
                content_html = str(soup)
            
            final_markdown += md(
                content_html,
                heading_style="ATX",
                bullets="-",
                strip=['script', 'style']
            )
            method = "BeautifulSoup + markdownify"
        
        # 표 추가
        if tables_text:
            final_markdown += "\n\n---\n\n## 추출된 표 데이터\n\n" + tables_text
        
        # 5. 정리 및 포맷팅 (연속된 빈 줄 제거)
        lines = final_markdown.split('\n')
        cleaned_lines = []
        prev_empty = False
        for line in lines:
            is_empty = not line.strip()
            if is_empty and prev_empty:
                continue
            cleaned_lines.append(line)
            prev_empty = is_empty
        
        final_markdown = '\n'.join(cleaned_lines).strip()
        
        # RAG 최적화 적용
        optimized_markdown, rag_config = _optimize_markdown_for_rag(final_markdown)
        
        # RAG 설정을 기존 설정과 병합
        # 사용자가 제공한 separators가 단일 문자열인 경우를 고려
        if rag_config['strategy'] == 'table_aware':
            # 표 중심 문서는 더 큰 청크 사용
            rag_config['primary_separator'] = "\n---\n"  # 표 구분자
            rag_config['fallback_separator'] = "\n- "     # 리스트 항목
        elif rag_config['strategy'] == 'sentence_aware':
            # 긴 텍스트는 문장 단위 분할
            rag_config['primary_separator'] = ". "
            rag_config['fallback_separator'] = "\n"
        else:
            # 균형잡힌 접근
            rag_config['primary_separator'] = "\n- "
            rag_config['fallback_separator'] = "\n\n"
        
        result = {
            'success': True,
            'result': {
                'markdown': optimized_markdown,  # 최적화된 마크다운 사용
                'stats': {
                    'original_size': len(html_content),
                    'markdown_size': len(optimized_markdown),
                    'original_markdown_size': len(final_markdown),
                    'tables_found': tables_count,
                    'method': method,
                    'optimization_applied': True
                },
                'rag_config': rag_config,  # RAG 추천 설정 추가
                'converted_at': datetime.now().isoformat()
            }
        }
        
        logger.info(f"[MCP] ari_html_to_markdown 완료 - 마크다운 크기: {len(final_markdown)} chars, 표 {tables_count}개")
        return result
        
    except Exception as e:
        logger.error(f"[MCP] Enhanced HTML to Markdown 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}




@mcp.tool
def ari_extract_navigation(html_content: str) -> Dict[str, Any]:
    """
    HTML에서 Confluence 페이지 트리 네비게이션 메뉴를 추출하는 도구
    
    - 현재 페이지 ID 및 부모 페이지 ID 추출
    - 페이지 트리 구조 파싱 (최상위 페이지 및 전체 계층 구조)
    - 각 페이지의 제목, URL, 레벨, 하위 페이지 존재 여부 등 메타데이터 포함
    
    Args:
        html_content: HTML 본문 문자열
    
    Returns:
        네비게이션 메뉴 정보 딕셔너리 (success, result 포함)
    """
    logger.info(f"[MCP] ari_extract_navigation 호출됨 - HTML 크기: {len(html_content)} chars")
    
    try:
        from bs4 import BeautifulSoup
        import re
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # 현재 페이지 ID 추출 (meta 태그에서)
        current_page_id = None
        page_id_meta = soup.find('meta', {'name': 'ajs-page-id'})
        if page_id_meta and page_id_meta.get('content'):
            current_page_id = page_id_meta.get('content')
        
        # 부모 페이지 ID 추출
        parent_page_id = None
        parent_id_meta = soup.find('meta', {'name': 'ajs-parent-page-id'})
        if parent_id_meta and parent_id_meta.get('content'):
            parent_page_id = parent_id_meta.get('content')
        
        logger.info(f"[MCP] 페이지 정보: current_page_id={current_page_id}, parent_page_id={parent_page_id}")
        
        # 페이지 트리 컨테이너 찾기
        page_tree_container = soup.find('div', class_='plugin_pagetree_children_list')
        
        if not page_tree_container:
            logger.warning("[MCP] 페이지 트리 컨테이너를 찾을 수 없습니다")
            return {
                'success': True,
                'result': {
                    'current_page_id': current_page_id,
                    'parent_page_id': parent_page_id,
                    'root_pages': [],
                    'all_pages': []
                }
            }
        
        # 최상위 ul 태그 찾기
        root_ul = page_tree_container.find('ul', class_='plugin_pagetree_children_list')
        
        all_pages = []
        root_pages = []
        
        if root_ul:
            # 재귀적으로 모든 페이지 아이템 추출
            all_pages = _extract_page_items_recursive(root_ul, level=0)
            
            # 최상위 페이지만 필터링 (level 0)
            root_pages = [page for page in all_pages if page['level'] == 0]
        
        logger.info(f"[MCP] 네비게이션 메뉴 추출 완료: 총 {len(all_pages)}개 페이지, 최상위 {len(root_pages)}개")
        
        return {
            'success': True,
            'result': {
                'current_page_id': current_page_id,
                'parent_page_id': parent_page_id,
                'root_pages': root_pages,
                'all_pages': all_pages
            }
        }
        
    except Exception as e:
        logger.error(f"[MCP] 네비게이션 메뉴 추출 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def _extract_page_items_recursive(ul_element, level: int = 0) -> list:
    """
    재귀적으로 페이지 트리에서 모든 아이템을 추출
    
    Args:
        ul_element: BeautifulSoup ul 엘리먼트
        level: 현재 계층 깊이
        
    Returns:
        페이지 아이템 딕셔너리 리스트
    """
    import re
    items = []
    
    # 직접 자식 li 태그만 처리
    for li in ul_element.find_all('li', recursive=False):
        try:
            # 페이지 ID 추출
            toggle_link = li.find('a', class_='plugin_pagetree_childtoggle')
            page_id = None
            is_expanded = False
            
            if toggle_link:
                page_id = toggle_link.get('data-page-id')
                aria_expanded = toggle_link.get('aria-expanded', 'false')
                is_expanded = aria_expanded == 'true'
            
            # 페이지 제목과 URL 추출
            content_span = li.find('span', class_='plugin_pagetree_children_span')
            title = None
            url = None
            
            if content_span:
                link = content_span.find('a')
                if link:
                    title = link.get_text(strip=True)
                    url = link.get('href')
                    
                    # pageId가 없으면 URL에서 추출 시도
                    if not page_id and url:
                        match = re.search(r'pageId=(\d+)', url)
                        if match:
                            page_id = match.group(1)
            
            # 하위 페이지 컨테이너 확인
            children_container = li.find('div', class_='plugin_pagetree_children_container')
            has_children = False
            
            if children_container:
                child_ul = children_container.find('ul', class_='plugin_pagetree_children_list', recursive=False)
                has_children = child_ul is not None and len(child_ul.find_all('li', recursive=False)) > 0
            
            # 페이지 아이템 생성
            if page_id and title:
                item = {
                    'page_id': page_id,
                    'title': title,
                    'url': url,
                    'level': level,
                    'has_children': has_children,
                    'is_expanded': is_expanded
                }
                items.append(item)
                
                # 하위 페이지가 있고 펼쳐져 있으면 재귀 호출
                if has_children and children_container:
                    child_ul = children_container.find('ul', class_='plugin_pagetree_children_list', recursive=False)
                    if child_ul:
                        child_items = _extract_page_items_recursive(child_ul, level + 1)
                        items.extend(child_items)
        
        except Exception as e:
            logger.warning(f"[MCP] 페이지 아이템 추출 중 오류 (level {level}): {e}")
            continue
    
    return items


async def main():
    # Start ARI Processing MCP server
    logger.info("🚀 ARI Processing MCP Server 시작 중...")
    logger.info("📍 서버 주소: http://0.0.0.0:4200/my-custom-path")
    logger.info("🔧 사용 가능한 도구: health_check, ari_html_to_markdown, ari_extract_navigation")
    
    await mcp.run_async(
        transport="http",
        host="0.0.0.0",
        port=4200,
        path="/my-custom-path",
        log_level="info",
    )

if __name__ == "__main__":
    asyncio.run(main())