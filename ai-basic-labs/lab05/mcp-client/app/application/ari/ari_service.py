"""ARI HTML Processing Service"""
import logging
import tempfile
from typing import List, Dict, Any, Optional
import re
try:
    from markdownify import markdownify as md_convert
except Exception:  # 런타임 환경에 따라 미설치 가능
    md_convert = None  # graceful fallback

try:
    import pymupdf4llm
    import fitz  # PyMuPDF
except ImportError:
    pymupdf4llm = None
    fitz = None

from fastapi import UploadFile
import uuid
import os
import json
from datetime import datetime
from bs4 import BeautifulSoup
from app.infrastructure.mcp.mcp_service import mcp_service

logger = logging.getLogger(__name__)

class AriService:
    """ARI HTML 파일 처리 서비스"""
    
    def __init__(self):
        self.temp_dir = "/tmp/ari_html"
        self.output_dir = "/tmp/ari_json"
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def process_html_files(self, files: List[UploadFile]) -> Dict[str, Any]:
        """
        HTML 파일들을 처리하여 header, footer, sidebar를 제외하고 JSON 형태로 저장
        
        Args:
            files: 업로드된 HTML 파일들
            
        Returns:
            처리 결과 정보
        """
        try:
            if not files:
                raise ValueError("업로드된 파일이 없습니다")
            
            # 파일 저장 및 처리
            processed_files = []
            total_size = 0
            
            for file in files:
                if not file.filename.endswith('.html'):
                    logger.warning(f"HTML이 아닌 파일 무시: {file.filename}")
                    continue
                
                # 고유한 파일명 생성
                file_id = str(uuid.uuid4())
                file_path = os.path.join(self.temp_dir, f"{file_id}_{file.filename}")
                json_path = os.path.join(self.output_dir, f"{file_id}.json")
                
                # 파일 저장
                content = await file.read()
                total_size += len(content)
                
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                # HTML에서 header, footer, sidebar 제외하여 JSON으로 변환
                processed_data = await self._extract_main_content(content.decode('utf-8', errors='ignore'))
                
                # JSON 파일로 저장
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
                processed_files.append({
                    'original_filename': file.filename,
                    'file_id': file_id,
                    'file_path': file_path,
                    'json_path': json_path,
                    'size': len(content),
                    'processed_data': processed_data,
                    'upload_time': datetime.now().isoformat()
                })
                
                logger.info(f"HTML 파일 처리 완료: {file.filename} ({len(content)} bytes)")
            
            return {
                'success': True,
                'processed_files': processed_files,
                'total_files': len(processed_files),
                'total_size': total_size,
                'message': f"{len(processed_files)}개의 HTML 파일이 성공적으로 처리되었습니다"
            }
            
        except Exception as e:
            logger.error(f"HTML 파일 처리 중 오류: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f"HTML 파일 처리 중 오류가 발생했습니다: {str(e)}"
            }

    async def process_html_files_complete(self, files: List[UploadFile]) -> Dict[str, Any]:
        """
        HTML 파일들을 완전히 처리하여 구조화된 JSON까지 변환
        
        Args:
            files: 업로드된 HTML 파일들
            
        Returns:
            완전 처리된 결과 정보 (마크다운 + 구조화된 JSON 포함)
        """
        try:
            if not files:
                raise ValueError("업로드된 파일이 없습니다")
            
            processed_files = []
            total_size = 0
            
            for file in files:
                if not file.filename.endswith('.html'):
                    logger.warning(f"HTML이 아닌 파일 무시: {file.filename}")
                    continue
                
                # 고유한 파일명 생성
                file_id = str(uuid.uuid4())
                file_path = os.path.join(self.temp_dir, f"{file_id}_{file.filename}")
                json_path = os.path.join(self.output_dir, f"{file_id}.json")
                
                # 파일 저장
                content = await file.read()
                total_size += len(content)
                
                with open(file_path, 'wb') as f:
                    f.write(content)
                
                # HTML 콘텐츠 디코딩
                html_content = content.decode('utf-8', errors='ignore')
                
                # 1단계: 기본 메타데이터 추출 (개선된 버전)
                basic_data = await self._extract_main_content(html_content)
                
                # 2단계: 마크다운 변환 (개선된 버전 사용)
                markdown_content = self.extract_markdown(html_content)
                
                # 3단계: 마크다운을 구조화된 JSON으로 변환
                json_result = self.ari_markdown_to_json(markdown_content)
                contents = json_result.get('contents', []) if json_result.get('success') else []
                
                # 폴백: 마크다운을 텍스트 단락으로 반환
                if not contents:
                    contents = [{"id": 1, "type": "text", "title": "", "data": markdown_content}]
                
                # 새로운 구조로 통합된 데이터 구성
                processed_data = {
                    'title': basic_data.get('title', ''),
                    'breadcrumbs': basic_data.get('breadcrumbs', []),
                    'content': {
                        'text': basic_data['content']['text'],
                        'markdown': markdown_content,
                        'contents': contents
                    },
                    'metadata': {
                        **basic_data.get('metadata', {}),
                        'markdown_length': len(markdown_content),
                        'contents_count': len(contents)
                    }
                }
                
                # JSON 파일로 저장
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=2)
                
                processed_files.append({
                    'original_filename': file.filename,
                    'file_id': file_id,
                    'file_path': file_path,
                    'json_path': json_path,
                    'size': len(content),
                    'processed_data': processed_data,
                    'contents': contents,
                    'markdown': markdown_content,
                    'upload_time': datetime.now().isoformat()
                })
                
                logger.info(f"HTML 파일 완전 처리 완료: {file.filename} ({len(content)} bytes)")
            
            return {
                'success': True,
                'processed_files': processed_files,
                'total_files': len(processed_files),
                'total_size': total_size,
                'message': f"{len(processed_files)}개의 HTML 파일이 성공적으로 완전 처리되었습니다"
            }
            
        except Exception as e:
            logger.error(f"HTML 파일 완전 처리 중 오류: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': f"HTML 파일 완전 처리 중 오류가 발생했습니다: {str(e)}"
            }
    
    async def _extract_main_content(self, html_content: str) -> Dict[str, Any]:
        """
        HTML에서 header, footer, sidebar를 제외한 메인 콘텐츠를 추출하여 JSON으로 변환
        
        Args:
            html_content: 원본 HTML 내용
            
        Returns:
            추출된 콘텐츠의 JSON 데이터
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 페이지 제목과 브레드크럼 추출 (요소 제거 전에 먼저 수행)
            page_title = ""
            breadcrumbs = []
            urls = []
            pagetree = []
            
            # 1. 페이지 제목 추출 (h1#title-text)
            title_element = soup.find('h1', {'id': 'title-text'})
            if title_element:
                page_title = title_element.get_text(strip=True)
            
            # 2. 브레드크럼 추출 (ol#breadcrumbs)
            breadcrumb_element = soup.find('ol', {'id': 'breadcrumbs'})
            if breadcrumb_element:
                for li in breadcrumb_element.find_all('li'):
                    # ellipsis 버튼 제외
                    if li.get('id') == 'ellipsis':
                        continue
                        
                    link = li.find('a')
                    if link:
                        breadcrumbs.append({
                            'text': link.get_text(strip=True),
                            'href': link.get('href', '')
                        })
                    else:
                        # 링크가 없는 경우 (예: 현재 페이지)
                        span = li.find('span')
                        if span:
                            breadcrumbs.append({
                                'text': span.get_text(strip=True),
                                'href': ''
                            })
            
            # 3. 페이지 트리 추출 (ia-secondary-content)
            pagetree_element = soup.find('div', {'class': 'ia-secondary-content'})
            if pagetree_element:
                pagetree = self._extract_pagetree(pagetree_element)
            
            # 4. 메인 콘텐츠에서 URL 추출 (요소 제거 전에 수행)
            main_content = soup.find('div', {'id': 'main-content'})
            if not main_content:
                main_content = soup.find('div', {'class': 'wiki-content'})
            if not main_content:
                main_content = soup.find('main') or soup.find('body') or soup
                
            if main_content:
                urls = self._extract_urls(main_content)
            
            # 제거할 요소들 (header, footer, sidebar, nav 등 + Confluence 특화)
            # 단, 페이지 제목과 브레드크럼은 유지
            elements_to_remove = [
                'header', 'footer', 'nav', 'aside', 'sidebar',
                '.header', '.footer', '.nav', '.aside', '.sidebar',
                '.navigation', '.menu',
                
                # Confluence 특화 UI 요소들 (제목/브레드크럼 제외)
                'div.aui-page-header-actions',     # 페이지 액션 버튼들
                'div.page-actions',               # 페이지 액션들
                'div.aui-toolbar2',               # 툴바
                'div.comment-container',          # 댓글 컨테이너
                'div.like-button-container',      # 좋아요 버튼
                'div.page-labels',                # 페이지 라벨
                'div.comment-actions',            # 댓글 액션
                'span.st-table-filter',           # 스마트 테이블 필터
                'svg',                            # SVG 아이콘들
                'div.confluence-information-macro', # 정보 매크로
                'div.aui-message',                # 메시지
                'div.page-metadata-modification-info', # 수정 정보
                '.aui-page-header-actions',       # 페이지 헤더 액션
                '.like-button-container',         # 좋아요 버튼 (클래스)
                '.page-labels',                   # 페이지 라벨 (클래스)
                
                # 메타데이터 배너는 제거하되 제목/브레드크럼은 유지
                'div#page-metadata-banner',       # 메타데이터 배너
                'ul.banner',                      # 배너 리스트
            ]
            
            # 요소 제거
            for selector in elements_to_remove:
                for element in soup.select(selector):
                    element.decompose()
            
            # 메인 콘텐츠 추출 - Confluence 특화
            main_content = None
            
            # 1. Confluence의 main-content 영역 우선 찾기
            main_content = soup.find('div', {'id': 'main-content'})
            
            # 2. wiki-content 클래스가 있는 div 찾기
            if not main_content:
                main_content = soup.find('div', {'class': 'wiki-content'})
            
            # 3. 일반적인 main 태그 찾기
            if not main_content:
                main_content = soup.find('main')
            
            # 4. body 태그 찾기
            if not main_content:
                main_content = soup.find('body')
            
            # 5. 최후의 수단으로 전체 soup 사용
            if not main_content:
                main_content = soup
            
            # 텍스트 추출
            text_content = main_content.get_text(separator=' ', strip=True)
            
            # HTML 제목 추출 (fallback)
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else ""
            
            # 페이지 제목이 없으면 HTML 제목 사용
            if not page_title:
                page_title = title_text
            
            # 원복: 메인 콘텐츠만 추출하도록 간소화 (이미지/첨부/댓글/테이블 비활성화)
            images: List[Dict[str, Any]] = []
            structured_tables: List[Dict[str, Any]] = []
            tables_markdown: List[str] = []
            attachments: List[Dict[str, Any]] = []
            comments: List[Dict[str, Any]] = []
            
            # 위에서 파싱한 결과 사용

            # 새로운 JSON 구조로 구성
            result = {
                'title': page_title or title_text,  # html-title 사용 (중복 제거)
                'breadcrumbs': breadcrumbs,
                'content': {
                    'text': text_content
                },
                'metadata': {
                    'img': images,
                    'urls': urls,
                    'pagetree': pagetree,
                    'extracted_at': datetime.now().isoformat(),
                    'content_length': len(text_content),
                    'tables_markdown': tables_markdown,
                    'structured_tables': structured_tables,
                    'attachments': attachments,
                    'comments': comments
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"HTML 콘텐츠 추출 중 오류: {e}")
            return {
                'content': {
                    'text': '콘텐츠 추출 중 오류가 발생했습니다.'
                },
                'metadata': {
                    'title': 'Error',
                    'extracted_at': datetime.now().isoformat(),
                    'content_length': 0,
                    'error': str(e),
                    'tables_markdown': [],
                    'images': [],
                    'attachments': [],
                    'comments': []
                }
            }


    def extract_clean_html(self, html_content: str) -> str:
        """exclude 요소 제거 후 페이지 제목, 브레드크럼, main-content 원문 HTML을 그대로 반환"""
        soup = BeautifulSoup(html_content, 'html.parser')
        elements_to_remove = [
            'header', 'footer', 'nav', 'aside', 'sidebar',
            '.header', '.footer', '.nav', '.aside', '.sidebar',
            '.navigation', '.menu',
            'div.aui-page-header-actions', 'div.page-actions', 'div.aui-toolbar2', 
            'div.comment-container', 'div.like-button-container', 'div.page-labels', 
            'div.comment-actions', 'span.st-table-filter', 'svg',
            'div.confluence-information-macro', 'div.aui-message', 'div.page-metadata-modification-info',
            '.aui-page-header-actions', '.like-button-container', '.page-labels',
            'div#page-metadata-banner', 'ul.banner',
        ]
        for selector in elements_to_remove:
            for el in soup.select(selector):
                el.decompose()

        # 페이지 제목과 브레드크럼을 포함한 컨테이너 생성
        result_parts = []
        
        # 1. 페이지 제목 추가
        title_element = soup.find('h1', {'id': 'title-text'})
        if title_element:
            result_parts.append(f"<h1>{title_element.decode_contents()}</h1>")
        
        # 2. 브레드크럼 추가
        breadcrumb_element = soup.find('ol', {'id': 'breadcrumbs'})
        if breadcrumb_element:
            result_parts.append(f"<nav aria-label='이동 경로'>{breadcrumb_element.decode_contents()}</nav>")
        else:
            # 대안: breadcrumbs 클래스가 있는 요소 찾기
            breadcrumb_alt = soup.find('div', class_='breadcrumbs')
            if breadcrumb_alt:
                result_parts.append(f"<nav aria-label='이동 경로'>{breadcrumb_alt.decode_contents()}</nav>")
        
        # 3. 메인 콘텐츠 추가
        main_content = soup.find('div', {'id': 'main-content'})
        if not main_content:
            main_content = soup.find('div', {'class': 'wiki-content'})
        if not main_content:
            main_content = soup.find('main') or soup.find('body') or soup

        if main_content:
            try:
                main_html = main_content.decode_contents() if hasattr(main_content, 'decode_contents') else str(main_content)
                result_parts.append(main_html)
            except Exception:
                result_parts.append(str(main_content))

        return '\n'.join(result_parts)

    def _parse_tables(self, root: BeautifulSoup) -> Dict[str, Any]:
        """중첩/병합 테이블을 처리하여 구조화 + 마크다운 생성"""
        structured: List[Dict[str, Any]] = []
        markdowns: List[str] = []

        def extract_text(el) -> str:
            # 줄바꿈 로직 제거 - 원본 텍스트 유지
            text = el.get_text(separator=' ', strip=True) or ''
            return text.strip()

        def limit_text(text: str, limit: int = None) -> str:
            # 내용 생략 문제 해결 - 제한 없이 모든 내용 표시
            return text if text else ""

        def get_table_rows(table_el) -> List[Any]:
            rows: List[Any] = []
            for sec_name in ['thead', 'tbody', 'tfoot']:
                for sec in table_el.find_all(sec_name, recursive=False):
                    rows.extend(sec.find_all('tr', recursive=False))
            rows.extend(table_el.find_all('tr', recursive=False))
            return rows

        def build_grid(table_el) -> Dict[str, Any]:
            rows = get_table_rows(table_el)
            grid: List[List[str]] = []
            span_map: Dict[tuple, Dict[str, int]] = {}
            max_cols = 0
            
            for r_idx, tr in enumerate(rows):
                if len(grid) <= r_idx:
                    grid.append([])
                c_idx = 0
                
                while (r_idx, c_idx) in span_map:
                    grid[r_idx].append('')
                    span_map[(r_idx, c_idx)]['remaining_rowspan'] -= 1
                    if span_map[(r_idx, c_idx)]['remaining_rowspan'] > 0:
                        span_map[(r_idx + 1, c_idx)] = span_map[(r_idx, c_idx)].copy()
                    del span_map[(r_idx, c_idx)]
                    c_idx += 1

                for cell in tr.find_all(['td', 'th'], recursive=False):
                    cell_text = extract_text(cell)
                    rowspan = int(cell.get('rowspan', 1) or 1)
                    colspan = int(cell.get('colspan', 1) or 1)

                    grid[r_idx].append(cell_text)
                    c_idx += 1
                    for _ in range(colspan - 1):
                        grid[r_idx].append('')
                        c_idx += 1

                    if rowspan > 1:
                        for rs in range(1, rowspan):
                            for cs in range(colspan):
                                span_map[(r_idx + rs, (c_idx - colspan) + cs)] = {
                                    'text': cell_text,
                                    'remaining_rowspan': rowspan - rs
                                }
                max_cols = max(max_cols, len(grid[r_idx]))

            for r in grid:
                if len(r) < max_cols:
                    r.extend([''] * (max_cols - len(r)))

            # 빈 열 제거
            col_count = max_cols
            used: List[bool] = [False] * col_count
            for row in grid:
                for idx, val in enumerate(row):
                    if idx < col_count and (val or '').strip():
                        used[idx] = True
            keep_indices = [i for i, u in enumerate(used) if u]
            if keep_indices:
                grid = [[row[i] for i in keep_indices] for row in grid]
                max_cols = len(keep_indices)

            return {'grid': grid, 'cols': max_cols}

        def is_header_cell(cell) -> bool:
            """셀이 헤더인지 판단"""
            # 1. th 태그인 경우
            if cell.name == 'th':
                return True
            
            # 2. td 태그지만 내부에 strong/b 태그가 있는 경우
            if cell.name == 'td':
                # p > strong 또는 직접 strong/b 태그 확인
                strong_tags = cell.find_all(['strong', 'b'])
                if strong_tags:
                    # strong 태그의 텍스트가 셀 전체 텍스트의 대부분을 차지하는지 확인
                    cell_text = extract_text(cell).strip()
                    strong_text = ' '.join(extract_text(tag).strip() for tag in strong_tags)
                    if strong_text and len(strong_text) >= len(cell_text) * 0.7:  # 70% 이상
                        return True
                
                # 3. CSS 클래스 기반 헤더 감지 (Confluence 테이블)
                cell_classes = cell.get('class', [])
                if any('highlight' in str(cls) for cls in cell_classes):  # highlight-grey 등
                    return True
            
            return False

        def detect_headers(table_el, grid_obj) -> Dict[str, Any]:
            header_rows: List[List[str]] = []
            thead = table_el.find('thead')
            
            if thead and thead.find_all('tr'):
                for tr in thead.find_all('tr', recursive=False):
                    if tr.find_all(['th', 'td']):
                        expanded: List[str] = []
                        for cell in tr.find_all(['th', 'td'], recursive=False):
                            txt = extract_text(cell)
                            span = int(cell.get('colspan', 1) or 1)
                            expanded.extend([txt] * max(1, span))
                        header_rows.append(expanded)
            else:
                # 복잡한 테이블 헤더 감지 개선
                body_rows: List[Any] = []
                for tbody in table_el.find_all('tbody', recursive=False):
                    body_rows.extend(tbody.find_all('tr', recursive=False))
                if not body_rows:
                    body_rows = table_el.find_all('tr', recursive=False)
                
                # 상위 2-3행에서 헤더 패턴 찾기
                max_scan = min(3, len(body_rows))
                collected = 0
                
                for i, tr in enumerate(body_rows[:max_scan]):
                    cells = tr.find_all(['th', 'td'], recursive=False)
                    if not cells:
                        continue
                    
                    # 헤더 가능성 체크 - 엄격한 조건만 사용
                    is_likely_header = False
                    
                    # 1. th 태그 또는 strong 태그가 있는 경우
                    if any(is_header_cell(c) for c in cells):
                        is_likely_header = True
                    
                    # 2. rowspan/colspan이 있는 첫 번째 행인 경우 (복잡한 헤더 구조)
                    elif i == 0 and any(  # 첫 번째 행만 체크
                        int(c.get('rowspan', 1) or 1) > 1 or int(c.get('colspan', 1) or 1) > 1 
                        for c in cells
                    ):
                        is_likely_header = True
                    
                    if is_likely_header and collected < 3:  # 최대 3행까지 헤더로 인식
                        expanded: List[str] = []
                        for cell in cells:
                            txt = extract_text(cell).strip()
                            # 빈 셀이나 전각 공백은 빈 문자열로 처리
                            if txt == '　' or not txt:
                                txt = ''
                            span = int(cell.get('colspan', 1) or 1)
                            expanded.extend([txt] * max(1, span))
                        header_rows.append(expanded)
                        collected += 1
                    elif collected > 0:
                        # 헤더 행 이후 일반 데이터 행이 나오면 중단
                        break

            cols = grid_obj['cols']
            
            # 휴리스틱 방법 제거 - thead, th 태그만 사용
            if not header_rows:
                # 헤더가 없으면 기본 컬럼명 생성
                headers = [f"컬럼{i+1}" for i in range(cols)]
                return {'headers': headers, 'header_rows_count': 0}

            # 헤더 행 길이 보정
            norm_rows: List[List[str]] = []
            for row in header_rows:
                row = row[:cols] + [''] * max(0, cols - len(row))
                norm_rows.append(row)

            # 다중 헤더 행 병합 - 계층적 헤더명 생성
            headers = []
            for c in range(cols):
                name_parts = []
                for r in range(len(norm_rows)):
                    if norm_rows[r][c] and norm_rows[r][c].strip():
                        name_parts.append(norm_rows[r][c].strip())
                
                if name_parts:
                    # 중복 제거하면서 계층 구조 유지
                    unique_parts = []
                    for part in name_parts:
                        if part not in unique_parts:
                            unique_parts.append(part)
                    
                    if len(unique_parts) == 1:
                        name = unique_parts[0]
                    else:
                        # 계층적 헤더명: "상위헤더 > 하위헤더"
                        name = ' > '.join(unique_parts)
                else:
                    name = f"컬럼{c+1}"
                
                headers.append(name)
                
            return {'headers': headers, 'header_rows_count': len(norm_rows)}

        def preprocess_markdown_text(text: str) -> str:
            """마크다운 문법 전처리"""
            if not text:
                return text
            
            # 마크다운 특수문자 이스케이프
            text = text.replace('|', '\\|')  # 테이블 구분자
            text = text.replace('*', '\\*')  # 볼드/이탤릭
            text = text.replace('_', '\\_')  # 언더스코어
            text = text.replace('#', '\\#')  # 헤더
            text = text.replace('[', '\\[')  # 링크
            text = text.replace(']', '\\]')  # 링크
            text = text.replace('`', '\\`')  # 코드
            
            return text

        def grid_to_markdown(grid_obj, headers: List[str], header_rows_count: int, title: Optional[str]) -> str:
            lines = []
            if title:
                lines.append(f"### {title}")
                lines.append("")  # 제목 후 빈 줄
                
            # 헤더 - 마크다운 전처리 적용
            processed_headers = [preprocess_markdown_text(h) for h in headers]
            lines.append('|' + '|'.join(processed_headers) + '|')
            lines.append('|' + '|'.join(' --- ' for _ in headers) + '|')
            
            # 데이터 행 - 마크다운 전처리 적용
            data_rows = grid_obj['grid'][header_rows_count if header_rows_count > 0 else 1:]
            for row in data_rows:
                preview_vals = [preprocess_markdown_text(limit_text(str(v))) for v in row[:len(headers)]]
                if all(v.strip() == '' for v in preview_vals):
                    continue
                lines.append('|' + '|'.join(preview_vals) + '|')
            
            lines.append("")  # 테이블 후 빈 줄
            return '\n'.join(lines)

        # 테이블 파싱 실행
        table_index = 0
        for tbl in root.find_all('table'):
            try:
                table_index += 1
                grid_obj = build_grid(tbl)
                header_info = detect_headers(tbl, grid_obj)
                headers = header_info['headers']
                header_rows_count = header_info['header_rows_count']
                
                # 마크다운 생성
                table_name = f"테이블 {table_index}"
                caption = tbl.find('caption')
                if caption:
                    cap = extract_text(caption)
                    if cap:
                        table_name = cap
                        
                markdowns.append(grid_to_markdown(
                    grid_obj, headers, header_rows_count, table_name
                ))
                
            except Exception as e:
                logger.warning(f"Table parse failed at index {table_index}: {e}")
                continue

        return {'structured': structured, 'markdown': markdowns}

    def extract_markdown(self, html_content: str) -> str:
        """하이브리드 방식: 테이블은 기존 로직으로, 나머지는 pymupdf4llm/markdownify로 처리"""
        cleaned_html = self.extract_clean_html(html_content)
        soup = BeautifulSoup(cleaned_html, 'html.parser')
        
        # 1. 테이블을 별도로 파싱하여 마크다운 생성
        table_markdowns = []
        try:
            parsed_tables = self._parse_tables(soup)
            table_markdowns = parsed_tables.get('markdown', [])
        except Exception as e:
            logger.warning(f"Table parsing failed: {e}")
        
        # 2. 테이블을 제거한 HTML에서 나머지 텍스트 추출
        for table in soup.find_all('table'):
            table.decompose()
        
        remaining_html = str(soup)
        
        # 3. 나머지 내용을 마크다운으로 변환
        remaining_markdown = ""
        
        # 3-1. pymupdf4llm 시도
        if pymupdf4llm is not None and remaining_html.strip():
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_html:
                    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
</head>
<body>
    {remaining_html}
</body>
</html>"""
                    temp_html.write(full_html)
                    temp_html_path = temp_html.name
                
                markdown_result = pymupdf4llm.to_markdown(temp_html_path)
                
                try:
                    os.unlink(temp_html_path)
                except:
                    pass
                
                if markdown_result and markdown_result.strip():
                    remaining_markdown = markdown_result
                    
            except Exception as e:
                logger.warning(f"pymupdf4llm conversion failed: {e}")
                try:
                    if 'temp_html_path' in locals():
                        os.unlink(temp_html_path)
                except:
                    pass
        
        # 3-2. markdownify 폴백
        if not remaining_markdown and md_convert is not None:
            try:
                remaining_markdown = md_convert(
                    remaining_html,
                    heading_style="ATX",
                    strip=['style', 'script']
                )
            except Exception as e:
                logger.warning(f"markdownify failed: {e}")
        
        # 3-3. 최종 폴백
        if not remaining_markdown:
            try:
                soup_remaining = BeautifulSoup(remaining_html, 'html.parser')
                remaining_markdown = soup_remaining.get_text('\n', strip=True)
            except Exception:
                remaining_markdown = remaining_html
        
        # 4. 결과 조합 - 테이블 간 적절한 줄바꿈 추가
        result_parts = []
        
        if remaining_markdown.strip():
            result_parts.append(remaining_markdown.strip())
        
        if table_markdowns:
            # 테이블들을 추가하되, 각 테이블 사이에 충분한 간격 확보
            for i, table_md in enumerate(table_markdowns):
                if i > 0:  # 첫 번째 테이블이 아닌 경우
                    result_parts.append("")  # 테이블 간 빈 줄 추가
                result_parts.append(table_md.strip())
        
        return '\n\n'.join(result_parts) if result_parts else ""

    def ari_markdown_to_json(self, markdown_content: str) -> Dict[str, Any]:
        """
        ARI 마크다운을 최종 JSON 포맷(contents 배열)으로 변환합니다.
        - 헤더(#..)는 이후 text 블록의 title로 사용
        - 마크다운 테이블(|...| + | --- |)을 table 항목으로 파싱
        - 그 외 연속 텍스트를 하나의 text 항목으로 누적하여 추가
        리턴: { success, contents: [...] }
        """
        try:
            if markdown_content is None:
                return {"success": False, "error": "마크다운이 비어있습니다"}

            lines = markdown_content.splitlines()
            contents: List[Dict[str, Any]] = []
            buffer: List[str] = []  # 텍스트 누적 버퍼
            current_title: str = ""
            idx = 0
            i = 0

            def flush_text_buffer():
                nonlocal idx, buffer
                if buffer and any(s.strip() for s in buffer):
                    text = "\n".join([s.rstrip() for s in buffer]).strip()
                    if text:
                        idx += 1
                        contents.append({
                            "id": idx,
                            "type": "text",
                            "title": current_title,
                            "data": text
                        })
                buffer = []

            while i < len(lines):
                line = lines[i]

                # 제목 라인
                if re.match(r"^\s*#{1,6}\s+", line):
                    # 기존 텍스트 버퍼를 먼저 비움
                    flush_text_buffer()
                    # 현재 제목 갱신
                    current_title = re.sub(r"^\s*#{1,6}\s+", "", line).strip()
                    i += 1
                    continue

                # 테이블 감지: 현재 줄이 헤더 라인, 다음 줄이 구분선
                if '|' in line:
                    # 테이블 헤더 후보와 구분선 검사
                    header_candidate = line.strip()
                    if i + 1 < len(lines):
                        separator = lines[i + 1].strip()
                        if re.match(r"^\|\s*:?\-+\s*(\|\s*:?\-+\s*)+\|$", separator):
                            # 텍스트 버퍼를 먼저 비움
                            flush_text_buffer()

                            # 헤더 파싱
                            raw_headers = [h.strip() for h in header_candidate.strip('|').split('|')]
                            headers = [h for h in raw_headers if h != ""]

                            # 데이터 행 수집
                            i += 2  # 헤더와 구분선 건너뜀
                            rows = []
                            row_id = 0
                            while i < len(lines) and '|' in lines[i] and not re.match(r"^\s*#", lines[i]):
                                row_line = lines[i].strip()
                                if not row_line:
                                    break
                                raw_cells = [c.strip() for c in row_line.strip('|').split('|')]
                                # 셀 수가 헤더보다 적으면 보정
                                while len(raw_cells) < len(headers):
                                    raw_cells.append("")
                                data = {headers[j]: raw_cells[j] if j < len(raw_cells) else "" for j in range(len(headers))}
                                row_id += 1
                                rows.append({"row_id": row_id, "data": data})
                                i += 1

                            idx += 1
                            contents.append({
                                "id": idx,
                                "type": "table",
                                "headers": headers,
                                "rows": rows
                            })
                            continue

                # 빈 줄이면 텍스트 버퍼를 플러시
                if not line.strip():
                    flush_text_buffer()
                    i += 1
                    continue

                # 그 외는 텍스트 버퍼에 누적
                buffer.append(line)
                i += 1

            # 루프 종료 후 남은 텍스트 반영
            flush_text_buffer()

            return {"success": True, "contents": contents}
        except Exception as e:
            logger.error(f"ari_markdown_to_json 실패: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_urls(self, content_element) -> List[Dict[str, str]]:
        """메인 콘텐츠에서 링크 추출"""
        urls = []
        for link in content_element.find_all('a', href=True):
            href = link.get('href', '')
            text = link.get_text(strip=True)
            if href and text:  # 빈 링크나 텍스트 제외
                urls.append({
                    'text': text,
                    'href': href
                })
        return urls
    
    def _extract_pagetree(self, pagetree_element) -> List[Dict[str, Any]]:
        """페이지 트리 구조 추출 (사용자 제공 코드 기반)"""
        def extract_page_info(li_element) -> Optional[Dict]:
            """
            li 요소에서 페이지 정보를 추출
            """
            # span 안의 링크 요소 찾기 (viewpage.action이 포함된 링크만)
            span = li_element.find('span', class_='plugin_pagetree_children_span')
            if not span:
                return None
                
            link = span.find('a', href=lambda x: x and ('viewpage.action' in x or '/display/' in x))
            if not link:
                return None
                
            # 페이지 정보 추출
            page_info = {
                'text': link.text.strip(),
                'href': link.get('href', '')
            }
            
            # pageId 추출 (href에서)
            if 'pageId=' in page_info['href']:
                import re
                match = re.search(r'pageId=(\d+)', page_info['href'])
                if match:
                    page_info['page_id'] = match.group(1)
            
            # 자식 요소들 찾기
            children_container = li_element.find('div', class_='plugin_pagetree_children_container')
            if children_container:
                children_ul = children_container.find('ul', class_='plugin_pagetree_children_list')
                if children_ul:
                    children = []
                    for child_li in children_ul.find_all('li', recursive=False):
                        child_info = extract_page_info(child_li)
                        if child_info:
                            children.append(child_info)
                    
                    if children:
                        page_info['children'] = children
            
            return page_info
        
        # 메인 ul 요소 찾기
        main_ul = pagetree_element.find('ul', class_='plugin_pagetree_children_list')
        if not main_ul:
            return []
        
        # 최상위 li 요소들 처리
        result = []
        for li in main_ul.find_all('li', recursive=False):
            page_info = extract_page_info(li)
            if page_info:
                result.append(page_info)
        
        return result


# 싱글톤 인스턴스
ari_service = AriService()
