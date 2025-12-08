"""LLM Service - Handles OpenAI API interactions"""
from typing import List, Dict, Any, AsyncGenerator, Optional
import json
import logging
import asyncio
import httpx
import urllib3
from datetime import datetime
import tiktoken
import numpy as np
from openai import AsyncOpenAI
from app.config import settings
from app.infrastructure.mcp.mcp_service import mcp_service
from app.exceptions.base import LLMQueryError

logger = logging.getLogger(__name__)

# 임베딩 모델 사용하지 않음 - OpenAI API만 사용
EMBEDDING_AVAILABLE = False
logger.info("임베딩 모델 사용 안함 - OpenAI API만으로 의도 분류")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
no_ssl_httpx = httpx.AsyncClient(verify=False)


class LLMService:
    """Service class for managing LLM interactions"""
    
    def __init__(self):
        # Azure OpenAI 클라이언트 초기화 (우선순위)
        self._azure_client = None
        self._openai_client = None
        self.azure_available = False
        
        if settings.azure_openai_enabled:
            logger.info("🔵 Azure OpenAI 클라이언트 초기화 (Standard OpenAI 방식)")
            # Azure OpenAI를 Standard OpenAI 방식으로 사용 (/openai/v1/ 경로 포함)
            azure_base_url = f"{settings.azure_openai_api_base}/openai/v1/"
            logger.info(f"🔗 Azure Base URL: {azure_base_url}")
            
            self._azure_client = AsyncOpenAI(
                api_key=settings.azure_openai_api_key,
                base_url=azure_base_url,
                http_client=no_ssl_httpx
            )
            self.azure_model = settings.azure_openai_model
            self.azure_available = True
        
        # 일반 OpenAI 클라이언트 초기화 (폴백용)
        if settings.openai_api_key:
            logger.info("🟢 일반 OpenAI 클라이언트 초기화 (폴백용)")
            self._openai_client = AsyncOpenAI(api_key=settings.openai_api_key, http_client=no_ssl_httpx)
            self.openai_model = settings.openai_model
        
        # 기본 클라이언트 설정
        self._client = self._azure_client if self._azure_client else self._openai_client
        self.current_model = self.azure_model if self._azure_client else self.openai_model
        self.is_azure = bool(self._azure_client)
            
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    async def _call_llm_with_fallback(self, **kwargs) -> Any:
        """
        Azure OpenAI 호출 후 실패 시 일반 OpenAI로 자동 폴백
        
        Args:
            **kwargs: OpenAI API에 전달할 파라미터들
            
        Returns:
            API 응답 객체
        """
        # 1차 시도: Azure OpenAI (설정된 경우)
        if self._azure_client and self.azure_available:
            try:
                logger.info(f"🔵 Azure OpenAI API 호출 시작: model={self.azure_model}")
                response = await self._azure_client.chat.completions.create(
                    model=self.azure_model,
                    **kwargs
                )
                logger.info(f"✅ Azure OpenAI API 응답 성공")
                return response
                
            except Exception as azure_error:
                error_msg = str(azure_error)
                error_code = getattr(azure_error, 'status_code', None)
                
                # 403 에러 또는 연결 실패 시 폴백
                if error_code == 403 or '403' in error_msg:
                    logger.warning(f"⚠️ Azure OpenAI 403 Forbidden - 내부망 접근 불가")
                    logger.warning(f"🔄 일반 OpenAI로 폴백 시도...")
                elif 'timeout' in error_msg.lower() or 'connection' in error_msg.lower():
                    logger.warning(f"⚠️ Azure OpenAI 연결 실패: {error_msg[:100]}")
                    logger.warning(f"🔄 일반 OpenAI로 폴백 시도...")
                else:
                    # 다른 에러는 그대로 raise
                    logger.error(f"❌ Azure OpenAI API 오류: {error_msg[:200]}")
                    raise
                
                # Azure 사용 불가능으로 마킹
                self.azure_available = False
        
        # 2차 시도: 일반 OpenAI (폴백)
        if self._openai_client:
            try:
                logger.info(f"🟢 일반 OpenAI API 호출 시작: model={self.openai_model}")
                response = await self._openai_client.chat.completions.create(
                    model=self.openai_model,
                    **kwargs
                )
                logger.info(f"✅ 일반 OpenAI API 응답 성공")
                return response
                
            except Exception as openai_error:
                logger.error(f"❌ 일반 OpenAI API 오류: {str(openai_error)[:200]}")
                raise LLMQueryError(f"OpenAI API 호출 실패: {str(openai_error)}")
        
        # 사용 가능한 클라이언트가 없음
        raise LLMQueryError("사용 가능한 LLM 클라이언트가 없습니다 (Azure, OpenAI 모두 실패)")
    
    
    def _format_tools_for_openai(self, available_tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert MCP tools format to OpenAI tools format
        
        MCP format might be: {"name": "tool_name", "description": "...", "parameters": {...}}
        OpenAI expects: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
        """
        formatted_tools = []
        for tool in available_tools:
            # Check if already in OpenAI format
            if "type" in tool and tool["type"] == "function" and "function" in tool:
                formatted_tools.append(tool)
            else:
                # Convert from MCP format to OpenAI format
                formatted_tool = {
                    "type": "function",
                    "function": {
                        "name": tool.get("name", ""),
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters", {
                            "type": "object",
                            "properties": {},
                            "required": []
                        })
                    }
                }
                formatted_tools.append(formatted_tool)
        return formatted_tools
    
    async def query_with_raw_result_and_html(self, question: str, available_tools: List[Dict[str, Any]], html_content: str) -> tuple[str, List[str]]:
        """
        HTML을 마크다운으로 변환 (MCP 도구 호출 방식)
        - 사용자가 "추출해줘", "요약해줘" 등의 요청을 하면 ari_html_to_markdown 도구 호출
        - trafilatura + BeautifulSoup + markdownify로 RAG 친화적인 변환
        """
        try:
            logger.info(f"🤖 HTML 처리 시작 - 사용자 질문: {question}")
            logger.info(f"📄 HTML 크기: {len(html_content):,} bytes")
            
            # 사용된 도구들 추적
            used_tools = []
            
            # 1단계: 모델이 의도 분류하여 도구 선택
            formatted_tools = self._format_tools_for_openai(available_tools)
            
            system_prompt = """You are an AI assistant that analyzes user requests and determines the appropriate action.

When a user uploads an HTML file, analyze their question to understand their intent:

1. **HTML Processing Intent** (추출/요약/분석/변환 요청):
   - Keywords: "추출", "요약", "분석", "변환", "정리", "파싱", "extract", "summarize", "parse", "convert"
   - Action: Call ari_html_to_markdown to process the HTML file
   - Example: "HTML 내용을 추출해줘", "이 파일을 요약해줘", "내용 정리해줘"

2. **General Question** (일반 질문):
   - User asks about something unrelated to HTML processing
   - Action: Answer directly without calling tools
   - Example: "이 파일이 뭐야?", "HTML이란?", "오늘 날씨는?"

Available tools:
- ari_html_to_markdown: Converts HTML to RAG-friendly markdown format (USE THIS for HTML processing)
- health_check: System status check

**Instructions:**
- Carefully analyze the user's question to understand their intent
- If they want to extract/summarize/analyze the HTML content → call ari_html_to_markdown
- If they ask a general question → answer directly without tools
- Always respond in Korean for final answers"""

            user_message = f"""사용자가 HTML 파일을 업로드했습니다.

사용자 질문: {question}

위 질문의 의도를 분석하여 적절한 도구를 호출하거나 답변해주세요."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # 모델이 의도 분류 및 도구 선택
            logger.info("🤖 모델이 사용자 의도를 분석하여 도구 선택 중...")
            response = await self._call_llm_with_fallback(
                messages=messages,
                tools=formatted_tools,
                tool_choice="auto",
                timeout=60,
                max_tokens=500
            )
            
            message = response.choices[0].message
            
            if not message.tool_calls:
                # 도구 호출 없음 = 일반 질문으로 판단
                logger.info("💬 모델이 도구 호출 없이 직접 답변하기로 결정")
                answer = message.content or "죄송합니다. 요청을 처리할 수 없습니다."
                # HTML 처리 요청인데 도구 호출이 없으면 에러
                if any(keyword in question.lower() for keyword in ['추출', '요약', '분석', '변환', '정리', 'extract', 'parse', 'convert']):
                    logger.error("❌ HTML 처리 요청이지만 모델이 도구를 호출하지 않음")
                    raise LLMQueryError("HTML 처리 요청이지만 적절한 도구를 선택하지 못했습니다. API 키를 확인해주세요.")
                return answer, []
            
            # 2단계: HTML → 마크다운 변환
            logger.info(f"🔧 모델이 {len(message.tool_calls)}개 도구 호출 결정")
            
            tool_call = message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            logger.info(f"📄 선택된 도구: {function_name}")
            used_tools.append(function_name)
            
            # HTML 처리 도구에 실제 HTML 내용 전달
            if function_name == "ari_html_to_markdown":
                function_args["html_content"] = html_content
                logger.info(f"📄 HTML 내용 전달: {len(html_content)} bytes")
            
            # 도구 실행
            try:
                result = await mcp_service.call_tool(function_name, function_args)
                
                # fastmcp CallToolResult 구조 파싱
                markdown_content = ""
                success = False
                
                if hasattr(result, 'content') and result.content:
                    first_content = result.content[0]
                    if hasattr(first_content, 'text'):
                        json_data = json.loads(first_content.text)
                        logger.info(f"📄 도구 실행 결과: {json_data.get('success')}")
                        
                        if json_data.get('success') and 'result' in json_data:
                            result_data = json_data['result']
                            if 'markdown' in result_data:
                                success = True
                                markdown_content = result_data['markdown']
                                logger.info(f"✅ 마크다운 변환 완료: {len(markdown_content):,} 자")
                
                if not success:
                    logger.error("❌ HTML → 마크다운 변환 실패")
                    raise LLMQueryError("HTML → 마크다운 변환에 실패했습니다")
                
                # 3단계: 크기 제한 (1MB)
                max_response_size = 1024 * 1024  # 1MB
                content_size = len(markdown_content.encode('utf-8'))
                
                if content_size > max_response_size:
                    logger.warning(f"⚠️ 응답이 너무 큽니다 ({content_size:,} bytes). 1MB로 제한합니다.")
                    content_bytes = markdown_content.encode('utf-8')
                    truncated_bytes = content_bytes[:max_response_size-100]
                    
                    try:
                        markdown_content = truncated_bytes.decode('utf-8')
                    except UnicodeDecodeError:
                        for i in range(1, 11):
                            try:
                                markdown_content = truncated_bytes[:-i].decode('utf-8')
                                break
                            except UnicodeDecodeError:
                                continue
                    
                    markdown_content += "\n\n... (내용이 너무 길어 일부만 표시됩니다)"
                    logger.info(f"📏 응답 크기 조정됨: {len(markdown_content):,} characters")
                
                logger.info(f"✅ HTML 처리 완료 - 최종 크기: {len(markdown_content):,} characters")
                return markdown_content, used_tools
                    
            except Exception as e:
                logger.error(f"❌ 도구 실행 오류: {e}")
                raise LLMQueryError(f"MCP 도구 실행 중 오류가 발생했습니다: {str(e)}")
            
        except Exception as e:
            logger.error(f"HTML 처리 플로우 실패: {e}")
            # LLM 실패 시 예외를 발생시켜 상위 레이어에서 적절히 처리하도록 함
            raise LLMQueryError(f"HTML 처리 중 오류가 발생했습니다: {str(e)}")
    
    async def generate_response(self, prompt: str) -> str:
        """
        Generate a simple response using OpenAI Chat Completions API
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Generated response as string
        """
        try:
            # 토큰 수 계산 및 제한 확인
            tokens = self.tokenizer.encode(prompt)
            token_count = len(tokens)
            
            # RAG용으로는 gpt-4o-mini 사용 (더 저렴하고 토큰 제한이 큼)
            model = "gpt-4o-mini"  # RAG 응답 생성용
            
            # 토큰 제한 확인 (gpt-4o-mini는 128k context)
            max_tokens = 20000  # 더 안전한 제한
            if token_count > max_tokens:
                logger.warning(f"Prompt too long ({token_count} tokens), truncating to {max_tokens}")
                tokens = tokens[:max_tokens]
                prompt = self.tokenizer.decode(tokens)
                token_count = max_tokens
            
            logger.info(f"LLM request: {token_count} tokens")
            
            # Azure와 OpenAI에서 다른 모델 사용
            if self._azure_client and self.azure_available:
                # Azure는 설정된 모델 사용
                response = await self._call_llm_with_fallback(
                    messages=[
                        {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다. 주어진 정보를 바탕으로 정확하고 유용한 답변을 제공합니다."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,  # 응답 토큰 수 증가
                    temperature=0.7
                )
            elif self._openai_client:
                # 일반 OpenAI는 gpt-4o-mini 사용 (RAG용)
                response = await self._openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "당신은 도움이 되는 AI 어시스턴트입니다. 주어진 정보를 바탕으로 정확하고 유용한 답변을 제공합니다."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.7
                )
                logger.info(f"✅ 일반 OpenAI (gpt-4o-mini) 응답 성공")
            else:
                raise LLMQueryError("사용 가능한 LLM 클라이언트가 없습니다")
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise LLMQueryError(f"Failed to generate response: {str(e)}")
    
    async def generate_response_stream(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        토큰 단위 스트리밍 응답 생성 (ChatGPT 스타일)
        Azure → OpenAI 자동 폴백 지원
        
        Args:
            prompt: 사용자 프롬프트
            system_prompt: 시스템 프롬프트 (선택)
            
        Yields:
            토큰 단위 문자열
        """
        # 토큰 수 계산 및 제한 확인
        tokens = self.tokenizer.encode(prompt)
        token_count = len(tokens)
        
        max_tokens = 20000
        if token_count > max_tokens:
            logger.warning(f"Prompt too long ({token_count} tokens), truncating")
            tokens = tokens[:max_tokens]
            prompt = self.tokenizer.decode(tokens)
        
        logger.info(f"🌊 스트리밍 요청: {token_count} tokens")
        
        messages = [
            {
                "role": "system", 
                "content": system_prompt or "당신은 도움이 되는 AI 어시스턴트입니다. 주어진 정보를 바탕으로 정확하고 유용한 답변을 제공합니다."
            },
            {"role": "user", "content": prompt}
        ]
        
        # 1차 시도: Azure OpenAI (설정된 경우)
        if self._azure_client and self.azure_available:
            try:
                logger.info(f"🔵 Azure OpenAI 스트리밍 시작: model={self.azure_model}")
                stream = await self._azure_client.chat.completions.create(
                    model=self.azure_model,
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.7,
                    stream=True
                )
                
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                
                logger.info("✅ Azure OpenAI 스트리밍 완료")
                return  # 성공 시 종료
                
            except Exception as azure_error:
                error_msg = str(azure_error)
                error_code = getattr(azure_error, 'status_code', None)
                
                if error_code == 403 or '403' in error_msg:
                    logger.warning(f"⚠️ Azure OpenAI 403 Forbidden - 일반 OpenAI로 폴백")
                elif 'timeout' in error_msg.lower() or 'connection' in error_msg.lower():
                    logger.warning(f"⚠️ Azure OpenAI 연결 실패 - 일반 OpenAI로 폴백")
                else:
                    logger.error(f"❌ Azure OpenAI 스트리밍 오류: {error_msg[:200]}")
                    raise LLMQueryError(f"스트리밍 응답 생성 실패: {error_msg}")
                
                self.azure_available = False
        
        # 2차 시도: 일반 OpenAI (폴백)
        if self._openai_client:
            try:
                logger.info(f"🟢 일반 OpenAI 스트리밍 시작: model=gpt-4o-mini")
                stream = await self._openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=2000,
                    temperature=0.7,
                    stream=True
                )
                
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                
                logger.info("✅ 일반 OpenAI 스트리밍 완료")
                return
                
            except Exception as openai_error:
                logger.error(f"❌ 일반 OpenAI 스트리밍 오류: {str(openai_error)[:200]}")
                raise LLMQueryError(f"스트리밍 응답 생성 실패: {str(openai_error)}")
        
        # 사용 가능한 클라이언트가 없음
        raise LLMQueryError("사용 가능한 LLM 클라이언트가 없습니다 (Azure, OpenAI 모두 실패)")

# Global service instance
llm_service = LLMService()
