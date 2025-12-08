"""MCP Client Service - Handles all MCP server interactions"""
from typing import List, Dict, Any, Optional
import asyncio
import json
import logging
from fastmcp import Client
from app.config import settings
from app.exceptions.base import MCPConnectionError, MCPToolExecutionError

logger = logging.getLogger(__name__)

class MCPService:
    """Service class for managing MCP client operations"""
    
    def __init__(self):
        self._client: Optional[Client] = None
        self._tools_cache: List[Dict[str, Any]] = []
        self._connection_lock = asyncio.Lock()
        self._tool_usage_stats: Dict[str, int] = {}  # 도구 사용 통계
        
    async def initialize(self) -> None:
        """Initialize MCP client connection"""
        async with self._connection_lock:
            if self._client is not None:
                return
                
            try:
                self._client = Client(settings.mcp_server_url)
                await self._client.__aenter__()
                
                # Cache available tools
                await self._refresh_tools_cache()
                
                logger.info(f"MCP Client connected to {settings.mcp_server_url}")
                tool_names = []
                for tool in self._tools_cache:
                    if isinstance(tool, dict) and 'function' in tool:
                        tool_names.append(tool['function'].get('name', 'unknown'))
                    else:
                        tool_names.append('unknown')
                logger.info(f"Available tools: {tool_names}")
                
            except Exception as e:
                logger.error(f"Failed to initialize MCP client: {e}")
                self._client = None
                raise MCPConnectionError(f"MCP client initialization failed: {str(e)}")
    
    async def shutdown(self) -> None:
        """Cleanup MCP client connection"""
        if self._client:
            try:
                await self._client.__aexit__(None, None, None)
                logger.info("MCP Client connection closed")
            except Exception as e:
                logger.error(f"Error during MCP client shutdown: {e}")
            finally:
                self._client = None
                self._tools_cache = []
    
    async def _refresh_tools_cache(self) -> None:
        """Refresh the cached tools list"""
        if not self._client:
            raise MCPConnectionError("MCP client not initialized")
            
        try:
            from app.utils.schema_converter import to_openai_schema
            
            logger.info("📡 MCP 도구 목록 요청 중...")
            mcp_tools = await self._client.list_tools()
            logger.info(f"📡 MCP 서버에서 {len(mcp_tools)}개 도구 받음")
            
            self._tools_cache = []
            for i, tool in enumerate(mcp_tools):
                try:
                    converted_tool = to_openai_schema(tool)
                    self._tools_cache.append(converted_tool)
                    logger.debug(f"✅ 도구 {i+1} 변환 완료: {converted_tool.get('function', {}).get('name', 'unknown')}")
                except Exception as tool_error:
                    logger.error(f"❌ 도구 {i+1} 변환 실패: {tool_error}, 도구 데이터: {tool}")
                    continue
            
            logger.info(f"✅ 총 {len(self._tools_cache)}개 도구 캐시 완료")
            
        except Exception as e:
            logger.error(f"Failed to refresh tools cache: {e}")
            # 빈 캐시로 설정하여 서비스가 계속 동작할 수 있도록 함
            self._tools_cache = []
            logger.warning("⚠️ 도구 캐시를 빈 상태로 설정하여 서비스 계속 진행")
    
    @property
    def is_connected(self) -> bool:
        """Check if MCP client is connected"""
        return self._client is not None and self._client.is_connected()
    
    @property
    def available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools"""
        return self._tools_cache.copy()
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool on the MCP server"""
        if not self.is_connected:
            raise MCPConnectionError("MCP client not connected")
        
        try:
            # 인자 요약 (HTML 내용은 크기만 표시)
            args_summary = {}
            for key, value in arguments.items():
                if key == "html_content" and isinstance(value, str):
                    args_summary[key] = f"<HTML content: {len(value)} chars>"
                elif key == "markdown_content" and isinstance(value, str):
                    args_summary[key] = f"<Markdown content: {len(value)} chars>"
                else:
                    args_summary[key] = value
            
            logger.info(f"🚀 Calling MCP tool: {tool_name} with args: {args_summary}")
            result = await self._client.call_tool(tool_name, arguments)
            
            # 사용 통계 업데이트
            self._tool_usage_stats[tool_name] = self._tool_usage_stats.get(tool_name, 0) + 1
            
            # 결과 요약 (fastmcp CallToolResult 구조 고려)
            if hasattr(result, 'content') and result.content:
                # fastmcp CallToolResult 구조
                try:
                    first_content = result.content[0]
                    if hasattr(first_content, 'text'):
                        json_data = json.loads(first_content.text)
                        result_summary = {
                            "type": "CallToolResult", 
                            "success": json_data.get('success'),
                            "has_result": 'result' in json_data,
                            "has_contents": 'contents' in json_data
                        }
                        if 'result' in json_data and isinstance(json_data['result'], dict):
                            result_data = json_data['result']
                            if 'markdown' in result_data:
                                result_summary["markdown_length"] = len(result_data['markdown'])
                        logger.info(f"✅ Tool '{tool_name}' executed successfully. Result summary: {result_summary}")
                except:
                    logger.info(f"✅ Tool '{tool_name}' executed successfully. Result type: {type(result)}")
            elif isinstance(result, dict):
                result_summary = {"type": "dict", "keys": list(result.keys())}
                if 'result' in result and isinstance(result['result'], dict):
                    result_data = result['result']
                    if 'markdown' in result_data:
                        result_summary["markdown_length"] = len(result_data['markdown'])
                logger.info(f"✅ Tool '{tool_name}' executed successfully. Result summary: {result_summary}")
            else:
                logger.info(f"✅ Tool '{tool_name}' executed successfully. Result type: {type(result)}")
            
            logger.info(f"📊 Tool usage count for '{tool_name}': {self._tool_usage_stats[tool_name]}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Tool execution failed - {tool_name}: {e}")
            raise MCPToolExecutionError(f"Failed to execute tool '{tool_name}': {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on MCP connection"""
        return {
            "connected": self.is_connected,
            "server_url": settings.mcp_server_url,
            "tools_available": len(self._tools_cache),
            "tools": [tool["name"] for tool in self._tools_cache],
            "tool_usage_stats": self._tool_usage_stats.copy()
        }
    
    def get_usage_stats(self) -> Dict[str, int]:
        """Get tool usage statistics"""
        return self._tool_usage_stats.copy()

# Global service instance
mcp_service = MCPService()
