"""Base exceptions for the application"""

class MCPConnectionError(Exception):
    """Exception raised when MCP connection fails"""
    pass

class MCPToolExecutionError(Exception):
    """Exception raised when MCP tool execution fails"""
    pass

class LLMQueryError(Exception):
    """Exception raised when LLM query fails"""
    pass
