"""Schema converter utilities for MCP tools"""

def to_openai_schema(mcp_tool):
    """Convert MCP tool schema to OpenAI function calling schema"""
    # Handle both dict and object formats
    if hasattr(mcp_tool, 'name'):
        # Object format
        name = mcp_tool.name
        description = getattr(mcp_tool, 'description', '') or ""
        input_schema = getattr(mcp_tool, 'inputSchema', None) or {"type": "object", "properties": {}}
    elif isinstance(mcp_tool, dict):
        # Dict format
        name = mcp_tool.get('name', '')
        description = mcp_tool.get('description', '')
        input_schema = mcp_tool.get('inputSchema', {"type": "object", "properties": {}})
    else:
        # Fallback
        name = str(mcp_tool)
        description = ""
        input_schema = {"type": "object", "properties": {}}
    
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": input_schema
        }
    }
