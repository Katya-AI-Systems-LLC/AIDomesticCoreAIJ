"""
Agent Tools
===========

Tools for AI agents to interact with the world.
"""

from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolParameter:
    """Tool parameter definition."""
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum: List[Any] = None


@dataclass
class ToolSchema:
    """Tool schema for function calling."""
    name: str
    description: str
    parameters: List[ToolParameter]
    
    def to_openai_schema(self) -> Dict:
        """Convert to OpenAI function schema."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


class Tool(ABC):
    """
    Base tool class.
    
    Example:
        >>> class Calculator(Tool):
        ...     name = "calculator"
        ...     description = "Perform math calculations"
        ...     
        ...     async def execute(self, expression: str) -> str:
        ...         return str(eval(expression))
    """
    
    name: str = "tool"
    description: str = "A tool"
    
    def __init__(self):
        """Initialize tool."""
        self._call_count = 0
        self._total_time = 0.0
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute tool.
        
        Args:
            **kwargs: Tool arguments
            
        Returns:
            Tool result
        """
        pass
    
    def get_schema(self) -> ToolSchema:
        """Get tool schema."""
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=self._get_parameters()
        )
    
    def _get_parameters(self) -> List[ToolParameter]:
        """Get parameters from execute signature."""
        import inspect
        
        sig = inspect.signature(self.execute)
        params = []
        
        for name, param in sig.parameters.items():
            if name in ('self', 'kwargs'):
                continue
            
            param_type = "string"
            if param.annotation != inspect.Parameter.empty:
                if param.annotation == int:
                    param_type = "integer"
                elif param.annotation == float:
                    param_type = "number"
                elif param.annotation == bool:
                    param_type = "boolean"
                elif param.annotation == list:
                    param_type = "array"
            
            params.append(ToolParameter(
                name=name,
                type=param_type,
                description=f"Parameter {name}",
                required=param.default == inspect.Parameter.empty
            ))
        
        return params
    
    def __repr__(self) -> str:
        return f"Tool(name='{self.name}')"


class ToolRegistry:
    """
    Registry for agent tools.
    
    Example:
        >>> registry = ToolRegistry()
        >>> registry.register(Calculator())
        >>> tool = registry.get("calculator")
    """
    
    def __init__(self):
        """Initialize registry."""
        self._tools: Dict[str, Tool] = {}
        
        # Register built-in tools
        self._register_builtins()
    
    def _register_builtins(self):
        """Register built-in tools."""
        self.register(SearchTool())
        self.register(CalculatorTool())
        self.register(WebBrowserTool())
        self.register(CodeExecutorTool())
        self.register(FileReaderTool())
    
    def register(self, tool: Tool):
        """Register tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Tool registered: {tool.name}")
    
    def unregister(self, name: str):
        """Unregister tool."""
        if name in self._tools:
            del self._tools[name]
    
    def get(self, name: str) -> Optional[Tool]:
        """Get tool by name."""
        return self._tools.get(name)
    
    def list(self) -> List[Tool]:
        """List all tools."""
        return list(self._tools.values())
    
    def get_schemas(self) -> List[Dict]:
        """Get all tool schemas."""
        return [t.get_schema().to_openai_schema() for t in self._tools.values()]


# Built-in Tools

class SearchTool(Tool):
    """Web search tool."""
    
    name = "search"
    description = "Search the web for information. Returns relevant search results."
    
    async def execute(self, query: str) -> str:
        """Execute web search."""
        # Simulated search results
        return f"Search results for '{query}':\n1. Result about {query}\n2. More info on {query}"


class CalculatorTool(Tool):
    """Calculator tool."""
    
    name = "calculator"
    description = "Perform mathematical calculations. Supports +, -, *, /, ^, sqrt, etc."
    
    async def execute(self, expression: str) -> str:
        """Evaluate math expression."""
        import math
        
        # Safe evaluation
        allowed = {
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'log': math.log,
            'exp': math.exp,
            'pi': math.pi,
            'e': math.e,
            'abs': abs,
            'round': round,
            'pow': pow
        }
        
        try:
            # Replace ^ with **
            expr = expression.replace('^', '**')
            result = eval(expr, {"__builtins__": {}}, allowed)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"


class WebBrowserTool(Tool):
    """Web browser tool."""
    
    name = "browser"
    description = "Fetch and read content from a URL."
    
    async def execute(self, url: str) -> str:
        """Fetch URL content."""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    text = await response.text()
                    # Return first 2000 chars
                    return text[:2000]
        except:
            return f"[Simulated content from {url}]"


class CodeExecutorTool(Tool):
    """Code execution tool."""
    
    name = "code_executor"
    description = "Execute Python code safely. Returns stdout and result."
    
    async def execute(self, code: str) -> str:
        """Execute Python code."""
        import io
        import sys
        
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()
        
        try:
            # Execute in restricted namespace
            namespace = {"__builtins__": {"print": print, "range": range, "len": len}}
            exec(code, namespace)
            output = captured.getvalue()
            return output if output else "Code executed successfully"
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            sys.stdout = old_stdout


class FileReaderTool(Tool):
    """File reader tool."""
    
    name = "file_reader"
    description = "Read content from a file."
    
    async def execute(self, path: str) -> str:
        """Read file content."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content[:5000]  # Limit size
        except Exception as e:
            return f"Error reading file: {str(e)}"


class APICallTool(Tool):
    """API call tool."""
    
    name = "api_call"
    description = "Make HTTP API calls."
    
    async def execute(self, url: str,
                      method: str = "GET",
                      headers: Dict = None,
                      body: str = None) -> str:
        """Make API call."""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                kwargs = {"headers": headers or {}}
                if body:
                    kwargs["data"] = body
                
                async with session.request(method, url, **kwargs) as response:
                    return await response.text()
        except:
            return f"[Simulated API response from {url}]"
