"""
Base Agent
==========

Core agent implementation.
"""

from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time
import logging

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent states."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class AgentConfig:
    """Agent configuration."""
    name: str
    description: str = ""
    model: str = "gpt-4"
    temperature: float = 0.7
    max_iterations: int = 10
    timeout: float = 300.0
    verbose: bool = False


@dataclass
class AgentAction:
    """Agent action."""
    tool: str
    tool_input: Dict[str, Any]
    thought: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentObservation:
    """Observation from tool execution."""
    result: Any
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass 
class AgentStep:
    """Single agent step."""
    action: AgentAction
    observation: AgentObservation


@dataclass
class AgentResult:
    """Agent execution result."""
    output: Any
    steps: List[AgentStep]
    total_time: float
    iterations: int
    success: bool
    error: Optional[str] = None


class Agent(ABC):
    """
    Base AI agent.
    
    Features:
    - Tool execution
    - Reasoning chain
    - Memory integration
    - Error handling
    - Streaming output
    
    Example:
        >>> agent = MyAgent(config, tools=[search, calculator])
        >>> result = await agent.run("What is 2+2?")
    """
    
    def __init__(self, config: AgentConfig,
                 tools: List['Tool'] = None,
                 memory: 'AgentMemory' = None,
                 llm_client: Any = None):
        """
        Initialize agent.
        
        Args:
            config: Agent configuration
            tools: Available tools
            memory: Agent memory
            llm_client: LLM client for reasoning
        """
        self.config = config
        self.tools = {t.name: t for t in (tools or [])}
        self.memory = memory
        self.llm_client = llm_client
        
        self._state = AgentState.IDLE
        self._current_task: Optional[str] = None
        self._steps: List[AgentStep] = []
        
        # Callbacks
        self._on_step: List[Callable] = []
        self._on_state_change: List[Callable] = []
        
        logger.info(f"Agent initialized: {config.name}")
    
    @abstractmethod
    async def plan(self, task: str) -> AgentAction:
        """
        Plan next action.
        
        Args:
            task: Current task
            
        Returns:
            Next action to take
        """
        pass
    
    async def run(self, task: str,
                  context: Dict = None) -> AgentResult:
        """
        Run agent on task.
        
        Args:
            task: Task to complete
            context: Additional context
            
        Returns:
            AgentResult
        """
        start_time = time.time()
        self._current_task = task
        self._steps = []
        
        self._set_state(AgentState.THINKING)
        
        # Add to memory
        if self.memory:
            self.memory.add("user", task)
        
        iteration = 0
        final_output = None
        error = None
        
        try:
            while iteration < self.config.max_iterations:
                iteration += 1
                
                if self.config.verbose:
                    logger.info(f"Iteration {iteration}")
                
                # Plan next action
                action = await self.plan(task)
                
                # Check for completion
                if action.tool == "finish":
                    final_output = action.tool_input.get("output", "")
                    break
                
                # Execute tool
                self._set_state(AgentState.EXECUTING)
                observation = await self._execute_tool(action)
                
                # Record step
                step = AgentStep(action=action, observation=observation)
                self._steps.append(step)
                
                # Fire callbacks
                for callback in self._on_step:
                    callback(step)
                
                # Add to memory
                if self.memory:
                    self.memory.add("assistant", f"Action: {action.tool}\nResult: {observation.result}")
                
                self._set_state(AgentState.THINKING)
                
                # Check timeout
                if time.time() - start_time > self.config.timeout:
                    error = "Timeout exceeded"
                    break
            
            if iteration >= self.config.max_iterations:
                error = "Max iterations exceeded"
            
        except Exception as e:
            error = str(e)
            logger.error(f"Agent error: {e}")
            self._set_state(AgentState.ERROR)
        
        self._set_state(AgentState.COMPLETED if not error else AgentState.ERROR)
        
        return AgentResult(
            output=final_output,
            steps=self._steps,
            total_time=time.time() - start_time,
            iterations=iteration,
            success=error is None,
            error=error
        )
    
    async def _execute_tool(self, action: AgentAction) -> AgentObservation:
        """Execute tool action."""
        start_time = time.time()
        
        if action.tool not in self.tools:
            return AgentObservation(
                result=None,
                success=False,
                error=f"Unknown tool: {action.tool}",
                execution_time=0
            )
        
        tool = self.tools[action.tool]
        
        try:
            result = await tool.execute(**action.tool_input)
            
            return AgentObservation(
                result=result,
                success=True,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return AgentObservation(
                result=None,
                success=False,
                error=str(e),
                execution_time=time.time() - start_time
            )
    
    def add_tool(self, tool: 'Tool'):
        """Add tool to agent."""
        self.tools[tool.name] = tool
    
    def remove_tool(self, name: str):
        """Remove tool from agent."""
        if name in self.tools:
            del self.tools[name]
    
    def get_tools_description(self) -> str:
        """Get formatted tools description."""
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        return "\n".join(descriptions)
    
    def _set_state(self, state: AgentState):
        """Set agent state."""
        if self._state != state:
            old_state = self._state
            self._state = state
            
            for callback in self._on_state_change:
                callback(old_state, state)
    
    def get_state(self) -> AgentState:
        """Get current state."""
        return self._state
    
    def on_step(self, callback: Callable[[AgentStep], None]):
        """Register step callback."""
        self._on_step.append(callback)
    
    def on_state_change(self, callback: Callable):
        """Register state change callback."""
        self._on_state_change.append(callback)
    
    def reset(self):
        """Reset agent state."""
        self._state = AgentState.IDLE
        self._current_task = None
        self._steps = []
        
        if self.memory:
            self.memory.clear()
    
    def __repr__(self) -> str:
        return f"Agent(name='{self.config.name}', tools={list(self.tools.keys())})"
