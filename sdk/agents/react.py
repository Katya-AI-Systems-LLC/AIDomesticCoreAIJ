"""
ReAct Agent
===========

Reasoning and Acting agent implementation.
"""

from typing import Dict, Any, Optional, List
from .base import Agent, AgentConfig, AgentAction, AgentState
from .tools import Tool
from .memory import AgentMemory
import re
import logging

logger = logging.getLogger(__name__)


class ReActAgent(Agent):
    """
    ReAct (Reasoning + Acting) Agent.
    
    Implements the ReAct pattern:
    1. Thought - reason about what to do
    2. Action - select and execute a tool
    3. Observation - observe the result
    4. Repeat until done
    
    Example:
        >>> agent = ReActAgent(config, tools=[search, calculator])
        >>> result = await agent.run("What is the population of France?")
    """
    
    SYSTEM_PROMPT = """You are a helpful AI assistant that solves tasks step by step.

Available tools:
{tools}

Use this format:
Thought: Think about what to do next
Action: tool_name
Action Input: {{"param": "value"}}
Observation: (result will be provided)
... (repeat as needed)
Thought: I now have the final answer
Action: finish
Action Input: {{"output": "final answer"}}

Task: {task}

Begin!"""

    CONTINUE_PROMPT = """Previous steps:
{history}

Observation: {observation}

Continue reasoning:"""
    
    def __init__(self, config: AgentConfig,
                 tools: List[Tool] = None,
                 memory: AgentMemory = None,
                 llm_client: Any = None):
        """
        Initialize ReAct agent.
        
        Args:
            config: Agent configuration
            tools: Available tools
            memory: Agent memory
            llm_client: LLM client
        """
        super().__init__(config, tools, memory, llm_client)
        
        # Add finish tool
        self.tools["finish"] = FinishTool()
        
        # Scratchpad for reasoning
        self._scratchpad = ""
        
        logger.info(f"ReAct Agent initialized: {config.name}")
    
    async def plan(self, task: str) -> AgentAction:
        """
        Plan next action using ReAct reasoning.
        
        Args:
            task: Current task
            
        Returns:
            Next action
        """
        # Build prompt
        if not self._scratchpad:
            prompt = self.SYSTEM_PROMPT.format(
                tools=self.get_tools_description(),
                task=task
            )
        else:
            # Get last observation
            last_obs = ""
            if self._steps:
                last_obs = str(self._steps[-1].observation.result)
            
            prompt = self.CONTINUE_PROMPT.format(
                history=self._scratchpad,
                observation=last_obs
            )
        
        # Get LLM response
        if self.llm_client:
            response = await self.llm_client.generate(
                prompt,
                temperature=self.config.temperature
            )
            text = response.text
        else:
            # Simulated response
            text = self._simulate_response(task)
        
        # Parse action
        action = self._parse_action(text)
        
        # Update scratchpad
        self._scratchpad += f"\n{text}"
        
        return action
    
    def _parse_action(self, text: str) -> AgentAction:
        """Parse action from LLM response."""
        # Extract thought
        thought_match = re.search(r'Thought:\s*(.+?)(?=Action:|$)', text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        # Extract action
        action_match = re.search(r'Action:\s*(\w+)', text)
        action_name = action_match.group(1) if action_match else "finish"
        
        # Extract action input
        input_match = re.search(r'Action Input:\s*(\{.*?\})', text, re.DOTALL)
        
        if input_match:
            try:
                import json
                action_input = json.loads(input_match.group(1))
            except:
                action_input = {"input": input_match.group(1)}
        else:
            # Try to extract simple input
            simple_match = re.search(r'Action Input:\s*(.+?)(?=\n|$)', text)
            action_input = {"input": simple_match.group(1).strip()} if simple_match else {}
        
        return AgentAction(
            tool=action_name,
            tool_input=action_input,
            thought=thought
        )
    
    def _simulate_response(self, task: str) -> str:
        """Generate simulated response."""
        # Simple simulation for testing
        if "calculate" in task.lower() or "math" in task.lower():
            return """Thought: I need to use the calculator for this math problem.
Action: calculator
Action Input: {"expression": "2+2"}"""
        
        elif "search" in task.lower() or "find" in task.lower():
            return """Thought: I should search for information.
Action: search
Action Input: {"query": "relevant information"}"""
        
        else:
            return """Thought: I can answer this directly.
Action: finish
Action Input: {"output": "Based on my knowledge, here is the answer."}"""
    
    def reset(self):
        """Reset agent state."""
        super().reset()
        self._scratchpad = ""


class FinishTool(Tool):
    """Tool to indicate task completion."""
    
    name = "finish"
    description = "Use this when you have the final answer. Input should be the final output."
    
    async def execute(self, output: str = "", **kwargs) -> str:
        """Return final output."""
        return output


class ConversationalReActAgent(ReActAgent):
    """
    Conversational ReAct agent with memory.
    
    Maintains conversation context across turns.
    """
    
    CONVERSATION_PROMPT = """You are a helpful AI assistant having a conversation.

Previous conversation:
{history}

Available tools:
{tools}

Current message: {message}

Respond naturally, using tools when needed. Use the ReAct format:
Thought: ...
Action: ...
Action Input: ...

Or to respond directly:
Thought: I can respond to this directly
Action: finish
Action Input: {{"output": "your response"}}

Begin!"""
    
    async def chat(self, message: str) -> str:
        """
        Handle chat message.
        
        Args:
            message: User message
            
        Returns:
            Agent response
        """
        # Add to memory
        if self.memory:
            self.memory.add("user", message)
        
        # Run agent
        result = await self.run(message)
        
        # Add response to memory
        if self.memory and result.output:
            self.memory.add("assistant", result.output)
        
        return result.output or "I couldn't generate a response."
    
    async def plan(self, task: str) -> AgentAction:
        """Plan with conversation context."""
        # Get conversation history
        history = ""
        if self.memory:
            for item in self.memory.get_recent(10):
                history += f"{item.role.capitalize()}: {item.content}\n"
        
        prompt = self.CONVERSATION_PROMPT.format(
            history=history,
            tools=self.get_tools_description(),
            message=task
        )
        
        if self.llm_client:
            response = await self.llm_client.generate(prompt)
            text = response.text
        else:
            text = self._simulate_response(task)
        
        return self._parse_action(text)


class CodeAgent(ReActAgent):
    """
    Code-focused ReAct agent.
    
    Specialized for coding tasks.
    """
    
    CODE_PROMPT = """You are an expert programmer. Write clean, efficient code.

Task: {task}

Available tools:
{tools}

Use the ReAct format to plan and execute:
Thought: Analyze the coding task
Action: code_executor (to run code) or finish (with code as output)
Action Input: {{"code": "..."}} or {{"output": "..."}}

Begin!"""
    
    async def plan(self, task: str) -> AgentAction:
        """Plan for coding task."""
        prompt = self.CODE_PROMPT.format(
            task=task,
            tools=self.get_tools_description()
        )
        
        if self.llm_client:
            response = await self.llm_client.generate(prompt)
            return self._parse_action(response.text)
        
        # Generate simple code
        return AgentAction(
            tool="finish",
            tool_input={"output": f"# Solution for: {task}\npass"},
            thought="Generating code solution"
        )
