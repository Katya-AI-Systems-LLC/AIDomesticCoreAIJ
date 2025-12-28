"""
Multi-Agent Orchestrator
========================

Orchestrate multiple AI agents.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Agent roles in multi-agent system."""
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"
    SPECIALIST = "specialist"


@dataclass
class AgentMessage:
    """Message between agents."""
    sender: str
    receiver: str
    content: Any
    message_type: str = "request"
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None


@dataclass
class TeamConfig:
    """Team configuration."""
    name: str
    agents: List[str]
    coordinator: str
    communication: str = "broadcast"  # broadcast, direct, hierarchical


class MultiAgentOrchestrator:
    """
    Multi-agent orchestration system.
    
    Features:
    - Agent team management
    - Message routing
    - Task delegation
    - Consensus building
    - Parallel execution
    
    Example:
        >>> orchestrator = MultiAgentOrchestrator()
        >>> orchestrator.add_agent(researcher, AgentRole.RESEARCHER)
        >>> orchestrator.add_agent(executor, AgentRole.EXECUTOR)
        >>> result = await orchestrator.execute_task("Research and build X")
    """
    
    def __init__(self, max_rounds: int = 10,
                 timeout: float = 300.0):
        """
        Initialize orchestrator.
        
        Args:
            max_rounds: Maximum communication rounds
            timeout: Overall timeout
        """
        self.max_rounds = max_rounds
        self.timeout = timeout
        
        # Registered agents
        self._agents: Dict[str, Any] = {}
        self._roles: Dict[str, AgentRole] = {}
        
        # Teams
        self._teams: Dict[str, TeamConfig] = {}
        
        # Message queue
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._message_history: List[AgentMessage] = []
        
        # Callbacks
        self._on_message: List[Callable] = []
        
        logger.info("Multi-Agent Orchestrator initialized")
    
    def add_agent(self, agent: Any,
                  role: AgentRole = AgentRole.SPECIALIST,
                  agent_id: str = None):
        """
        Add agent to orchestrator.
        
        Args:
            agent: Agent instance
            role: Agent role
            agent_id: Agent identifier
        """
        agent_id = agent_id or agent.config.name
        
        self._agents[agent_id] = agent
        self._roles[agent_id] = role
        
        logger.info(f"Agent added: {agent_id} ({role.value})")
    
    def remove_agent(self, agent_id: str):
        """Remove agent from orchestrator."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            del self._roles[agent_id]
    
    def create_team(self, name: str,
                    agent_ids: List[str],
                    coordinator: str = None) -> TeamConfig:
        """
        Create agent team.
        
        Args:
            name: Team name
            agent_ids: Agent IDs in team
            coordinator: Coordinator agent ID
            
        Returns:
            TeamConfig
        """
        # Use first agent as coordinator if not specified
        coordinator = coordinator or agent_ids[0]
        
        team = TeamConfig(
            name=name,
            agents=agent_ids,
            coordinator=coordinator
        )
        
        self._teams[name] = team
        
        logger.info(f"Team created: {name} with {len(agent_ids)} agents")
        return team
    
    async def execute_task(self, task: str,
                           team: str = None,
                           strategy: str = "sequential") -> Dict:
        """
        Execute task with agents.
        
        Args:
            task: Task to execute
            team: Team name (None = all agents)
            strategy: Execution strategy
            
        Returns:
            Execution result
        """
        start_time = time.time()
        
        # Get agents
        if team and team in self._teams:
            agent_ids = self._teams[team].agents
        else:
            agent_ids = list(self._agents.keys())
        
        if not agent_ids:
            return {"error": "No agents available", "success": False}
        
        # Execute based on strategy
        if strategy == "sequential":
            result = await self._execute_sequential(task, agent_ids)
        elif strategy == "parallel":
            result = await self._execute_parallel(task, agent_ids)
        elif strategy == "debate":
            result = await self._execute_debate(task, agent_ids)
        else:
            result = await self._execute_sequential(task, agent_ids)
        
        return {
            "result": result,
            "agents_used": agent_ids,
            "strategy": strategy,
            "total_time": time.time() - start_time,
            "messages": len(self._message_history),
            "success": True
        }
    
    async def _execute_sequential(self, task: str,
                                   agent_ids: List[str]) -> Any:
        """Sequential execution."""
        result = None
        current_task = task
        
        for agent_id in agent_ids:
            agent = self._agents[agent_id]
            
            # Add context from previous agent
            context = {"previous_result": result} if result else None
            
            result = await agent.run(current_task, context)
            
            # Use output as next input if continuing
            if result.output:
                current_task = f"Continue from: {result.output}"
        
        return result
    
    async def _execute_parallel(self, task: str,
                                 agent_ids: List[str]) -> List[Any]:
        """Parallel execution."""
        tasks = []
        
        for agent_id in agent_ids:
            agent = self._agents[agent_id]
            tasks.append(agent.run(task))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return [r for r in results if not isinstance(r, Exception)]
    
    async def _execute_debate(self, task: str,
                               agent_ids: List[str],
                               rounds: int = 3) -> Any:
        """Debate-style execution."""
        responses = {}
        
        for round_num in range(rounds):
            for agent_id in agent_ids:
                agent = self._agents[agent_id]
                
                # Include other responses as context
                context = {
                    "round": round_num,
                    "other_responses": {
                        k: v for k, v in responses.items() if k != agent_id
                    }
                }
                
                result = await agent.run(task, context)
                responses[agent_id] = result.output
        
        # Synthesize final answer
        return {
            "final_responses": responses,
            "rounds": rounds
        }
    
    async def send_message(self, sender: str,
                           receiver: str,
                           content: Any,
                           message_type: str = "request") -> AgentMessage:
        """
        Send message between agents.
        
        Args:
            sender: Sender agent ID
            receiver: Receiver agent ID
            content: Message content
            message_type: Message type
            
        Returns:
            AgentMessage
        """
        message = AgentMessage(
            sender=sender,
            receiver=receiver,
            content=content,
            message_type=message_type
        )
        
        await self._message_queue.put(message)
        self._message_history.append(message)
        
        # Fire callbacks
        for callback in self._on_message:
            callback(message)
        
        return message
    
    async def broadcast(self, sender: str,
                        content: Any,
                        team: str = None):
        """Broadcast message to all agents."""
        if team and team in self._teams:
            receivers = self._teams[team].agents
        else:
            receivers = list(self._agents.keys())
        
        for receiver in receivers:
            if receiver != sender:
                await self.send_message(sender, receiver, content, "broadcast")
    
    def delegate_task(self, task: str,
                      agent_id: str = None) -> str:
        """
        Delegate task to best agent.
        
        Args:
            task: Task to delegate
            agent_id: Specific agent (None = auto-select)
            
        Returns:
            Selected agent ID
        """
        if agent_id and agent_id in self._agents:
            return agent_id
        
        # Auto-select based on role
        for aid, role in self._roles.items():
            if role == AgentRole.EXECUTOR:
                return aid
        
        # Return first available
        return next(iter(self._agents.keys()), None)
    
    def get_agent(self, agent_id: str) -> Optional[Any]:
        """Get agent by ID."""
        return self._agents.get(agent_id)
    
    def list_agents(self) -> Dict[str, AgentRole]:
        """List all agents with roles."""
        return self._roles.copy()
    
    def on_message(self, callback: Callable):
        """Register message callback."""
        self._on_message.append(callback)
    
    def get_message_history(self) -> List[AgentMessage]:
        """Get message history."""
        return self._message_history.copy()
    
    def __repr__(self) -> str:
        return f"MultiAgentOrchestrator(agents={len(self._agents)}, teams={len(self._teams)})"
