"""
Agent Planner
=============

Planning and task decomposition for agents.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class Task:
    """Task definition."""
    task_id: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    dependencies: List[str] = field(default_factory=list)
    subtasks: List[str] = field(default_factory=list)
    result: Any = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


@dataclass
class Plan:
    """Execution plan."""
    goal: str
    tasks: List[Task]
    created_at: float = field(default_factory=time.time)
    estimated_steps: int = 0


class AgentPlanner:
    """
    Agent planning system.
    
    Features:
    - Goal decomposition
    - Dependency management
    - Priority scheduling
    - Plan revision
    
    Example:
        >>> planner = AgentPlanner(llm_client)
        >>> plan = await planner.create_plan("Build a web app")
        >>> next_task = planner.get_next_task()
    """
    
    PLANNING_PROMPT = """You are a task planning assistant. Given a goal, break it down into actionable tasks.

Goal: {goal}

Available tools: {tools}

Create a step-by-step plan. For each step, specify:
1. Task description
2. Required tools
3. Dependencies on previous steps

Output as JSON:
{{
    "tasks": [
        {{"id": "1", "description": "...", "tools": [...], "depends_on": []}}
    ]
}}
"""
    
    def __init__(self, llm_client: Any = None):
        """
        Initialize planner.
        
        Args:
            llm_client: LLM client for planning
        """
        self.llm_client = llm_client
        
        self._current_plan: Optional[Plan] = None
        self._tasks: Dict[str, Task] = {}
        
        logger.info("Agent Planner initialized")
    
    async def create_plan(self, goal: str,
                          available_tools: List[str] = None) -> Plan:
        """
        Create execution plan.
        
        Args:
            goal: Goal to achieve
            available_tools: Available tool names
            
        Returns:
            Plan
        """
        tools_str = ", ".join(available_tools or ["search", "calculator", "browser"])
        
        # Generate plan with LLM
        if self.llm_client:
            prompt = self.PLANNING_PROMPT.format(goal=goal, tools=tools_str)
            response = await self.llm_client.generate(prompt)
            tasks = self._parse_plan(response.text)
        else:
            # Simple fallback plan
            tasks = [
                Task(task_id="1", description=f"Analyze: {goal}", priority=1),
                Task(task_id="2", description=f"Execute: {goal}", priority=2, dependencies=["1"]),
                Task(task_id="3", description="Verify results", priority=3, dependencies=["2"])
            ]
        
        plan = Plan(
            goal=goal,
            tasks=tasks,
            estimated_steps=len(tasks)
        )
        
        self._current_plan = plan
        self._tasks = {t.task_id: t for t in tasks}
        
        logger.info(f"Plan created with {len(tasks)} tasks")
        return plan
    
    def _parse_plan(self, response: str) -> List[Task]:
        """Parse LLM plan response."""
        import json
        import re
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                tasks = []
                
                for item in data.get("tasks", []):
                    tasks.append(Task(
                        task_id=str(item.get("id", len(tasks) + 1)),
                        description=item.get("description", ""),
                        dependencies=item.get("depends_on", [])
                    ))
                
                return tasks
        except:
            pass
        
        # Fallback
        return [Task(task_id="1", description=response[:200])]
    
    def get_next_task(self) -> Optional[Task]:
        """Get next task to execute."""
        if not self._current_plan:
            return None
        
        # Find first pending task with satisfied dependencies
        for task in self._current_plan.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            
            # Check dependencies
            deps_satisfied = all(
                self._tasks.get(dep, Task("", "")).status == TaskStatus.COMPLETED
                for dep in task.dependencies
            )
            
            if deps_satisfied:
                return task
        
        return None
    
    def complete_task(self, task_id: str, result: Any = None):
        """Mark task as completed."""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = time.time()
            
            logger.info(f"Task completed: {task_id}")
    
    def fail_task(self, task_id: str, error: str = None):
        """Mark task as failed."""
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.status = TaskStatus.FAILED
            task.result = error
            
            logger.warning(f"Task failed: {task_id}")
    
    def revise_plan(self, feedback: str) -> Plan:
        """Revise plan based on feedback."""
        # Add revision task
        if self._current_plan:
            revision_task = Task(
                task_id=f"revision_{len(self._tasks) + 1}",
                description=f"Address feedback: {feedback}",
                priority=0
            )
            self._current_plan.tasks.append(revision_task)
            self._tasks[revision_task.task_id] = revision_task
        
        return self._current_plan
    
    def get_progress(self) -> Dict:
        """Get plan progress."""
        if not self._current_plan:
            return {"progress": 0, "completed": 0, "total": 0}
        
        completed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED)
        total = len(self._tasks)
        
        return {
            "progress": completed / total if total > 0 else 0,
            "completed": completed,
            "total": total,
            "pending": sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING),
            "failed": sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED)
        }
    
    def get_plan(self) -> Optional[Plan]:
        """Get current plan."""
        return self._current_plan


class TaskDecomposer:
    """
    Decompose complex tasks into subtasks.
    
    Example:
        >>> decomposer = TaskDecomposer()
        >>> subtasks = decomposer.decompose("Create a REST API")
    """
    
    def __init__(self, llm_client: Any = None,
                 max_depth: int = 3):
        """
        Initialize decomposer.
        
        Args:
            llm_client: LLM client
            max_depth: Maximum decomposition depth
        """
        self.llm_client = llm_client
        self.max_depth = max_depth
    
    async def decompose(self, task: str,
                        context: Dict = None,
                        depth: int = 0) -> List[Task]:
        """
        Decompose task into subtasks.
        
        Args:
            task: Task description
            context: Additional context
            depth: Current depth
            
        Returns:
            List of subtasks
        """
        if depth >= self.max_depth:
            return [Task(task_id=f"leaf_{depth}", description=task)]
        
        # Use LLM to decompose
        subtasks = await self._llm_decompose(task, context)
        
        # Recursively decompose if needed
        result = []
        for i, subtask in enumerate(subtasks):
            if self._is_atomic(subtask.description):
                result.append(subtask)
            else:
                children = await self.decompose(
                    subtask.description,
                    context,
                    depth + 1
                )
                subtask.subtasks = [c.task_id for c in children]
                result.append(subtask)
                result.extend(children)
        
        return result
    
    async def _llm_decompose(self, task: str,
                             context: Dict = None) -> List[Task]:
        """Use LLM to decompose task."""
        # Simple rule-based decomposition
        if "and" in task.lower():
            parts = task.lower().split(" and ")
            return [
                Task(task_id=f"sub_{i}", description=part.strip())
                for i, part in enumerate(parts)
            ]
        
        return [Task(task_id="main", description=task)]
    
    def _is_atomic(self, task: str) -> bool:
        """Check if task is atomic (can't be decomposed further)."""
        # Simple heuristic
        return len(task.split()) < 10 and "and" not in task.lower()
