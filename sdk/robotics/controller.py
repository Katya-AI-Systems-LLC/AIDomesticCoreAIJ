"""
Robot Controller
================

High-level robot control system.
"""

from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class RobotState(Enum):
    """Robot states."""
    IDLE = "idle"
    MOVING = "moving"
    EXECUTING = "executing"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"
    CHARGING = "charging"


class ControlMode(Enum):
    """Control modes."""
    MANUAL = "manual"
    AUTONOMOUS = "autonomous"
    TELEOPERATION = "teleoperation"
    FOLLOW = "follow"


@dataclass
class RobotPose:
    """Robot pose in 3D space."""
    position: np.ndarray  # x, y, z
    orientation: np.ndarray  # quaternion or euler
    timestamp: float = field(default_factory=time.time)


@dataclass
class JointState:
    """Robot joint state."""
    positions: List[float]
    velocities: List[float]
    efforts: List[float]
    names: List[str]


@dataclass
class Velocity:
    """Robot velocity command."""
    linear: np.ndarray  # x, y, z
    angular: np.ndarray  # roll, pitch, yaw rates


class RobotController:
    """
    Robot controller system.
    
    Features:
    - Motion control
    - State management
    - Safety monitoring
    - Task execution
    - Multi-robot coordination
    
    Example:
        >>> robot = RobotController("robot_001")
        >>> robot.move_to(target_pose)
        >>> robot.execute_task("pick_and_place", params)
    """
    
    def __init__(self, robot_id: str,
                 robot_type: str = "mobile",
                 num_joints: int = 0):
        """
        Initialize robot controller.
        
        Args:
            robot_id: Robot identifier
            robot_type: Robot type (mobile, arm, humanoid)
            num_joints: Number of joints for arm robots
        """
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.num_joints = num_joints
        
        # State
        self._state = RobotState.IDLE
        self._control_mode = ControlMode.MANUAL
        self._pose = RobotPose(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.0, 1.0])
        )
        self._joint_state = JointState(
            positions=[0.0] * num_joints,
            velocities=[0.0] * num_joints,
            efforts=[0.0] * num_joints,
            names=[f"joint_{i}" for i in range(num_joints)]
        )
        
        # Velocity
        self._current_velocity = Velocity(
            linear=np.zeros(3),
            angular=np.zeros(3)
        )
        
        # Limits
        self._max_linear_velocity = 1.0  # m/s
        self._max_angular_velocity = 1.0  # rad/s
        
        # Tasks
        self._current_task: Optional[Dict] = None
        self._task_queue: List[Dict] = []
        
        # Callbacks
        self._on_state_change: List[Callable] = []
        self._on_goal_reached: List[Callable] = []
        
        # Safety
        self._emergency_stop = False
        self._obstacles: List[Dict] = []
        
        logger.info(f"Robot Controller initialized: {robot_id} ({robot_type})")
    
    def set_control_mode(self, mode: ControlMode):
        """Set control mode."""
        self._control_mode = mode
        logger.info(f"Control mode set to {mode.value}")
    
    def set_velocity(self, linear: np.ndarray,
                     angular: np.ndarray):
        """
        Set velocity command.
        
        Args:
            linear: Linear velocity (x, y, z)
            angular: Angular velocity (roll, pitch, yaw rates)
        """
        if self._emergency_stop:
            logger.warning("Emergency stop active, ignoring velocity command")
            return
        
        # Apply limits
        linear_mag = np.linalg.norm(linear)
        if linear_mag > self._max_linear_velocity:
            linear = linear / linear_mag * self._max_linear_velocity
        
        angular_mag = np.linalg.norm(angular)
        if angular_mag > self._max_angular_velocity:
            angular = angular / angular_mag * self._max_angular_velocity
        
        self._current_velocity = Velocity(linear=linear, angular=angular)
        
        if np.linalg.norm(linear) > 0 or np.linalg.norm(angular) > 0:
            self._set_state(RobotState.MOVING)
        else:
            self._set_state(RobotState.IDLE)
    
    def stop(self):
        """Stop robot motion."""
        self._current_velocity = Velocity(
            linear=np.zeros(3),
            angular=np.zeros(3)
        )
        self._set_state(RobotState.IDLE)
    
    def emergency_stop(self):
        """Activate emergency stop."""
        self._emergency_stop = True
        self.stop()
        self._set_state(RobotState.EMERGENCY_STOP)
        logger.warning(f"Emergency stop activated for {self.robot_id}")
    
    def reset_emergency_stop(self):
        """Reset emergency stop."""
        self._emergency_stop = False
        self._set_state(RobotState.IDLE)
        logger.info(f"Emergency stop reset for {self.robot_id}")
    
    async def move_to(self, target: RobotPose,
                      tolerance: float = 0.1) -> bool:
        """
        Move robot to target pose.
        
        Args:
            target: Target pose
            tolerance: Position tolerance
            
        Returns:
            True if goal reached
        """
        if self._emergency_stop:
            return False
        
        self._set_state(RobotState.MOVING)
        
        # Compute path (simplified)
        direction = target.position - self._pose.position
        distance = np.linalg.norm(direction)
        
        while distance > tolerance:
            if self._emergency_stop:
                return False
            
            # Check obstacles
            if self._check_collision(self._pose.position + direction * 0.1):
                logger.warning("Obstacle detected, stopping")
                self.stop()
                return False
            
            # Update position (simulated)
            step = direction / distance * min(0.1, distance)
            self._pose.position = self._pose.position + step
            self._pose.timestamp = time.time()
            
            distance = np.linalg.norm(target.position - self._pose.position)
            
            await self._sleep(0.1)
        
        self._pose.orientation = target.orientation
        self._set_state(RobotState.IDLE)
        
        # Fire callbacks
        for callback in self._on_goal_reached:
            callback(target)
        
        logger.info(f"Goal reached: {target.position}")
        return True
    
    async def _sleep(self, duration: float):
        """Async sleep."""
        import asyncio
        await asyncio.sleep(duration)
    
    def _check_collision(self, position: np.ndarray) -> bool:
        """Check for collision at position."""
        for obstacle in self._obstacles:
            obs_pos = np.array(obstacle["position"])
            obs_radius = obstacle.get("radius", 0.5)
            
            if np.linalg.norm(position - obs_pos) < obs_radius:
                return True
        
        return False
    
    def set_joint_positions(self, positions: List[float],
                            velocities: List[float] = None):
        """Set joint positions (for arm robots)."""
        if len(positions) != self.num_joints:
            raise ValueError(f"Expected {self.num_joints} positions")
        
        self._joint_state.positions = positions
        if velocities:
            self._joint_state.velocities = velocities
    
    def get_joint_state(self) -> JointState:
        """Get current joint state."""
        return self._joint_state
    
    async def execute_task(self, task_name: str,
                           params: Dict = None) -> bool:
        """
        Execute a task.
        
        Args:
            task_name: Task name
            params: Task parameters
            
        Returns:
            True if successful
        """
        self._current_task = {
            "name": task_name,
            "params": params or {},
            "started_at": time.time()
        }
        
        self._set_state(RobotState.EXECUTING)
        
        try:
            if task_name == "pick_and_place":
                return await self._task_pick_and_place(params)
            elif task_name == "patrol":
                return await self._task_patrol(params)
            elif task_name == "follow":
                return await self._task_follow(params)
            else:
                logger.warning(f"Unknown task: {task_name}")
                return False
                
        finally:
            self._current_task = None
            self._set_state(RobotState.IDLE)
    
    async def _task_pick_and_place(self, params: Dict) -> bool:
        """Pick and place task."""
        pick_pos = params.get("pick_position")
        place_pos = params.get("place_position")
        
        if not pick_pos or not place_pos:
            return False
        
        # Move to pick position
        await self.move_to(RobotPose(position=np.array(pick_pos), orientation=np.array([0,0,0,1])))
        
        # Simulate pick
        await self._sleep(0.5)
        logger.info("Object picked")
        
        # Move to place position
        await self.move_to(RobotPose(position=np.array(place_pos), orientation=np.array([0,0,0,1])))
        
        # Simulate place
        await self._sleep(0.5)
        logger.info("Object placed")
        
        return True
    
    async def _task_patrol(self, params: Dict) -> bool:
        """Patrol task."""
        waypoints = params.get("waypoints", [])
        
        for wp in waypoints:
            pose = RobotPose(position=np.array(wp), orientation=np.array([0,0,0,1]))
            if not await self.move_to(pose):
                return False
        
        return True
    
    async def _task_follow(self, params: Dict) -> bool:
        """Follow target task."""
        # Simplified follow implementation
        return True
    
    def add_obstacle(self, position: List[float], radius: float = 0.5):
        """Add obstacle to environment."""
        self._obstacles.append({
            "position": position,
            "radius": radius
        })
    
    def clear_obstacles(self):
        """Clear all obstacles."""
        self._obstacles = []
    
    def _set_state(self, state: RobotState):
        """Set robot state."""
        if self._state != state:
            old_state = self._state
            self._state = state
            
            for callback in self._on_state_change:
                callback(old_state, state)
    
    def get_state(self) -> RobotState:
        """Get current state."""
        return self._state
    
    def get_pose(self) -> RobotPose:
        """Get current pose."""
        return self._pose
    
    def on_state_change(self, callback: Callable):
        """Register state change callback."""
        self._on_state_change.append(callback)
    
    def on_goal_reached(self, callback: Callable):
        """Register goal reached callback."""
        self._on_goal_reached.append(callback)
    
    def get_status(self) -> Dict:
        """Get robot status."""
        return {
            "robot_id": self.robot_id,
            "robot_type": self.robot_type,
            "state": self._state.value,
            "control_mode": self._control_mode.value,
            "position": self._pose.position.tolist(),
            "orientation": self._pose.orientation.tolist(),
            "velocity": {
                "linear": self._current_velocity.linear.tolist(),
                "angular": self._current_velocity.angular.tolist()
            },
            "emergency_stop": self._emergency_stop,
            "current_task": self._current_task
        }
    
    def __repr__(self) -> str:
        return f"RobotController(id='{self.robot_id}', state={self._state.value})"
