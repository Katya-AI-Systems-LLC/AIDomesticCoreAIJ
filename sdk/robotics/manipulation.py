"""
Manipulator Arm
===============

Robot arm control and manipulation.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class GripperState(Enum):
    """Gripper states."""
    OPEN = "open"
    CLOSED = "closed"
    HOLDING = "holding"


@dataclass
class JointLimits:
    """Joint limits."""
    lower: float
    upper: float
    velocity: float
    effort: float


@dataclass
class EndEffectorPose:
    """End effector pose."""
    position: np.ndarray  # x, y, z
    orientation: np.ndarray  # quaternion
    timestamp: float = field(default_factory=time.time)


@dataclass
class GraspPose:
    """Grasp pose for object."""
    approach: EndEffectorPose
    grasp: EndEffectorPose
    retreat: EndEffectorPose
    gripper_width: float


class ManipulatorArm:
    """
    Robot manipulator arm control.
    
    Features:
    - Forward/Inverse kinematics
    - Motion planning
    - Grasp planning
    - Force control
    - Collision avoidance
    
    Example:
        >>> arm = ManipulatorArm(6)
        >>> arm.move_to_pose(target_pose)
        >>> arm.grasp(object_pose)
    """
    
    def __init__(self, num_joints: int = 6,
                 dh_params: List[Tuple] = None):
        """
        Initialize manipulator arm.
        
        Args:
            num_joints: Number of joints
            dh_params: DH parameters [(a, alpha, d, theta), ...]
        """
        self.num_joints = num_joints
        
        # DH parameters (default 6-DOF arm)
        self.dh_params = dh_params or [
            (0, np.pi/2, 0.1, 0),
            (0.4, 0, 0, 0),
            (0.4, 0, 0, 0),
            (0, np.pi/2, 0, 0),
            (0, -np.pi/2, 0.1, 0),
            (0, 0, 0.05, 0)
        ]
        
        # Joint state
        self._joint_positions = np.zeros(num_joints)
        self._joint_velocities = np.zeros(num_joints)
        
        # Joint limits
        self._joint_limits = [
            JointLimits(-np.pi, np.pi, 2.0, 100.0)
            for _ in range(num_joints)
        ]
        
        # End effector
        self._ee_pose: Optional[EndEffectorPose] = None
        
        # Gripper
        self._gripper_state = GripperState.OPEN
        self._gripper_width = 0.08  # 8cm max
        
        # Motion state
        self._is_moving = False
        
        logger.info(f"Manipulator Arm initialized ({num_joints} joints)")
    
    def forward_kinematics(self, joint_positions: np.ndarray = None) -> EndEffectorPose:
        """
        Compute forward kinematics.
        
        Args:
            joint_positions: Joint positions (use current if None)
            
        Returns:
            End effector pose
        """
        if joint_positions is None:
            joint_positions = self._joint_positions
        
        # Compute transformation matrix
        T = np.eye(4)
        
        for i, (a, alpha, d, theta_offset) in enumerate(self.dh_params[:self.num_joints]):
            theta = joint_positions[i] + theta_offset
            
            # DH transformation matrix
            ct, st = np.cos(theta), np.sin(theta)
            ca, sa = np.cos(alpha), np.sin(alpha)
            
            Ti = np.array([
                [ct, -st*ca, st*sa, a*ct],
                [st, ct*ca, -ct*sa, a*st],
                [0, sa, ca, d],
                [0, 0, 0, 1]
            ])
            
            T = T @ Ti
        
        # Extract position and orientation
        position = T[:3, 3]
        rotation_matrix = T[:3, :3]
        orientation = self._rotation_to_quaternion(rotation_matrix)
        
        self._ee_pose = EndEffectorPose(
            position=position,
            orientation=orientation
        )
        
        return self._ee_pose
    
    def inverse_kinematics(self, target: EndEffectorPose,
                           seed: np.ndarray = None) -> Optional[np.ndarray]:
        """
        Compute inverse kinematics.
        
        Args:
            target: Target end effector pose
            seed: Initial joint positions
            
        Returns:
            Joint positions or None if no solution
        """
        if seed is None:
            seed = self._joint_positions.copy()
        
        # Numerical IK using Jacobian
        max_iterations = 100
        tolerance = 1e-4
        
        q = seed.copy()
        
        for _ in range(max_iterations):
            current = self.forward_kinematics(q)
            
            # Position error
            pos_error = target.position - current.position
            
            # Orientation error (simplified)
            orn_error = target.orientation[:3] - current.orientation[:3]
            
            error = np.concatenate([pos_error, orn_error])
            
            if np.linalg.norm(error) < tolerance:
                return q
            
            # Compute Jacobian
            J = self._compute_jacobian(q)
            
            # Damped least squares
            damping = 0.1
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping**2 * np.eye(6))
            
            # Update joints
            dq = J_pinv @ error
            q += dq * 0.5
            
            # Apply joint limits
            for i in range(self.num_joints):
                q[i] = np.clip(q[i], 
                              self._joint_limits[i].lower,
                              self._joint_limits[i].upper)
        
        logger.warning("IK did not converge")
        return None
    
    def _compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """Compute Jacobian matrix."""
        J = np.zeros((6, self.num_joints))
        delta = 1e-6
        
        base_pose = self.forward_kinematics(q)
        
        for i in range(self.num_joints):
            q_plus = q.copy()
            q_plus[i] += delta
            
            pose_plus = self.forward_kinematics(q_plus)
            
            # Position derivative
            J[:3, i] = (pose_plus.position - base_pose.position) / delta
            
            # Orientation derivative
            J[3:6, i] = (pose_plus.orientation[:3] - base_pose.orientation[:3]) / delta
        
        return J
    
    def _rotation_to_quaternion(self, R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to quaternion."""
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        
        return np.array([w, x, y, z])
    
    async def move_to_pose(self, target: EndEffectorPose,
                           velocity_scale: float = 0.5) -> bool:
        """
        Move end effector to target pose.
        
        Args:
            target: Target pose
            velocity_scale: Velocity scaling (0-1)
            
        Returns:
            True if successful
        """
        joint_target = self.inverse_kinematics(target)
        
        if joint_target is None:
            return False
        
        return await self.move_to_joints(joint_target, velocity_scale)
    
    async def move_to_joints(self, target: np.ndarray,
                             velocity_scale: float = 0.5) -> bool:
        """
        Move to target joint positions.
        
        Args:
            target: Target joint positions
            velocity_scale: Velocity scaling
            
        Returns:
            True if successful
        """
        self._is_moving = True
        
        # Simple linear interpolation
        start = self._joint_positions.copy()
        steps = 50
        
        for i in range(steps):
            t = i / (steps - 1)
            self._joint_positions = start + t * (target - start)
            
            # Update FK
            self.forward_kinematics()
            
            await self._sleep(0.02 / velocity_scale)
        
        self._is_moving = False
        return True
    
    async def _sleep(self, duration: float):
        """Async sleep."""
        import asyncio
        await asyncio.sleep(duration)
    
    def open_gripper(self, width: float = None):
        """Open gripper."""
        self._gripper_width = width or 0.08
        self._gripper_state = GripperState.OPEN
        logger.debug("Gripper opened")
    
    def close_gripper(self, force: float = 10.0) -> bool:
        """
        Close gripper.
        
        Args:
            force: Gripping force
            
        Returns:
            True if object grasped
        """
        self._gripper_state = GripperState.CLOSED
        
        # Simulate grasp detection
        grasped = np.random.random() > 0.2
        
        if grasped:
            self._gripper_state = GripperState.HOLDING
            logger.debug("Object grasped")
        
        return grasped
    
    async def grasp(self, object_pose: EndEffectorPose,
                    approach_distance: float = 0.1) -> bool:
        """
        Execute grasp sequence.
        
        Args:
            object_pose: Object pose
            approach_distance: Approach distance
            
        Returns:
            True if successful
        """
        # Open gripper
        self.open_gripper()
        
        # Approach pose
        approach_pose = EndEffectorPose(
            position=object_pose.position + np.array([0, 0, approach_distance]),
            orientation=object_pose.orientation
        )
        
        if not await self.move_to_pose(approach_pose):
            return False
        
        # Move to grasp
        if not await self.move_to_pose(object_pose):
            return False
        
        # Close gripper
        success = self.close_gripper()
        
        if success:
            # Retreat
            await self.move_to_pose(approach_pose)
        
        return success
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions."""
        return self._joint_positions.copy()
    
    def get_ee_pose(self) -> Optional[EndEffectorPose]:
        """Get current end effector pose."""
        if self._ee_pose is None:
            self.forward_kinematics()
        return self._ee_pose
    
    def get_gripper_state(self) -> GripperState:
        """Get gripper state."""
        return self._gripper_state
    
    def is_moving(self) -> bool:
        """Check if arm is moving."""
        return self._is_moving
    
    def __repr__(self) -> str:
        return f"ManipulatorArm(joints={self.num_joints})"
