"""
Navigation System
=================

Robot navigation and path planning.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import heapq
import time
import logging

logger = logging.getLogger(__name__)


class PlannerType(Enum):
    """Path planner types."""
    ASTAR = "astar"
    RRT = "rrt"
    PRM = "prm"
    DIJKSTRA = "dijkstra"
    DWA = "dwa"  # Dynamic Window Approach


@dataclass
class Waypoint:
    """Navigation waypoint."""
    position: np.ndarray
    orientation: Optional[np.ndarray] = None
    speed: float = 1.0
    tolerance: float = 0.1


@dataclass
class Path:
    """Navigation path."""
    waypoints: List[Waypoint]
    total_distance: float
    estimated_time: float
    planner: PlannerType


@dataclass
class OccupancyGrid:
    """2D occupancy grid map."""
    data: np.ndarray
    resolution: float  # meters per cell
    origin: np.ndarray  # world coordinates of (0,0)
    width: int
    height: int


class NavigationSystem:
    """
    Robot navigation system.
    
    Features:
    - Multiple path planners
    - Obstacle avoidance
    - Dynamic replanning
    - Localization
    - Map management
    
    Example:
        >>> nav = NavigationSystem()
        >>> nav.load_map(occupancy_grid)
        >>> path = nav.plan_path(start, goal)
    """
    
    def __init__(self, planner: PlannerType = PlannerType.ASTAR):
        """
        Initialize navigation system.
        
        Args:
            planner: Path planner type
        """
        self.planner_type = planner
        
        # Map
        self._map: Optional[OccupancyGrid] = None
        
        # Current path
        self._current_path: Optional[Path] = None
        
        # Localization
        self._position = np.array([0.0, 0.0, 0.0])
        self._orientation = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Parameters
        self._inflation_radius = 0.3  # meters
        self._robot_radius = 0.25
        
        logger.info(f"Navigation System initialized (planner={planner.value})")
    
    def load_map(self, grid: OccupancyGrid):
        """Load occupancy grid map."""
        self._map = grid
        logger.info(f"Map loaded: {grid.width}x{grid.height}")
    
    def create_empty_map(self, width: int, height: int,
                         resolution: float = 0.05,
                         origin: np.ndarray = None) -> OccupancyGrid:
        """Create empty occupancy grid."""
        grid = OccupancyGrid(
            data=np.zeros((height, width), dtype=np.uint8),
            resolution=resolution,
            origin=origin if origin is not None else np.array([0.0, 0.0]),
            width=width,
            height=height
        )
        self._map = grid
        return grid
    
    def add_obstacle(self, position: np.ndarray, radius: float):
        """Add circular obstacle to map."""
        if self._map is None:
            return
        
        # Convert to grid coordinates
        gx = int((position[0] - self._map.origin[0]) / self._map.resolution)
        gy = int((position[1] - self._map.origin[1]) / self._map.resolution)
        grid_radius = int(radius / self._map.resolution)
        
        # Fill obstacle cells
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                if dx*dx + dy*dy <= grid_radius*grid_radius:
                    x, y = gx + dx, gy + dy
                    if 0 <= x < self._map.width and 0 <= y < self._map.height:
                        self._map.data[y, x] = 255  # Occupied
    
    def plan_path(self, start: np.ndarray,
                  goal: np.ndarray) -> Optional[Path]:
        """
        Plan path from start to goal.
        
        Args:
            start: Start position
            goal: Goal position
            
        Returns:
            Path or None if no path found
        """
        if self._map is None:
            logger.warning("No map loaded")
            return None
        
        # Convert to grid coordinates
        start_grid = self._world_to_grid(start)
        goal_grid = self._world_to_grid(goal)
        
        # Plan based on planner type
        if self.planner_type == PlannerType.ASTAR:
            grid_path = self._astar(start_grid, goal_grid)
        elif self.planner_type == PlannerType.DIJKSTRA:
            grid_path = self._dijkstra(start_grid, goal_grid)
        elif self.planner_type == PlannerType.RRT:
            grid_path = self._rrt(start_grid, goal_grid)
        else:
            grid_path = self._astar(start_grid, goal_grid)
        
        if not grid_path:
            logger.warning("No path found")
            return None
        
        # Convert to world coordinates
        waypoints = []
        total_distance = 0.0
        
        for i, (gx, gy) in enumerate(grid_path):
            world_pos = self._grid_to_world(gx, gy)
            waypoints.append(Waypoint(position=world_pos))
            
            if i > 0:
                prev_pos = waypoints[i-1].position
                total_distance += np.linalg.norm(world_pos - prev_pos)
        
        path = Path(
            waypoints=waypoints,
            total_distance=total_distance,
            estimated_time=total_distance / 0.5,  # Assuming 0.5 m/s
            planner=self.planner_type
        )
        
        self._current_path = path
        logger.info(f"Path planned: {len(waypoints)} waypoints, {total_distance:.2f}m")
        
        return path
    
    def _world_to_grid(self, position: np.ndarray) -> Tuple[int, int]:
        """Convert world to grid coordinates."""
        gx = int((position[0] - self._map.origin[0]) / self._map.resolution)
        gy = int((position[1] - self._map.origin[1]) / self._map.resolution)
        return (gx, gy)
    
    def _grid_to_world(self, gx: int, gy: int) -> np.ndarray:
        """Convert grid to world coordinates."""
        wx = gx * self._map.resolution + self._map.origin[0]
        wy = gy * self._map.resolution + self._map.origin[1]
        return np.array([wx, wy, 0.0])
    
    def _astar(self, start: Tuple[int, int],
               goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* path planning."""
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_neighbors(pos):
            neighbors = []
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (-1,1), (1,-1), (-1,-1)]:
                nx, ny = pos[0] + dx, pos[1] + dy
                if 0 <= nx < self._map.width and 0 <= ny < self._map.height:
                    if self._map.data[ny, nx] < 128:  # Not occupied
                        neighbors.append((nx, ny))
            return neighbors
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))
            
            for neighbor in get_neighbors(current):
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # No path found
    
    def _dijkstra(self, start: Tuple[int, int],
                  goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Dijkstra path planning."""
        return self._astar(start, goal)  # A* with h=0 is Dijkstra
    
    def _rrt(self, start: Tuple[int, int],
             goal: Tuple[int, int],
             max_iterations: int = 1000) -> List[Tuple[int, int]]:
        """RRT path planning."""
        tree = {start: None}
        
        for _ in range(max_iterations):
            # Random sample
            if np.random.random() < 0.1:
                sample = goal
            else:
                sample = (
                    np.random.randint(0, self._map.width),
                    np.random.randint(0, self._map.height)
                )
            
            # Find nearest node
            nearest = min(tree.keys(), key=lambda n: 
                         (n[0]-sample[0])**2 + (n[1]-sample[1])**2)
            
            # Extend towards sample
            direction = np.array([sample[0]-nearest[0], sample[1]-nearest[1]])
            dist = np.linalg.norm(direction)
            
            if dist > 0:
                direction = direction / dist
                step = min(10, dist)
                new_node = (
                    int(nearest[0] + direction[0] * step),
                    int(nearest[1] + direction[1] * step)
                )
                
                # Check collision
                if self._is_valid(new_node):
                    tree[new_node] = nearest
                    
                    # Check if goal reached
                    if abs(new_node[0] - goal[0]) < 5 and abs(new_node[1] - goal[1]) < 5:
                        tree[goal] = new_node
                        
                        # Reconstruct path
                        path = [goal]
                        current = goal
                        while tree[current] is not None:
                            current = tree[current]
                            path.append(current)
                        return list(reversed(path))
        
        return []
    
    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is valid."""
        x, y = pos
        if not (0 <= x < self._map.width and 0 <= y < self._map.height):
            return False
        return self._map.data[y, x] < 128
    
    def set_position(self, position: np.ndarray,
                     orientation: np.ndarray = None):
        """Update robot position (localization)."""
        self._position = position
        if orientation is not None:
            self._orientation = orientation
    
    def get_next_waypoint(self) -> Optional[Waypoint]:
        """Get next waypoint in current path."""
        if self._current_path is None or not self._current_path.waypoints:
            return None
        
        # Find closest waypoint ahead
        for wp in self._current_path.waypoints:
            dist = np.linalg.norm(wp.position[:2] - self._position[:2])
            if dist > wp.tolerance:
                return wp
        
        return None
    
    def is_goal_reached(self) -> bool:
        """Check if goal is reached."""
        if self._current_path is None or not self._current_path.waypoints:
            return True
        
        final_wp = self._current_path.waypoints[-1]
        dist = np.linalg.norm(final_wp.position[:2] - self._position[:2])
        return dist <= final_wp.tolerance
    
    def clear_path(self):
        """Clear current path."""
        self._current_path = None
    
    def __repr__(self) -> str:
        return f"NavigationSystem(planner={self.planner_type.value})"
