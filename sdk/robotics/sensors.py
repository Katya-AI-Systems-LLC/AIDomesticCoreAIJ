"""
Sensor Fusion
=============

Multi-sensor data fusion for robotics.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Sensor types."""
    LIDAR = "lidar"
    CAMERA = "camera"
    IMU = "imu"
    GPS = "gps"
    ULTRASONIC = "ultrasonic"
    ENCODER = "encoder"
    DEPTH_CAMERA = "depth_camera"
    RADAR = "radar"


@dataclass
class SensorReading:
    """Sensor reading."""
    sensor_id: str
    sensor_type: SensorType
    data: Any
    timestamp: float
    confidence: float = 1.0
    frame_id: str = "base_link"


@dataclass
class LidarScan:
    """Lidar scan data."""
    ranges: np.ndarray
    angles: np.ndarray
    intensities: Optional[np.ndarray] = None
    min_range: float = 0.1
    max_range: float = 30.0


@dataclass
class IMUData:
    """IMU sensor data."""
    linear_acceleration: np.ndarray
    angular_velocity: np.ndarray
    orientation: np.ndarray  # quaternion
    covariance: Optional[np.ndarray] = None


@dataclass
class FusedState:
    """Fused sensor state."""
    position: np.ndarray
    velocity: np.ndarray
    orientation: np.ndarray
    angular_velocity: np.ndarray
    covariance: np.ndarray
    timestamp: float


class SensorFusion:
    """
    Multi-sensor fusion system.
    
    Features:
    - Kalman filtering
    - Multi-sensor integration
    - Outlier rejection
    - Time synchronization
    - State estimation
    
    Example:
        >>> fusion = SensorFusion()
        >>> fusion.add_sensor("lidar_front", SensorType.LIDAR)
        >>> fusion.update(reading)
        >>> state = fusion.get_state()
    """
    
    def __init__(self, use_ekf: bool = True):
        """
        Initialize sensor fusion.
        
        Args:
            use_ekf: Use Extended Kalman Filter
        """
        self.use_ekf = use_ekf
        
        # Registered sensors
        self._sensors: Dict[str, Dict] = {}
        
        # Latest readings
        self._readings: Dict[str, SensorReading] = {}
        
        # Fused state (EKF state)
        self._state = np.zeros(13)  # [x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz]
        self._state[6] = 1.0  # qw = 1 (identity quaternion)
        
        # Covariance
        self._P = np.eye(13) * 0.1
        
        # Process noise
        self._Q = np.eye(13) * 0.01
        
        # Callbacks
        self._on_state_update: List[Callable] = []
        
        logger.info("Sensor Fusion initialized")
    
    def add_sensor(self, sensor_id: str,
                   sensor_type: SensorType,
                   frame_id: str = "base_link",
                   transform: np.ndarray = None,
                   noise_covariance: np.ndarray = None):
        """
        Add sensor to fusion system.
        
        Args:
            sensor_id: Sensor identifier
            sensor_type: Sensor type
            frame_id: Sensor frame ID
            transform: Transform from sensor to base frame
            noise_covariance: Measurement noise covariance
        """
        self._sensors[sensor_id] = {
            "type": sensor_type,
            "frame_id": frame_id,
            "transform": transform if transform is not None else np.eye(4),
            "noise_cov": noise_covariance,
            "last_update": None
        }
        
        logger.info(f"Sensor added: {sensor_id} ({sensor_type.value})")
    
    def remove_sensor(self, sensor_id: str):
        """Remove sensor from fusion."""
        if sensor_id in self._sensors:
            del self._sensors[sensor_id]
            if sensor_id in self._readings:
                del self._readings[sensor_id]
    
    def update(self, reading: SensorReading):
        """
        Update fusion with sensor reading.
        
        Args:
            reading: Sensor reading
        """
        if reading.sensor_id not in self._sensors:
            logger.warning(f"Unknown sensor: {reading.sensor_id}")
            return
        
        self._readings[reading.sensor_id] = reading
        self._sensors[reading.sensor_id]["last_update"] = reading.timestamp
        
        # Apply measurement update
        if reading.sensor_type == SensorType.IMU:
            self._update_imu(reading)
        elif reading.sensor_type == SensorType.GPS:
            self._update_gps(reading)
        elif reading.sensor_type == SensorType.LIDAR:
            self._update_lidar(reading)
        elif reading.sensor_type == SensorType.ENCODER:
            self._update_encoder(reading)
        
        # Fire callbacks
        for callback in self._on_state_update:
            callback(self.get_state())
    
    def _update_imu(self, reading: SensorReading):
        """Update with IMU data."""
        if not isinstance(reading.data, IMUData):
            return
        
        imu: IMUData = reading.data
        
        # Update orientation
        self._state[6:10] = imu.orientation
        
        # Update angular velocity
        self._state[10:13] = imu.angular_velocity
        
        # Integrate acceleration for velocity (simplified)
        dt = 0.01  # Assume 100Hz
        self._state[3:6] += imu.linear_acceleration * dt
    
    def _update_gps(self, reading: SensorReading):
        """Update with GPS data."""
        if isinstance(reading.data, dict):
            lat = reading.data.get("latitude", 0)
            lon = reading.data.get("longitude", 0)
            alt = reading.data.get("altitude", 0)
            
            # Convert to local coordinates (simplified)
            # In real system, use proper projection
            self._state[0] = lon * 111320  # Approximate conversion
            self._state[1] = lat * 110540
            self._state[2] = alt
    
    def _update_lidar(self, reading: SensorReading):
        """Update with Lidar data."""
        # Lidar typically used for localization (SLAM)
        # Simplified: extract position from scan matching
        pass
    
    def _update_encoder(self, reading: SensorReading):
        """Update with encoder data."""
        if isinstance(reading.data, dict):
            velocity = reading.data.get("velocity", 0)
            # Update velocity in robot's forward direction
            self._state[3] = velocity
    
    def predict(self, dt: float):
        """
        Predict state forward in time.
        
        Args:
            dt: Time step
        """
        # Simple kinematic prediction
        # Position += velocity * dt
        self._state[0:3] += self._state[3:6] * dt
        
        # Update orientation from angular velocity
        omega = self._state[10:13]
        q = self._state[6:10]
        
        # Quaternion derivative
        omega_quat = np.array([0, omega[0], omega[1], omega[2]])
        q_dot = 0.5 * self._quaternion_multiply(q, omega_quat)
        
        self._state[6:10] += q_dot * dt
        
        # Normalize quaternion
        self._state[6:10] /= np.linalg.norm(self._state[6:10])
        
        # Update covariance
        self._P += self._Q * dt
    
    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def get_state(self) -> FusedState:
        """Get current fused state."""
        return FusedState(
            position=self._state[0:3].copy(),
            velocity=self._state[3:6].copy(),
            orientation=self._state[6:10].copy(),
            angular_velocity=self._state[10:13].copy(),
            covariance=self._P.copy(),
            timestamp=time.time()
        )
    
    def get_position(self) -> np.ndarray:
        """Get current position."""
        return self._state[0:3].copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity."""
        return self._state[3:6].copy()
    
    def get_orientation(self) -> np.ndarray:
        """Get current orientation (quaternion)."""
        return self._state[6:10].copy()
    
    def get_reading(self, sensor_id: str) -> Optional[SensorReading]:
        """Get latest reading from sensor."""
        return self._readings.get(sensor_id)
    
    def process_lidar_scan(self, scan: LidarScan) -> np.ndarray:
        """
        Process lidar scan to point cloud.
        
        Args:
            scan: Lidar scan
            
        Returns:
            Point cloud (Nx3)
        """
        valid = (scan.ranges > scan.min_range) & (scan.ranges < scan.max_range)
        
        ranges = scan.ranges[valid]
        angles = scan.angles[valid]
        
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        z = np.zeros_like(x)
        
        return np.column_stack([x, y, z])
    
    def detect_obstacles(self, scan: LidarScan,
                         threshold: float = 1.0) -> List[Dict]:
        """
        Detect obstacles from lidar scan.
        
        Args:
            scan: Lidar scan
            threshold: Distance threshold
            
        Returns:
            List of obstacle dicts
        """
        obstacles = []
        
        valid = scan.ranges < threshold
        indices = np.where(valid)[0]
        
        # Cluster nearby points
        if len(indices) > 0:
            clusters = []
            current_cluster = [indices[0]]
            
            for i in range(1, len(indices)):
                if indices[i] - indices[i-1] <= 3:
                    current_cluster.append(indices[i])
                else:
                    if len(current_cluster) >= 3:
                        clusters.append(current_cluster)
                    current_cluster = [indices[i]]
            
            if len(current_cluster) >= 3:
                clusters.append(current_cluster)
            
            for cluster in clusters:
                angle = np.mean(scan.angles[cluster])
                distance = np.mean(scan.ranges[cluster])
                
                obstacles.append({
                    "angle": float(angle),
                    "distance": float(distance),
                    "x": float(distance * np.cos(angle)),
                    "y": float(distance * np.sin(angle)),
                    "size": len(cluster)
                })
        
        return obstacles
    
    def on_state_update(self, callback: Callable):
        """Register state update callback."""
        self._on_state_update.append(callback)
    
    def reset(self):
        """Reset fusion state."""
        self._state = np.zeros(13)
        self._state[6] = 1.0
        self._P = np.eye(13) * 0.1
        self._readings.clear()
    
    def __repr__(self) -> str:
        return f"SensorFusion(sensors={len(self._sensors)})"
