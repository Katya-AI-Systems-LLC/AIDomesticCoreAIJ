"""
Device Manager
==============

Edge device discovery and management.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
import logging

logger = logging.getLogger(__name__)


class DeviceStatus(Enum):
    """Device status."""
    ONLINE = "online"
    OFFLINE = "offline"
    BUSY = "busy"
    ERROR = "error"
    UPDATING = "updating"


class DeviceCapability(Enum):
    """Device capabilities."""
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    TPU = "tpu"
    CAMERA = "camera"
    MICROPHONE = "microphone"
    GPS = "gps"
    BLUETOOTH = "bluetooth"
    WIFI = "wifi"
    CELLULAR = "cellular"


@dataclass
class EdgeDevice:
    """Edge device representation."""
    device_id: str
    name: str
    device_type: str
    platform: str
    capabilities: List[DeviceCapability]
    ip_address: Optional[str]
    status: DeviceStatus
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    battery_level: Optional[float] = None
    firmware_version: str = "1.0.0"
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DeviceGroup:
    """Group of edge devices."""
    group_id: str
    name: str
    devices: List[str]
    created_at: float = field(default_factory=time.time)


class DeviceManager:
    """
    Edge device manager.
    
    Features:
    - Device discovery
    - Health monitoring
    - Remote management
    - OTA updates
    - Device grouping
    
    Example:
        >>> manager = DeviceManager()
        >>> manager.start_discovery()
        >>> devices = manager.get_devices()
    """
    
    DEVICE_TYPES = [
        "raspberry_pi", "jetson_nano", "jetson_xavier",
        "coral_dev", "intel_ncs", "android", "ios",
        "esp32", "arduino", "generic"
    ]
    
    def __init__(self, discovery_port: int = 5555,
                 heartbeat_interval: int = 30):
        """
        Initialize device manager.
        
        Args:
            discovery_port: UDP port for discovery
            heartbeat_interval: Heartbeat interval in seconds
        """
        self.discovery_port = discovery_port
        self.heartbeat_interval = heartbeat_interval
        
        # Devices
        self._devices: Dict[str, EdgeDevice] = {}
        
        # Groups
        self._groups: Dict[str, DeviceGroup] = {}
        
        # Event handlers
        self._on_device_connected: List[Callable] = []
        self._on_device_disconnected: List[Callable] = []
        self._on_device_status_changed: List[Callable] = []
        
        # Discovery running
        self._discovery_running = False
        
        logger.info(f"Device Manager initialized (port={discovery_port})")
    
    def register_device(self, device_id: str,
                        name: str,
                        device_type: str,
                        platform: str,
                        capabilities: List[DeviceCapability],
                        ip_address: str = None,
                        metadata: Dict = None) -> EdgeDevice:
        """
        Register edge device.
        
        Args:
            device_id: Unique device ID
            name: Device name
            device_type: Device type
            platform: Platform/OS
            capabilities: Device capabilities
            ip_address: IP address
            metadata: Additional metadata
            
        Returns:
            EdgeDevice
        """
        device = EdgeDevice(
            device_id=device_id,
            name=name,
            device_type=device_type,
            platform=platform,
            capabilities=capabilities,
            ip_address=ip_address,
            status=DeviceStatus.ONLINE,
            metadata=metadata or {}
        )
        
        self._devices[device_id] = device
        
        # Fire event
        for handler in self._on_device_connected:
            try:
                handler(device)
            except Exception as e:
                logger.error(f"Handler error: {e}")
        
        logger.info(f"Device registered: {name} ({device_id})")
        return device
    
    def unregister_device(self, device_id: str):
        """Unregister device."""
        if device_id in self._devices:
            device = self._devices[device_id]
            
            # Remove from groups
            for group in self._groups.values():
                if device_id in group.devices:
                    group.devices.remove(device_id)
            
            # Fire event
            for handler in self._on_device_disconnected:
                try:
                    handler(device)
                except Exception as e:
                    logger.error(f"Handler error: {e}")
            
            del self._devices[device_id]
            logger.info(f"Device unregistered: {device_id}")
    
    def update_heartbeat(self, device_id: str,
                         cpu_usage: float = None,
                         memory_usage: float = None,
                         battery_level: float = None):
        """
        Update device heartbeat.
        
        Args:
            device_id: Device ID
            cpu_usage: CPU usage (0-100)
            memory_usage: Memory usage (0-100)
            battery_level: Battery level (0-100)
        """
        if device_id not in self._devices:
            return
        
        device = self._devices[device_id]
        device.last_heartbeat = time.time()
        
        if cpu_usage is not None:
            device.cpu_usage = cpu_usage
        if memory_usage is not None:
            device.memory_usage = memory_usage
        if battery_level is not None:
            device.battery_level = battery_level
        
        # Update status if was offline
        if device.status == DeviceStatus.OFFLINE:
            self._update_status(device_id, DeviceStatus.ONLINE)
    
    def _update_status(self, device_id: str, status: DeviceStatus):
        """Update device status."""
        if device_id not in self._devices:
            return
        
        device = self._devices[device_id]
        old_status = device.status
        device.status = status
        
        if old_status != status:
            for handler in self._on_device_status_changed:
                try:
                    handler(device, old_status, status)
                except Exception as e:
                    logger.error(f"Handler error: {e}")
    
    def check_heartbeats(self):
        """Check device heartbeats and mark offline."""
        current_time = time.time()
        timeout = self.heartbeat_interval * 3
        
        for device in self._devices.values():
            if current_time - device.last_heartbeat > timeout:
                if device.status != DeviceStatus.OFFLINE:
                    self._update_status(device.device_id, DeviceStatus.OFFLINE)
    
    def get_device(self, device_id: str) -> Optional[EdgeDevice]:
        """Get device by ID."""
        return self._devices.get(device_id)
    
    def get_devices(self, status: DeviceStatus = None,
                    device_type: str = None,
                    capability: DeviceCapability = None) -> List[EdgeDevice]:
        """
        Get devices with optional filtering.
        
        Args:
            status: Filter by status
            device_type: Filter by device type
            capability: Filter by capability
            
        Returns:
            List of devices
        """
        devices = list(self._devices.values())
        
        if status:
            devices = [d for d in devices if d.status == status]
        
        if device_type:
            devices = [d for d in devices if d.device_type == device_type]
        
        if capability:
            devices = [d for d in devices if capability in d.capabilities]
        
        return devices
    
    def create_group(self, name: str,
                     device_ids: List[str] = None) -> DeviceGroup:
        """
        Create device group.
        
        Args:
            name: Group name
            device_ids: Initial device IDs
            
        Returns:
            DeviceGroup
        """
        group_id = hashlib.sha256(f"{name}_{time.time()}".encode()).hexdigest()[:16]
        
        group = DeviceGroup(
            group_id=group_id,
            name=name,
            devices=device_ids or []
        )
        
        self._groups[group_id] = group
        
        logger.info(f"Group created: {name} ({group_id})")
        return group
    
    def add_to_group(self, group_id: str, device_id: str) -> bool:
        """Add device to group."""
        if group_id not in self._groups:
            return False
        
        if device_id not in self._devices:
            return False
        
        if device_id not in self._groups[group_id].devices:
            self._groups[group_id].devices.append(device_id)
        
        return True
    
    def remove_from_group(self, group_id: str, device_id: str) -> bool:
        """Remove device from group."""
        if group_id not in self._groups:
            return False
        
        if device_id in self._groups[group_id].devices:
            self._groups[group_id].devices.remove(device_id)
        
        return True
    
    def get_group(self, group_id: str) -> Optional[DeviceGroup]:
        """Get group by ID."""
        return self._groups.get(group_id)
    
    def get_group_devices(self, group_id: str) -> List[EdgeDevice]:
        """Get devices in group."""
        if group_id not in self._groups:
            return []
        
        return [
            self._devices[did] for did in self._groups[group_id].devices
            if did in self._devices
        ]
    
    async def send_command(self, device_id: str,
                           command: str,
                           params: Dict = None) -> Dict:
        """
        Send command to device.
        
        Args:
            device_id: Target device
            command: Command name
            params: Command parameters
            
        Returns:
            Command result
        """
        if device_id not in self._devices:
            return {"error": "Device not found"}
        
        device = self._devices[device_id]
        
        if device.status == DeviceStatus.OFFLINE:
            return {"error": "Device offline"}
        
        # Simulated command execution
        logger.info(f"Command sent to {device_id}: {command}")
        
        return {
            "device_id": device_id,
            "command": command,
            "status": "executed",
            "timestamp": time.time()
        }
    
    async def deploy_model(self, device_id: str,
                           model_path: str,
                           model_name: str) -> bool:
        """
        Deploy model to device.
        
        Args:
            device_id: Target device
            model_path: Model file path
            model_name: Model name
            
        Returns:
            True if successful
        """
        if device_id not in self._devices:
            return False
        
        self._update_status(device_id, DeviceStatus.UPDATING)
        
        # Simulated deployment
        logger.info(f"Deploying {model_name} to {device_id}")
        
        self._update_status(device_id, DeviceStatus.ONLINE)
        return True
    
    def on_device_connected(self, handler: Callable):
        """Register device connected handler."""
        self._on_device_connected.append(handler)
    
    def on_device_disconnected(self, handler: Callable):
        """Register device disconnected handler."""
        self._on_device_disconnected.append(handler)
    
    def on_device_status_changed(self, handler: Callable):
        """Register status changed handler."""
        self._on_device_status_changed.append(handler)
    
    def get_statistics(self) -> Dict:
        """Get device statistics."""
        devices = list(self._devices.values())
        
        return {
            "total_devices": len(devices),
            "online": len([d for d in devices if d.status == DeviceStatus.ONLINE]),
            "offline": len([d for d in devices if d.status == DeviceStatus.OFFLINE]),
            "busy": len([d for d in devices if d.status == DeviceStatus.BUSY]),
            "groups": len(self._groups),
            "avg_cpu_usage": sum(d.cpu_usage for d in devices) / len(devices) if devices else 0,
            "avg_memory_usage": sum(d.memory_usage for d in devices) / len(devices) if devices else 0
        }
    
    def __repr__(self) -> str:
        return f"DeviceManager(devices={len(self._devices)}, groups={len(self._groups)})"
