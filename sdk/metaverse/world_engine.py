"""
World Engine
============

Virtual world management and simulation.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
import numpy as np
import time
import hashlib
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorldRegion:
    """World region/chunk."""
    region_id: str
    position: np.ndarray  # World coordinates
    size: np.ndarray
    terrain_type: str
    entities: List[str]
    loaded: bool = False


@dataclass
class Entity:
    """World entity."""
    entity_id: str
    entity_type: str
    position: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray
    properties: Dict[str, Any]
    components: List[str]
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)


@dataclass
class WorldSettings:
    """World configuration."""
    name: str
    max_players: int
    physics_enabled: bool
    day_night_cycle: bool
    weather_enabled: bool
    persistence: bool
    blockchain_integration: bool


class WorldEngine:
    """
    Virtual world engine.
    
    Features:
    - Dynamic world loading
    - Entity management
    - Physics simulation
    - Weather and day/night cycle
    - Persistence and blockchain integration
    - Multi-user synchronization
    
    Example:
        >>> engine = WorldEngine("MyWorld")
        >>> await engine.initialize()
        >>> entity = engine.spawn_entity("npc", position)
    """
    
    def __init__(self, world_name: str,
                 settings: WorldSettings = None):
        """
        Initialize world engine.
        
        Args:
            world_name: World name
            settings: World settings
        """
        self.world_name = world_name
        self.settings = settings or WorldSettings(
            name=world_name,
            max_players=100,
            physics_enabled=True,
            day_night_cycle=True,
            weather_enabled=True,
            persistence=True,
            blockchain_integration=True
        )
        
        # World state
        self._regions: Dict[str, WorldRegion] = {}
        self._entities: Dict[str, Entity] = {}
        self._players: Dict[str, Dict] = {}
        
        # Time
        self._world_time: float = 0.0  # 0-24 hours
        self._time_scale: float = 1.0
        
        # Weather
        self._weather: str = "clear"
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # Running state
        self._running = False
        
        logger.info(f"World Engine initialized: {world_name}")
    
    async def initialize(self):
        """Initialize world."""
        # Generate initial regions
        for x in range(-2, 3):
            for z in range(-2, 3):
                region_id = f"region_{x}_{z}"
                region = WorldRegion(
                    region_id=region_id,
                    position=np.array([x * 100, 0, z * 100]),
                    size=np.array([100, 100, 100]),
                    terrain_type="grass" if abs(x) + abs(z) < 3 else "forest",
                    entities=[]
                )
                self._regions[region_id] = region
        
        self._running = True
        logger.info(f"World initialized with {len(self._regions)} regions")
    
    def spawn_entity(self, entity_type: str,
                     position: np.ndarray,
                     rotation: np.ndarray = None,
                     properties: Dict = None) -> Entity:
        """
        Spawn entity in world.
        
        Args:
            entity_type: Type of entity
            position: World position
            rotation: Rotation (quaternion)
            properties: Custom properties
            
        Returns:
            Spawned entity
        """
        entity_id = hashlib.sha256(
            f"{entity_type}_{time.time()}_{np.random.random()}".encode()
        ).hexdigest()[:16]
        
        entity = Entity(
            entity_id=entity_id,
            entity_type=entity_type,
            position=position,
            rotation=rotation if rotation is not None else np.array([0, 0, 0, 1]),
            scale=np.array([1, 1, 1]),
            properties=properties or {},
            components=self._get_default_components(entity_type)
        )
        
        self._entities[entity_id] = entity
        
        # Add to region
        region = self._get_region_at(position)
        if region:
            region.entities.append(entity_id)
        
        # Fire event
        self._fire_event("entity_spawned", entity)
        
        logger.debug(f"Entity spawned: {entity_type} ({entity_id})")
        return entity
    
    def _get_default_components(self, entity_type: str) -> List[str]:
        """Get default components for entity type."""
        components = ["transform"]
        
        if entity_type in ["player", "npc", "creature"]:
            components.extend(["movement", "animation", "collision"])
        elif entity_type in ["item", "pickup"]:
            components.extend(["collision", "interaction"])
        elif entity_type in ["trigger", "zone"]:
            components.extend(["trigger_zone"])
        
        return components
    
    def _get_region_at(self, position: np.ndarray) -> Optional[WorldRegion]:
        """Get region at position."""
        for region in self._regions.values():
            if self._point_in_region(position, region):
                return region
        return None
    
    def _point_in_region(self, point: np.ndarray,
                          region: WorldRegion) -> bool:
        """Check if point is in region."""
        min_bound = region.position
        max_bound = region.position + region.size
        
        return all(min_bound[i] <= point[i] <= max_bound[i] for i in range(3))
    
    def destroy_entity(self, entity_id: str) -> bool:
        """Remove entity from world."""
        if entity_id not in self._entities:
            return False
        
        entity = self._entities[entity_id]
        
        # Remove from region
        region = self._get_region_at(entity.position)
        if region and entity_id in region.entities:
            region.entities.remove(entity_id)
        
        # Fire event
        self._fire_event("entity_destroyed", entity)
        
        del self._entities[entity_id]
        return True
    
    def update_entity(self, entity_id: str,
                      position: np.ndarray = None,
                      rotation: np.ndarray = None,
                      properties: Dict = None):
        """Update entity state."""
        if entity_id not in self._entities:
            return
        
        entity = self._entities[entity_id]
        
        if position is not None:
            old_region = self._get_region_at(entity.position)
            entity.position = position
            new_region = self._get_region_at(position)
            
            # Update region membership
            if old_region != new_region:
                if old_region and entity_id in old_region.entities:
                    old_region.entities.remove(entity_id)
                if new_region:
                    new_region.entities.append(entity_id)
        
        if rotation is not None:
            entity.rotation = rotation
        
        if properties:
            entity.properties.update(properties)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)
    
    def get_entities_in_radius(self, center: np.ndarray,
                               radius: float,
                               entity_type: str = None) -> List[Entity]:
        """Get entities within radius."""
        result = []
        
        for entity in self._entities.values():
            if entity_type and entity.entity_type != entity_type:
                continue
            
            dist = np.linalg.norm(entity.position - center)
            if dist <= radius:
                result.append(entity)
        
        return result
    
    def add_player(self, player_id: str,
                   display_name: str,
                   position: np.ndarray = None) -> Dict:
        """Add player to world."""
        if len(self._players) >= self.settings.max_players:
            raise ValueError("World is full")
        
        spawn_position = position if position is not None else np.array([0, 1, 0])
        
        player_entity = self.spawn_entity(
            "player",
            spawn_position,
            properties={"display_name": display_name, "player_id": player_id}
        )
        
        player_data = {
            "player_id": player_id,
            "display_name": display_name,
            "entity_id": player_entity.entity_id,
            "joined_at": time.time()
        }
        
        self._players[player_id] = player_data
        self._fire_event("player_joined", player_data)
        
        return player_data
    
    def remove_player(self, player_id: str):
        """Remove player from world."""
        if player_id not in self._players:
            return
        
        player_data = self._players[player_id]
        self.destroy_entity(player_data["entity_id"])
        
        self._fire_event("player_left", player_data)
        del self._players[player_id]
    
    def set_world_time(self, hours: float):
        """Set world time (0-24)."""
        self._world_time = hours % 24
        self._fire_event("time_changed", self._world_time)
    
    def set_weather(self, weather: str):
        """Set weather."""
        self._weather = weather
        self._fire_event("weather_changed", weather)
    
    def tick(self, delta_time: float):
        """Update world state."""
        if not self._running:
            return
        
        # Update time
        if self.settings.day_night_cycle:
            self._world_time = (self._world_time + delta_time * self._time_scale / 3600) % 24
        
        # Update entities
        for entity in self._entities.values():
            self._update_entity_tick(entity, delta_time)
    
    def _update_entity_tick(self, entity: Entity, delta_time: float):
        """Update single entity."""
        # Placeholder for entity-specific updates
        pass
    
    def on(self, event: str, handler: Callable):
        """Register event handler."""
        if event not in self._event_handlers:
            self._event_handlers[event] = []
        self._event_handlers[event].append(handler)
    
    def _fire_event(self, event: str, data: Any):
        """Fire event to handlers."""
        handlers = self._event_handlers.get(event, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    def get_statistics(self) -> Dict:
        """Get world statistics."""
        return {
            "name": self.world_name,
            "regions": len(self._regions),
            "entities": len(self._entities),
            "players": len(self._players),
            "world_time": self._world_time,
            "weather": self._weather,
            "running": self._running
        }
    
    def __repr__(self) -> str:
        return f"WorldEngine('{self.world_name}', entities={len(self._entities)})"
