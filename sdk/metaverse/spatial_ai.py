"""
Spatial AI
==========

3D-aware AI for metaverse environments.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class SpatialObject:
    """Object in 3D space."""
    object_id: str
    position: np.ndarray  # x, y, z
    rotation: np.ndarray  # quaternion
    scale: np.ndarray
    object_type: str
    properties: Dict[str, Any]
    bounds: Tuple[np.ndarray, np.ndarray]  # min, max


@dataclass
class SpatialContext:
    """Spatial context for AI."""
    user_position: np.ndarray
    user_rotation: np.ndarray
    gaze_direction: np.ndarray
    nearby_objects: List[SpatialObject]
    environment_type: str
    lighting: Dict[str, float]
    audio_sources: List[Dict]


@dataclass
class SpatialInteraction:
    """Spatial interaction event."""
    interaction_id: str
    interaction_type: str  # gaze, grab, point, voice
    target_object: Optional[str]
    position: np.ndarray
    timestamp: float
    parameters: Dict[str, Any]


class SpatialAI:
    """
    3D-aware AI for metaverse.
    
    Features:
    - Spatial understanding
    - Object recognition in 3D
    - Gesture and gaze tracking
    - Natural language + spatial reasoning
    - Environment-aware responses
    
    Example:
        >>> ai = SpatialAI()
        >>> response = await ai.process_spatial_query(
        ...     "What's on my left?", context
        ... )
    """
    
    def __init__(self, model: str = "spatial-gpt"):
        """
        Initialize Spatial AI.
        
        Args:
            model: AI model to use
        """
        self.model = model
        
        # Spatial memory
        self._objects: Dict[str, SpatialObject] = {}
        
        # Interaction history
        self._interactions: List[SpatialInteraction] = []
        
        # Conversation context
        self._conversation_history: List[Dict] = []
        
        logger.info(f"Spatial AI initialized with model: {model}")
    
    def register_object(self, obj: SpatialObject):
        """Register object in spatial memory."""
        self._objects[obj.object_id] = obj
    
    def update_object_position(self, object_id: str,
                                position: np.ndarray,
                                rotation: np.ndarray = None):
        """Update object position."""
        if object_id in self._objects:
            self._objects[object_id].position = position
            if rotation is not None:
                self._objects[object_id].rotation = rotation
    
    async def process_spatial_query(self, query: str,
                                     context: SpatialContext) -> Dict:
        """
        Process query with spatial context.
        
        Args:
            query: User query
            context: Spatial context
            
        Returns:
            AI response with spatial awareness
        """
        # Analyze spatial relationships
        spatial_info = self._analyze_spatial_context(context)
        
        # Generate response
        response = await self._generate_spatial_response(query, spatial_info, context)
        
        # Store in conversation history
        self._conversation_history.append({
            "role": "user",
            "content": query,
            "spatial_context": spatial_info
        })
        
        self._conversation_history.append({
            "role": "assistant",
            "content": response["text"]
        })
        
        return response
    
    def _analyze_spatial_context(self, context: SpatialContext) -> Dict:
        """Analyze spatial relationships."""
        user_pos = context.user_position
        gaze = context.gaze_direction
        
        # Find objects in different directions
        objects_ahead = []
        objects_left = []
        objects_right = []
        objects_behind = []
        
        for obj in context.nearby_objects:
            direction = obj.position - user_pos
            distance = np.linalg.norm(direction)
            direction = direction / (distance + 1e-6)
            
            # Calculate angle relative to gaze
            dot = np.dot(direction[:2], gaze[:2])
            cross = direction[0] * gaze[1] - direction[1] * gaze[0]
            
            obj_info = {
                "id": obj.object_id,
                "type": obj.object_type,
                "distance": distance,
                "position": obj.position.tolist()
            }
            
            if dot > 0.7:  # Ahead
                objects_ahead.append(obj_info)
            elif dot < -0.7:  # Behind
                objects_behind.append(obj_info)
            elif cross > 0:  # Right
                objects_right.append(obj_info)
            else:  # Left
                objects_left.append(obj_info)
        
        return {
            "ahead": objects_ahead,
            "left": objects_left,
            "right": objects_right,
            "behind": objects_behind,
            "environment": context.environment_type,
            "user_position": user_pos.tolist(),
            "gaze_direction": gaze.tolist()
        }
    
    async def _generate_spatial_response(self, query: str,
                                          spatial_info: Dict,
                                          context: SpatialContext) -> Dict:
        """Generate spatially-aware response."""
        # Identify spatial keywords
        spatial_keywords = ["left", "right", "ahead", "behind", "near", "far",
                          "above", "below", "here", "there", "this", "that"]
        
        query_lower = query.lower()
        relevant_direction = None
        
        for keyword in spatial_keywords:
            if keyword in query_lower:
                relevant_direction = keyword
                break
        
        # Build response based on spatial context
        response_text = ""
        highlighted_objects = []
        actions = []
        
        if "left" in query_lower and spatial_info["left"]:
            objs = spatial_info["left"]
            response_text = f"On your left, I see {len(objs)} object(s): "
            response_text += ", ".join([o["type"] for o in objs])
            highlighted_objects = [o["id"] for o in objs]
            
        elif "right" in query_lower and spatial_info["right"]:
            objs = spatial_info["right"]
            response_text = f"On your right, I see {len(objs)} object(s): "
            response_text += ", ".join([o["type"] for o in objs])
            highlighted_objects = [o["id"] for o in objs]
            
        elif "ahead" in query_lower or "front" in query_lower:
            objs = spatial_info["ahead"]
            if objs:
                response_text = f"Ahead of you, there are {len(objs)} object(s): "
                response_text += ", ".join([o["type"] for o in objs])
            else:
                response_text = "There's nothing directly ahead of you."
            highlighted_objects = [o["id"] for o in objs]
            
        else:
            # General description
            total_objects = (len(spatial_info["ahead"]) + len(spatial_info["left"]) +
                           len(spatial_info["right"]) + len(spatial_info["behind"]))
            response_text = f"I can see {total_objects} objects around you. "
            
            if spatial_info["ahead"]:
                response_text += f"{len(spatial_info['ahead'])} ahead, "
            if spatial_info["left"]:
                response_text += f"{len(spatial_info['left'])} on your left, "
            if spatial_info["right"]:
                response_text += f"{len(spatial_info['right'])} on your right."
        
        return {
            "text": response_text,
            "highlighted_objects": highlighted_objects,
            "actions": actions,
            "spatial_context": spatial_info
        }
    
    async def process_gesture(self, gesture_type: str,
                               position: np.ndarray,
                               direction: Optional[np.ndarray] = None) -> Dict:
        """
        Process gesture interaction.
        
        Args:
            gesture_type: Type of gesture (point, grab, wave, etc.)
            position: Hand/controller position
            direction: Pointing/gesture direction
            
        Returns:
            Interaction result
        """
        interaction = SpatialInteraction(
            interaction_id=f"int_{len(self._interactions)}",
            interaction_type=gesture_type,
            target_object=None,
            position=position,
            timestamp=time.time(),
            parameters={"direction": direction.tolist() if direction is not None else None}
        )
        
        # Find target object
        if direction is not None:
            target = self._raycast(position, direction)
            if target:
                interaction.target_object = target.object_id
        
        self._interactions.append(interaction)
        
        return {
            "interaction_id": interaction.interaction_id,
            "type": gesture_type,
            "target": interaction.target_object,
            "position": position.tolist()
        }
    
    def _raycast(self, origin: np.ndarray,
                  direction: np.ndarray,
                  max_distance: float = 100.0) -> Optional[SpatialObject]:
        """Simple raycast for object selection."""
        closest = None
        closest_dist = max_distance
        
        for obj in self._objects.values():
            # Simple sphere intersection
            to_obj = obj.position - origin
            proj_dist = np.dot(to_obj, direction)
            
            if proj_dist > 0 and proj_dist < closest_dist:
                # Check if ray passes near object center
                closest_point = origin + direction * proj_dist
                dist_to_center = np.linalg.norm(closest_point - obj.position)
                
                # Assume object has radius ~1
                if dist_to_center < 1.0:
                    closest = obj
                    closest_dist = proj_dist
        
        return closest
    
    def get_nearby_objects(self, position: np.ndarray,
                           radius: float = 10.0) -> List[SpatialObject]:
        """Get objects within radius."""
        nearby = []
        
        for obj in self._objects.values():
            dist = np.linalg.norm(obj.position - position)
            if dist <= radius:
                nearby.append(obj)
        
        return sorted(nearby, key=lambda o: np.linalg.norm(o.position - position))
    
    def __repr__(self) -> str:
        return f"SpatialAI(model='{self.model}', objects={len(self._objects)})"
