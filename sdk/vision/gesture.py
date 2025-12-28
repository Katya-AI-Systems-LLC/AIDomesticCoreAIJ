"""
Gesture Recognition Module
==========================

Hand and body gesture recognition.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GestureType(Enum):
    """Types of gestures."""
    HAND = "hand"
    BODY = "body"
    FACE = "face"


class HandGesture(Enum):
    """Hand gesture types."""
    OPEN_PALM = "open_palm"
    CLOSED_FIST = "closed_fist"
    POINTING = "pointing"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    PEACE = "peace"
    OK = "ok"
    PINCH = "pinch"
    GRAB = "grab"
    WAVE = "wave"
    UNKNOWN = "unknown"


@dataclass
class HandLandmarks:
    """Hand landmark positions."""
    wrist: Tuple[float, float, float]
    thumb: List[Tuple[float, float, float]]
    index: List[Tuple[float, float, float]]
    middle: List[Tuple[float, float, float]]
    ring: List[Tuple[float, float, float]]
    pinky: List[Tuple[float, float, float]]


@dataclass
class GestureDetection:
    """A detected gesture."""
    gesture_type: GestureType
    gesture_name: str
    confidence: float
    landmarks: Optional[Any] = None
    bbox: Optional[Tuple[int, int, int, int]] = None


class GestureRecognizer:
    """
    Gesture recognition for hands and body.
    
    Features:
    - Hand gesture recognition
    - Hand landmark detection
    - Body pose estimation
    - Gesture tracking over time
    
    Example:
        >>> recognizer = GestureRecognizer()
        >>> gestures = recognizer.detect_gestures(image)
        >>> for g in gestures:
        ...     print(f"{g.gesture_name}: {g.confidence:.2f}")
    """
    
    def __init__(self, model: str = "default",
                 confidence_threshold: float = 0.5,
                 language: str = "en"):
        """
        Initialize gesture recognizer.
        
        Args:
            model: Model name
            confidence_threshold: Detection threshold
            language: Language for messages
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.language = language
        
        self._hand_detector = None
        self._gesture_history: List[GestureDetection] = []
        
        logger.info(f"Gesture recognizer initialized: {model}")
    
    def load_models(self):
        """Load gesture recognition models."""
        try:
            import mediapipe as mp
            self._hand_detector = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=self.confidence_threshold
            )
            logger.info("Gesture models loaded")
        except ImportError:
            logger.warning("mediapipe not installed, using simulation")
            self._hand_detector = "simulated"
    
    def detect_gestures(self, image: np.ndarray) -> List[GestureDetection]:
        """
        Detect gestures in image.
        
        Args:
            image: Input image
            
        Returns:
            List of detected gestures
        """
        if self._hand_detector is None:
            self.load_models()
        
        if self._hand_detector == "simulated":
            return self._simulate_detection(image)
        
        try:
            import cv2
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self._hand_detector.process(rgb_image)
            
            gestures = []
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks = self._extract_landmarks(hand_landmarks, image.shape)
                    gesture = self._classify_gesture(landmarks)
                    gestures.append(gesture)
            
            return gestures
            
        except Exception as e:
            logger.error(f"Gesture detection failed: {e}")
            return self._simulate_detection(image)
    
    def _simulate_detection(self, image: np.ndarray) -> List[GestureDetection]:
        """Simulate gesture detection."""
        h, w = image.shape[:2]
        
        num_gestures = np.random.randint(0, 3)
        gestures = []
        
        for _ in range(num_gestures):
            gesture = np.random.choice(list(HandGesture))
            confidence = np.random.uniform(self.confidence_threshold, 1.0)
            
            x = np.random.randint(0, w - 100)
            y = np.random.randint(0, h - 100)
            
            gestures.append(GestureDetection(
                gesture_type=GestureType.HAND,
                gesture_name=gesture.value,
                confidence=confidence,
                bbox=(x, y, 100, 100)
            ))
        
        return gestures
    
    def _extract_landmarks(self, hand_landmarks, image_shape) -> HandLandmarks:
        """Extract hand landmarks."""
        h, w = image_shape[:2]
        
        def get_point(landmark):
            return (landmark.x * w, landmark.y * h, landmark.z)
        
        landmarks = hand_landmarks.landmark
        
        return HandLandmarks(
            wrist=get_point(landmarks[0]),
            thumb=[get_point(landmarks[i]) for i in range(1, 5)],
            index=[get_point(landmarks[i]) for i in range(5, 9)],
            middle=[get_point(landmarks[i]) for i in range(9, 13)],
            ring=[get_point(landmarks[i]) for i in range(13, 17)],
            pinky=[get_point(landmarks[i]) for i in range(17, 21)]
        )
    
    def _classify_gesture(self, landmarks: HandLandmarks) -> GestureDetection:
        """Classify gesture from landmarks."""
        # Simple gesture classification based on finger positions
        
        # Check if fingers are extended
        thumb_extended = self._is_finger_extended(landmarks.thumb, landmarks.wrist)
        index_extended = self._is_finger_extended(landmarks.index, landmarks.wrist)
        middle_extended = self._is_finger_extended(landmarks.middle, landmarks.wrist)
        ring_extended = self._is_finger_extended(landmarks.ring, landmarks.wrist)
        pinky_extended = self._is_finger_extended(landmarks.pinky, landmarks.wrist)
        
        extended_count = sum([
            thumb_extended, index_extended, middle_extended,
            ring_extended, pinky_extended
        ])
        
        # Classify based on extended fingers
        if extended_count == 5:
            gesture = HandGesture.OPEN_PALM
        elif extended_count == 0:
            gesture = HandGesture.CLOSED_FIST
        elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
            gesture = HandGesture.POINTING
        elif thumb_extended and not index_extended and not middle_extended:
            gesture = HandGesture.THUMBS_UP
        elif index_extended and middle_extended and not ring_extended and not pinky_extended:
            gesture = HandGesture.PEACE
        else:
            gesture = HandGesture.UNKNOWN
        
        return GestureDetection(
            gesture_type=GestureType.HAND,
            gesture_name=gesture.value,
            confidence=0.85,
            landmarks=landmarks
        )
    
    def _is_finger_extended(self, finger_landmarks: List[Tuple],
                            wrist: Tuple) -> bool:
        """Check if a finger is extended."""
        if len(finger_landmarks) < 4:
            return False
        
        tip = finger_landmarks[-1]
        base = finger_landmarks[0]
        
        # Finger is extended if tip is farther from wrist than base
        tip_dist = np.sqrt((tip[0] - wrist[0])**2 + (tip[1] - wrist[1])**2)
        base_dist = np.sqrt((base[0] - wrist[0])**2 + (base[1] - wrist[1])**2)
        
        return tip_dist > base_dist * 1.2
    
    def track_gesture(self, gesture: GestureDetection):
        """Track gesture over time."""
        self._gesture_history.append(gesture)
        
        # Keep only recent history
        if len(self._gesture_history) > 30:
            self._gesture_history = self._gesture_history[-30:]
    
    def get_gesture_sequence(self) -> List[str]:
        """Get recent gesture sequence."""
        return [g.gesture_name for g in self._gesture_history]
    
    def detect_gesture_pattern(self, pattern: List[str]) -> bool:
        """Check if a gesture pattern was performed."""
        sequence = self.get_gesture_sequence()
        
        if len(sequence) < len(pattern):
            return False
        
        # Check if pattern exists in sequence
        for i in range(len(sequence) - len(pattern) + 1):
            if sequence[i:i + len(pattern)] == pattern:
                return True
        
        return False
    
    def clear_history(self):
        """Clear gesture history."""
        self._gesture_history.clear()
    
    def __repr__(self) -> str:
        return f"GestureRecognizer(model='{self.model}')"
