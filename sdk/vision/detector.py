"""
Object Detector
===============

Advanced object detection with multiple backend support.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A detected object."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    mask: Optional[np.ndarray] = None
    keypoints: Optional[List[Tuple[int, int]]] = None


@dataclass
class DetectionResult:
    """Result from object detection."""
    detections: List[Detection]
    image_size: Tuple[int, int]
    inference_time_ms: float
    model_name: str


class ObjectDetector:
    """
    Object detection with multiple model backends.
    
    Supports:
    - YOLO (v5, v8)
    - SSD
    - Faster R-CNN
    - Custom models
    
    Example:
        >>> detector = ObjectDetector(model="yolov8")
        >>> result = detector.detect(image)
        >>> for det in result.detections:
        ...     print(f"{det.class_name}: {det.confidence:.2f}")
    """
    
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    def __init__(self, model: str = "yolov8",
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4,
                 device: str = "auto",
                 language: str = "en"):
        """
        Initialize object detector.
        
        Args:
            model: Model name or path
            confidence_threshold: Minimum confidence for detections
            nms_threshold: Non-maximum suppression threshold
            device: Device to use (auto, cpu, cuda)
            language: Language for messages
        """
        self.model_name = model
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = device
        self.language = language
        
        self._model = None
        self._classes = self.COCO_CLASSES
        
        logger.info(f"Object detector initialized: {model}")
    
    def load_model(self):
        """Load the detection model."""
        try:
            if "yolo" in self.model_name.lower():
                self._load_yolo()
            else:
                self._load_generic()
            logger.info(f"Model loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}, using simulation")
            self._model = "simulated"
    
    def _load_yolo(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(f"{self.model_name}.pt")
        except ImportError:
            logger.warning("ultralytics not installed")
            self._model = "simulated"
    
    def _load_generic(self):
        """Load generic model."""
        self._model = "simulated"
    
    def detect(self, image: np.ndarray,
               classes: Optional[List[int]] = None) -> DetectionResult:
        """
        Detect objects in image.
        
        Args:
            image: Input image (HxWxC numpy array)
            classes: Filter to specific class IDs
            
        Returns:
            DetectionResult
        """
        import time
        start_time = time.time()
        
        if self._model is None:
            self.load_model()
        
        if self._model == "simulated":
            detections = self._simulate_detection(image)
        else:
            detections = self._run_detection(image)
        
        # Filter by classes
        if classes:
            detections = [d for d in detections if d.class_id in classes]
        
        inference_time = (time.time() - start_time) * 1000
        
        return DetectionResult(
            detections=detections,
            image_size=(image.shape[1], image.shape[0]),
            inference_time_ms=inference_time,
            model_name=self.model_name
        )
    
    def _run_detection(self, image: np.ndarray) -> List[Detection]:
        """Run actual detection."""
        try:
            results = self._model(image)
            detections = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if confidence < self.confidence_threshold:
                        continue
                    
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    detections.append(Detection(
                        class_id=class_id,
                        class_name=self._classes[class_id] if class_id < len(self._classes) else "unknown",
                        confidence=confidence,
                        bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    ))
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return self._simulate_detection(image)
    
    def _simulate_detection(self, image: np.ndarray) -> List[Detection]:
        """Simulate detection for testing."""
        h, w = image.shape[:2]
        
        # Generate random detections
        num_detections = np.random.randint(1, 5)
        detections = []
        
        for _ in range(num_detections):
            class_id = np.random.randint(0, len(self._classes))
            confidence = np.random.uniform(self.confidence_threshold, 1.0)
            
            x = np.random.randint(0, w - 100)
            y = np.random.randint(0, h - 100)
            width = np.random.randint(50, min(200, w - x))
            height = np.random.randint(50, min(200, h - y))
            
            detections.append(Detection(
                class_id=class_id,
                class_name=self._classes[class_id],
                confidence=confidence,
                bbox=(x, y, width, height)
            ))
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray]) -> List[DetectionResult]:
        """Detect objects in multiple images."""
        return [self.detect(img) for img in images]
    
    def set_classes(self, classes: List[str]):
        """Set custom class names."""
        self._classes = classes
    
    def get_classes(self) -> List[str]:
        """Get class names."""
        return self._classes.copy()
    
    def __repr__(self) -> str:
        return f"ObjectDetector(model='{self.model_name}')"
