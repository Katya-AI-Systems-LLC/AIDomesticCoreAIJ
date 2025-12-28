"""
Video Stream Processor
======================

Real-time video stream processing and analysis.
"""

from typing import Dict, Any, Optional, List, Callable, Generator
from dataclasses import dataclass
from enum import Enum
import numpy as np
import time
import logging
import threading
import queue

logger = logging.getLogger(__name__)


class StreamSource(Enum):
    """Video stream sources."""
    CAMERA = "camera"
    FILE = "file"
    URL = "url"
    RTSP = "rtsp"


@dataclass
class FrameResult:
    """Result from frame processing."""
    frame_number: int
    timestamp: float
    detections: List[Any]
    metadata: Dict[str, Any]
    processing_time_ms: float


class VideoStreamProcessor:
    """
    Real-time video stream processing.
    
    Features:
    - Multiple source support (camera, file, URL, RTSP)
    - Frame-by-frame processing
    - Pipeline-based analysis
    - Async processing
    - Recording capability
    
    Example:
        >>> processor = VideoStreamProcessor()
        >>> processor.add_pipeline(object_detector)
        >>> for result in processor.process_stream(source):
        ...     print(f"Frame {result.frame_number}: {len(result.detections)} objects")
    """
    
    def __init__(self, buffer_size: int = 30,
                 target_fps: Optional[float] = None,
                 language: str = "en"):
        """
        Initialize video processor.
        
        Args:
            buffer_size: Frame buffer size
            target_fps: Target processing FPS
            language: Language for messages
        """
        self.buffer_size = buffer_size
        self.target_fps = target_fps
        self.language = language
        
        # Processing pipeline
        self._pipeline: List[Callable] = []
        
        # Stream state
        self._running = False
        self._frame_count = 0
        self._start_time = 0.0
        
        # Frame buffer
        self._frame_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        
        # Recording
        self._recording = False
        self._recorder = None
        
        logger.info("Video stream processor initialized")
    
    def add_pipeline(self, processor: Callable):
        """
        Add a processor to the pipeline.
        
        Args:
            processor: Callable that takes frame and returns results
        """
        self._pipeline.append(processor)
        logger.info(f"Added processor to pipeline: {processor}")
    
    def remove_pipeline(self, processor: Callable):
        """Remove a processor from the pipeline."""
        if processor in self._pipeline:
            self._pipeline.remove(processor)
    
    def clear_pipeline(self):
        """Clear all processors from pipeline."""
        self._pipeline.clear()
    
    def process_stream(self, source: str,
                       source_type: StreamSource = StreamSource.CAMERA,
                       max_frames: Optional[int] = None) -> Generator[FrameResult, None, None]:
        """
        Process video stream.
        
        Args:
            source: Source path/URL/camera index
            source_type: Type of source
            max_frames: Maximum frames to process
            
        Yields:
            FrameResult for each processed frame
        """
        self._running = True
        self._frame_count = 0
        self._start_time = time.time()
        
        try:
            cap = self._open_source(source, source_type)
            
            while self._running:
                ret, frame = self._read_frame(cap)
                
                if not ret:
                    break
                
                # Process frame
                result = self._process_frame(frame)
                yield result
                
                self._frame_count += 1
                
                if max_frames and self._frame_count >= max_frames:
                    break
                
                # FPS limiting
                if self.target_fps:
                    self._limit_fps()
            
        finally:
            self._running = False
            if cap:
                self._close_source(cap)
    
    def _open_source(self, source: str, source_type: StreamSource):
        """Open video source."""
        try:
            import cv2
            
            if source_type == StreamSource.CAMERA:
                cap = cv2.VideoCapture(int(source) if source.isdigit() else 0)
            elif source_type == StreamSource.FILE:
                cap = cv2.VideoCapture(source)
            elif source_type in [StreamSource.URL, StreamSource.RTSP]:
                cap = cv2.VideoCapture(source)
            else:
                cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open source: {source}")
            
            return cap
            
        except ImportError:
            logger.warning("OpenCV not installed, using simulation")
            return "simulated"
    
    def _read_frame(self, cap) -> tuple:
        """Read frame from source."""
        if cap == "simulated":
            # Generate random frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            time.sleep(0.033)  # ~30 FPS
            return True, frame
        
        return cap.read()
    
    def _close_source(self, cap):
        """Close video source."""
        if cap != "simulated" and hasattr(cap, 'release'):
            cap.release()
    
    def _process_frame(self, frame: np.ndarray) -> FrameResult:
        """Process a single frame through pipeline."""
        start_time = time.time()
        
        all_detections = []
        metadata = {}
        
        for processor in self._pipeline:
            try:
                result = processor(frame)
                
                if hasattr(result, 'detections'):
                    all_detections.extend(result.detections)
                elif isinstance(result, list):
                    all_detections.extend(result)
                
                if hasattr(result, 'metadata'):
                    metadata.update(result.metadata)
                    
            except Exception as e:
                logger.error(f"Pipeline processor error: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        
        return FrameResult(
            frame_number=self._frame_count,
            timestamp=time.time() - self._start_time,
            detections=all_detections,
            metadata=metadata,
            processing_time_ms=processing_time
        )
    
    def _limit_fps(self):
        """Limit processing to target FPS."""
        if not self.target_fps:
            return
        
        elapsed = time.time() - self._start_time
        expected_time = self._frame_count / self.target_fps
        
        if elapsed < expected_time:
            time.sleep(expected_time - elapsed)
    
    def start_recording(self, output_path: str,
                        fps: float = 30.0,
                        codec: str = "mp4v"):
        """
        Start recording processed frames.
        
        Args:
            output_path: Output file path
            fps: Recording FPS
            codec: Video codec
        """
        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self._recorder = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
            self._recording = True
            logger.info(f"Started recording to: {output_path}")
        except ImportError:
            logger.warning("OpenCV not installed, recording disabled")
    
    def stop_recording(self):
        """Stop recording."""
        if self._recorder:
            self._recorder.release()
            self._recorder = None
        self._recording = False
        logger.info("Stopped recording")
    
    def record_frame(self, frame: np.ndarray):
        """Record a frame."""
        if self._recording and self._recorder:
            self._recorder.write(frame)
    
    def stop(self):
        """Stop processing."""
        self._running = False
    
    def get_fps(self) -> float:
        """Get current processing FPS."""
        if self._start_time == 0:
            return 0.0
        
        elapsed = time.time() - self._start_time
        if elapsed == 0:
            return 0.0
        
        return self._frame_count / elapsed
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "frames_processed": self._frame_count,
            "running": self._running,
            "fps": self.get_fps(),
            "pipeline_size": len(self._pipeline),
            "recording": self._recording
        }
    
    def __repr__(self) -> str:
        return f"VideoStreamProcessor(pipeline={len(self._pipeline)})"
