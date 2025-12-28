"""
Edge Inference
==============

High-performance edge inference engine.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)


class InferenceMode(Enum):
    """Inference modes."""
    SYNC = "sync"
    ASYNC = "async"
    BATCH = "batch"
    STREAM = "stream"


@dataclass
class InferenceRequest:
    """Inference request."""
    request_id: str
    inputs: Any
    callback: Optional[Callable] = None
    priority: int = 0
    timestamp: float = 0.0


@dataclass
class InferenceResponse:
    """Inference response."""
    request_id: str
    outputs: Any
    latency_ms: float
    success: bool
    error: Optional[str] = None


class EdgeInference:
    """
    High-performance edge inference engine.
    
    Features:
    - Synchronous and asynchronous inference
    - Batch processing
    - Request queuing
    - Priority scheduling
    - Streaming inference
    
    Example:
        >>> engine = EdgeInference(runtime)
        >>> result = engine.infer(input_data)
        >>> engine.infer_async(input_data, callback)
    """
    
    def __init__(self, runtime: Any,
                 mode: InferenceMode = InferenceMode.SYNC,
                 max_batch_size: int = 8,
                 max_queue_size: int = 100):
        """
        Initialize inference engine.
        
        Args:
            runtime: EdgeRuntime instance
            mode: Default inference mode
            max_batch_size: Maximum batch size
            max_queue_size: Maximum queue size
        """
        self.runtime = runtime
        self.mode = mode
        self.max_batch_size = max_batch_size
        self.max_queue_size = max_queue_size
        
        # Request queue
        self._queue = queue.PriorityQueue(maxsize=max_queue_size)
        
        # Worker thread
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Callbacks
        self._callbacks: Dict[str, Callable] = {}
        
        # Stats
        self._request_count = 0
        
        logger.info(f"Edge Inference initialized (mode={mode.value})")
    
    def start(self):
        """Start async inference worker."""
        if self._running:
            return
        
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        
        logger.info("Inference worker started")
    
    def stop(self):
        """Stop async inference worker."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        
        logger.info("Inference worker stopped")
    
    def _worker_loop(self):
        """Worker loop for async inference."""
        while self._running:
            try:
                # Get batch of requests
                batch = []
                
                try:
                    # Wait for first request
                    priority, request = self._queue.get(timeout=0.1)
                    batch.append(request)
                    
                    # Try to fill batch
                    while len(batch) < self.max_batch_size:
                        try:
                            priority, request = self._queue.get_nowait()
                            batch.append(request)
                        except queue.Empty:
                            break
                    
                except queue.Empty:
                    continue
                
                # Process batch
                self._process_batch(batch)
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    def _process_batch(self, batch: List[InferenceRequest]):
        """Process a batch of requests."""
        for request in batch:
            response = self._execute_single(request)
            
            # Call callback if registered
            callback = self._callbacks.pop(request.request_id, None)
            if callback:
                try:
                    callback(response)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
    
    def _execute_single(self, request: InferenceRequest) -> InferenceResponse:
        """Execute single inference request."""
        start_time = time.time()
        
        try:
            outputs = self.runtime.infer(request.inputs)
            latency = (time.time() - start_time) * 1000
            
            return InferenceResponse(
                request_id=request.request_id,
                outputs=outputs,
                latency_ms=latency,
                success=True
            )
            
        except Exception as e:
            return InferenceResponse(
                request_id=request.request_id,
                outputs=None,
                latency_ms=(time.time() - start_time) * 1000,
                success=False,
                error=str(e)
            )
    
    def infer(self, inputs: Any) -> InferenceResponse:
        """
        Synchronous inference.
        
        Args:
            inputs: Input data
            
        Returns:
            InferenceResponse
        """
        self._request_count += 1
        request_id = f"req_{self._request_count}"
        
        request = InferenceRequest(
            request_id=request_id,
            inputs=inputs,
            timestamp=time.time()
        )
        
        return self._execute_single(request)
    
    def infer_async(self, inputs: Any,
                    callback: Callable[[InferenceResponse], None],
                    priority: int = 0) -> str:
        """
        Asynchronous inference.
        
        Args:
            inputs: Input data
            callback: Callback function
            priority: Request priority (lower = higher priority)
            
        Returns:
            Request ID
        """
        self._request_count += 1
        request_id = f"req_{self._request_count}"
        
        request = InferenceRequest(
            request_id=request_id,
            inputs=inputs,
            callback=callback,
            priority=priority,
            timestamp=time.time()
        )
        
        self._callbacks[request_id] = callback
        self._queue.put((priority, request))
        
        return request_id
    
    def infer_batch(self, inputs_list: List[Any]) -> List[InferenceResponse]:
        """
        Batch inference.
        
        Args:
            inputs_list: List of inputs
            
        Returns:
            List of responses
        """
        responses = []
        
        for inputs in inputs_list:
            response = self.infer(inputs)
            responses.append(response)
        
        return responses
    
    def infer_stream(self, input_generator: Any,
                     callback: Callable[[InferenceResponse], None]):
        """
        Streaming inference.
        
        Args:
            input_generator: Generator yielding inputs
            callback: Callback for each result
        """
        for inputs in input_generator:
            response = self.infer(inputs)
            callback(response)
    
    def get_queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()
    
    def clear_queue(self):
        """Clear pending requests."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
    
    def __repr__(self) -> str:
        return f"EdgeInference(mode={self.mode.value}, queue={self.get_queue_size()})"
