"""
Stream Pipeline
===============

Data processing pipelines for streams.
"""

from typing import Dict, Any, Optional, List, Callable, AsyncGenerator, TypeVar, Generic
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import time
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class StreamProcessor(ABC, Generic[T, R]):
    """
    Base stream processor.
    
    Transforms data in a stream.
    """
    
    @abstractmethod
    async def process(self, item: T) -> R:
        """Process single item."""
        pass
    
    async def process_batch(self, items: List[T]) -> List[R]:
        """Process batch of items."""
        return [await self.process(item) for item in items]


class MapProcessor(StreamProcessor[T, R]):
    """Map processor - applies function to each item."""
    
    def __init__(self, func: Callable[[T], R]):
        self.func = func
    
    async def process(self, item: T) -> R:
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(item)
        return self.func(item)


class FilterProcessor(StreamProcessor[T, T]):
    """Filter processor - filters items."""
    
    def __init__(self, predicate: Callable[[T], bool]):
        self.predicate = predicate
    
    async def process(self, item: T) -> Optional[T]:
        if asyncio.iscoroutinefunction(self.predicate):
            keep = await self.predicate(item)
        else:
            keep = self.predicate(item)
        return item if keep else None


class AggregateProcessor(StreamProcessor[T, Any]):
    """Aggregate processor - aggregates items."""
    
    def __init__(self, window_size: int = 10,
                 aggregator: Callable[[List[T]], Any] = None):
        self.window_size = window_size
        self.aggregator = aggregator or (lambda x: x)
        self._buffer: List[T] = []
    
    async def process(self, item: T) -> Optional[Any]:
        self._buffer.append(item)
        
        if len(self._buffer) >= self.window_size:
            result = self.aggregator(self._buffer)
            self._buffer = []
            return result
        
        return None


@dataclass
class PipelineStats:
    """Pipeline statistics."""
    items_processed: int = 0
    items_filtered: int = 0
    errors: int = 0
    total_time: float = 0.0
    avg_latency: float = 0.0


class StreamPipeline(Generic[T]):
    """
    Stream processing pipeline.
    
    Features:
    - Chained processors
    - Async processing
    - Error handling
    - Backpressure
    - Statistics
    
    Example:
        >>> pipeline = StreamPipeline()
        >>> pipeline.map(lambda x: x * 2)
        >>> pipeline.filter(lambda x: x > 10)
        >>> async for result in pipeline.process(data_stream):
        ...     print(result)
    """
    
    def __init__(self, buffer_size: int = 100):
        """
        Initialize pipeline.
        
        Args:
            buffer_size: Internal buffer size
        """
        self.buffer_size = buffer_size
        
        self._processors: List[StreamProcessor] = []
        self._stats = PipelineStats()
        
        self._error_handler: Optional[Callable] = None
        
        logger.info("StreamPipeline initialized")
    
    def map(self, func: Callable) -> 'StreamPipeline':
        """Add map processor."""
        self._processors.append(MapProcessor(func))
        return self
    
    def filter(self, predicate: Callable) -> 'StreamPipeline':
        """Add filter processor."""
        self._processors.append(FilterProcessor(predicate))
        return self
    
    def aggregate(self, window_size: int,
                  aggregator: Callable = None) -> 'StreamPipeline':
        """Add aggregate processor."""
        self._processors.append(AggregateProcessor(window_size, aggregator))
        return self
    
    def add_processor(self, processor: StreamProcessor) -> 'StreamPipeline':
        """Add custom processor."""
        self._processors.append(processor)
        return self
    
    def on_error(self, handler: Callable) -> 'StreamPipeline':
        """Set error handler."""
        self._error_handler = handler
        return self
    
    async def process(self, source: AsyncGenerator[T, None]) -> AsyncGenerator[Any, None]:
        """
        Process stream.
        
        Args:
            source: Source stream
            
        Yields:
            Processed items
        """
        async for item in source:
            start_time = time.time()
            
            try:
                result = await self._process_item(item)
                
                if result is not None:
                    self._stats.items_processed += 1
                    yield result
                else:
                    self._stats.items_filtered += 1
                
                latency = time.time() - start_time
                self._stats.total_time += latency
                self._stats.avg_latency = (
                    self._stats.total_time / 
                    (self._stats.items_processed + self._stats.items_filtered)
                )
                
            except Exception as e:
                self._stats.errors += 1
                
                if self._error_handler:
                    self._error_handler(e, item)
                else:
                    logger.error(f"Pipeline error: {e}")
    
    async def _process_item(self, item: Any) -> Any:
        """Process single item through pipeline."""
        result = item
        
        for processor in self._processors:
            result = await processor.process(result)
            
            if result is None:
                return None
        
        return result
    
    async def process_list(self, items: List[T]) -> List[Any]:
        """Process list of items."""
        results = []
        
        async def generator():
            for item in items:
                yield item
        
        async for result in self.process(generator()):
            results.append(result)
        
        return results
    
    def get_stats(self) -> PipelineStats:
        """Get pipeline statistics."""
        return self._stats
    
    def reset_stats(self):
        """Reset statistics."""
        self._stats = PipelineStats()
    
    def clear(self):
        """Clear all processors."""
        self._processors.clear()
    
    def __repr__(self) -> str:
        return f"StreamPipeline(processors={len(self._processors)})"


class ParallelPipeline(StreamPipeline[T]):
    """
    Parallel stream processing.
    
    Processes items concurrently.
    """
    
    def __init__(self, concurrency: int = 4,
                 buffer_size: int = 100):
        """
        Initialize parallel pipeline.
        
        Args:
            concurrency: Max concurrent operations
            buffer_size: Buffer size
        """
        super().__init__(buffer_size)
        self.concurrency = concurrency
        self._semaphore = asyncio.Semaphore(concurrency)
    
    async def process(self, source: AsyncGenerator[T, None]) -> AsyncGenerator[Any, None]:
        """Process stream with concurrency."""
        tasks = []
        results_queue: asyncio.Queue = asyncio.Queue()
        
        async def process_with_semaphore(item):
            async with self._semaphore:
                result = await self._process_item(item)
                await results_queue.put(result)
        
        producer_done = False
        
        async def producer():
            nonlocal producer_done
            async for item in source:
                task = asyncio.create_task(process_with_semaphore(item))
                tasks.append(task)
            producer_done = True
        
        producer_task = asyncio.create_task(producer())
        
        while not producer_done or tasks:
            try:
                result = await asyncio.wait_for(results_queue.get(), timeout=0.1)
                if result is not None:
                    self._stats.items_processed += 1
                    yield result
                else:
                    self._stats.items_filtered += 1
            except asyncio.TimeoutError:
                # Clean up completed tasks
                tasks = [t for t in tasks if not t.done()]
        
        await producer_task


class BatchPipeline(StreamPipeline[T]):
    """
    Batch stream processing.
    
    Groups items into batches.
    """
    
    def __init__(self, batch_size: int = 10,
                 timeout: float = 5.0):
        """
        Initialize batch pipeline.
        
        Args:
            batch_size: Batch size
            timeout: Batch timeout
        """
        super().__init__()
        self.batch_size = batch_size
        self.timeout = timeout
    
    async def process(self, source: AsyncGenerator[T, None]) -> AsyncGenerator[List[Any], None]:
        """Process stream in batches."""
        batch = []
        last_emit = time.time()
        
        async for item in source:
            result = await self._process_item(item)
            
            if result is not None:
                batch.append(result)
            
            # Emit batch if full or timeout
            should_emit = (
                len(batch) >= self.batch_size or
                (batch and time.time() - last_emit >= self.timeout)
            )
            
            if should_emit:
                yield batch
                batch = []
                last_emit = time.time()
        
        # Emit remaining
        if batch:
            yield batch
