"""
Metrics and Monitoring
======================

Performance monitoring and metrics collection.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import time
import threading
import logging

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric data point."""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Centralized metrics collection.
    
    Collects:
    - Counters
    - Gauges
    - Histograms
    - Timers
    
    Example:
        >>> metrics = MetricsCollector()
        >>> metrics.increment("requests_total")
        >>> metrics.gauge("active_connections", 42)
        >>> with metrics.timer("operation_duration"):
        ...     do_operation()
    """
    
    def __init__(self, prefix: str = "aiplatform"):
        """
        Initialize metrics collector.
        
        Args:
            prefix: Metric name prefix
        """
        self.prefix = prefix
        
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, List[float]] = defaultdict(list)
        
        self._lock = threading.Lock()
        self._callbacks: List[Callable] = []
    
    def _full_name(self, name: str) -> str:
        """Get full metric name with prefix."""
        return f"{self.prefix}_{name}"
    
    def increment(self, name: str, value: float = 1.0,
                  tags: Optional[Dict[str, str]] = None):
        """
        Increment a counter.
        
        Args:
            name: Metric name
            value: Increment value
            tags: Optional tags
        """
        full_name = self._full_name(name)
        
        with self._lock:
            self._counters[full_name] += value
        
        self._emit(MetricPoint(full_name, value, time.time(), tags or {}))
    
    def decrement(self, name: str, value: float = 1.0,
                  tags: Optional[Dict[str, str]] = None):
        """Decrement a counter."""
        self.increment(name, -value, tags)
    
    def gauge(self, name: str, value: float,
              tags: Optional[Dict[str, str]] = None):
        """
        Set a gauge value.
        
        Args:
            name: Metric name
            value: Gauge value
            tags: Optional tags
        """
        full_name = self._full_name(name)
        
        with self._lock:
            self._gauges[full_name] = value
        
        self._emit(MetricPoint(full_name, value, time.time(), tags or {}))
    
    def histogram(self, name: str, value: float,
                  tags: Optional[Dict[str, str]] = None):
        """
        Record a histogram value.
        
        Args:
            name: Metric name
            value: Value to record
            tags: Optional tags
        """
        full_name = self._full_name(name)
        
        with self._lock:
            self._histograms[full_name].append(value)
            # Keep only last 1000 values
            if len(self._histograms[full_name]) > 1000:
                self._histograms[full_name] = self._histograms[full_name][-1000:]
        
        self._emit(MetricPoint(full_name, value, time.time(), tags or {}))
    
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """
        Context manager for timing operations.
        
        Args:
            name: Metric name
            tags: Optional tags
            
        Returns:
            Timer context manager
        """
        return TimerContext(self, name, tags or {})
    
    def record_time(self, name: str, duration: float,
                    tags: Optional[Dict[str, str]] = None):
        """Record a timing value."""
        full_name = self._full_name(name)
        
        with self._lock:
            self._timers[full_name].append(duration)
            if len(self._timers[full_name]) > 1000:
                self._timers[full_name] = self._timers[full_name][-1000:]
        
        self._emit(MetricPoint(full_name, duration, time.time(), tags or {}))
    
    def get_counter(self, name: str) -> float:
        """Get counter value."""
        return self._counters.get(self._full_name(name), 0.0)
    
    def get_gauge(self, name: str) -> Optional[float]:
        """Get gauge value."""
        return self._gauges.get(self._full_name(name))
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        full_name = self._full_name(name)
        values = self._histograms.get(full_name, [])
        
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "count": n,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "mean": sum(values) / n,
            "p50": sorted_values[n // 2],
            "p90": sorted_values[int(n * 0.9)],
            "p99": sorted_values[int(n * 0.99)]
        }
    
    def get_timer_stats(self, name: str) -> Dict[str, float]:
        """Get timer statistics."""
        full_name = self._full_name(name)
        values = self._timers.get(full_name, [])
        
        if not values:
            return {}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            "count": n,
            "min_ms": sorted_values[0] * 1000,
            "max_ms": sorted_values[-1] * 1000,
            "mean_ms": (sum(values) / n) * 1000,
            "p50_ms": sorted_values[n // 2] * 1000,
            "p95_ms": sorted_values[int(n * 0.95)] * 1000
        }
    
    def register_callback(self, callback: Callable[[MetricPoint], None]):
        """Register metric callback."""
        self._callbacks.append(callback)
    
    def _emit(self, point: MetricPoint):
        """Emit metric to callbacks."""
        for callback in self._callbacks:
            try:
                callback(point)
            except Exception as e:
                logger.error(f"Metric callback error: {e}")
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {k: self.get_histogram_stats(k.replace(f"{self.prefix}_", ""))
                          for k in self._histograms},
            "timers": {k: self.get_timer_stats(k.replace(f"{self.prefix}_", ""))
                      for k in self._timers}
        }
    
    def reset(self):
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str,
                 tags: Dict[str, str]):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = 0.0
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        duration = time.time() - self.start_time
        self.collector.record_time(self.name, duration, self.tags)


class PerformanceMonitor:
    """
    High-level performance monitoring.
    
    Monitors:
    - Request latency
    - Throughput
    - Error rates
    - Resource usage
    
    Example:
        >>> monitor = PerformanceMonitor()
        >>> with monitor.track("inference"):
        ...     result = model.predict(data)
        >>> print(monitor.get_report())
    """
    
    def __init__(self):
        self.metrics = MetricsCollector(prefix="perf")
        self._active_operations: Dict[str, float] = {}
    
    def track(self, operation: str, tags: Optional[Dict[str, str]] = None):
        """Track an operation."""
        return self.metrics.timer(f"{operation}_latency", tags)
    
    def record_success(self, operation: str):
        """Record successful operation."""
        self.metrics.increment(f"{operation}_success")
    
    def record_error(self, operation: str, error_type: str = "unknown"):
        """Record failed operation."""
        self.metrics.increment(f"{operation}_error", tags={"type": error_type})
    
    def record_throughput(self, operation: str, count: int = 1):
        """Record throughput."""
        self.metrics.increment(f"{operation}_throughput", count)
    
    def set_active_operations(self, count: int):
        """Set active operations gauge."""
        self.metrics.gauge("active_operations", count)
    
    def get_report(self) -> Dict[str, Any]:
        """Get performance report."""
        all_metrics = self.metrics.get_all_metrics()
        
        # Calculate error rates
        error_rates = {}
        for key in all_metrics["counters"]:
            if "_success" in key:
                base = key.replace("_success", "")
                success = all_metrics["counters"].get(f"{base}_success", 0)
                error = all_metrics["counters"].get(f"{base}_error", 0)
                total = success + error
                if total > 0:
                    error_rates[base] = error / total
        
        return {
            "latencies": all_metrics["timers"],
            "throughput": {k: v for k, v in all_metrics["counters"].items()
                          if "_throughput" in k},
            "error_rates": error_rates,
            "gauges": all_metrics["gauges"]
        }


# Global metrics instance
_global_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    global _global_metrics
    
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    
    return _global_metrics
