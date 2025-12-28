"""
Metrics Aggregator
==================

Aggregate and analyze system metrics.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import math
import logging

logger = logging.getLogger(__name__)


class AggregationType(Enum):
    """Aggregation types."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"


@dataclass
class MetricPoint:
    """Single metric data point."""
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class AggregatedMetric:
    """Aggregated metric result."""
    name: str
    aggregation: AggregationType
    value: float
    count: int
    start_time: float
    end_time: float
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsAggregator:
    """
    Metrics aggregation system.
    
    Features:
    - Time-series metrics
    - Multiple aggregations
    - Downsampling
    - Tag-based filtering
    - Real-time computation
    
    Example:
        >>> aggregator = MetricsAggregator()
        >>> aggregator.record("latency", 50.5, {"model": "gpt4"})
        >>> result = aggregator.aggregate("latency", AggregationType.P95)
    """
    
    def __init__(self, retention_seconds: int = 3600,
                 bucket_size_seconds: int = 60):
        """
        Initialize metrics aggregator.
        
        Args:
            retention_seconds: Data retention period
            bucket_size_seconds: Time bucket size
        """
        self.retention_seconds = retention_seconds
        self.bucket_size_seconds = bucket_size_seconds
        
        # Metrics storage: name -> list of points
        self._metrics: Dict[str, List[MetricPoint]] = {}
        
        # Pre-computed aggregations
        self._aggregations: Dict[str, Dict[str, AggregatedMetric]] = {}
        
        logger.info(f"Metrics Aggregator initialized (retention={retention_seconds}s)")
    
    def record(self, name: str, value: float,
               tags: Dict[str, str] = None,
               timestamp: float = None):
        """
        Record metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Metric tags
            timestamp: Custom timestamp
        """
        point = MetricPoint(
            value=value,
            timestamp=timestamp or time.time(),
            tags=tags or {}
        )
        
        if name not in self._metrics:
            self._metrics[name] = []
        
        self._metrics[name].append(point)
        
        # Cleanup old data
        self._cleanup(name)
    
    def _cleanup(self, name: str):
        """Remove old data points."""
        cutoff = time.time() - self.retention_seconds
        
        if name in self._metrics:
            self._metrics[name] = [
                p for p in self._metrics[name] if p.timestamp >= cutoff
            ]
    
    def aggregate(self, name: str,
                  aggregation: AggregationType,
                  start_time: float = None,
                  end_time: float = None,
                  tags: Dict[str, str] = None) -> Optional[AggregatedMetric]:
        """
        Aggregate metric values.
        
        Args:
            name: Metric name
            aggregation: Aggregation type
            start_time: Start of time range
            end_time: End of time range
            tags: Filter by tags
            
        Returns:
            AggregatedMetric or None
        """
        if name not in self._metrics:
            return None
        
        end_time = end_time or time.time()
        start_time = start_time or (end_time - self.retention_seconds)
        
        # Filter points
        points = self._metrics[name]
        points = [p for p in points if start_time <= p.timestamp <= end_time]
        
        if tags:
            points = [
                p for p in points
                if all(p.tags.get(k) == v for k, v in tags.items())
            ]
        
        if not points:
            return None
        
        values = [p.value for p in points]
        
        # Compute aggregation
        if aggregation == AggregationType.SUM:
            result = sum(values)
        elif aggregation == AggregationType.AVG:
            result = sum(values) / len(values)
        elif aggregation == AggregationType.MIN:
            result = min(values)
        elif aggregation == AggregationType.MAX:
            result = max(values)
        elif aggregation == AggregationType.COUNT:
            result = len(values)
        elif aggregation == AggregationType.P50:
            result = self._percentile(values, 50)
        elif aggregation == AggregationType.P95:
            result = self._percentile(values, 95)
        elif aggregation == AggregationType.P99:
            result = self._percentile(values, 99)
        else:
            result = 0
        
        return AggregatedMetric(
            name=name,
            aggregation=aggregation,
            value=result,
            count=len(points),
            start_time=start_time,
            end_time=end_time,
            tags=tags or {}
        )
    
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        
        lower = int(index)
        upper = lower + 1
        
        if upper >= len(sorted_values):
            return sorted_values[-1]
        
        fraction = index - lower
        return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])
    
    def get_timeseries(self, name: str,
                       start_time: float = None,
                       end_time: float = None,
                       bucket_seconds: int = None) -> List[Tuple[float, float]]:
        """
        Get time-series data.
        
        Args:
            name: Metric name
            start_time: Start time
            end_time: End time
            bucket_seconds: Bucket size for downsampling
            
        Returns:
            List of (timestamp, value) tuples
        """
        if name not in self._metrics:
            return []
        
        end_time = end_time or time.time()
        start_time = start_time or (end_time - self.retention_seconds)
        bucket_seconds = bucket_seconds or self.bucket_size_seconds
        
        points = [
            p for p in self._metrics[name]
            if start_time <= p.timestamp <= end_time
        ]
        
        if not points:
            return []
        
        # Bucket data
        buckets: Dict[int, List[float]] = {}
        
        for point in points:
            bucket_key = int(point.timestamp // bucket_seconds)
            if bucket_key not in buckets:
                buckets[bucket_key] = []
            buckets[bucket_key].append(point.value)
        
        # Average within buckets
        result = []
        for bucket_key in sorted(buckets.keys()):
            timestamp = bucket_key * bucket_seconds
            avg_value = sum(buckets[bucket_key]) / len(buckets[bucket_key])
            result.append((timestamp, avg_value))
        
        return result
    
    def get_rate(self, name: str,
                 window_seconds: int = 60) -> float:
        """
        Get rate (events per second).
        
        Args:
            name: Metric name
            window_seconds: Window size
            
        Returns:
            Rate per second
        """
        if name not in self._metrics:
            return 0
        
        cutoff = time.time() - window_seconds
        count = sum(1 for p in self._metrics[name] if p.timestamp >= cutoff)
        
        return count / window_seconds
    
    def get_histogram(self, name: str,
                      buckets: List[float] = None,
                      start_time: float = None,
                      end_time: float = None) -> Dict[str, int]:
        """
        Get histogram of values.
        
        Args:
            name: Metric name
            buckets: Bucket boundaries
            start_time: Start time
            end_time: End time
            
        Returns:
            Histogram counts
        """
        if name not in self._metrics:
            return {}
        
        buckets = buckets or [10, 50, 100, 250, 500, 1000, 5000]
        
        end_time = end_time or time.time()
        start_time = start_time or (end_time - self.retention_seconds)
        
        values = [
            p.value for p in self._metrics[name]
            if start_time <= p.timestamp <= end_time
        ]
        
        histogram = {f"<={b}": 0 for b in buckets}
        histogram[f">{buckets[-1]}"] = 0
        
        for value in values:
            for bucket in buckets:
                if value <= bucket:
                    histogram[f"<={bucket}"] += 1
                    break
            else:
                histogram[f">{buckets[-1]}"] += 1
        
        return histogram
    
    def get_summary(self, name: str,
                    window_seconds: int = 300) -> Dict[str, float]:
        """
        Get metric summary.
        
        Args:
            name: Metric name
            window_seconds: Window size
            
        Returns:
            Summary statistics
        """
        start_time = time.time() - window_seconds
        
        return {
            "count": self.aggregate(name, AggregationType.COUNT, start_time).value if self.aggregate(name, AggregationType.COUNT, start_time) else 0,
            "avg": self.aggregate(name, AggregationType.AVG, start_time).value if self.aggregate(name, AggregationType.AVG, start_time) else 0,
            "min": self.aggregate(name, AggregationType.MIN, start_time).value if self.aggregate(name, AggregationType.MIN, start_time) else 0,
            "max": self.aggregate(name, AggregationType.MAX, start_time).value if self.aggregate(name, AggregationType.MAX, start_time) else 0,
            "p50": self.aggregate(name, AggregationType.P50, start_time).value if self.aggregate(name, AggregationType.P50, start_time) else 0,
            "p95": self.aggregate(name, AggregationType.P95, start_time).value if self.aggregate(name, AggregationType.P95, start_time) else 0,
            "p99": self.aggregate(name, AggregationType.P99, start_time).value if self.aggregate(name, AggregationType.P99, start_time) else 0,
            "rate": self.get_rate(name, min(window_seconds, 60))
        }
    
    def list_metrics(self) -> List[str]:
        """List all metric names."""
        return list(self._metrics.keys())
    
    def clear(self, name: str = None):
        """Clear metrics data."""
        if name:
            self._metrics.pop(name, None)
        else:
            self._metrics.clear()
    
    def __repr__(self) -> str:
        return f"MetricsAggregator(metrics={len(self._metrics)})"
