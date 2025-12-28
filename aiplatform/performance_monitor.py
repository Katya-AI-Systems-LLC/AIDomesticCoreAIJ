"""
Performance Monitoring Module for AIPlatform

This module provides performance monitoring, metrics collection,
and profiling capabilities for AIPlatform operations.
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import logging
from functools import wraps
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Performance metric data."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class OperationMetrics:
    """Metrics for a specific operation."""
    operation_name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    errors: int = 0
    last_call: Optional[datetime] = None
    
    def update(self, duration: float, success: bool = True):
        """Update metrics with new operation."""
        self.total_calls += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.avg_time = self.total_time / self.total_calls
        self.last_call = datetime.now()
        
        if not success:
            self.errors += 1
    
    def get_success_rate(self) -> float:
        """Get success rate."""
        if self.total_calls == 0:
            return 0.0
        return (self.total_calls - self.errors) / self.total_calls


class PerformanceMonitor:
    """Performance monitoring system."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.operation_metrics: Dict[str, OperationMetrics] = {}
        self.system_metrics: List[PerformanceMetric] = []
        self.monitoring = False
        self.monitor_thread = None
        self.monitor_interval = 5.0  # seconds
        self.max_metrics = 10000  # Maximum metrics to keep
        
        # Performance decorators registry
        self.decorators: Dict[str, Callable] = {}
    
    def start_monitoring(self):
        """Start background monitoring."""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
            self.monitor_thread.start()
            logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Performance monitoring stopped")
    
    def _monitor_system(self):
        """Background system monitoring."""
        while self.monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.add_metric("system_cpu_usage", cpu_percent, "%")
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.add_metric("system_memory_usage", memory.percent, "%")
                self.add_metric("system_memory_available", memory.available, "bytes")
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.add_metric("system_disk_usage", disk_percent, "%")
                
                # Process info
                process = psutil.Process()
                self.add_metric("process_memory", process.memory_info().rss, "bytes")
                self.add_metric("process_cpu", process.cpu_percent(), "%")
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(self.monitor_interval)
    
    def add_metric(self, name: str, value: float, unit: str, tags: Dict[str, str] = None):
        """Add a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            tags=tags or {}
        )
        
        self.metrics.append(metric)
        
        # Limit metrics count
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
    
    def record_operation(self, operation_name: str, duration: float, success: bool = True):
        """Record operation performance."""
        if operation_name not in self.operation_metrics:
            self.operation_metrics[operation_name] = OperationMetrics(operation_name)
        
        self.operation_metrics[operation_name].update(duration, success)
        
        # Also add as metric
        self.add_metric(
            f"operation_{operation_name}_duration",
            duration,
            "seconds",
            {"success": str(success)}
        )
    
    def get_metrics_summary(self, operation_name: str = None) -> Dict[str, Any]:
        """Get metrics summary."""
        if operation_name:
            if operation_name in self.operation_metrics:
                metrics = self.operation_metrics[operation_name]
                return {
                    'operation': operation_name,
                    'total_calls': metrics.total_calls,
                    'total_time': metrics.total_time,
                    'avg_time': metrics.avg_time,
                    'min_time': metrics.min_time if metrics.min_time != float('inf') else 0,
                    'max_time': metrics.max_time,
                    'success_rate': metrics.get_success_rate(),
                    'errors': metrics.errors,
                    'last_call': metrics.last_call.isoformat() if metrics.last_call else None
                }
            else:
                return {}
        else:
            # Return summary for all operations
            return {
                name: self.get_metrics_summary(name)
                for name in self.operation_metrics.keys()
            }
    
    def get_system_metrics(self, minutes: int = 5) -> List[PerformanceMetric]:
        """Get recent system metrics."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [
            metric for metric in self.system_metrics
            if metric.timestamp >= cutoff_time
        ]
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get detailed statistics for an operation."""
        if operation_name not in self.operation_metrics:
            return {}
        
        # Get all metrics for this operation
        operation_metrics = [
            m for m in self.metrics
            if m.name == f"operation_{operation_name}_duration"
        ]
        
        if not operation_metrics:
            return {}
        
        durations = [m.value for m in operation_metrics]
        
        return {
            'operation': operation_name,
            'count': len(durations),
            'mean': statistics.mean(durations),
            'median': statistics.median(durations),
            'stdev': statistics.stdev(durations) if len(durations) > 1 else 0,
            'min': min(durations),
            'max': max(durations),
            'p95': self._percentile(durations, 95),
            'p99': self._percentile(durations, 99)
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def export_metrics(self, filename: str, format: str = 'json'):
        """Export metrics to file."""
        try:
            if format == 'json':
                data = {
                    'export_time': datetime.now().isoformat(),
                    'operation_metrics': {
                        name: asdict(metrics) for name, metrics in self.operation_metrics.items()
                    },
                    'recent_metrics': [
                        asdict(metric) for metric in self.metrics[-100:]
                    ]
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Metrics exported to {filename}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
    
    def clear_metrics(self):
        """Clear all metrics."""
        self.metrics.clear()
        self.operation_metrics.clear()
        self.system_metrics.clear()
        logger.info("Performance metrics cleared")
    
    def create_performance_decorator(self, operation_name: str = None):
        """Create performance monitoring decorator."""
        def decorator(func: Callable) -> Callable:
            name = operation_name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                success = True
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    raise
                finally:
                    duration = time.time() - start_time
                    self.record_operation(name, duration, success)
            
            return wrapper
        return decorator
    
    def profile_function(self, func: Callable) -> Callable:
        """Profile a function and return detailed stats."""
        name = f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss
            
            try:
                result = func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                success = False
                raise
            finally:
                duration = time.time() - start_time
                end_memory = psutil.Process().memory_info().rss
                memory_delta = end_memory - start_memory
                
                # Record detailed metrics
                self.add_metric(f"profile_{name}_duration", duration, "seconds")
                self.add_metric(f"profile_{name}_memory_delta", memory_delta, "bytes")
                self.record_operation(name, duration, success)
        
        return wrapper
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            'report_time': datetime.now().isoformat(),
            'monitoring_active': self.monitoring,
            'total_metrics': len(self.metrics),
            'total_operations': len(self.operation_metrics),
            'operations_summary': {},
            'system_status': {}
        }
        
        # Operations summary
        for op_name, metrics in self.operation_metrics.items():
            report['operations_summary'][op_name] = {
                'calls': metrics.total_calls,
                'avg_time': metrics.avg_time,
                'success_rate': metrics.get_success_rate(),
                'errors': metrics.errors
            }
        
        # System status
        recent_system = self.get_system_metrics(1)  # Last minute
        if recent_system:
            cpu_metrics = [m for m in recent_system if m.name == "system_cpu_usage"]
            mem_metrics = [m for m in recent_system if m.name == "system_memory_usage"]
            
            if cpu_metrics:
                report['system_status']['cpu_avg'] = statistics.mean([m.value for m in cpu_metrics])
            if mem_metrics:
                report['system_status']['memory_avg'] = statistics.mean([m.value for m in mem_metrics])
        
        return report


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def monitor_performance(operation_name: str = None):
    """Decorator for monitoring function performance."""
    return performance_monitor.create_performance_decorator(operation_name)


def profile_function(func: Callable) -> Callable:
    """Decorator for profiling function performance."""
    return performance_monitor.profile_function(func)


def get_performance_report() -> Dict[str, Any]:
    """Get current performance report."""
    return performance_monitor.get_performance_report()


if __name__ == "__main__":
    # Test performance monitoring
    print("=== Performance Monitoring Test ===")
    
    # Start monitoring
    performance_monitor.start_monitoring()
    
    # Test decorator
    @monitor_performance("test_operation")
    def test_function(delay: float = 0.1):
        """Test function with delay."""
        time.sleep(delay)
        return "test result"
    
    @profile_function
    def test_profile_function():
        """Test profiling function."""
        # Simulate some work
        data = [i for i in range(1000)]
        return sum(data)
    
    # Run test functions
    print("Running test functions...")
    for i in range(5):
        result = test_function(0.05)
        profile_result = test_profile_function()
    
    # Wait for some system metrics
    time.sleep(2)
    
    # Get performance report
    report = performance_monitor.get_performance_report()
    print(f"\nPerformance Report:")
    print(f"  Total metrics: {report['total_metrics']}")
    print(f"  Total operations: {report['total_operations']}")
    print(f"  Monitoring active: {report['monitoring_active']}")
    
    # Show operation metrics
    metrics_summary = performance_monitor.get_metrics_summary()
    print(f"\nOperation Metrics:")
    for op_name, metrics in metrics_summary.items():
        print(f"  {op_name}:")
        print(f"    Calls: {metrics['total_calls']}")
        print(f"    Avg time: {metrics['avg_time']:.4f}s")
        print(f"    Success rate: {metrics['success_rate']:.2%}")
    
    # Export metrics
    performance_monitor.export_metrics("performance_metrics.json")
    
    # Stop monitoring
    performance_monitor.stop_monitoring()
    
    print("\n=== Performance Monitoring Test Complete ===")
