"""
Anomaly Detector
================

Detect anomalies in system metrics and behavior.
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Anomaly types."""
    SPIKE = "spike"
    DROP = "drop"
    DRIFT = "drift"
    OUTLIER = "outlier"
    PATTERN_BREAK = "pattern_break"


class Severity(Enum):
    """Anomaly severity."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Detected anomaly."""
    anomaly_id: str
    anomaly_type: AnomalyType
    severity: Severity
    metric_name: str
    value: float
    expected_value: float
    deviation: float
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectorConfig:
    """Anomaly detector configuration."""
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    min_samples: int = 30
    sensitivity: float = 0.8
    window_size: int = 100


class AnomalyDetector:
    """
    Anomaly detection system.
    
    Features:
    - Statistical anomaly detection
    - Z-score analysis
    - IQR-based outliers
    - Pattern recognition
    - Real-time alerts
    
    Example:
        >>> detector = AnomalyDetector()
        >>> detector.train("latency", historical_data)
        >>> anomalies = detector.detect("latency", new_value)
    """
    
    def __init__(self, config: DetectorConfig = None):
        """
        Initialize anomaly detector.
        
        Args:
            config: Detector configuration
        """
        self.config = config or DetectorConfig()
        
        # Trained models per metric
        self._models: Dict[str, Dict] = {}
        
        # Recent values for online learning
        self._windows: Dict[str, List[float]] = {}
        
        # Detected anomalies
        self._anomalies: List[Anomaly] = []
        
        # Alert handlers
        self._alert_handlers: List = []
        
        self._anomaly_count = 0
        
        logger.info("Anomaly Detector initialized")
    
    def train(self, metric_name: str,
              values: List[float],
              timestamps: List[float] = None):
        """
        Train detector on historical data.
        
        Args:
            metric_name: Metric name
            values: Historical values
            timestamps: Corresponding timestamps
        """
        if len(values) < self.config.min_samples:
            logger.warning(f"Insufficient data for training {metric_name}")
            return
        
        values_array = np.array(values)
        
        # Compute statistics
        mean = np.mean(values_array)
        std = np.std(values_array)
        median = np.median(values_array)
        
        # IQR
        q1 = np.percentile(values_array, 25)
        q3 = np.percentile(values_array, 75)
        iqr = q3 - q1
        
        # Store model
        self._models[metric_name] = {
            "mean": mean,
            "std": std,
            "median": median,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "min": np.min(values_array),
            "max": np.max(values_array),
            "trained_at": time.time(),
            "sample_count": len(values)
        }
        
        # Initialize window
        self._windows[metric_name] = list(values[-self.config.window_size:])
        
        logger.info(f"Trained model for {metric_name} on {len(values)} samples")
    
    def detect(self, metric_name: str,
               value: float,
               timestamp: float = None) -> List[Anomaly]:
        """
        Detect anomalies in new value.
        
        Args:
            metric_name: Metric name
            value: New value
            timestamp: Value timestamp
            
        Returns:
            List of detected anomalies
        """
        timestamp = timestamp or time.time()
        anomalies = []
        
        if metric_name not in self._models:
            # Auto-train with online learning
            if metric_name not in self._windows:
                self._windows[metric_name] = []
            
            self._windows[metric_name].append(value)
            
            if len(self._windows[metric_name]) >= self.config.min_samples:
                self.train(metric_name, self._windows[metric_name])
            
            return anomalies
        
        model = self._models[metric_name]
        
        # Update window
        self._windows[metric_name].append(value)
        if len(self._windows[metric_name]) > self.config.window_size:
            self._windows[metric_name].pop(0)
        
        # Z-score detection
        zscore_anomaly = self._detect_zscore(metric_name, value, model, timestamp)
        if zscore_anomaly:
            anomalies.append(zscore_anomaly)
        
        # IQR detection
        iqr_anomaly = self._detect_iqr(metric_name, value, model, timestamp)
        if iqr_anomaly and not zscore_anomaly:
            anomalies.append(iqr_anomaly)
        
        # Spike/Drop detection
        trend_anomaly = self._detect_trend(metric_name, value, timestamp)
        if trend_anomaly:
            anomalies.append(trend_anomaly)
        
        # Store anomalies
        self._anomalies.extend(anomalies)
        
        # Fire alerts
        for anomaly in anomalies:
            self._fire_alert(anomaly)
        
        return anomalies
    
    def _detect_zscore(self, metric_name: str, value: float,
                       model: Dict, timestamp: float) -> Optional[Anomaly]:
        """Z-score based detection."""
        if model["std"] == 0:
            return None
        
        zscore = abs(value - model["mean"]) / model["std"]
        
        if zscore > self.config.zscore_threshold:
            self._anomaly_count += 1
            
            severity = self._calculate_severity(zscore, self.config.zscore_threshold)
            anomaly_type = AnomalyType.SPIKE if value > model["mean"] else AnomalyType.DROP
            
            return Anomaly(
                anomaly_id=f"anom_{self._anomaly_count}",
                anomaly_type=anomaly_type,
                severity=severity,
                metric_name=metric_name,
                value=value,
                expected_value=model["mean"],
                deviation=zscore,
                timestamp=timestamp,
                context={"detection_method": "zscore", "threshold": self.config.zscore_threshold}
            )
        
        return None
    
    def _detect_iqr(self, metric_name: str, value: float,
                    model: Dict, timestamp: float) -> Optional[Anomaly]:
        """IQR based detection."""
        lower = model["q1"] - self.config.iqr_multiplier * model["iqr"]
        upper = model["q3"] + self.config.iqr_multiplier * model["iqr"]
        
        if value < lower or value > upper:
            self._anomaly_count += 1
            
            deviation = max(
                abs(value - lower) / model["iqr"] if value < lower else 0,
                abs(value - upper) / model["iqr"] if value > upper else 0
            )
            
            severity = self._calculate_severity(deviation, 1.0)
            
            return Anomaly(
                anomaly_id=f"anom_{self._anomaly_count}",
                anomaly_type=AnomalyType.OUTLIER,
                severity=severity,
                metric_name=metric_name,
                value=value,
                expected_value=model["median"],
                deviation=deviation,
                timestamp=timestamp,
                context={
                    "detection_method": "iqr",
                    "lower_bound": lower,
                    "upper_bound": upper
                }
            )
        
        return None
    
    def _detect_trend(self, metric_name: str, value: float,
                      timestamp: float) -> Optional[Anomaly]:
        """Detect sudden spikes or drops."""
        window = self._windows.get(metric_name, [])
        
        if len(window) < 5:
            return None
        
        recent_mean = np.mean(window[-5:-1]) if len(window) > 5 else np.mean(window[:-1])
        
        if recent_mean == 0:
            return None
        
        change_ratio = abs(value - recent_mean) / recent_mean
        
        if change_ratio > 0.5:  # 50% change
            self._anomaly_count += 1
            
            severity = self._calculate_severity(change_ratio, 0.5)
            anomaly_type = AnomalyType.SPIKE if value > recent_mean else AnomalyType.DROP
            
            return Anomaly(
                anomaly_id=f"anom_{self._anomaly_count}",
                anomaly_type=anomaly_type,
                severity=severity,
                metric_name=metric_name,
                value=value,
                expected_value=recent_mean,
                deviation=change_ratio,
                timestamp=timestamp,
                context={"detection_method": "trend", "change_ratio": change_ratio}
            )
        
        return None
    
    def _calculate_severity(self, deviation: float,
                            threshold: float) -> Severity:
        """Calculate anomaly severity."""
        ratio = deviation / threshold
        
        if ratio > 4:
            return Severity.CRITICAL
        elif ratio > 2.5:
            return Severity.HIGH
        elif ratio > 1.5:
            return Severity.MEDIUM
        else:
            return Severity.LOW
    
    def _fire_alert(self, anomaly: Anomaly):
        """Fire alert for anomaly."""
        for handler in self._alert_handlers:
            try:
                handler(anomaly)
            except Exception as e:
                logger.error(f"Alert handler error: {e}")
        
        logger.warning(
            f"Anomaly detected: {anomaly.metric_name} - "
            f"{anomaly.anomaly_type.value} ({anomaly.severity.value})"
        )
    
    def on_alert(self, handler):
        """Register alert handler."""
        self._alert_handlers.append(handler)
    
    def get_anomalies(self, metric_name: str = None,
                      severity: Severity = None,
                      since: float = None) -> List[Anomaly]:
        """Get detected anomalies."""
        anomalies = self._anomalies.copy()
        
        if metric_name:
            anomalies = [a for a in anomalies if a.metric_name == metric_name]
        
        if severity:
            anomalies = [a for a in anomalies if a.severity == severity]
        
        if since:
            anomalies = [a for a in anomalies if a.timestamp >= since]
        
        return anomalies
    
    def get_model_info(self, metric_name: str) -> Optional[Dict]:
        """Get trained model info."""
        return self._models.get(metric_name)
    
    def __repr__(self) -> str:
        return f"AnomalyDetector(models={len(self._models)}, anomalies={len(self._anomalies)})"
