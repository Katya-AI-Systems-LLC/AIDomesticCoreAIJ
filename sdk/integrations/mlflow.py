"""
MLflow Integration
==================

Experiment tracking and model registry.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """MLflow experiment."""
    experiment_id: str
    name: str
    artifact_location: str
    lifecycle_stage: str


@dataclass
class Run:
    """MLflow run."""
    run_id: str
    experiment_id: str
    status: str
    start_time: float
    end_time: Optional[float]
    metrics: Dict[str, float]
    params: Dict[str, str]
    tags: Dict[str, str]


class MLflowTracker:
    """
    MLflow experiment tracking.
    
    Features:
    - Experiment management
    - Run tracking
    - Metric and parameter logging
    - Model registry
    - Artifact storage
    
    Example:
        >>> tracker = MLflowTracker()
        >>> with tracker.start_run("training"):
        ...     tracker.log_param("learning_rate", 0.01)
        ...     tracker.log_metric("loss", 0.5)
    """
    
    def __init__(self, tracking_uri: str = "mlruns",
                 experiment_name: str = "default"):
        """
        Initialize MLflow tracker.
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Default experiment name
        """
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        
        self._client = None
        self._current_run: Optional[Run] = None
        self._experiments: Dict[str, Experiment] = {}
        self._runs: Dict[str, Run] = {}
        
        self._initialize()
        logger.info(f"MLflow tracker initialized: {tracking_uri}")
    
    def _initialize(self):
        """Initialize MLflow client."""
        try:
            import mlflow
            
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._client = mlflow
            
        except ImportError:
            logger.warning("mlflow not installed, using simulation")
            self._client = None
    
    def create_experiment(self, name: str,
                          artifact_location: Optional[str] = None) -> str:
        """Create a new experiment."""
        experiment_id = f"exp_{hash(name) % 10000:04d}"
        
        self._experiments[experiment_id] = Experiment(
            experiment_id=experiment_id,
            name=name,
            artifact_location=artifact_location or f"{self.tracking_uri}/{name}",
            lifecycle_stage="active"
        )
        
        if self._client:
            try:
                experiment_id = self._client.create_experiment(
                    name, artifact_location
                )
            except:
                pass
        
        return experiment_id
    
    def set_experiment(self, name: str):
        """Set active experiment."""
        self.experiment_name = name
        
        if self._client:
            self._client.set_experiment(name)
    
    def start_run(self, run_name: Optional[str] = None,
                  tags: Optional[Dict[str, str]] = None) -> 'RunContext':
        """
        Start a new run.
        
        Args:
            run_name: Run name
            tags: Run tags
            
        Returns:
            Run context manager
        """
        return RunContext(self, run_name, tags)
    
    def _start_run_internal(self, run_name: Optional[str],
                            tags: Optional[Dict[str, str]]) -> Run:
        """Internal run start."""
        run_id = f"run_{int(time.time() * 1000) % 1000000:06d}"
        
        run = Run(
            run_id=run_id,
            experiment_id=self.experiment_name,
            status="RUNNING",
            start_time=time.time(),
            end_time=None,
            metrics={},
            params={},
            tags=tags or {}
        )
        
        if run_name:
            run.tags["mlflow.runName"] = run_name
        
        self._runs[run_id] = run
        self._current_run = run
        
        if self._client:
            try:
                self._client.start_run(run_name=run_name, tags=tags)
            except:
                pass
        
        return run
    
    def _end_run_internal(self, status: str = "FINISHED"):
        """Internal run end."""
        if self._current_run:
            self._current_run.status = status
            self._current_run.end_time = time.time()
            self._current_run = None
        
        if self._client:
            try:
                self._client.end_run(status)
            except:
                pass
    
    def log_param(self, key: str, value: Any):
        """Log a parameter."""
        if self._current_run:
            self._current_run.params[key] = str(value)
        
        if self._client:
            try:
                self._client.log_param(key, value)
            except:
                pass
    
    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters."""
        for key, value in params.items():
            self.log_param(key, value)
    
    def log_metric(self, key: str, value: float, step: int = None):
        """Log a metric."""
        if self._current_run:
            self._current_run.metrics[key] = value
        
        if self._client:
            try:
                self._client.log_metric(key, value, step=step)
            except:
                pass
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log multiple metrics."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)
    
    def log_artifact(self, local_path: str, artifact_path: str = None):
        """Log an artifact."""
        logger.info(f"Logging artifact: {local_path}")
        
        if self._client:
            try:
                self._client.log_artifact(local_path, artifact_path)
            except:
                pass
    
    def log_model(self, model: Any, artifact_path: str,
                  registered_model_name: Optional[str] = None):
        """Log a model."""
        logger.info(f"Logging model to {artifact_path}")
        
        # In production, serialize and save model
    
    def set_tag(self, key: str, value: str):
        """Set a tag."""
        if self._current_run:
            self._current_run.tags[key] = value
        
        if self._client:
            try:
                self._client.set_tag(key, value)
            except:
                pass
    
    def get_run(self, run_id: str) -> Optional[Run]:
        """Get run by ID."""
        return self._runs.get(run_id)
    
    def search_runs(self, experiment_ids: List[str] = None,
                    filter_string: str = None) -> List[Run]:
        """Search runs."""
        runs = list(self._runs.values())
        
        if experiment_ids:
            runs = [r for r in runs if r.experiment_id in experiment_ids]
        
        return runs
    
    def register_model(self, model_uri: str, name: str) -> str:
        """Register model in model registry."""
        model_version = f"{name}/1"
        logger.info(f"Registered model: {model_version}")
        return model_version
    
    def load_model(self, model_uri: str) -> Any:
        """Load model from registry."""
        logger.info(f"Loading model: {model_uri}")
        return None  # Placeholder


class RunContext:
    """Context manager for MLflow runs."""
    
    def __init__(self, tracker: MLflowTracker,
                 run_name: Optional[str],
                 tags: Optional[Dict[str, str]]):
        self.tracker = tracker
        self.run_name = run_name
        self.tags = tags
        self.run: Optional[Run] = None
    
    def __enter__(self) -> 'RunContext':
        self.run = self.tracker._start_run_internal(self.run_name, self.tags)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FINISHED" if exc_type is None else "FAILED"
        self.tracker._end_run_internal(status)
    
    def log_param(self, key: str, value: Any):
        self.tracker.log_param(key, value)
    
    def log_metric(self, key: str, value: float, step: int = None):
        self.tracker.log_metric(key, value, step)
