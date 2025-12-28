"""
Analytics Tracker
=================

Track and analyze AI system events and metrics.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Analytics event types."""
    INFERENCE = "inference"
    TRAINING = "training"
    ERROR = "error"
    USER_ACTION = "user_action"
    SYSTEM = "system"
    QUANTUM = "quantum"
    BLOCKCHAIN = "blockchain"


@dataclass
class AnalyticsEvent:
    """Analytics event."""
    event_id: str
    event_type: EventType
    name: str
    properties: Dict[str, Any]
    timestamp: float
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    device_id: Optional[str] = None


@dataclass
class Session:
    """User/device session."""
    session_id: str
    user_id: Optional[str]
    device_id: Optional[str]
    started_at: float
    last_activity: float
    events_count: int = 0
    properties: Dict[str, Any] = field(default_factory=dict)


class AnalyticsTracker:
    """
    Analytics tracking system.
    
    Features:
    - Event tracking
    - Session management
    - Real-time streaming
    - Batch processing
    - Privacy-aware tracking
    
    Example:
        >>> tracker = AnalyticsTracker()
        >>> tracker.track("inference_complete", {"latency": 50})
        >>> tracker.track_inference(model="gpt4", latency=50)
    """
    
    def __init__(self, app_id: str = "aiplatform",
                 flush_interval: int = 60,
                 batch_size: int = 100,
                 enable_pii_filter: bool = True):
        """
        Initialize analytics tracker.
        
        Args:
            app_id: Application identifier
            flush_interval: Flush interval in seconds
            batch_size: Batch size for sending
            enable_pii_filter: Filter PII data
        """
        self.app_id = app_id
        self.flush_interval = flush_interval
        self.batch_size = batch_size
        self.enable_pii_filter = enable_pii_filter
        
        # Events buffer
        self._events: List[AnalyticsEvent] = []
        
        # Sessions
        self._sessions: Dict[str, Session] = {}
        self._current_session: Optional[Session] = None
        
        # Handlers
        self._event_handlers: List[Callable] = []
        
        # Stats
        self._total_events = 0
        self._last_flush = time.time()
        
        logger.info(f"Analytics Tracker initialized: {app_id}")
    
    def start_session(self, user_id: str = None,
                      device_id: str = None,
                      properties: Dict = None) -> Session:
        """
        Start new tracking session.
        
        Args:
            user_id: User identifier
            device_id: Device identifier
            properties: Session properties
            
        Returns:
            Session
        """
        session_id = hashlib.sha256(
            f"{user_id}_{device_id}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            device_id=device_id,
            started_at=time.time(),
            last_activity=time.time(),
            properties=properties or {}
        )
        
        self._sessions[session_id] = session
        self._current_session = session
        
        self.track("session_start", {"session_id": session_id})
        
        logger.info(f"Session started: {session_id}")
        return session
    
    def end_session(self, session_id: str = None):
        """End tracking session."""
        session = self._sessions.get(
            session_id or (self._current_session.session_id if self._current_session else None)
        )
        
        if session:
            duration = time.time() - session.started_at
            
            self.track("session_end", {
                "session_id": session.session_id,
                "duration": duration,
                "events_count": session.events_count
            })
            
            if self._current_session and self._current_session.session_id == session.session_id:
                self._current_session = None
    
    def track(self, event_name: str,
              properties: Dict = None,
              event_type: EventType = EventType.SYSTEM) -> str:
        """
        Track an event.
        
        Args:
            event_name: Event name
            properties: Event properties
            event_type: Event type
            
        Returns:
            Event ID
        """
        event_id = f"evt_{self._total_events}_{int(time.time() * 1000) % 1000000}"
        
        # Filter PII if enabled
        safe_props = self._filter_pii(properties or {}) if self.enable_pii_filter else (properties or {})
        
        event = AnalyticsEvent(
            event_id=event_id,
            event_type=event_type,
            name=event_name,
            properties=safe_props,
            timestamp=time.time(),
            session_id=self._current_session.session_id if self._current_session else None,
            user_id=self._current_session.user_id if self._current_session else None,
            device_id=self._current_session.device_id if self._current_session else None
        )
        
        self._events.append(event)
        self._total_events += 1
        
        # Update session
        if self._current_session:
            self._current_session.last_activity = time.time()
            self._current_session.events_count += 1
        
        # Fire handlers
        for handler in self._event_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
        
        # Auto flush
        if len(self._events) >= self.batch_size:
            self.flush()
        
        return event_id
    
    def track_inference(self, model: str,
                        latency_ms: float,
                        input_tokens: int = 0,
                        output_tokens: int = 0,
                        success: bool = True,
                        error: str = None):
        """Track model inference."""
        self.track("inference", {
            "model": model,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "success": success,
            "error": error
        }, EventType.INFERENCE)
    
    def track_training(self, model: str,
                       epoch: int,
                       loss: float,
                       metrics: Dict = None):
        """Track training progress."""
        self.track("training", {
            "model": model,
            "epoch": epoch,
            "loss": loss,
            "metrics": metrics or {}
        }, EventType.TRAINING)
    
    def track_quantum(self, operation: str,
                      qubits: int,
                      shots: int,
                      backend: str,
                      execution_time: float):
        """Track quantum operation."""
        self.track("quantum_operation", {
            "operation": operation,
            "qubits": qubits,
            "shots": shots,
            "backend": backend,
            "execution_time": execution_time
        }, EventType.QUANTUM)
    
    def track_blockchain(self, chain: str,
                         operation: str,
                         tx_hash: str = None,
                         gas_used: int = 0,
                         success: bool = True):
        """Track blockchain operation."""
        self.track("blockchain_operation", {
            "chain": chain,
            "operation": operation,
            "tx_hash": tx_hash,
            "gas_used": gas_used,
            "success": success
        }, EventType.BLOCKCHAIN)
    
    def track_error(self, error_type: str,
                    message: str,
                    stack_trace: str = None,
                    context: Dict = None):
        """Track error event."""
        self.track("error", {
            "error_type": error_type,
            "message": message,
            "stack_trace": stack_trace,
            "context": context or {}
        }, EventType.ERROR)
    
    def _filter_pii(self, properties: Dict) -> Dict:
        """Filter PII from properties."""
        pii_fields = [
            "email", "phone", "ssn", "password", "credit_card",
            "address", "ip_address", "name", "dob"
        ]
        
        filtered = {}
        for key, value in properties.items():
            if key.lower() in pii_fields:
                filtered[key] = "[REDACTED]"
            elif isinstance(value, dict):
                filtered[key] = self._filter_pii(value)
            else:
                filtered[key] = value
        
        return filtered
    
    def flush(self) -> int:
        """
        Flush events to storage/API.
        
        Returns:
            Number of events flushed
        """
        if not self._events:
            return 0
        
        events_to_flush = self._events.copy()
        self._events = []
        
        # In production, send to analytics backend
        logger.debug(f"Flushed {len(events_to_flush)} events")
        
        self._last_flush = time.time()
        return len(events_to_flush)
    
    def on_event(self, handler: Callable):
        """Register event handler."""
        self._event_handlers.append(handler)
    
    def get_events(self, event_type: EventType = None,
                   since: float = None,
                   limit: int = 100) -> List[AnalyticsEvent]:
        """Get tracked events."""
        events = self._events.copy()
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        return events[-limit:]
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics."""
        return {
            "app_id": self.app_id,
            "total_events": self._total_events,
            "buffered_events": len(self._events),
            "active_sessions": len([s for s in self._sessions.values() 
                                   if time.time() - s.last_activity < 1800]),
            "total_sessions": len(self._sessions),
            "last_flush": self._last_flush
        }
    
    def export_events(self, format: str = "json") -> str:
        """Export events."""
        events_data = [
            {
                "event_id": e.event_id,
                "type": e.event_type.value,
                "name": e.name,
                "properties": e.properties,
                "timestamp": e.timestamp
            }
            for e in self._events
        ]
        
        if format == "json":
            return json.dumps(events_data, indent=2)
        
        return str(events_data)
    
    def __repr__(self) -> str:
        return f"AnalyticsTracker(app_id='{self.app_id}', events={self._total_events})"
