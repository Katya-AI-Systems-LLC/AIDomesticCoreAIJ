"""
Event System
============

Event emitting and handling.
"""

from typing import Dict, Any, Optional, List, Callable, Awaitable
from dataclasses import dataclass, field
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Event data."""
    name: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class EventEmitter:
    """
    Event emitter pattern implementation.
    
    Features:
    - Sync and async handlers
    - Once listeners
    - Wildcard events
    - Error handling
    
    Example:
        >>> emitter = EventEmitter()
        >>> emitter.on("data", lambda e: print(e.data))
        >>> emitter.emit("data", {"value": 42})
    """
    
    def __init__(self, max_listeners: int = 100):
        """
        Initialize event emitter.
        
        Args:
            max_listeners: Maximum listeners per event
        """
        self.max_listeners = max_listeners
        
        self._listeners: Dict[str, List[Callable]] = {}
        self._once_listeners: Dict[str, List[Callable]] = {}
        
        logger.debug("EventEmitter initialized")
    
    def on(self, event: str, handler: Callable):
        """
        Register event handler.
        
        Args:
            event: Event name
            handler: Handler function
        """
        if event not in self._listeners:
            self._listeners[event] = []
        
        if len(self._listeners[event]) >= self.max_listeners:
            logger.warning(f"Max listeners reached for event: {event}")
            return
        
        self._listeners[event].append(handler)
    
    def once(self, event: str, handler: Callable):
        """Register one-time handler."""
        if event not in self._once_listeners:
            self._once_listeners[event] = []
        
        self._once_listeners[event].append(handler)
    
    def off(self, event: str, handler: Callable = None):
        """
        Remove event handler.
        
        Args:
            event: Event name
            handler: Specific handler (None = all)
        """
        if handler is None:
            self._listeners.pop(event, None)
            self._once_listeners.pop(event, None)
        else:
            if event in self._listeners:
                self._listeners[event] = [
                    h for h in self._listeners[event] if h != handler
                ]
    
    def emit(self, event: str, data: Any = None,
             source: str = None) -> int:
        """
        Emit event synchronously.
        
        Args:
            event: Event name
            data: Event data
            source: Event source
            
        Returns:
            Number of handlers called
        """
        evt = Event(name=event, data=data, source=source)
        count = 0
        
        # Regular listeners
        for handler in self._listeners.get(event, []):
            try:
                handler(evt)
                count += 1
            except Exception as e:
                logger.error(f"Handler error for {event}: {e}")
        
        # Wildcard listeners
        for handler in self._listeners.get("*", []):
            try:
                handler(evt)
                count += 1
            except Exception as e:
                logger.error(f"Wildcard handler error: {e}")
        
        # Once listeners
        once_handlers = self._once_listeners.pop(event, [])
        for handler in once_handlers:
            try:
                handler(evt)
                count += 1
            except Exception as e:
                logger.error(f"Once handler error: {e}")
        
        return count
    
    async def emit_async(self, event: str, data: Any = None,
                         source: str = None) -> int:
        """Emit event asynchronously."""
        evt = Event(name=event, data=data, source=source)
        count = 0
        
        handlers = (
            self._listeners.get(event, []) +
            self._listeners.get("*", []) +
            self._once_listeners.pop(event, [])
        )
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(evt)
                else:
                    handler(evt)
                count += 1
            except Exception as e:
                logger.error(f"Async handler error: {e}")
        
        return count
    
    def listeners(self, event: str) -> List[Callable]:
        """Get listeners for event."""
        return self._listeners.get(event, []).copy()
    
    def event_names(self) -> List[str]:
        """Get all event names."""
        return list(self._listeners.keys())
    
    def remove_all_listeners(self):
        """Remove all listeners."""
        self._listeners.clear()
        self._once_listeners.clear()


class EventBus:
    """
    Global event bus for pub/sub.
    
    Features:
    - Topic-based messaging
    - Pattern matching
    - Persistence option
    - Replay capability
    
    Example:
        >>> bus = EventBus()
        >>> bus.subscribe("user.*", handler)
        >>> bus.publish("user.created", user_data)
    """
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize event bus.
        
        Args:
            history_size: Number of events to keep
        """
        self.history_size = history_size
        
        self._subscribers: Dict[str, List[Callable]] = {}
        self._history: List[Event] = []
        
        logger.info("EventBus initialized")
    
    def subscribe(self, topic: str, handler: Callable) -> str:
        """
        Subscribe to topic.
        
        Args:
            topic: Topic pattern (supports wildcards)
            handler: Handler function
            
        Returns:
            Subscription ID
        """
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        
        self._subscribers[topic].append(handler)
        
        sub_id = f"{topic}_{len(self._subscribers[topic])}"
        logger.debug(f"Subscribed to {topic}: {sub_id}")
        
        return sub_id
    
    def unsubscribe(self, topic: str, handler: Callable = None):
        """Unsubscribe from topic."""
        if topic in self._subscribers:
            if handler:
                self._subscribers[topic] = [
                    h for h in self._subscribers[topic] if h != handler
                ]
            else:
                del self._subscribers[topic]
    
    def publish(self, topic: str, data: Any = None,
                source: str = None) -> int:
        """
        Publish event to topic.
        
        Args:
            topic: Event topic
            data: Event data
            source: Event source
            
        Returns:
            Number of subscribers notified
        """
        evt = Event(name=topic, data=data, source=source)
        
        # Add to history
        self._history.append(evt)
        if len(self._history) > self.history_size:
            self._history.pop(0)
        
        count = 0
        
        # Find matching subscribers
        for pattern, handlers in self._subscribers.items():
            if self._matches(topic, pattern):
                for handler in handlers:
                    try:
                        handler(evt)
                        count += 1
                    except Exception as e:
                        logger.error(f"Subscriber error: {e}")
        
        return count
    
    async def publish_async(self, topic: str, data: Any = None) -> int:
        """Publish event asynchronously."""
        evt = Event(name=topic, data=data)
        
        self._history.append(evt)
        if len(self._history) > self.history_size:
            self._history.pop(0)
        
        count = 0
        
        for pattern, handlers in self._subscribers.items():
            if self._matches(topic, pattern):
                for handler in handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(evt)
                        else:
                            handler(evt)
                        count += 1
                    except Exception as e:
                        logger.error(f"Async subscriber error: {e}")
        
        return count
    
    def _matches(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern."""
        if pattern == "*":
            return True
        
        if "*" not in pattern:
            return topic == pattern
        
        # Simple wildcard matching
        import fnmatch
        return fnmatch.fnmatch(topic, pattern)
    
    def replay(self, topic: str = None,
               since: float = None,
               limit: int = 100) -> List[Event]:
        """
        Replay historical events.
        
        Args:
            topic: Filter by topic
            since: Events after timestamp
            limit: Maximum events
            
        Returns:
            List of events
        """
        events = self._history.copy()
        
        if topic:
            events = [e for e in events if self._matches(e.name, topic)]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        return events[-limit:]
    
    def get_topics(self) -> List[str]:
        """Get all subscribed topics."""
        return list(self._subscribers.keys())
    
    def clear_history(self):
        """Clear event history."""
        self._history.clear()
