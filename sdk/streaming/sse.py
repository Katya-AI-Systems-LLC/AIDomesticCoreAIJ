"""
Server-Sent Events
==================

SSE server and client.
"""

from typing import Dict, Any, Optional, List, Callable, AsyncGenerator
from dataclasses import dataclass, field
import asyncio
import json
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class SSEMessage:
    """SSE message."""
    data: Any
    event: Optional[str] = None
    id: Optional[str] = None
    retry: Optional[int] = None
    
    def format(self) -> str:
        """Format as SSE string."""
        lines = []
        
        if self.id:
            lines.append(f"id: {self.id}")
        if self.event:
            lines.append(f"event: {self.event}")
        if self.retry:
            lines.append(f"retry: {self.retry}")
        
        data = json.dumps(self.data) if not isinstance(self.data, str) else self.data
        for line in data.split("\n"):
            lines.append(f"data: {line}")
        
        return "\n".join(lines) + "\n\n"


class SSEServer:
    """
    Server-Sent Events server.
    
    Features:
    - Event streaming
    - Client management
    - Heartbeat
    - Event history
    
    Example:
        >>> sse = SSEServer()
        >>> @app.route("/events")
        ... async def events():
        ...     return sse.stream()
        >>> sse.send("update", {"value": 42})
    """
    
    def __init__(self, heartbeat_interval: int = 30,
                 history_size: int = 100):
        """
        Initialize SSE server.
        
        Args:
            heartbeat_interval: Heartbeat interval
            history_size: Event history size
        """
        self.heartbeat_interval = heartbeat_interval
        self.history_size = history_size
        
        self._clients: Dict[str, asyncio.Queue] = {}
        self._history: List[SSEMessage] = []
        self._event_id = 0
        
        logger.info("SSE Server initialized")
    
    async def stream(self, client_id: str = None,
                     last_event_id: str = None) -> AsyncGenerator[str, None]:
        """
        Create SSE stream.
        
        Args:
            client_id: Client identifier
            last_event_id: Last received event ID
            
        Yields:
            SSE formatted strings
        """
        client_id = client_id or f"client_{len(self._clients) + 1}"
        queue: asyncio.Queue = asyncio.Queue()
        self._clients[client_id] = queue
        
        # Replay missed events
        if last_event_id:
            for msg in self._history:
                if msg.id and int(msg.id) > int(last_event_id):
                    yield msg.format()
        
        try:
            while True:
                try:
                    message = await asyncio.wait_for(
                        queue.get(),
                        timeout=self.heartbeat_interval
                    )
                    yield message.format()
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield ": heartbeat\n\n"
        finally:
            del self._clients[client_id]
    
    def send(self, event: str, data: Any,
             client_id: str = None):
        """
        Send event to clients.
        
        Args:
            event: Event name
            data: Event data
            client_id: Specific client (None = all)
        """
        self._event_id += 1
        
        message = SSEMessage(
            data=data,
            event=event,
            id=str(self._event_id)
        )
        
        # Add to history
        self._history.append(message)
        if len(self._history) > self.history_size:
            self._history.pop(0)
        
        # Send to clients
        if client_id:
            if client_id in self._clients:
                self._clients[client_id].put_nowait(message)
        else:
            for queue in self._clients.values():
                queue.put_nowait(message)
    
    def send_data(self, data: Any):
        """Send data without event name."""
        self.send(None, data)
    
    def get_clients(self) -> List[str]:
        """Get connected clients."""
        return list(self._clients.keys())
    
    def disconnect(self, client_id: str):
        """Disconnect client."""
        if client_id in self._clients:
            del self._clients[client_id]


class SSEClient:
    """
    Server-Sent Events client.
    
    Example:
        >>> client = SSEClient("http://localhost:8000/events")
        >>> async for event in client.stream():
        ...     print(event)
    """
    
    def __init__(self, url: str):
        """
        Initialize SSE client.
        
        Args:
            url: SSE endpoint URL
        """
        self.url = url
        self._last_event_id: Optional[str] = None
        self._connected = False
    
    async def stream(self) -> AsyncGenerator[SSEMessage, None]:
        """
        Stream events from server.
        
        Yields:
            SSEMessage objects
        """
        try:
            import aiohttp
            
            headers = {}
            if self._last_event_id:
                headers["Last-Event-ID"] = self._last_event_id
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.url, headers=headers) as response:
                    self._connected = True
                    
                    buffer = ""
                    async for chunk in response.content:
                        buffer += chunk.decode("utf-8")
                        
                        while "\n\n" in buffer:
                            event_str, buffer = buffer.split("\n\n", 1)
                            message = self._parse_event(event_str)
                            
                            if message:
                                if message.id:
                                    self._last_event_id = message.id
                                yield message
            
        except ImportError:
            logger.error("aiohttp library not installed")
        except Exception as e:
            logger.error(f"SSE connection error: {e}")
        finally:
            self._connected = False
    
    def _parse_event(self, event_str: str) -> Optional[SSEMessage]:
        """Parse SSE event string."""
        if not event_str.strip():
            return None
        
        data_lines = []
        event = None
        event_id = None
        retry = None
        
        for line in event_str.split("\n"):
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
            elif line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("id:"):
                event_id = line[3:].strip()
            elif line.startswith("retry:"):
                try:
                    retry = int(line[6:].strip())
                except:
                    pass
        
        if not data_lines:
            return None
        
        data = "\n".join(data_lines)
        
        try:
            data = json.loads(data)
        except:
            pass
        
        return SSEMessage(
            data=data,
            event=event,
            id=event_id,
            retry=retry
        )
    
    @property
    def connected(self) -> bool:
        """Check if connected."""
        return self._connected
