"""
WebSocket
=========

WebSocket server and client.
"""

from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
import asyncio
import json
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class WSMessage:
    """WebSocket message."""
    type: str
    data: Any
    client_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


class WebSocketServer:
    """
    WebSocket server.
    
    Features:
    - Client management
    - Broadcasting
    - Rooms/channels
    - Message routing
    
    Example:
        >>> server = WebSocketServer()
        >>> server.on_message(handler)
        >>> await server.start("localhost", 8765)
    """
    
    def __init__(self, ping_interval: int = 30):
        """
        Initialize WebSocket server.
        
        Args:
            ping_interval: Ping interval in seconds
        """
        self.ping_interval = ping_interval
        
        self._clients: Dict[str, Any] = {}
        self._rooms: Dict[str, Set[str]] = {}
        
        self._on_connect: List[Callable] = []
        self._on_disconnect: List[Callable] = []
        self._on_message: List[Callable] = []
        
        self._running = False
        self._server = None
        
        logger.info("WebSocket Server initialized")
    
    async def start(self, host: str = "localhost",
                    port: int = 8765):
        """
        Start WebSocket server.
        
        Args:
            host: Server host
            port: Server port
        """
        try:
            import websockets
            
            self._running = True
            self._server = await websockets.serve(
                self._handle_client,
                host,
                port,
                ping_interval=self.ping_interval
            )
            
            logger.info(f"WebSocket server started on ws://{host}:{port}")
            
        except ImportError:
            logger.error("websockets library not installed")
    
    async def stop(self):
        """Stop server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
    
    async def _handle_client(self, websocket, path):
        """Handle client connection."""
        client_id = f"client_{len(self._clients) + 1}_{int(time.time())}"
        self._clients[client_id] = websocket
        
        # Fire connect callbacks
        for callback in self._on_connect:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(client_id)
                else:
                    callback(client_id)
            except Exception as e:
                logger.error(f"Connect callback error: {e}")
        
        try:
            async for message in websocket:
                await self._handle_message(client_id, message)
        finally:
            # Cleanup
            del self._clients[client_id]
            
            # Remove from rooms
            for room in self._rooms.values():
                room.discard(client_id)
            
            # Fire disconnect callbacks
            for callback in self._on_disconnect:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(client_id)
                    else:
                        callback(client_id)
                except Exception as e:
                    logger.error(f"Disconnect callback error: {e}")
    
    async def _handle_message(self, client_id: str, raw_message: str):
        """Handle incoming message."""
        try:
            data = json.loads(raw_message)
        except:
            data = raw_message
        
        message = WSMessage(
            type=data.get("type", "message") if isinstance(data, dict) else "message",
            data=data.get("data", data) if isinstance(data, dict) else data,
            client_id=client_id
        )
        
        for callback in self._on_message:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message)
                else:
                    callback(message)
            except Exception as e:
                logger.error(f"Message callback error: {e}")
    
    async def send(self, client_id: str, data: Any):
        """Send message to client."""
        if client_id in self._clients:
            websocket = self._clients[client_id]
            message = json.dumps(data) if not isinstance(data, str) else data
            await websocket.send(message)
    
    async def broadcast(self, data: Any, exclude: List[str] = None):
        """Broadcast to all clients."""
        exclude = exclude or []
        message = json.dumps(data) if not isinstance(data, str) else data
        
        for client_id, websocket in self._clients.items():
            if client_id not in exclude:
                try:
                    await websocket.send(message)
                except:
                    pass
    
    async def send_to_room(self, room: str, data: Any):
        """Send to room members."""
        if room not in self._rooms:
            return
        
        message = json.dumps(data) if not isinstance(data, str) else data
        
        for client_id in self._rooms[room]:
            if client_id in self._clients:
                try:
                    await self._clients[client_id].send(message)
                except:
                    pass
    
    def join_room(self, client_id: str, room: str):
        """Add client to room."""
        if room not in self._rooms:
            self._rooms[room] = set()
        self._rooms[room].add(client_id)
    
    def leave_room(self, client_id: str, room: str):
        """Remove client from room."""
        if room in self._rooms:
            self._rooms[room].discard(client_id)
    
    def on_connect(self, callback: Callable):
        """Register connect handler."""
        self._on_connect.append(callback)
    
    def on_disconnect(self, callback: Callable):
        """Register disconnect handler."""
        self._on_disconnect.append(callback)
    
    def on_message(self, callback: Callable):
        """Register message handler."""
        self._on_message.append(callback)
    
    def get_clients(self) -> List[str]:
        """Get connected client IDs."""
        return list(self._clients.keys())
    
    def get_rooms(self) -> Dict[str, int]:
        """Get rooms with member counts."""
        return {room: len(members) for room, members in self._rooms.items()}


class WebSocketClient:
    """
    WebSocket client.
    
    Example:
        >>> client = WebSocketClient("ws://localhost:8765")
        >>> await client.connect()
        >>> await client.send({"type": "hello"})
    """
    
    def __init__(self, url: str):
        """
        Initialize WebSocket client.
        
        Args:
            url: WebSocket server URL
        """
        self.url = url
        
        self._websocket = None
        self._connected = False
        
        self._on_message: List[Callable] = []
        self._on_connect: List[Callable] = []
        self._on_disconnect: List[Callable] = []
    
    async def connect(self):
        """Connect to server."""
        try:
            import websockets
            
            self._websocket = await websockets.connect(self.url)
            self._connected = True
            
            for callback in self._on_connect:
                callback()
            
            logger.info(f"Connected to {self.url}")
            
        except ImportError:
            logger.error("websockets library not installed")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
    
    async def disconnect(self):
        """Disconnect from server."""
        if self._websocket:
            await self._websocket.close()
            self._connected = False
            
            for callback in self._on_disconnect:
                callback()
    
    async def send(self, data: Any):
        """Send message to server."""
        if self._websocket:
            message = json.dumps(data) if not isinstance(data, str) else data
            await self._websocket.send(message)
    
    async def receive(self) -> Optional[Any]:
        """Receive message from server."""
        if self._websocket:
            message = await self._websocket.recv()
            try:
                return json.loads(message)
            except:
                return message
        return None
    
    async def listen(self):
        """Listen for messages."""
        if not self._websocket:
            return
        
        try:
            async for message in self._websocket:
                try:
                    data = json.loads(message)
                except:
                    data = message
                
                for callback in self._on_message:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
        except:
            self._connected = False
    
    def on_message(self, callback: Callable):
        """Register message handler."""
        self._on_message.append(callback)
    
    def on_connect(self, callback: Callable):
        """Register connect handler."""
        self._on_connect.append(callback)
    
    def on_disconnect(self, callback: Callable):
        """Register disconnect handler."""
        self._on_disconnect.append(callback)
    
    @property
    def connected(self) -> bool:
        """Check if connected."""
        return self._connected
