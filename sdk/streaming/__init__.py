"""
Streaming Module
================

Real-time data streaming and event processing.

Features:
- Event streaming
- WebSocket handling
- Server-Sent Events
- Message queues
- Stream processing
"""

from .events import EventEmitter, EventBus
from .websocket import WebSocketServer, WebSocketClient
from .sse import SSEServer, SSEClient
from .pipeline import StreamPipeline, StreamProcessor

__all__ = [
    "EventEmitter",
    "EventBus",
    "WebSocketServer",
    "WebSocketClient",
    "SSEServer",
    "SSEClient",
    "StreamPipeline",
    "StreamProcessor"
]
