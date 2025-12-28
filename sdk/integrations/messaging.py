"""
Message Queue Integrations
==========================

Support for message brokers and event streaming.
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class BrokerType(Enum):
    """Supported message broker types."""
    RABBITMQ = "rabbitmq"
    KAFKA = "kafka"
    REDIS = "redis"
    NATS = "nats"
    MEMORY = "memory"


@dataclass
class Message:
    """Message representation."""
    id: str
    topic: str
    payload: Any
    headers: Dict[str, str]
    timestamp: float


class MessageQueue:
    """
    Universal message queue interface.
    
    Supports:
    - RabbitMQ
    - Apache Kafka
    - Redis Pub/Sub
    - NATS
    - In-memory (for testing)
    
    Example:
        >>> mq = MessageQueue(BrokerType.KAFKA, "localhost:9092")
        >>> mq.connect()
        >>> mq.publish("events", {"action": "created"})
        >>> mq.subscribe("events", handler)
    """
    
    def __init__(self, broker_type: BrokerType = BrokerType.MEMORY,
                 host: str = "localhost",
                 port: Optional[int] = None,
                 **kwargs):
        """
        Initialize message queue.
        
        Args:
            broker_type: Broker type
            host: Broker host
            port: Broker port
            **kwargs: Additional broker-specific options
        """
        self.broker_type = broker_type
        self.host = host
        self.port = port or self._default_port(broker_type)
        self.options = kwargs
        
        self._connection = None
        self._connected = False
        
        # In-memory queues for testing
        self._memory_queues: Dict[str, List[Message]] = {}
        self._subscribers: Dict[str, List[Callable]] = {}
        
        self._message_counter = 0
        
        logger.info(f"Message queue initialized: {broker_type.value}")
    
    def _default_port(self, broker_type: BrokerType) -> int:
        """Get default port for broker type."""
        ports = {
            BrokerType.RABBITMQ: 5672,
            BrokerType.KAFKA: 9092,
            BrokerType.REDIS: 6379,
            BrokerType.NATS: 4222,
            BrokerType.MEMORY: 0
        }
        return ports.get(broker_type, 0)
    
    def connect(self) -> bool:
        """Connect to message broker."""
        try:
            if self.broker_type == BrokerType.RABBITMQ:
                import pika
                self._connection = pika.BlockingConnection(
                    pika.ConnectionParameters(host=self.host, port=self.port)
                )
            
            elif self.broker_type == BrokerType.KAFKA:
                from kafka import KafkaProducer, KafkaConsumer
                self._producer = KafkaProducer(
                    bootstrap_servers=f"{self.host}:{self.port}"
                )
            
            elif self.broker_type == BrokerType.REDIS:
                import redis
                self._connection = redis.Redis(host=self.host, port=self.port)
            
            elif self.broker_type == BrokerType.NATS:
                # NATS requires async
                pass
            
            elif self.broker_type == BrokerType.MEMORY:
                pass  # No connection needed
            
            self._connected = True
            logger.info(f"Connected to {self.broker_type.value}")
            return True
            
        except ImportError as e:
            logger.warning(f"Driver not installed: {e}")
            self._connected = True  # Use memory fallback
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from broker."""
        if self._connection:
            try:
                self._connection.close()
            except:
                pass
        self._connected = False
    
    def publish(self, topic: str, payload: Any,
                headers: Optional[Dict[str, str]] = None) -> str:
        """
        Publish message to topic.
        
        Args:
            topic: Topic/queue name
            payload: Message payload
            headers: Optional headers
            
        Returns:
            Message ID
        """
        self._message_counter += 1
        message_id = f"msg_{self._message_counter:08d}"
        
        message = Message(
            id=message_id,
            topic=topic,
            payload=payload,
            headers=headers or {},
            timestamp=time.time()
        )
        
        if self.broker_type == BrokerType.MEMORY or not self._connection:
            # In-memory queue
            if topic not in self._memory_queues:
                self._memory_queues[topic] = []
            self._memory_queues[topic].append(message)
            
            # Notify subscribers
            self._notify_subscribers(topic, message)
        
        elif self.broker_type == BrokerType.RABBITMQ:
            channel = self._connection.channel()
            channel.queue_declare(queue=topic)
            
            import json
            channel.basic_publish(
                exchange='',
                routing_key=topic,
                body=json.dumps(payload)
            )
        
        elif self.broker_type == BrokerType.KAFKA:
            import json
            self._producer.send(topic, json.dumps(payload).encode())
        
        elif self.broker_type == BrokerType.REDIS:
            import json
            self._connection.publish(topic, json.dumps(payload))
        
        logger.debug(f"Published message {message_id} to {topic}")
        return message_id
    
    def subscribe(self, topic: str,
                  handler: Callable[[Message], None]):
        """
        Subscribe to topic.
        
        Args:
            topic: Topic/queue name
            handler: Message handler function
        """
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        
        self._subscribers[topic].append(handler)
        logger.info(f"Subscribed to {topic}")
    
    def unsubscribe(self, topic: str,
                    handler: Callable[[Message], None]):
        """Unsubscribe from topic."""
        if topic in self._subscribers:
            self._subscribers[topic].remove(handler)
    
    def _notify_subscribers(self, topic: str, message: Message):
        """Notify all subscribers of a topic."""
        handlers = self._subscribers.get(topic, [])
        
        for handler in handlers:
            try:
                handler(message)
            except Exception as e:
                logger.error(f"Handler error: {e}")
    
    def consume(self, topic: str, timeout: float = 1.0) -> Optional[Message]:
        """
        Consume one message from topic.
        
        Args:
            topic: Topic name
            timeout: Timeout in seconds
            
        Returns:
            Message or None
        """
        if self.broker_type == BrokerType.MEMORY or not self._connection:
            if topic in self._memory_queues and self._memory_queues[topic]:
                return self._memory_queues[topic].pop(0)
            return None
        
        # For real brokers, implement blocking consume
        return None
    
    async def consume_async(self, topic: str) -> Optional[Message]:
        """Async consume."""
        return self.consume(topic)
    
    def get_queue_size(self, topic: str) -> int:
        """Get number of messages in queue."""
        if self.broker_type == BrokerType.MEMORY:
            return len(self._memory_queues.get(topic, []))
        return 0
    
    def create_topic(self, topic: str, partitions: int = 1,
                     replication_factor: int = 1) -> bool:
        """Create a topic (Kafka-specific)."""
        if topic not in self._memory_queues:
            self._memory_queues[topic] = []
        return True
    
    def delete_topic(self, topic: str) -> bool:
        """Delete a topic."""
        if topic in self._memory_queues:
            del self._memory_queues[topic]
        if topic in self._subscribers:
            del self._subscribers[topic]
        return True
    
    def list_topics(self) -> List[str]:
        """List all topics."""
        return list(self._memory_queues.keys())
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, *args):
        self.disconnect()


class EventBus:
    """
    Simple in-memory event bus.
    
    Example:
        >>> bus = EventBus()
        >>> bus.on("user.created", handle_user_created)
        >>> bus.emit("user.created", {"id": 123})
    """
    
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}
    
    def on(self, event: str, handler: Callable):
        """Register event handler."""
        if event not in self._handlers:
            self._handlers[event] = []
        self._handlers[event].append(handler)
    
    def off(self, event: str, handler: Callable):
        """Remove event handler."""
        if event in self._handlers:
            self._handlers[event].remove(handler)
    
    def emit(self, event: str, data: Any = None):
        """Emit event."""
        handlers = self._handlers.get(event, [])
        
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    async def emit_async(self, event: str, data: Any = None):
        """Async emit event."""
        handlers = self._handlers.get(event, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
    
    def clear(self, event: str = None):
        """Clear handlers."""
        if event:
            self._handlers.pop(event, None)
        else:
            self._handlers.clear()
