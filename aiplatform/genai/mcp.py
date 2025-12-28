"""
MCP (Model Coordination Protocol) Interface for AIPlatform SDK

This module provides implementation of the Model Coordination Protocol
for coordinating multiple AI models and systems.
"""

import logging
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import uuid

from ..exceptions import GenAIError
from .models import GenAIModel, ModelResponse

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@dataclass
class MCPMessage:
    """MCP message format."""
    message_id: str
    sender: str
    recipient: str
    content: Any
    timestamp: datetime
    message_type: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class MCPRequest:
    """MCP request format."""
    request_id: str
    sender: str
    recipient: str
    action: str
    parameters: Dict[str, Any]
    timestamp: datetime
    timeout: int = 30

@dataclass
class MCPResponse:
    """MCP response format."""
    request_id: str
    recipient: str
    sender: str
    status: str  # "success", "error", "timeout"
    result: Any
    timestamp: datetime
    error_message: Optional[str] = None

class MCPInterface:
    """
    Model Coordination Protocol Interface.
    
    Provides implementation of the MCP for coordinating
    multiple AI models and systems in a distributed environment.
    """
    
    def __init__(self, node_id: str, config: Optional[Dict] = None):
        """
        Initialize MCP interface.
        
        Args:
            node_id (str): Unique identifier for this MCP node
            config (dict, optional): MCP configuration
        """
        self._node_id = node_id
        self._config = config or {}
        self._is_initialized = False
        self._connected_nodes = {}
        self._message_handlers = {}
        self._request_handlers = {}
        self._pending_requests = {}
        
        # Initialize MCP
        self._initialize_mcp()
        
        logger.info(f"MCP interface initialized for node: {node_id}")
    
    def _initialize_mcp(self):
        """Initialize MCP system."""
        try:
            # In a real implementation, this would initialize the MCP network
            # For simulation, we'll create placeholder information
            self._mcp_info = {
                "node_id": self._node_id,
                "version": "1.0.0",
                "protocol_version": "MCP/1.0",
                "status": "initialized",
                "capabilities": ["message_routing", "request_response", "model_coordination"]
            }
            
            self._is_initialized = True
            logger.debug(f"MCP initialized for node {self._node_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP: {e}")
            raise GenAIError(f"MCP initialization failed: {e}")
    
    def register_message_handler(self, message_type: str, 
                              handler: Callable[[MCPMessage], None]) -> bool:
        """
        Register message handler.
        
        Args:
            message_type (str): Type of messages to handle
            handler (callable): Function to handle messages
            
        Returns:
            bool: True if registered successfully, False otherwise
        """
        try:
            if not callable(handler):
                raise ValueError("Handler must be callable")
            
            self._message_handlers[message_type] = handler
            logger.debug(f"Message handler registered for type: {message_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register message handler: {e}")
            return False
    
    def register_request_handler(self, action: str, 
                               handler: Callable[[MCPRequest], MCPResponse]) -> bool:
        """
        Register request handler.
        
        Args:
            action (str): Action to handle
            handler (callable): Function to handle requests
            
        Returns:
            bool: True if registered successfully, False otherwise
        """
        try:
            if not callable(handler):
                raise ValueError("Handler must be callable")
            
            self._request_handlers[action] = handler
            logger.debug(f"Request handler registered for action: {action}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register request handler: {e}")
            return False
    
    def send_message(self, recipient: str, content: Any, 
                   message_type: str = "generic", metadata: Optional[Dict] = None) -> str:
        """
        Send message to recipient.
        
        Args:
            recipient (str): Recipient node ID
            content (Any): Message content
            message_type (str): Type of message
            metadata (dict, optional): Additional metadata
            
        Returns:
            str: Message ID
        """
        try:
            if not self._is_initialized:
                raise GenAIError("MCP not initialized")
            
            # Create message
            message_id = f"msg_{uuid.uuid4().hex[:12]}"
            
            message = MCPMessage(
                message_id=message_id,
                sender=self._node_id,
                recipient=recipient,
                content=content,
                timestamp=datetime.now(),
                message_type=message_type,
                metadata=metadata
            )
            
            # In a real implementation, this would send the message over the network
            # For simulation, we'll process it locally
            self._process_message(message)
            
            logger.debug(f"Message sent: {message_id} to {recipient}")
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise GenAIError(f"Message sending failed: {e}")
    
    def send_request(self, recipient: str, action: str, parameters: Dict[str, Any],
                   timeout: int = 30) -> MCPResponse:
        """
        Send request to recipient.
        
        Args:
            recipient (str): Recipient node ID
            action (str): Action to perform
            parameters (dict): Request parameters
            timeout (int): Request timeout in seconds
            
        Returns:
            MCPResponse: Response from recipient
        """
        try:
            if not self._is_initialized:
                raise GenAIError("MCP not initialized")
            
            # Create request
            request_id = f"req_{uuid.uuid4().hex[:12]}"
            
            request = MCPRequest(
                request_id=request_id,
                sender=self._node_id,
                recipient=recipient,
                action=action,
                parameters=parameters,
                timestamp=datetime.now(),
                timeout=timeout
            )
            
            # Store pending request
            self._pending_requests[request_id] = {
                "request": request,
                "timestamp": datetime.now()
            }
            
            # In a real implementation, this would send the request over the network
            # For simulation, we'll process it locally and generate a response
            response = self._process_request(request)
            
            # Remove from pending requests
            if request_id in self._pending_requests:
                del self._pending_requests[request_id]
            
            logger.debug(f"Request sent: {request_id} to {recipient}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to send request: {e}")
            raise GenAIError(f"Request sending failed: {e}")
    
    def _process_message(self, message: MCPMessage):
        """
        Process incoming message.
        
        Args:
            message (MCPMessage): Message to process
        """
        try:
            # Check if we have a handler for this message type
            if message.message_type in self._message_handlers:
                handler = self._message_handlers[message.message_type]
                try:
                    handler(message)
                except Exception as e:
                    logger.error(f"Message handler error: {e}")
            else:
                # Default message handling
                logger.debug(f"Received message: {message.message_id} from {message.sender}")
                logger.debug(f"Message content: {message.content}")
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
    
    def _process_request(self, request: MCPRequest) -> MCPResponse:
        """
        Process incoming request.
        
        Args:
            request (MCPRequest): Request to process
            
        Returns:
            MCPResponse: Generated response
        """
        try:
            # Check if we have a handler for this action
            if request.action in self._request_handlers:
                handler = self._request_handlers[request.action]
                try:
                    return handler(request)
                except Exception as e:
                    logger.error(f"Request handler error: {e}")
                    return MCPResponse(
                        request_id=request.request_id,
                        recipient=request.sender,
                        sender=self._node_id,
                        status="error",
                        result=None,
                        timestamp=datetime.now(),
                        error_message=str(e)
                    )
            else:
                # Default request handling - simulate processing
                result = self._simulate_request_processing(request)
                
                return MCPResponse(
                    request_id=request.request_id,
                    recipient=request.sender,
                    sender=self._node_id,
                    status="success",
                    result=result,
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            logger.error(f"Failed to process request: {e}")
            return MCPResponse(
                request_id=request.request_id,
                recipient=request.sender,
                sender=self._node_id,
                status="error",
                result=None,
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    def _simulate_request_processing(self, request: MCPRequest) -> Any:
        """
        Simulate request processing.
        
        Args:
            request (MCPRequest): Request to process
            
        Returns:
            Any: Processing result
        """
        # Simulate different actions
        action = request.action.lower()
        
        if "generate" in action:
            return f"Generated content for request {request.request_id}"
        elif "analyze" in action:
            return {"analysis_result": "success", "confidence": 0.95}
        elif "process" in action:
            return {"processed": True, "items": len(request.parameters.get("data", []))}
        elif "coordinate" in action:
            return {"coordinated": True, "nodes": [self._node_id, "node_002", "node_003"]}
        else:
            return f"Processed request {request.request_id} with action {action}"
    
    def broadcast_message(self, content: Any, message_type: str = "broadcast",
                        target_nodes: Optional[List[str]] = None) -> List[str]:
        """
        Broadcast message to multiple nodes.
        
        Args:
            content (Any): Message content
            message_type (str): Type of message
            target_nodes (list, optional): Specific nodes to broadcast to
            
        Returns:
            list: List of message IDs
        """
        try:
            if not self._is_initialized:
                raise GenAIError("MCP not initialized")
            
            # If no target nodes specified, use all connected nodes
            if target_nodes is None:
                target_nodes = list(self._connected_nodes.keys())
            
            message_ids = []
            
            for node_id in target_nodes:
                if node_id != self._node_id:  # Don't send to self
                    message_id = self.send_message(
                        recipient=node_id,
                        content=content,
                        message_type=message_type
                    )
                    message_ids.append(message_id)
            
            logger.debug(f"Broadcast message sent to {len(message_ids)} nodes")
            return message_ids
            
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}")
            raise GenAIError(f"Message broadcasting failed: {e}")
    
    def coordinate_models(self, models: List[GenAIModel], task: str, 
                        parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate multiple models for a task.
        
        Args:
            models (list): List of GenAI models to coordinate
            task (str): Task to coordinate
            parameters (dict): Task parameters
            
        Returns:
            dict: Coordination results
        """
        try:
            if not self._is_initialized:
                raise GenAIError("MCP not initialized")
            
            results = {}
            model_responses = []
            
            # Send requests to each model
            for i, model in enumerate(models):
                try:
                    # Create model-specific parameters
                    model_params = parameters.copy()
                    model_params["model_index"] = i
                    model_params["model_info"] = model.get_model_info()
                    
                    # Send request to model (simulated as local processing)
                    response = self._process_model_request(model, task, model_params)
                    model_responses.append(response)
                    
                    results[f"model_{i}"] = {
                        "model": model.get_model_info(),
                        "response": response.content if hasattr(response, 'content') else str(response),
                        "status": "success"
                    }
                    
                except Exception as e:
                    logger.error(f"Model coordination error for model {i}: {e}")
                    results[f"model_{i}"] = {
                        "model": model.get_model_info() if hasattr(model, 'get_model_info') else str(model),
                        "error": str(e),
                        "status": "error"
                    }
            
            # Combine results
            combined_result = {
                "task": task,
                "models_coordinated": len(models),
                "successful_responses": len([r for r in model_responses if r]),
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.debug(f"Coordinated {len(models)} models for task: {task}")
            return combined_result
            
        except Exception as e:
            logger.error(f"Model coordination failed: {e}")
            raise GenAIError(f"Model coordination failed: {e}")
    
    def _process_model_request(self, model: GenAIModel, task: str, 
                             parameters: Dict[str, Any]) -> Any:
        """
        Process request for a specific model.
        
        Args:
            model (GenAIModel): Model to process request for
            task (str): Task to perform
            parameters (dict): Task parameters
            
        Returns:
            Any: Model response
        """
        # Simulate different tasks
        task_lower = task.lower()
        
        if "generate" in task_lower:
            prompt = parameters.get("prompt", "Generate content")
            return model.generate_text(prompt)
        elif "chat" in task_lower:
            messages = parameters.get("messages", [{"role": "user", "content": "Hello"}])
            return model.chat(messages)
        elif "embed" in task_lower:
            texts = parameters.get("texts", ["Sample text"])
            # Check if model supports embeddings
            model_info = model.get_model_info()
            if "embedding" in str(model_info.get("capabilities", [])):
                return model.generate_embeddings(texts)
            else:
                # Simulate embedding generation
                import numpy as np
                embeddings = [np.random.randn(768).tolist() for _ in texts]
                return {"embeddings": embeddings}
        else:
            # Default processing
            return f"Processed task '{task}' with model {model.get_model_info().get('model_name', 'unknown')}"
    
    def get_mcp_info(self) -> Dict[str, Any]:
        """
        Get MCP interface information.
        
        Returns:
            dict: MCP information
        """
        return {
            "node_id": self._node_id,
            "initialized": self._is_initialized,
            "connected_nodes": list(self._connected_nodes.keys()),
            "message_handlers": list(self._message_handlers.keys()),
            "request_handlers": list(self._request_handlers.keys()),
            "pending_requests": len(self._pending_requests),
            "mcp_info": self._mcp_info
        }
    
    def get_pending_requests(self) -> Dict[str, Any]:
        """
        Get pending requests.
        
        Returns:
            dict: Pending requests information
        """
        return {
            request_id: {
                "recipient": req_info["request"].recipient,
                "action": req_info["request"].action,
                "timestamp": req_info["timestamp"].isoformat()
            } for request_id, req_info in self._pending_requests.items()
        }
    
    def add_connected_node(self, node_id: str, node_info: Dict[str, Any]) -> bool:
        """
        Add connected node.
        
        Args:
            node_id (str): Node identifier
            node_info (dict): Node information
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        try:
            self._connected_nodes[node_id] = node_info
            logger.debug(f"Connected node added: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add connected node: {e}")
            return False
    
    def remove_connected_node(self, node_id: str) -> bool:
        """
        Remove connected node.
        
        Args:
            node_id (str): Node identifier
            
        Returns:
            bool: True if removed successfully, False otherwise
        """
        try:
            if node_id in self._connected_nodes:
                del self._connected_nodes[node_id]
                logger.debug(f"Connected node removed: {node_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to remove connected node: {e}")
            return False

# Utility functions for MCP
def create_mcp_interface(node_id: str, config: Optional[Dict] = None) -> MCPInterface:
    """
    Create MCP interface.
    
    Args:
        node_id (str): Unique identifier for this MCP node
        config (dict, optional): MCP configuration
        
    Returns:
        MCPInterface: Created MCP interface
    """
    return MCPInterface(node_id, config)

def coordinate_multiple_models(mcp: MCPInterface, models: List[GenAIModel], 
                            task: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Coordinate multiple models using MCP.
    
    Args:
        mcp (MCPInterface): MCP interface
        models (list): List of GenAI models to coordinate
        task (str): Task to coordinate
        parameters (dict): Task parameters
        
    Returns:
        dict: Coordination results
    """
    return mcp.coordinate_models(models, task, parameters)

# Example usage
def example_mcp():
    """Example of MCP usage."""
    # Create MCP interface
    mcp = create_mcp_interface(
        node_id="node_001",
        config={"network": "katya-mcp-network", "version": "1.0"}
    )
    
    # Register message handler
    def message_handler(msg: MCPMessage):
        print(f"Received message: {msg.content} from {msg.sender}")
    
    mcp.register_message_handler("test", message_handler)
    
    # Register request handler
    def request_handler(req: MCPRequest) -> MCPResponse:
        return MCPResponse(
            request_id=req.request_id,
            recipient=req.sender,
            sender="node_001",
            status="success",
            result=f"Processed request: {req.action}",
            timestamp=datetime.now()
        )
    
    mcp.register_request_handler("test_action", request_handler)
    
    # Send message
    message_id = mcp.send_message(
        recipient="node_002",
        content="Hello from node_001",
        message_type="test"
    )
    print(f"Message sent with ID: {message_id}")
    
    # Send request
    response = mcp.send_request(
        recipient="node_002",
        action="test_action",
        parameters={"data": "test_data"}
    )
    print(f"Request response: {response.status} - {response.result}")
    
    # Get MCP info
    mcp_info = mcp.get_mcp_info()
    print(f"MCP info: {mcp_info}")
    
    # Add connected nodes
    mcp.add_connected_node("node_002", {"type": "compute", "status": "online"})
    mcp.add_connected_node("node_003", {"type": "storage", "status": "online"})
    
    # Broadcast message
    broadcast_ids = mcp.broadcast_message(
        content="Broadcast message to all nodes",
        message_type="broadcast"
    )
    print(f"Broadcast to {len(broadcast_ids)} nodes")
    
    return mcp

# Advanced MCP coordination example
def advanced_mcp_coordination():
    """Advanced example of MCP coordination with multiple models."""
    # Create MCP interface
    mcp = create_mcp_interface("coordinator_node")
    
    # Simulate multiple models (in a real scenario, these would be actual model instances)
    class DummyModel:
        def __init__(self, name):
            self.name = name
        
        def get_model_info(self):
            return {"model_name": self.name, "provider": "dummy"}
        
        def generate_text(self, prompt):
            return f"[{self.name}] Response to: {prompt}"
        
        def chat(self, messages):
            return f"[{self.name}] Chat response"
    
    models = [
        DummyModel("Model_A"),
        DummyModel("Model_B"),
        DummyModel("Model_C")
    ]
    
    # Coordinate models
    coordination_result = mcp.coordinate_models(
        models=models,
        task="generate_analysis",
        parameters={
            "prompt": "Analyze the impact of quantum computing on AI",
            "max_tokens": 500,
            "temperature": 0.7
        }
    )
    
    print("Coordination Result:")
    print(json.dumps(coordination_result, indent=2, default=str))
    
    return mcp, coordination_result