"""
Serialization Utilities
=======================

Data serialization and deserialization.
"""

from typing import Any, Dict, Optional, Type, Union
from dataclasses import dataclass, asdict, is_dataclass
import json
import pickle
import base64
import gzip
import numpy as np
import logging

logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder with numpy support."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__numpy__": True,
                "dtype": str(obj.dtype),
                "shape": obj.shape,
                "data": base64.b64encode(obj.tobytes()).decode()
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, bytes):
            return {
                "__bytes__": True,
                "data": base64.b64encode(obj).decode()
            }
        if is_dataclass(obj):
            return {
                "__dataclass__": obj.__class__.__name__,
                "data": asdict(obj)
            }
        return super().default(obj)


def numpy_decoder(obj: Dict) -> Any:
    """JSON decoder hook for numpy arrays."""
    if "__numpy__" in obj:
        data = base64.b64decode(obj["data"])
        arr = np.frombuffer(data, dtype=obj["dtype"])
        return arr.reshape(obj["shape"])
    if "__bytes__" in obj:
        return base64.b64decode(obj["data"])
    return obj


def serialize(data: Any, format: str = "json",
              compress: bool = False) -> bytes:
    """
    Serialize data to bytes.
    
    Args:
        data: Data to serialize
        format: Serialization format (json, pickle, msgpack)
        compress: Whether to compress
        
    Returns:
        Serialized bytes
    """
    if format == "json":
        serialized = json.dumps(data, cls=NumpyEncoder).encode()
    elif format == "pickle":
        serialized = pickle.dumps(data)
    elif format == "msgpack":
        try:
            import msgpack
            serialized = msgpack.packb(data, default=_msgpack_default)
        except ImportError:
            logger.warning("msgpack not installed, using pickle")
            serialized = pickle.dumps(data)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    if compress:
        serialized = gzip.compress(serialized)
    
    return serialized


def deserialize(data: bytes, format: str = "json",
                compressed: bool = False) -> Any:
    """
    Deserialize bytes to data.
    
    Args:
        data: Serialized bytes
        format: Serialization format
        compressed: Whether data is compressed
        
    Returns:
        Deserialized data
    """
    if compressed:
        data = gzip.decompress(data)
    
    if format == "json":
        return json.loads(data.decode(), object_hook=numpy_decoder)
    elif format == "pickle":
        return pickle.loads(data)
    elif format == "msgpack":
        try:
            import msgpack
            return msgpack.unpackb(data, raw=False, object_hook=_msgpack_hook)
        except ImportError:
            logger.warning("msgpack not installed, using pickle")
            return pickle.loads(data)
    else:
        raise ValueError(f"Unknown format: {format}")


def _msgpack_default(obj):
    """Default handler for msgpack."""
    if isinstance(obj, np.ndarray):
        return {
            "__numpy__": True,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": obj.tobytes()
        }
    if isinstance(obj, bytes):
        return {"__bytes__": True, "data": obj}
    raise TypeError(f"Unknown type: {type(obj)}")


def _msgpack_hook(obj):
    """Object hook for msgpack."""
    if isinstance(obj, dict):
        if "__numpy__" in obj:
            arr = np.frombuffer(obj["data"], dtype=obj["dtype"])
            return arr.reshape(obj["shape"])
        if "__bytes__" in obj:
            return obj["data"]
    return obj


class ModelSerializer:
    """
    Serializer for ML models and weights.
    
    Supports:
    - NumPy arrays
    - PyTorch tensors
    - TensorFlow tensors
    - Model checkpoints
    
    Example:
        >>> serializer = ModelSerializer()
        >>> data = serializer.serialize_weights(weights)
        >>> weights = serializer.deserialize_weights(data)
    """
    
    @staticmethod
    def serialize_weights(weights: Dict[str, np.ndarray],
                          compress: bool = True) -> bytes:
        """Serialize model weights."""
        return serialize(weights, format="pickle", compress=compress)
    
    @staticmethod
    def deserialize_weights(data: bytes,
                            compressed: bool = True) -> Dict[str, np.ndarray]:
        """Deserialize model weights."""
        return deserialize(data, format="pickle", compressed=compressed)
    
    @staticmethod
    def save_checkpoint(path: str, model_state: Dict[str, Any],
                        optimizer_state: Optional[Dict] = None,
                        metadata: Optional[Dict] = None):
        """Save model checkpoint."""
        checkpoint = {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
            "metadata": metadata or {}
        }
        
        data = serialize(checkpoint, format="pickle", compress=True)
        
        with open(path, 'wb') as f:
            f.write(data)
    
    @staticmethod
    def load_checkpoint(path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        with open(path, 'rb') as f:
            data = f.read()
        
        return deserialize(data, format="pickle", compressed=True)


class QuantumStateSerializer:
    """
    Serializer for quantum states and circuits.
    
    Example:
        >>> serializer = QuantumStateSerializer()
        >>> data = serializer.serialize_state(statevector)
    """
    
    @staticmethod
    def serialize_state(state: np.ndarray, metadata: Optional[Dict] = None) -> bytes:
        """Serialize quantum state."""
        payload = {
            "state": state,
            "metadata": metadata or {}
        }
        return serialize(payload, format="pickle", compress=True)
    
    @staticmethod
    def deserialize_state(data: bytes) -> tuple:
        """Deserialize quantum state."""
        payload = deserialize(data, format="pickle", compressed=True)
        return payload["state"], payload.get("metadata", {})
    
    @staticmethod
    def serialize_circuit(circuit: Any) -> bytes:
        """Serialize quantum circuit."""
        try:
            from qiskit.qpy import dump
            from io import BytesIO
            
            buffer = BytesIO()
            dump(circuit, buffer)
            return gzip.compress(buffer.getvalue())
        except ImportError:
            # Fallback to pickle
            return serialize(circuit, format="pickle", compress=True)
    
    @staticmethod
    def deserialize_circuit(data: bytes) -> Any:
        """Deserialize quantum circuit."""
        try:
            from qiskit.qpy import load
            from io import BytesIO
            
            buffer = BytesIO(gzip.decompress(data))
            return load(buffer)[0]
        except ImportError:
            return deserialize(data, format="pickle", compressed=True)
