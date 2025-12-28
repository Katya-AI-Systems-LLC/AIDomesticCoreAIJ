"""
Exception classes for AIPlatform Quantum Infrastructure Zero SDK

This module defines custom exception classes for the SDK,
providing detailed error information for different components.
"""

class AIPlatformError(Exception):
    """Base exception class for AIPlatform SDK."""
    pass

class QuantumError(AIPlatformError):
    """Exception for quantum computing related errors."""
    pass

class QIZError(AIPlatformError):
    """Exception for Quantum Infrastructure Zero related errors."""
    pass

class FederatedError(AIPlatformError):
    """Exception for federated learning related errors."""
    pass

class VisionError(AIPlatformError):
    """Exception for computer vision related errors."""
    pass

class GenAIError(AIPlatformError):
    """Exception for generative AI related errors."""
    pass

class SecurityError(AIPlatformError):
    """Exception for security related errors."""
    pass

class ConfigurationError(AIPlatformError):
    """Exception for configuration related errors."""
    pass

class NetworkError(AIPlatformError):
    """Exception for network related errors."""
    pass

# Specific quantum errors
class QuantumBackendError(QuantumError):
    """Exception for quantum backend connection errors."""
    pass

class QuantumCircuitError(QuantumError):
    """Exception for quantum circuit related errors."""
    pass

class QuantumAlgorithmError(QuantumError):
    """Exception for quantum algorithm related errors."""
    pass

class QuantumCryptoError(QuantumError):
    """Exception for quantum cryptography related errors."""
    pass

# Specific QIZ errors
class QIZNodeError(QIZError):
    """Exception for QIZ node related errors."""
    pass

class QIZRoutingError(QIZError):
    """Exception for QIZ routing related errors."""
    pass

class QIZSecurityError(QIZError):
    """Exception for QIZ security related errors."""
    pass

# Specific federated errors
class FederatedTrainingError(FederatedError):
    """Exception for federated training related errors."""
    pass

class FederatedSecurityError(FederatedError):
    """Exception for federated security related errors."""
    pass

class ModelMarketplaceError(FederatedError):
    """Exception for model marketplace related errors."""
    pass

# Specific vision errors
class ObjectDetectionError(VisionError):
    """Exception for object detection related errors."""
    pass

class FaceRecognitionError(VisionError):
    """Exception for face recognition related errors."""
    pass

class GestureRecognitionError(VisionError):
    """Exception for gesture recognition related errors."""
    pass

# Specific GenAI errors
class TextGenerationError(GenAIError):
    """Exception for text generation related errors."""
    pass

class ImageGenerationError(GenAIError):
    """Exception for image generation related errors."""
    pass

class MultimodalError(GenAIError):
    """Exception for multimodal processing related errors."""
    pass

# Error codes
ERROR_CODES = {
    # Quantum errors
    "QP001": "Quantum backend connection failed",
    "QP002": "Quantum circuit execution timeout",
    "QP003": "Quantum algorithm convergence failed",
    "QP004": "Quantum cryptography validation failed",
    
    # QIZ errors
    "QZ001": "QIZ node discovery failed",
    "QZ002": "Quantum signature verification failed",
    "QZ003": "QMP protocol connection failed",
    "QZ004": "Zero-DNS resolution failed",
    
    # Federated errors
    "FF001": "Federated training synchronization error",
    "FF002": "Model aggregation failed",
    "FF003": "Federated security validation failed",
    "FF004": "Model marketplace transaction failed",
    
    # Vision errors
    "CV001": "Computer vision processing error",
    "CV002": "Object detection failed",
    "CV003": "Face recognition failed",
    "CV004": "Gesture recognition failed",
    
    # GenAI errors
    "GA001": "GenAI model inference failed",
    "GA002": "Text generation failed",
    "GA003": "Image generation failed",
    "GA004": "Multimodal processing failed",
    
    # Security errors
    "SE001": "Authentication failed",
    "SE002": "Authorization failed",
    "SE003": "Encryption failed",
    "SE004": "Signature verification failed",
    
    # Configuration errors
    "CF001": "Configuration file not found",
    "CF002": "Invalid configuration parameter",
    "CF003": "Configuration validation failed",
    
    # Network errors
    "NW001": "Network connection failed",
    "NW002": "Network timeout",
    "NW003": "Network protocol error",
}

def get_error_message(error_code: str) -> str:
    """
    Get error message for error code.
    
    Args:
        error_code (str): Error code
        
    Returns:
        str: Error message
    """
    return ERROR_CODES.get(error_code, "Unknown error")

def raise_error(error_code: str, details: str = "") -> None:
    """
    Raise appropriate exception for error code.
    
    Args:
        error_code (str): Error code
        details (str): Additional error details
    """
    message = get_error_message(error_code)
    if details:
        message += f": {details}"
    
    # Determine exception class based on error code prefix
    if error_code.startswith("QP"):
        raise QuantumError(message)
    elif error_code.startswith("QZ"):
        raise QIZError(message)
    elif error_code.startswith("FF"):
        raise FederatedError(message)
    elif error_code.startswith("CV"):
        raise VisionError(message)
    elif error_code.startswith("GA"):
        raise GenAIError(message)
    elif error_code.startswith("SE"):
        raise SecurityError(message)
    elif error_code.startswith("CF"):
        raise ConfigurationError(message)
    elif error_code.startswith("NW"):
        raise NetworkError(message)
    else:
        raise AIPlatformError(message)