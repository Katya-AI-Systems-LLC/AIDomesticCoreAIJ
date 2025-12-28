"""
Core module for AIPlatform Quantum Infrastructure Zero SDK

This module provides the main entry point for the AIPlatform SDK with internationalization support.
"""

from typing import Dict, Any, Optional, List
import logging

# Import i18n components
from .i18n import translate, detect_language, get_translator
from .i18n import LanguageDetector
from .i18n.vocabulary_manager import get_vocabulary_manager

# Import core components
from .exceptions import AIPlatformError

# Set up logging
logger = logging.getLogger(__name__)


class AIPlatform:
    """Main entry point for the AIPlatform SDK."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize AIPlatform.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.initialized = False
        self.language = self.config.get('language', 'en')
        self.supported_languages = ['en', 'ru', 'zh', 'ar']
        
        # Initialize i18n components
        self.translator = get_translator()
        self.language_detector = LanguageDetector()
        self.vocabulary_manager = get_vocabulary_manager()
        
        platform_term = self.vocabulary_manager.translate_term('AI Platform', 'core', self.language)
        logger.info(translate('initialize_platform', self.language) or f"Initializing {platform_term}")
    
    def initialize(self) -> bool:
        """
        Initialize the platform.
        
        Returns:
            bool: Initialization success
        """
        try:
            # Initialize all components
            self.initialized = True
            logger.info(translate('success', self.language) or "AI Platform initialized successfully")
            return True
        except Exception as e:
            logger.error(translate('error_occurred', self.language, error=str(e)) or f"Error occurred: {str(e)}")
            return False
    
    def run_demo(self) -> Dict[str, Any]:
        """
        Run a quick demonstration.
        
        Returns:
            dict: Demo results
        """
        if not self.initialized:
            self.initialize()
        
        logger.info(translate('run_demo', self.language) or "Running quick demonstration")
        
        # Simulate demo operations
        demo_results = {
            'status': 'success',
            'modules_tested': [
                translate('quantum_computing', self.language),
                translate('artificial_intelligence', self.language),
                translate('federated_learning', self.language),
                translate('computer_vision', self.language)
            ],
            'summary': translate('welcome', self.language),
            'language': self.language,
            'timestamp': '2025-01-01T00:00:00Z'
        }
        logger.info(translate('demo_completed', self.language) or f"Demo completed: {demo_results['summary']}")
        return demo_results
    

# Quantum Computing Methods
def create_quantum_circuit(self, qubits: int) -> Any:
    """
    Create quantum circuit with current language support.
    
    Args:
        qubits: Number of qubits
        
    Returns:
        QuantumCircuit: Created quantum circuit
    """
    from .quantum import QuantumCircuit
    circuit = QuantumCircuit(qubits, language=self.language)
    logger.info(translate('quantum_circuit_created', self.language) or "Quantum circuit created")
    return circuit

def create_vqe_solver(self, hamiltonian: Any) -> Any:
    """
    Create VQE solver with current language support.
    
    Args:
        hamiltonian: Hamiltonian matrix
        
    Returns:
        VQE: Created VQE solver
    """
    from .quantum import VQE
    vqe = VQE(hamiltonian, language=self.language)
    logger.info(translate('vqe_solver_created', self.language) or "VQE solver created")
    return vqe

def create_qaoa_solver(self, problem_graph: List[tuple], max_depth: int) -> Any:
    """
    Create QAOA solver with current language support.
    
    Args:
        problem_graph: Problem graph as list of edges
        max_depth: Maximum circuit depth
        
    Returns:
        QAOA: Created QAOA solver
    """
    from .quantum import QAOA
    qaoa = QAOA(problem_graph, max_depth, language=self.language)
    logger.info(translate('qaoa_solver_created', self.language) or "QAOA solver created")
    return qaoa

# QIZ Methods
def create_zero_dns(self) -> Any:
    """
    Create Zero-DNS system with current language support.
    
    Returns:
        ZeroDNS: Created Zero-DNS system
    """
    from .qiz import ZeroDNS
    dns = ZeroDNS(language=self.language)
    logger.info(translate('zero_dns_created', self.language) or "Zero-DNS system created")
    return dns

def create_qmp_node(self, node_id: str) -> Any:
    """
    Create QMP node with current language support.
    
    Args:
        node_id: Node identifier
        
    Returns:
        QuantumMeshProtocol: Created QMP node
    """
    from .qiz import QuantumMeshProtocol
    qmp = QuantumMeshProtocol(node_id, language=self.language)
    logger.info(translate('qmp_node_created', self.language) or "QMP node created")
    return qmp

# Federated Learning Methods
def create_federated_node(self, node_id: str, model: Any) -> Any:
    """
    Create federated node with current language support.
    
    Args:
        node_id: Node identifier
        model: Local model
        
    Returns:
        FederatedNode: Created federated node
    """
    from .federated import FederatedNode
    node = FederatedNode(node_id, model, language=self.language)
    logger.info(translate('federated_node_created', self.language) or "Federated node created")
    return node

def create_federated_coordinator(self) -> Any:
    """
    Create federated coordinator with current language support.
    
    Returns:
        FederatedCoordinator: Created federated coordinator
    """
    from .federated import FederatedCoordinator
    coordinator = FederatedCoordinator(language=self.language)
    logger.info(translate('federated_coordinator_created', self.language) or "Federated coordinator created")
    return coordinator

# Vision Methods
def create_object_detector(self, model_type: str = 'yolo') -> Any:
    """
    Create object detector with current language support.
    
    Args:
        model_type: Type of detection model
        
    Returns:
        ObjectDetector: Created object detector
    """
    from .vision import ObjectDetector
    detector = ObjectDetector(model_type, language=self.language)
    logger.info(translate('object_detector_created', self.language) or "Object detector created")
    return detector

def create_face_recognizer(self) -> Any:
    """
    Create face recognizer with current language support.
    
    Returns:
        FaceRecognizer: Created face recognizer
    """
    from .vision import FaceRecognizer
    recognizer = FaceRecognizer(language=self.language)
    logger.info(translate('face_recognizer_created', self.language) or "Face recognizer created")
    return recognizer

# GenAI Methods
def create_genai_model(self, model_name: str, api_key: Optional[str] = None) -> Any:
    """
    Create GenAI model with current language support.
    
    Args:
        model_name: Name of the model
        api_key: API key for cloud models
        
    Returns:
        GenAIModel: Created GenAI model
    """
    from .genai import GenAIModel
    model = GenAIModel(model_name, api_key, language=self.language)
    logger.info(translate('genai_model_created', self.language) or "GenAI model created")
    return model

# Security Methods
def create_didn(self) -> Any:
    """
    Create DIDN with current language support.
    
    Returns:
        DIDN: Created DIDN
    """
    from .security import DIDN
    didn = DIDN(language=self.language)
    logger.info(translate('didn_created', self.language) or "DIDN created")
    return didn

# Protocols Methods
def create_post_dns_protocol(self) -> Any:
    """
    Create Post-DNS protocol with current language support.
    
    Returns:
        PostDNSProtocol: Created Post-DNS protocol
    """
    from .protocols import PostDNSProtocol
    protocol = PostDNSProtocol(language=self.language)
    logger.info(translate('post_dns_protocol_created', self.language) or "Post-DNS protocol created")
    return protocol

    def get_version(self) -> str:
        """
        Get platform version.
        
        Returns:
            str: Version string
        """
        return "1.0.0"
    
    def get_capabilities(self) -> List[str]:
        """
        Get platform capabilities.
        
        Returns:
            list: List of capabilities
        """
        capabilities = [
            'quantum_computing',
            'computer_vision',
            'federated_learning',
            'genai_integration',
            'zero_infrastructure',
            'multilingual_support'
        ]
        
        # Translate capabilities if needed
        translated_capabilities = []
        for capability in capabilities:
            translated = translate(capability, self.language)
            translated_capabilities.append(translated if translated != capability else capability)
        
        return translated_capabilities
    
    def set_language(self, language: str) -> bool:
        """
        Set the platform language.
        
        Args:
            language (str): Language code
            
        Returns:
            bool: Success status
        """
        if language in self.supported_languages:
            self.language = language
            logger.info(translate('language_set', self.language) or f"Language set to: {language}")
            return True
        else:
            logger.warning(translate('unsupported_language', self.language) or f"Unsupported language: {language}")
            return False
    
    def detect_and_set_language(self, text: str) -> str:
        """
        Detect language from text and set it as the platform language.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            str: Detected language code
        """
        detected_language = self.language_detector.detect_language(text)
        self.set_language(detected_language)
        return detected_language
    
    def translate_text(self, text: str, target_language: str = None) -> str:
        """
        Translate text to the specified language.
        
        Args:
            text (str): Text to translate
            target_language (str, optional): Target language code
            
        Returns:
            str: Translated text
        """
        if target_language is None:
            target_language = self.language
        
        return self.translator.translate(text, target_language)
    
    def get_technical_term(self, term: str, domain: str, target_language: str = None) -> str:
        """
        Get translation of a technical term.
        
        Args:
            term (str): Technical term
            domain (str): Domain of the term
            target_language (str, optional): Target language code
            
        Returns:
            str: Translated technical term
        """
        if target_language is None:
            target_language = self.language
        
        return self.vocabulary_manager.translate_term(term, domain, target_language)
    
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported languages.
        
        Returns:
            list: List of supported language codes
        """
        return self.supported_languages.copy()
    
    def get_localized_documentation(self, doc_name: str) -> Dict[str, Any]:
        """
        Get localized documentation.
        
        Args:
            doc_name (str): Documentation name
            
        Returns:
            dict: Localized documentation
        """
        from .i18n.resource_manager import get_resource_manager
        resource_manager = get_resource_manager()
        
        try:
            return resource_manager.load_resource('documentation', doc_name, self.language)
        except Exception as e:
            logger.error(translate('error_loading_documentation', self.language, error=str(e)) or f"Error loading documentation: {e}")
            return {
                'title': f'Documentation: {doc_name}',
                'content': translate('documentation_not_available', self.language) or 'Documentation not available',
                'language': self.language
            }


# Convenience function for quick access
def create_platform(config: Optional[Dict] = None) -> AIPlatform:
    """
    Create and initialize an AIPlatform instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        AIPlatform: Initialized platform instance
    """
    platform = AIPlatform(config)
    platform.initialize()
    return platform


# Global instance for easy access
_platform = None


def get_platform() -> AIPlatform:
    """
    Get the global AIPlatform instance.
    
    Returns:
        AIPlatform: Global platform instance
    """
    global _platform
    if _platform is None:
        _platform = create_platform()
    return _platform


# Auto-initialize when module is imported
if __name__ != "__main__":
    # Only auto-initialize when not running as main module
    try:
        _platform = create_platform()
    except Exception:
        # Don't fail if auto-initialization fails
        pass