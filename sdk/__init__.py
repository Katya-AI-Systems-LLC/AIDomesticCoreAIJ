"""
AIPlatform Quantum Infrastructure Zero SDK
==========================================

Enterprise-Grade Quantum-AI Platform SDK for REChain® Network Solutions,
Katya AI Systems, and IBM Quantum integration.

This SDK provides:
- Quantum computing with IBM Qiskit Runtime
- Quantum Infrastructure Zero (QIZ) networking
- Federated Quantum AI training
- Computer Vision and Big Data processing
- Multimodal AI with GigaChat3-702B
- Quantum-Safe Cryptography
- Post-DNS protocols and QMP
- Multi-chain Blockchain (Ethereum, Solana, Polkadot)
- Metaverse and Spatial Computing
- Web3-Web6 Architecture

Copyright (c) 2025 REChain Network Solutions & Katya AI Systems
Licensed under Apache 2.0
"""

__version__ = "1.5.0"
__author__ = "REChain Network Solutions & Katya AI Systems"
__license__ = "Apache-2.0"

# Core SDK imports
from .quantum import (
    QuantumCircuitBuilder,
    QiskitRuntimeClient,
    VQESolver,
    QAOASolver,
    GroverSearch,
    ShorFactorization,
    QuantumSimulator,
    IBMQuantumBackend,
    AWSBraketClient,
    GoogleQuantumClient
)

from .qmp import (
    QuantumMeshProtocol,
    QMPNode,
    QMPRouter,
    QuantumSignature,
    MeshNetwork
)

from .post_dns import (
    PostDNSResolver,
    ZeroDNSRegistry,
    QuantumAddressing,
    ObjectSignatureRouter
)

from .federated import (
    FederatedCoordinator,
    QuantumFederatedNode,
    HybridTrainer,
    ModelMarketplace,
    NFTWeightManager
)

from .vision import (
    ObjectDetector,
    FaceRecognizer,
    GestureRecognizer,
    VideoStreamProcessor,
    Vision3DEngine,
    SLAMProcessor,
    WebXRIntegration
)

from .multimodal import (
    MultimodalProcessor,
    TextProcessor,
    AudioProcessor,
    VideoProcessor,
    Spatial3DProcessor,
    GigaChat3Client
)

from .genai import (
    OpenAIClient,
    ClaudeClient,
    LLaMAClient,
    KatyaGenAI,
    DiffusionModel,
    UnifiedGenAI
)

from .security import (
    KyberKEM,
    DilithiumSignature,
    DIDNManager,
    ZeroTrustManager,
    SecureKeyManager
)

from .protocols import (
    Web6Protocol,
    QIZProtocol,
    ZeroServer,
    DeployEngine
)

from .blockchain import (
    EthereumClient,
    SolanaClient,
    PolkadotClient,
    MultiChainBridge,
    DeFiProtocol,
    NFTManager,
    DAOGovernance,
    QuantumSafeSwap
)

from .metaverse import (
    SpatialAI,
    WorldEngine,
    AvatarSystem,
    MetaverseEconomy,
    CrossMetaversePortal,
    XRManager
)

from .edge import (
    EdgeRuntime,
    ModelOptimizer,
    EdgeInference,
    FederatedEdge,
    DeviceManager
)

from .analytics import (
    AnalyticsTracker,
    MetricsAggregator,
    AnomalyDetector,
    DashboardAPI,
    ReportGenerator
)

from .robotics import (
    RobotController,
    NavigationSystem,
    SensorFusion,
    ManipulatorArm,
    ROSBridge
)

from .nlp import (
    EmbeddingEngine,
    SemanticSearch,
    NERExtractor,
    SentimentAnalyzer,
    Translator,
    RAGPipeline
)

from .speech import (
    SpeechRecognizer,
    SpeechSynthesizer,
    SpeakerRecognition,
    VoiceActivityDetector
)

from .agents import (
    Agent,
    AgentState,
    Tool,
    ToolRegistry,
    AgentMemory,
    AgentPlanner,
    MultiAgentOrchestrator,
    ReActAgent
)

from .streaming import (
    EventEmitter,
    EventBus,
    WebSocketServer,
    WebSocketClient,
    SSEServer,
    StreamPipeline
)

__all__ = [
    # Quantum
    "QuantumCircuitBuilder",
    "QiskitRuntimeClient", 
    "VQESolver",
    "QAOASolver",
    "GroverSearch",
    "ShorFactorization",
    "QuantumSimulator",
    "IBMQuantumBackend",
    
    # QMP
    "QuantumMeshProtocol",
    "QMPNode",
    "QMPRouter",
    "QuantumSignature",
    "MeshNetwork",
    
    # Post-DNS
    "PostDNSResolver",
    "ZeroDNSRegistry",
    "QuantumAddressing",
    "ObjectSignatureRouter",
    
    # Federated
    "FederatedCoordinator",
    "QuantumFederatedNode",
    "HybridTrainer",
    "ModelMarketplace",
    "NFTWeightManager",
    
    # Vision
    "ObjectDetector",
    "FaceRecognizer",
    "GestureRecognizer",
    "VideoStreamProcessor",
    "Vision3DEngine",
    "SLAMProcessor",
    "WebXRIntegration",
    
    # Multimodal
    "MultimodalProcessor",
    "TextProcessor",
    "AudioProcessor",
    "VideoProcessor",
    "Spatial3DProcessor",
    "GigaChat3Client",
    
    # GenAI
    "OpenAIClient",
    "ClaudeClient",
    "LLaMAClient",
    "KatyaGenAI",
    "DiffusionModel",
    "UnifiedGenAI",
    
    # Security
    "KyberKEM",
    "DilithiumSignature",
    "DIDNManager",
    "ZeroTrustManager",
    "SecureKeyManager",
    
    # Protocols
    "Web6Protocol",
    "QIZProtocol",
    "ZeroServer",
    "DeployEngine",
    
    # Blockchain
    "EthereumClient",
    "SolanaClient",
    "PolkadotClient",
    "MultiChainBridge",
    "DeFiProtocol",
    "NFTManager",
    "DAOGovernance",
    "QuantumSafeSwap",
    
    # Metaverse
    "SpatialAI",
    "WorldEngine",
    "AvatarSystem",
    "MetaverseEconomy",
    "CrossMetaversePortal",
    "XRManager",
    
    # Edge AI
    "EdgeRuntime",
    "ModelOptimizer",
    "EdgeInference",
    "FederatedEdge",
    "DeviceManager",
    
    # Additional Quantum Backends
    "AWSBraketClient",
    "GoogleQuantumClient",
    
    # Analytics
    "AnalyticsTracker",
    "MetricsAggregator",
    "AnomalyDetector",
    "DashboardAPI",
    "ReportGenerator",
    
    # Robotics
    "RobotController",
    "NavigationSystem",
    "SensorFusion",
    "ManipulatorArm",
    "ROSBridge",
    
    # NLP
    "EmbeddingEngine",
    "SemanticSearch",
    "NERExtractor",
    "SentimentAnalyzer",
    "Translator",
    "RAGPipeline",
    
    # Speech
    "SpeechRecognizer",
    "SpeechSynthesizer",
    "SpeakerRecognition",
    "VoiceActivityDetector",
    
    # Agents
    "Agent",
    "AgentState",
    "Tool",
    "ToolRegistry",
    "AgentMemory",
    "AgentPlanner",
    "MultiAgentOrchestrator",
    "ReActAgent",
    
    # Streaming
    "EventEmitter",
    "EventBus",
    "WebSocketServer",
    "WebSocketClient",
    "SSEServer",
    "StreamPipeline"
]
