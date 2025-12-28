"""
Blockchain Module
=================

Multi-chain blockchain integration for AIPlatform SDK.

Supports:
- Ethereum & EVM chains
- Polkadot & Substrate
- Solana
- Cardano
- Avalanche
- BSC (Binance Smart Chain)

Features:
- Quantum-safe atomic swaps
- Cross-chain bridges
- Multi-chain DeFi
- DAO governance
- NFT operations
"""

from .ethereum import EthereumClient
from .solana import SolanaClient
from .polkadot import PolkadotClient
from .multichain import MultiChainBridge
from .defi import DeFiProtocol
from .nft import NFTManager
from .dao import DAOGovernance
from .quantum_safe import QuantumSafeSwap

__all__ = [
    "EthereumClient",
    "SolanaClient",
    "PolkadotClient",
    "MultiChainBridge",
    "DeFiProtocol",
    "NFTManager",
    "DAOGovernance",
    "QuantumSafeSwap"
]
