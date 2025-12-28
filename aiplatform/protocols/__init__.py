"""
Protocols module for AIPlatform SDK

This module provides implementations of advanced networking protocols
including Quantum Mesh Protocol (QMP) and Post-DNS protocols.
"""

from .qmp import QuantumMeshProtocol, QMPNode, QMPMessage
from .postdns import PostDNSProtocol, PostDNSRecord, PostDNSResolver
from ..exceptions import ProtocolError

__all__ = [
    'QuantumMeshProtocol',
    'QMPNode',
    'QMPMessage',
    'PostDNSProtocol',
    'PostDNSRecord',
    'PostDNSResolver',
    'ProtocolError'
]

__version__ = '1.0.0'
__author__ = 'REChain Network Solutions & Katya AI Systems'