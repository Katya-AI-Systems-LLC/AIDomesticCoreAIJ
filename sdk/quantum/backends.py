"""
IBM Quantum Backends Module
===========================

Support for IBM Quantum hardware backends:
- IBM Nighthawk (133 qubits)
- IBM Heron (156 qubits)
- IBM Eagle (127 qubits)
- IBM Osprey (433 qubits)
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ProcessorType(Enum):
    """IBM Quantum processor types."""
    FALCON = "falcon"
    HUMMINGBIRD = "hummingbird"
    EAGLE = "eagle"
    OSPREY = "osprey"
    HERON = "heron"
    NIGHTHAWK = "nighthawk"


@dataclass
class BackendProperties:
    """Properties of a quantum backend."""
    name: str
    processor_type: ProcessorType
    num_qubits: int
    quantum_volume: int
    clops: int  # Circuit Layer Operations Per Second
    basis_gates: List[str]
    coupling_map: List[List[int]]
    t1_times: List[float] = field(default_factory=list)  # microseconds
    t2_times: List[float] = field(default_factory=list)  # microseconds
    readout_errors: List[float] = field(default_factory=list)
    gate_errors: Dict[str, float] = field(default_factory=dict)
    max_shots: int = 100000
    max_circuits: int = 300


class IBMQuantumBackend:
    """
    Base class for IBM Quantum backends.
    
    Provides common functionality for interacting with IBM Quantum hardware.
    """
    
    def __init__(self, name: str, api_token: Optional[str] = None,
                 language: str = "en"):
        """
        Initialize IBM Quantum backend.
        
        Args:
            name: Backend name
            api_token: IBM Quantum API token
            language: Language for messages
        """
        self.name = name
        self.api_token = api_token
        self.language = language
        self._properties: Optional[BackendProperties] = None
        self._connected = False
        
        logger.info(f"IBM Quantum backend initialized: {name}")
    
    def connect(self) -> bool:
        """Connect to IBM Quantum backend."""
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            
            if self.api_token:
                service = QiskitRuntimeService(token=self.api_token)
            else:
                service = QiskitRuntimeService()
            
            backend = service.backend(self.name)
            self._load_properties(backend)
            self._connected = True
            
            logger.info(f"Connected to {self.name}")
            return True
            
        except ImportError:
            logger.warning("qiskit_ibm_runtime not installed, using simulated properties")
            self._load_simulated_properties()
            self._connected = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to {self.name}: {e}")
            return False
    
    def _load_properties(self, backend: Any):
        """Load properties from real backend."""
        config = backend.configuration()
        props = backend.properties()
        
        self._properties = BackendProperties(
            name=self.name,
            processor_type=self._get_processor_type(),
            num_qubits=config.n_qubits,
            quantum_volume=getattr(config, 'quantum_volume', 0),
            clops=getattr(config, 'clops', 0),
            basis_gates=config.basis_gates,
            coupling_map=config.coupling_map,
            max_shots=config.max_shots,
            max_circuits=getattr(config, 'max_experiments', 300)
        )
        
        if props:
            self._properties.t1_times = [props.t1(i) for i in range(config.n_qubits)]
            self._properties.t2_times = [props.t2(i) for i in range(config.n_qubits)]
            self._properties.readout_errors = [
                props.readout_error(i) for i in range(config.n_qubits)
            ]
    
    def _load_simulated_properties(self):
        """Load simulated properties for offline mode."""
        self._properties = self._get_default_properties()
    
    def _get_processor_type(self) -> ProcessorType:
        """Determine processor type from backend name."""
        name_lower = self.name.lower()
        if "nighthawk" in name_lower:
            return ProcessorType.NIGHTHAWK
        elif "heron" in name_lower:
            return ProcessorType.HERON
        elif "eagle" in name_lower:
            return ProcessorType.EAGLE
        elif "osprey" in name_lower:
            return ProcessorType.OSPREY
        else:
            return ProcessorType.FALCON
    
    def _get_default_properties(self) -> BackendProperties:
        """Get default properties for backend type."""
        raise NotImplementedError("Subclasses must implement _get_default_properties")
    
    @property
    def properties(self) -> Optional[BackendProperties]:
        """Get backend properties."""
        return self._properties
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to backend."""
        return self._connected
    
    def get_coupling_map(self) -> List[List[int]]:
        """Get qubit coupling map."""
        if self._properties:
            return self._properties.coupling_map
        return []
    
    def get_basis_gates(self) -> List[str]:
        """Get supported basis gates."""
        if self._properties:
            return self._properties.basis_gates
        return ["cx", "id", "rz", "sx", "x"]
    
    def estimate_execution_time(self, circuit_depth: int, 
                                 num_circuits: int = 1,
                                 shots: int = 1024) -> float:
        """
        Estimate execution time in seconds.
        
        Args:
            circuit_depth: Depth of circuit
            num_circuits: Number of circuits
            shots: Number of shots per circuit
            
        Returns:
            Estimated time in seconds
        """
        if self._properties and self._properties.clops > 0:
            layers = circuit_depth * num_circuits
            return layers / self._properties.clops * shots
        
        # Default estimation
        return circuit_depth * 0.001 * num_circuits + shots * 0.0001


class NighthawkBackend(IBMQuantumBackend):
    """
    IBM Nighthawk quantum processor backend.
    
    Nighthawk is IBM's latest quantum processor with 133 qubits
    and improved error rates.
    """
    
    def __init__(self, api_token: Optional[str] = None, language: str = "en"):
        super().__init__("ibm_nighthawk", api_token, language)
    
    def _get_default_properties(self) -> BackendProperties:
        """Get default Nighthawk properties."""
        import numpy as np
        
        num_qubits = 133
        
        # Generate heavy-hex coupling map
        coupling_map = self._generate_heavy_hex_coupling(num_qubits)
        
        return BackendProperties(
            name="ibm_nighthawk",
            processor_type=ProcessorType.NIGHTHAWK,
            num_qubits=num_qubits,
            quantum_volume=256,
            clops=15000,
            basis_gates=["cx", "id", "rz", "sx", "x", "ecr"],
            coupling_map=coupling_map,
            t1_times=[150.0] * num_qubits,  # microseconds
            t2_times=[100.0] * num_qubits,
            readout_errors=[0.01] * num_qubits,
            gate_errors={"cx": 0.005, "sx": 0.0001, "x": 0.0001},
            max_shots=100000,
            max_circuits=300
        )
    
    def _generate_heavy_hex_coupling(self, num_qubits: int) -> List[List[int]]:
        """Generate heavy-hex lattice coupling map."""
        coupling = []
        
        # Simplified heavy-hex pattern
        for i in range(num_qubits - 1):
            if i % 4 != 3:  # Skip some connections for heavy-hex
                coupling.append([i, i + 1])
        
        # Add some cross-connections
        for i in range(0, num_qubits - 10, 10):
            coupling.append([i, i + 10])
        
        return coupling


class HeronBackend(IBMQuantumBackend):
    """
    IBM Heron quantum processor backend.
    
    Heron is IBM's advanced quantum processor with 156 qubits
    and tunable couplers for improved two-qubit gates.
    """
    
    def __init__(self, api_token: Optional[str] = None, language: str = "en"):
        super().__init__("ibm_heron", api_token, language)
    
    def _get_default_properties(self) -> BackendProperties:
        """Get default Heron properties."""
        num_qubits = 156
        
        coupling_map = self._generate_heron_coupling(num_qubits)
        
        return BackendProperties(
            name="ibm_heron",
            processor_type=ProcessorType.HERON,
            num_qubits=num_qubits,
            quantum_volume=512,
            clops=20000,
            basis_gates=["cx", "cz", "id", "rz", "sx", "x", "ecr"],
            coupling_map=coupling_map,
            t1_times=[200.0] * num_qubits,
            t2_times=[150.0] * num_qubits,
            readout_errors=[0.008] * num_qubits,
            gate_errors={"cx": 0.003, "cz": 0.003, "sx": 0.00005, "x": 0.00005},
            max_shots=100000,
            max_circuits=300
        )
    
    def _generate_heron_coupling(self, num_qubits: int) -> List[List[int]]:
        """Generate Heron coupling map with tunable couplers."""
        coupling = []
        
        # Grid-like pattern with tunable couplers
        cols = 12
        rows = num_qubits // cols
        
        for row in range(rows):
            for col in range(cols):
                qubit = row * cols + col
                
                # Horizontal connection
                if col < cols - 1:
                    coupling.append([qubit, qubit + 1])
                
                # Vertical connection
                if row < rows - 1:
                    coupling.append([qubit, qubit + cols])
        
        return coupling


class EagleBackend(IBMQuantumBackend):
    """IBM Eagle quantum processor backend (127 qubits)."""
    
    def __init__(self, api_token: Optional[str] = None, language: str = "en"):
        super().__init__("ibm_eagle", api_token, language)
    
    def _get_default_properties(self) -> BackendProperties:
        num_qubits = 127
        
        return BackendProperties(
            name="ibm_eagle",
            processor_type=ProcessorType.EAGLE,
            num_qubits=num_qubits,
            quantum_volume=128,
            clops=10000,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=self._generate_eagle_coupling(num_qubits),
            t1_times=[100.0] * num_qubits,
            t2_times=[80.0] * num_qubits,
            readout_errors=[0.015] * num_qubits,
            gate_errors={"cx": 0.01, "sx": 0.0002},
            max_shots=100000,
            max_circuits=300
        )
    
    def _generate_eagle_coupling(self, num_qubits: int) -> List[List[int]]:
        coupling = []
        for i in range(num_qubits - 1):
            if i % 5 != 4:
                coupling.append([i, i + 1])
        return coupling


class OspreyBackend(IBMQuantumBackend):
    """IBM Osprey quantum processor backend (433 qubits)."""
    
    def __init__(self, api_token: Optional[str] = None, language: str = "en"):
        super().__init__("ibm_osprey", api_token, language)
    
    def _get_default_properties(self) -> BackendProperties:
        num_qubits = 433
        
        return BackendProperties(
            name="ibm_osprey",
            processor_type=ProcessorType.OSPREY,
            num_qubits=num_qubits,
            quantum_volume=64,
            clops=8000,
            basis_gates=["cx", "id", "rz", "sx", "x"],
            coupling_map=self._generate_osprey_coupling(num_qubits),
            t1_times=[80.0] * num_qubits,
            t2_times=[60.0] * num_qubits,
            readout_errors=[0.02] * num_qubits,
            gate_errors={"cx": 0.015, "sx": 0.0003},
            max_shots=100000,
            max_circuits=300
        )
    
    def _generate_osprey_coupling(self, num_qubits: int) -> List[List[int]]:
        coupling = []
        cols = 20
        rows = num_qubits // cols
        
        for row in range(rows):
            for col in range(cols):
                qubit = row * cols + col
                if col < cols - 1:
                    coupling.append([qubit, qubit + 1])
                if row < rows - 1:
                    coupling.append([qubit, qubit + cols])
        
        return coupling


def get_backend(name: str, api_token: Optional[str] = None,
                language: str = "en") -> IBMQuantumBackend:
    """
    Factory function to get appropriate backend.
    
    Args:
        name: Backend name
        api_token: IBM Quantum API token
        language: Language for messages
        
    Returns:
        Appropriate backend instance
    """
    name_lower = name.lower()
    
    if "nighthawk" in name_lower:
        return NighthawkBackend(api_token, language)
    elif "heron" in name_lower:
        return HeronBackend(api_token, language)
    elif "eagle" in name_lower:
        return EagleBackend(api_token, language)
    elif "osprey" in name_lower:
        return OspreyBackend(api_token, language)
    else:
        return IBMQuantumBackend(name, api_token, language)


def list_available_backends() -> List[Dict[str, Any]]:
    """List all available IBM Quantum backends."""
    backends = [
        {
            "name": "ibm_nighthawk",
            "qubits": 133,
            "processor": "Nighthawk",
            "status": "available"
        },
        {
            "name": "ibm_heron",
            "qubits": 156,
            "processor": "Heron",
            "status": "available"
        },
        {
            "name": "ibm_eagle",
            "qubits": 127,
            "processor": "Eagle",
            "status": "available"
        },
        {
            "name": "ibm_osprey",
            "qubits": 433,
            "processor": "Osprey",
            "status": "available"
        }
    ]
    
    return backends
