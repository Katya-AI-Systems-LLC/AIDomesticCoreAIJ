# AIPlatform SDK API Reference

## Table of Contents

- [Quantum Module](#quantum-module)
- [QMP Module](#qmp-module)
- [Post-DNS Module](#post-dns-module)
- [Federated Module](#federated-module)
- [Vision Module](#vision-module)
- [Multimodal Module](#multimodal-module)
- [GenAI Module](#genai-module)
- [Security Module](#security-module)
- [Protocols Module](#protocols-module)

---

## Quantum Module

### QuantumCircuitBuilder

High-level quantum circuit construction.

```python
from sdk.quantum import QuantumCircuitBuilder

circuit = QuantumCircuitBuilder(num_qubits=4)

# Single-qubit gates
circuit.h(0)        # Hadamard
circuit.x(1)        # Pauli-X
circuit.y(2)        # Pauli-Y
circuit.z(3)        # Pauli-Z
circuit.s(0)        # S gate
circuit.t(1)        # T gate

# Rotation gates
circuit.rx(0, theta=np.pi/4)
circuit.ry(1, theta=np.pi/2)
circuit.rz(2, theta=np.pi)

# Two-qubit gates
circuit.cx(0, 1)    # CNOT
circuit.cz(1, 2)    # CZ
circuit.swap(0, 3)  # SWAP

# Measurement
circuit.measure_all()

# Properties
print(circuit.depth)
print(circuit.num_qubits)
```

### VQESolver

Variational Quantum Eigensolver for molecular simulation.

```python
from sdk.quantum import VQESolver

vqe = VQESolver(num_qubits=4, ansatz="ry_rz")

hamiltonian = {
    "ZZ": [(0, 1, 0.5), (1, 2, 0.3)],
    "X": [(0, 0.2), (1, 0.2)],
    "Z": [(0, -0.5)]
}

result = await vqe.solve(hamiltonian, max_iterations=100)

print(f"Energy: {result.energy}")
print(f"Converged: {result.converged}")
```

### QAOASolver

Quantum Approximate Optimization Algorithm.

```python
from sdk.quantum import QAOASolver

qaoa = QAOASolver(num_qubits=5, p_layers=2)

edges = [(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0)]

result = await qaoa.solve_maxcut(edges)

print(f"Solution: {result.solution}")
print(f"Objective: {result.objective_value}")
```

---

## Vision Module

### ObjectDetector

Object detection using YOLO and other models.

```python
from sdk.vision import ObjectDetector

detector = ObjectDetector(model="yolov8", confidence_threshold=0.5)

result = detector.detect(image)

for detection in result.detections:
    print(f"{detection.class_name}: {detection.confidence:.2f}")
    print(f"  BBox: {detection.bbox}")
```

### FaceRecognizer

Face detection and recognition.

```python
from sdk.vision import FaceRecognizer

recognizer = FaceRecognizer()

# Detect faces
faces = recognizer.detect_faces(image)

# Register face
recognizer.register_face("person_id", face.embedding, {"name": "John"})

# Recognize
matches = recognizer.recognize(face.embedding, threshold=0.6)
```

### SLAMProcessor

Visual SLAM for localization and mapping.

```python
from sdk.vision import SLAMProcessor

slam = SLAMProcessor(method="orb")
slam.initialize(first_frame)

for frame in video_frames:
    pose = slam.process_frame(frame)
    if pose:
        print(f"Position: {pose.position}")

trajectory = slam.get_trajectory()
map_points = slam.get_map_points()
```

---

## Multimodal Module

### MultimodalProcessor

Unified multimodal processing.

```python
from sdk.multimodal import MultimodalProcessor
from sdk.multimodal.processor import ModalityType, ModalityInput

processor = MultimodalProcessor(fusion_method="attention")

result = processor.process([
    ModalityInput(ModalityType.TEXT, "Describe this"),
    ModalityInput(ModalityType.IMAGE, image_array),
    ModalityInput(ModalityType.AUDIO, audio_array)
])

print(f"Fused embedding: {result.fused_embedding.shape}")
print(f"Confidence: {result.confidence}")
```

### GigaChat3Client

GigaChat3-702B multimodal AI.

```python
from sdk.multimodal import GigaChat3Client
from sdk.multimodal.gigachat import GigaChatMessage, GigaChatRole

client = GigaChat3Client()

response = await client.chat([
    GigaChatMessage(GigaChatRole.USER, "Hello!", images=[image])
])

print(response.content)
```

---

## GenAI Module

### UnifiedGenAI

Unified interface for multiple AI providers.

```python
from sdk.genai import UnifiedGenAI, Provider

genai = UnifiedGenAI()

genai.add_provider(Provider.OPENAI, api_key="sk-...")
genai.add_provider(Provider.CLAUDE, api_key="sk-ant-...")
genai.add_provider(Provider.KATYA)

response = await genai.generate("Hello!")

print(f"Provider: {response.provider}")
print(f"Content: {response.content}")
```

### DiffusionModel

Image generation with diffusion models.

```python
from sdk.genai import DiffusionModel

model = DiffusionModel(model="sd-xl")

result = await model.generate(
    "A beautiful sunset",
    width=1024,
    height=1024,
    steps=30
)

image = result.output
```

---

## Security Module

### KyberKEM

Post-quantum key encapsulation.

```python
from sdk.security import KyberKEM

kyber = KyberKEM(security_level=768)

# Key generation
keypair = kyber.keygen()

# Encapsulation
ciphertext = kyber.encapsulate(keypair.public_key)

# Decapsulation
shared_secret = kyber.decapsulate(
    ciphertext.ciphertext, 
    keypair.secret_key
)
```

### ZeroTrustManager

Zero-trust security model.

```python
from sdk.security import ZeroTrustManager
from sdk.security.zero_trust import TrustLevel, SecurityContext

zt = ZeroTrustManager()

zt.register_policy("api/admin", required_trust=TrustLevel.HIGH)

context = SecurityContext(
    identity="user1",
    device_id="device1",
    location="office",
    timestamp=time.time(),
    trust_level=TrustLevel.MEDIUM
)

decision = zt.evaluate_access(context, "api/admin")
```

---

## Protocols Module

### QIZProtocol

Quantum Infrastructure Zero protocol.

```python
from sdk.protocols import QIZProtocol

qiz = QIZProtocol(node_id="node_001")

await qiz.connect("remote_node")
await qiz.send("remote_node", data)

@qiz.on_message(QIZMessageType.DATA)
async def handle_data(message):
    print(f"Received: {message.payload}")
```

### ZeroServer

Zero-infrastructure server.

```python
from sdk.protocols import ZeroServer

server = ZeroServer(port=8080)

@server.route("/api/data", methods=["GET", "POST"])
async def handle_data(ctx):
    return {"status": "ok"}

await server.start()
```

---

## Integrations

### KubernetesDeployer

```python
from sdk.integrations import KubernetesDeployer

k8s = KubernetesDeployer(namespace="production")
k8s.connect()

deployment = k8s.create_deployment(
    name="my-app",
    image="my-image:latest",
    replicas=3
)

k8s.scale_deployment("my-app", replicas=5)
```

### MLflowTracker

```python
from sdk.integrations import MLflowTracker

tracker = MLflowTracker(experiment_name="quantum-training")

with tracker.start_run("run-1"):
    tracker.log_param("learning_rate", 0.01)
    tracker.log_metric("loss", 0.5)
    tracker.log_metric("accuracy", 0.95)
```

---

## Utilities

### Configuration

```python
from sdk.utils import ConfigManager

config = ConfigManager.load("config.yaml")

backend = config.get("quantum.backend")
config.set("debug", True)
```

### Logging

```python
from sdk.utils import setup_logging, get_logger

setup_logging(level="DEBUG", format=LogFormat.COLORED)

logger = get_logger("my_module")
logger.info("Hello!")
```

### Metrics

```python
from sdk.utils import MetricsCollector

metrics = MetricsCollector()

metrics.increment("requests_total")
metrics.gauge("active_connections", 42)

with metrics.timer("operation_duration"):
    do_operation()
```

---

## CLI Usage

```bash
# Show information
aiplatform info

# Initialize project
aiplatform init my-project

# Run quantum circuit
aiplatform quantum run circuit.json --backend simulator

# Interactive chat
aiplatform genai chat --provider katya

# Generate keys
aiplatform security keygen --algorithm kyber --level 768
```
