# AIPlatform SDK CLI Guide

**Command Line Interface for Quantum-AI Platform**

---

## 🚀 **Getting Started**

The AIPlatform CLI provides a powerful command-line interface for accessing all features of the AIPlatform SDK. It allows you to work with quantum computing, computer vision, generative AI, and other advanced features directly from your terminal.

### Installation

```bash
# Install the AIPlatform SDK
pip install aiplatform

# Or install from source
git clone https://github.com/REChain-Network-Solutions/AIPlatform.git
cd AIPlatform
pip install -e .
```

### Basic Usage

```bash
# Show help
aiplatform --help

# Initialize platform
aiplatform init --language en

# Run a quick demo
aiplatform demo run
```

---

## 📋 **CLI Commands Overview**

### Core Commands

| Command | Description | Example |
|---------|-------------|---------|
| `init` | Initialize AIPlatform SDK | `aiplatform init --language ru` |
| `core status` | Show platform status | `aiplatform core status` |
| `core info` | Show platform information | `aiplatform core info` |

### Quantum Computing

| Command | Description | Example |
|---------|-------------|---------|
| `quantum create-circuit` | Create quantum circuit | `aiplatform quantum create-circuit --qubits 4` |
| `quantum apply-gates` | Apply quantum gates | `aiplatform quantum apply-gates --circuit-id qc1 --gates hadamard --targets 0` |

### QIZ Infrastructure

| Command | Description | Example |
|---------|-------------|---------|
| `qiz init` | Initialize QIZ infrastructure | `aiplatform qiz init` |
| `qiz status` | Show QIZ status | `aiplatform qiz status` |

### Federated Learning

| Command | Description | Example |
|---------|-------------|---------|
| `federated create-network` | Create federated network | `aiplatform federated create-network --nodes 5` |

### Computer Vision

| Command | Description | Example |
|---------|-------------|---------|
| `vision detect-objects` | Detect objects in image | `aiplatform vision detect-objects --image photo.jpg` |

### Generative AI

| Command | Description | Example |
|---------|-------------|---------|
| `genai generate-text` | Generate text with AI | `aiplatform genai generate-text --prompt "Explain quantum computing"` |

### Security

| Command | Description | Example |
|---------|-------------|---------|
| `security create-identity` | Create decentralized identity | `aiplatform security create-identity --entity-id user1 --public-key pubkey123` |

### Protocols

| Command | Description | Example |
|---------|-------------|---------|
| `protocols init-qmp` | Initialize Quantum Mesh Protocol | `aiplatform protocols init-qmp --network-id mynetwork` |

### Demos & Tests

| Command | Description | Example |
|---------|-------------|---------|
| `demo run` | Run comprehensive demo | `aiplatform demo run --language zh` |
| `test run` | Run tests | `aiplatform test run --components` |

---

## 🛠️ **Detailed Command Reference**

### Initialization

Initialize the AIPlatform SDK with your preferred language:

```bash
# Initialize with English (default)
aiplatform init

# Initialize with Russian
aiplatform init --language ru

# Initialize with Chinese
aiplatform init --language zh

# Initialize with Arabic
aiplatform init --language ar

# Initialize with configuration file
aiplatform init --config config.json
```

### Core Platform Commands

#### Show Platform Status

```bash
aiplatform core status
```

Output:
```
AIPlatform SDK Status:
  ✓ Core platform: Initialized
  ✓ Quantum computing: Available
  ✓ Computer vision: Available
  ✓ Generative AI: Available
  ✓ Security: Available
  ✓ Protocols: Available
```

#### Show Platform Information

```bash
aiplatform core info
```

Output:
```
AIPlatform SDK Information:
  Version: 1.0.0
  Modules: Quantum, QIZ, Federated, Vision, GenAI, Security, Protocols
  Languages: English, Russian, Chinese, Arabic
  License: Apache 2.0
```

### Quantum Computing Commands

#### Create Quantum Circuit

Create a quantum circuit with specified number of qubits:

```bash
# Create 4-qubit circuit
aiplatform quantum create-circuit --qubits 4

# Create 8-qubit circuit with Russian messages
aiplatform quantum create-circuit --qubits 8 --language ru
```

#### Apply Quantum Gates

Apply quantum gates to an existing circuit:

```bash
# Apply Hadamard gate to qubit 0
aiplatform quantum apply-gates --circuit-id qc1 --gates hadamard --targets 0

# Apply CNOT gate between qubits 0 and 1
aiplatform quantum apply-gates --circuit-id qc1 --gates cnot --targets 0 1

# Apply rotation gates
aiplatform quantum apply-gates --circuit-id qc1 --gates rotation-x rotation-y --targets 2 3
```

### QIZ Infrastructure Commands

#### Initialize QIZ Infrastructure

```bash
# Initialize QIZ infrastructure
aiplatform qiz init

# Check QIZ status
aiplatform qiz status
```

### Federated Learning Commands

#### Create Federated Network

Create a federated learning network with specified number of nodes:

```bash
# Create network with 3 nodes
aiplatform federated create-network --nodes 3

# Create network with 5 nodes in Chinese
aiplatform federated create-network --nodes 5 --language zh
```

### Computer Vision Commands

#### Detect Objects in Image

Detect and identify objects in an image file:

```bash
# Detect objects in image
aiplatform vision detect-objects --image path/to/image.jpg

# Detect objects with Arabic messages
aiplatform vision detect-objects --image photo.png --language ar
```

### Generative AI Commands

#### Generate Text

Generate text using advanced AI models:

```bash
# Generate text with default model
aiplatform genai generate-text --prompt "Explain quantum entanglement"

# Generate text with specific model and parameters
aiplatform genai generate-text \
  --prompt "Write a poem about quantum computing" \
  --model gigachat3-702b \
  --max-length 300 \
  --language ru
```

### Security Commands

#### Create Decentralized Identity

Create a decentralized identity for secure operations:

```bash
# Create identity
aiplatform security create-identity \
  --entity-id user123 \
  --public-key "-----BEGIN PUBLIC KEY-----..."

# Create identity with Chinese messages
aiplatform security create-identity \
  --entity-id 用户1 \
  --public-key "-----BEGIN PUBLIC KEY-----..." \
  --language zh
```

### Protocol Commands

#### Initialize Quantum Mesh Protocol

Set up Quantum Mesh Protocol for advanced networking:

```bash
# Initialize QMP with default network
aiplatform protocols init-qmp

# Initialize QMP with custom network ID
aiplatform protocols init-qmp --network-id quantum_network_001

# Initialize QMP with Arabic messages
aiplatform protocols init-qmp --network-id شبكة_كمومية --language ar
```

### Demo Commands

#### Run Comprehensive Demo

Run a complete demonstration of all AIPlatform capabilities:

```bash
# Run demo in English
aiplatform demo run

# Run demo in Russian
aiplatform demo run --language ru

# Run demo in Chinese
aiplatform demo run --language zh

# Run demo in Arabic
aiplatform demo run --language ar
```

### Test Commands

#### Run Tests

Execute various test suites to validate platform functionality:

```bash
# Run all tests
aiplatform test run

# Run only component tests
aiplatform test run --components

# Run only multilingual tests
aiplatform test run --multilingual

# Run only integration tests
aiplatform test run --integration

# Run only performance tests
aiplatform test run --performance

# Run only example tests
aiplatform test run --examples

# Run tests for specific languages
aiplatform test run --languages en ru

# Run comprehensive tests with all languages
aiplatform test run --languages en ru zh ar
```

---

## 🌍 **Multilingual Support**

The AIPlatform CLI fully supports multilingual operations in four languages:

### Supported Languages

| Language | Code | CLI Support |
|----------|------|------------|
| English | en | ✅ Full |
| Russian | ru | ✅ Full |
| Chinese | zh | ✅ Full |
| Arabic | ar | ✅ Full |

### Language-Specific Examples

#### Russian Commands

```bash
# Инициализация платформы на русском языке
aiplatform init --language ru

# Создание квантовой схемы
aiplatform quantum create-circuit --qubits 4 --language ru

# Генерация текста на русском
aiplatform genai generate-text --prompt "Объясните квантовую запутанность" --language ru
```

#### Chinese Commands

```bash
# 用中文初始化平台
aiplatform init --language zh

# 创建量子电路
aiplatform quantum create-circuit --qubits 4 --language zh

# 用中文生成文本
aiplatform genai generate-text --prompt "解释量子纠缠" --language zh
```

#### Arabic Commands

```bash
# تهيئة النظام باللغة العربية
aiplatform init --language ar

# إنشاء دائرة كمومية
aiplatform quantum create-circuit --qubits 4 --language ar

# توليد نص بالعربية
aiplatform genai generate-text --prompt "اشرح التشابك الكمومي" --language ar
```

---

## ⚡ **Advanced Usage**

### Batch Operations

Run multiple commands in sequence:

```bash
# Create quantum circuit and apply gates
aiplatform quantum create-circuit --qubits 4 && \
aiplatform quantum apply-gates --circuit-id qc1 --gates hadamard --targets 0 && \
aiplatform quantum apply-gates --circuit-id qc1 --gates cnot --targets 0 1
```

### Scripting

Create shell scripts for complex operations:

```bash
#!/bin/bash
# quantum_setup.sh

echo "Setting up quantum computing environment..."
aiplatform init --language en
aiplatform quantum create-circuit --qubits 8
aiplatform quantum apply-gates --circuit-id qc1 --gates hadamard --targets 0
aiplatform quantum apply-gates --circuit-id qc1 --gates cnot --targets 0 1
aiplatform quantum apply-gates --circuit-id qc1 --gates rotation-x --targets 2
echo "Quantum setup completed!"
```

### Environment Variables

Set environment variables for configuration:

```bash
# Set default language
export AIPLATFORM_LANGUAGE=ru

# Set default quantum backend
export AIPLATFORM_QUANTUM_BACKEND=ibm_simulator

# Run CLI with environment variables
aiplatform quantum create-circuit --qubits 4
```

---

## 🧪 **Testing and Validation**

### Run Specific Test Suites

```bash
# Run component tests only
aiplatform test run --components

# Run multilingual tests only
aiplatform test run --multilingual

# Run integration tests only
aiplatform test run --integration

# Run performance tests only
aiplatform test run --performance

# Run example tests only
aiplatform test run --examples
```

### Test with Different Languages

```bash
# Test with English only
aiplatform test run --languages en

# Test with Russian and Chinese
aiplatform test run --languages ru zh

# Test with all supported languages
aiplatform test run --languages en ru zh ar
```

---

## 📊 **Monitoring and Logging**

### Verbose Output

Get detailed information about operations:

```bash
# Run with verbose output
aiplatform --verbose quantum create-circuit --qubits 4
```

### Logging Configuration

Configure logging for debugging:

```bash
# Set log level
export AIPLATFORM_LOG_LEVEL=DEBUG

# Run with logging
aiplatform quantum create-circuit --qubits 4
```

---

## 🚨 **Troubleshooting**

### Common Issues

#### Module Not Found

```bash
# Install missing dependencies
pip install -r requirements.txt

# Or install specific modules
pip install qiskit opencv-python torch
```

#### Language Detection Issues

```bash
# Specify language explicitly
aiplatform init --language en

# Check available languages
aiplatform core info
```

#### Performance Issues

```bash
# Run performance tests
aiplatform test run --performance

# Check system resources
aiplatform core status
```

### Error Handling

The CLI provides detailed error messages:

```bash
# Example error output
Error: Image file not found: non_existent.jpg
Usage: aiplatform vision detect-objects --image PATH
```

---

## 📚 **Additional Resources**

### Documentation

- [Quick Start Guide](quickstart.md)
- [Quantum Integration Guide](quantum_integration.md)
- [Vision Module API](vision_api.md)
- [Federated Training Manual](federated_training.md)
- [Web6 & QIZ Architecture](web6_qiz.md)

### Examples

Check the `examples/` directory for comprehensive usage examples:

```bash
# Run platform demo
python aiplatform/examples/platform_demo.py

# Run integration tests
python aiplatform/examples/integration_test.py
```

### Support

For issues and support:
- GitHub Issues: https://github.com/REChain-Network-Solutions/AIPlatform/issues
- Discord: https://discord.gg/aiplatform
- Email: support@rechain.network

---

## 🏆 **Best Practices**

### 1. Start with Initialization

Always initialize the platform first:

```bash
aiplatform init --language en
```

### 2. Use Language Consistently

Set language once and use consistently:

```bash
aiplatform init --language ru
aiplatform quantum create-circuit --qubits 4  # Will use Russian
```

### 3. Test Regularly

Run tests to ensure everything works:

```bash
aiplatform test run --components
```

### 4. Use Demos for Learning

Run demos to understand capabilities:

```bash
aiplatform demo run --language zh
```

### 5. Check Status Regularly

Monitor platform status:

```bash
aiplatform core status
```

---

*"The future of quantum-AI computing is command-line accessible!"*

**AIPlatform SDK CLI** - Bringing the power of quantum-AI to your terminal, one command at a time.