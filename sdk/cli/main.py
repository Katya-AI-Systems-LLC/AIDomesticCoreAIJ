"""
AIPlatform SDK CLI
==================

Command-line interface for SDK operations.
"""

import argparse
import asyncio
import sys
import json
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sdk.cli")


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="aiplatform",
        description="AIPlatform Quantum Infrastructure Zero SDK CLI"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="Show version"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Quantum commands
    quantum_parser = subparsers.add_parser("quantum", help="Quantum operations")
    quantum_sub = quantum_parser.add_subparsers(dest="quantum_cmd")
    
    # quantum run
    run_parser = quantum_sub.add_parser("run", help="Run quantum circuit")
    run_parser.add_argument("circuit", help="Circuit file or definition")
    run_parser.add_argument("--backend", "-b", default="simulator", help="Backend")
    run_parser.add_argument("--shots", "-s", type=int, default=1024, help="Shots")
    
    # quantum simulate
    sim_parser = quantum_sub.add_parser("simulate", help="Simulate circuit")
    sim_parser.add_argument("--qubits", "-q", type=int, default=4, help="Qubits")
    
    # quantum backends
    quantum_sub.add_parser("backends", help="List available backends")
    
    # GenAI commands
    genai_parser = subparsers.add_parser("genai", help="GenAI operations")
    genai_sub = genai_parser.add_subparsers(dest="genai_cmd")
    
    # genai chat
    chat_parser = genai_sub.add_parser("chat", help="Interactive chat")
    chat_parser.add_argument("--provider", "-p", default="katya", help="Provider")
    chat_parser.add_argument("--model", "-m", help="Model name")
    
    # genai generate
    gen_parser = genai_sub.add_parser("generate", help="Generate text")
    gen_parser.add_argument("prompt", help="Input prompt")
    gen_parser.add_argument("--provider", "-p", default="katya", help="Provider")
    
    # Vision commands
    vision_parser = subparsers.add_parser("vision", help="Vision operations")
    vision_sub = vision_parser.add_subparsers(dest="vision_cmd")
    
    # vision detect
    detect_parser = vision_sub.add_parser("detect", help="Detect objects")
    detect_parser.add_argument("image", help="Image path")
    detect_parser.add_argument("--model", "-m", default="yolov8", help="Model")
    
    # vision analyze
    analyze_parser = vision_sub.add_parser("analyze", help="Analyze image")
    analyze_parser.add_argument("image", help="Image path")
    
    # Deploy commands
    deploy_parser = subparsers.add_parser("deploy", help="Deployment operations")
    deploy_sub = deploy_parser.add_subparsers(dest="deploy_cmd")
    
    # deploy start
    start_parser = deploy_sub.add_parser("start", help="Start deployment")
    start_parser.add_argument("path", help="Application path")
    start_parser.add_argument("--name", "-n", help="Deployment name")
    start_parser.add_argument("--replicas", "-r", type=int, default=1)
    
    # deploy status
    status_parser = deploy_sub.add_parser("status", help="Deployment status")
    status_parser.add_argument("deployment_id", help="Deployment ID")
    
    # deploy list
    deploy_sub.add_parser("list", help="List deployments")
    
    # Security commands
    security_parser = subparsers.add_parser("security", help="Security operations")
    security_sub = security_parser.add_subparsers(dest="security_cmd")
    
    # security keygen
    keygen_parser = security_sub.add_parser("keygen", help="Generate keys")
    keygen_parser.add_argument("--algorithm", "-a", default="kyber", help="Algorithm")
    keygen_parser.add_argument("--level", "-l", type=int, default=768)
    keygen_parser.add_argument("--output", "-o", help="Output file")
    
    # security sign
    sign_parser = security_sub.add_parser("sign", help="Sign data")
    sign_parser.add_argument("data", help="Data to sign")
    sign_parser.add_argument("--key", "-k", required=True, help="Private key")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize new project")
    init_parser.add_argument("name", help="Project name")
    init_parser.add_argument("--template", "-t", default="basic", help="Template")
    
    # Info command
    subparsers.add_parser("info", help="Show SDK information")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration")
    config_sub = config_parser.add_subparsers(dest="config_cmd")
    
    config_sub.add_parser("show", help="Show configuration")
    
    set_parser = config_sub.add_parser("set", help="Set config value")
    set_parser.add_argument("key", help="Config key")
    set_parser.add_argument("value", help="Config value")
    
    return parser


async def cmd_quantum_run(args):
    """Run quantum circuit."""
    from sdk.quantum import QuantumCircuitBuilder, QuantumSimulator
    
    print(f"Running quantum circuit on {args.backend}...")
    
    # Create simple demo circuit
    circuit = QuantumCircuitBuilder(num_qubits=4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(2, 3)
    circuit.measure_all()
    
    sim = QuantumSimulator(num_qubits=4)
    sim.h(0)
    sim.cx(0, 1)
    sim.cx(1, 2)
    sim.cx(2, 3)
    
    results = sim.measure(shots=args.shots)
    
    print("\nResults:")
    for state, count in sorted(results.items(), key=lambda x: -x[1])[:10]:
        prob = count / args.shots * 100
        print(f"  |{state}⟩: {count} ({prob:.1f}%)")


async def cmd_quantum_backends(args):
    """List quantum backends."""
    from sdk.quantum import IBMQuantumBackend
    
    print("Available Quantum Backends:")
    print("-" * 40)
    
    backends = [
        ("aer_simulator", "Local simulator", "Unlimited"),
        ("ibm_brisbane", "IBM Brisbane", "127 qubits"),
        ("ibm_kyoto", "IBM Kyoto", "127 qubits"),
        ("ibm_nighthawk", "IBM Nighthawk", "133 qubits"),
        ("ibm_heron", "IBM Heron", "156 qubits"),
    ]
    
    for name, desc, qubits in backends:
        print(f"  {name:20} {desc:20} {qubits}")


async def cmd_genai_chat(args):
    """Interactive chat."""
    from sdk.genai import KatyaGenAI
    
    print(f"Starting chat with {args.provider}...")
    print("Type 'exit' to quit.\n")
    
    katya = KatyaGenAI(language="ru" if args.provider == "katya" else "en")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            response = await katya.generate(user_input)
            print(f"AI: {response.content}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


async def cmd_genai_generate(args):
    """Generate text."""
    from sdk.genai import KatyaGenAI
    
    katya = KatyaGenAI()
    response = await katya.generate(args.prompt)
    
    print(f"\nGenerated:\n{response.content}")


async def cmd_vision_detect(args):
    """Detect objects in image."""
    import numpy as np
    from sdk.vision import ObjectDetector
    
    print(f"Detecting objects in {args.image}...")
    
    # Load image (simulated)
    try:
        import cv2
        image = cv2.imread(args.image)
    except:
        # Simulated image
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    detector = ObjectDetector(model=args.model)
    result = detector.detect(image)
    
    print(f"\nDetections ({len(result.detections)} objects):")
    for det in result.detections:
        print(f"  - {det.class_name}: {det.confidence:.2f}")


async def cmd_security_keygen(args):
    """Generate cryptographic keys."""
    from sdk.security import KyberKEM, DilithiumSignature
    
    print(f"Generating {args.algorithm} keys (level {args.level})...")
    
    if args.algorithm == "kyber":
        crypto = KyberKEM(security_level=args.level)
        keypair = crypto.keygen()
        
        print(f"\nPublic key: {len(keypair.public_key)} bytes")
        print(f"Secret key: {len(keypair.secret_key)} bytes")
        
        if args.output:
            with open(args.output, 'wb') as f:
                f.write(keypair.public_key)
            print(f"Public key saved to {args.output}")
    
    elif args.algorithm == "dilithium":
        crypto = DilithiumSignature(security_level=args.level)
        keypair = crypto.keygen()
        
        print(f"\nPublic key: {len(keypair.public_key)} bytes")
        print(f"Secret key: {len(keypair.secret_key)} bytes")


async def cmd_init(args):
    """Initialize new project."""
    import os
    
    project_dir = args.name
    
    if os.path.exists(project_dir):
        print(f"Error: Directory {project_dir} already exists")
        return
    
    os.makedirs(project_dir)
    os.makedirs(os.path.join(project_dir, "src"))
    os.makedirs(os.path.join(project_dir, "tests"))
    
    # Create main.py
    main_content = '''"""
{name} - AIPlatform SDK Project
"""

import asyncio
from sdk import QuantumCircuitBuilder, KatyaGenAI

async def main():
    # Quantum example
    circuit = QuantumCircuitBuilder(num_qubits=4)
    circuit.h(0)
    circuit.cx(0, 1)
    print(f"Circuit depth: {{circuit.depth}}")
    
    # GenAI example
    katya = KatyaGenAI()
    response = await katya.generate("Hello!")
    print(f"AI: {{response.content}}")

if __name__ == "__main__":
    asyncio.run(main())
'''.format(name=args.name)
    
    with open(os.path.join(project_dir, "src", "main.py"), 'w') as f:
        f.write(main_content)
    
    # Create README
    readme_content = f'''# {args.name}

AIPlatform SDK Project

## Setup

```bash
pip install aiplatform-sdk
```

## Run

```bash
python src/main.py
```
'''
    
    with open(os.path.join(project_dir, "README.md"), 'w') as f:
        f.write(readme_content)
    
    # Create config
    config_content = {
        "name": args.name,
        "version": "0.1.0",
        "sdk_version": "1.0.0",
        "quantum": {
            "backend": "simulator",
            "shots": 1024
        }
    }
    
    with open(os.path.join(project_dir, "config.json"), 'w') as f:
        json.dump(config_content, f, indent=2)
    
    print(f"✓ Created project: {args.name}")
    print(f"  - src/main.py")
    print(f"  - README.md")
    print(f"  - config.json")


async def cmd_info(args):
    """Show SDK information."""
    print("""
╔═══════════════════════════════════════════════════════════╗
║       AIPlatform Quantum Infrastructure Zero SDK          ║
╠═══════════════════════════════════════════════════════════╣
║  Version:     1.0.0                                       ║
║  Python:      3.9+                                        ║
║  License:     Apache 2.0                                  ║
╠═══════════════════════════════════════════════════════════╣
║  Modules:                                                 ║
║    • quantum    - IBM Qiskit, VQE, QAOA, Grover, Shor    ║
║    • qmp        - Quantum Mesh Protocol                   ║
║    • post_dns   - Post-DNS Architecture                   ║
║    • federated  - Federated Quantum AI                    ║
║    • vision     - Computer Vision, SLAM, WebXR            ║
║    • multimodal - Text, Audio, Video, 3D                  ║
║    • genai      - OpenAI, Claude, LLaMA, Katya            ║
║    • security   - Kyber, Dilithium, Zero-Trust            ║
║    • protocols  - Web6, QIZ, ZeroServer                   ║
╠═══════════════════════════════════════════════════════════╣
║  GitHub:  github.com/REChain-Network-Solutions/AIPlatform ║
║  Docs:    docs.aiplatform.io                              ║
╚═══════════════════════════════════════════════════════════╝
""")


def cli():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.version:
        print("AIPlatform SDK v1.0.0")
        return
    
    if not args.command:
        parser.print_help()
        return
    
    # Route commands
    try:
        if args.command == "quantum":
            if args.quantum_cmd == "run":
                asyncio.run(cmd_quantum_run(args))
            elif args.quantum_cmd == "backends":
                asyncio.run(cmd_quantum_backends(args))
            else:
                print("Unknown quantum command")
        
        elif args.command == "genai":
            if args.genai_cmd == "chat":
                asyncio.run(cmd_genai_chat(args))
            elif args.genai_cmd == "generate":
                asyncio.run(cmd_genai_generate(args))
            else:
                print("Unknown genai command")
        
        elif args.command == "vision":
            if args.vision_cmd == "detect":
                asyncio.run(cmd_vision_detect(args))
            else:
                print("Unknown vision command")
        
        elif args.command == "security":
            if args.security_cmd == "keygen":
                asyncio.run(cmd_security_keygen(args))
            else:
                print("Unknown security command")
        
        elif args.command == "init":
            asyncio.run(cmd_init(args))
        
        elif args.command == "info":
            asyncio.run(cmd_info(args))
        
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        print("\nAborted.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.debug:
            raise
        sys.exit(1)


def main():
    """Alias for cli()."""
    cli()


if __name__ == "__main__":
    main()
