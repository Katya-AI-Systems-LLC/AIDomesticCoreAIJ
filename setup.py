"""
AIPlatform Quantum Infrastructure Zero SDK - Setup Script

This script allows the AIPlatform SDK to be installed as a Python package.
"""

import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install

# Read the README file for the long description
def read_readme():
    """Read the README file."""
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from requirements.txt
def read_requirements():
    """Read requirements from requirements.txt."""
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return requirements

# Custom installation command to handle post-installation tasks
class PostInstallCommand(install):
    """Custom post-installation for installation mode."""
    
    def run(self):
        # Run the standard installation
        install.run(self)
        
        # Post-installation tasks
        print("AIPlatform SDK installation completed successfully!")
        print("To get started, run: python -c \"from aiplatform.core import AIPlatform; platform = AIPlatform()\"")

# Setup configuration
setup(
    name="aiplatform",
    version="1.0.0",
    author="REChain Network Solutions & Katya AI Systems",
    author_email="info@rechain.network",
    description="AIPlatform Quantum Infrastructure Zero SDK - Enterprise-Grade Quantum-AI Platform",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/REChain-Network-Solutions/AIPlatform",
    project_urls={
        "Bug Tracker": "https://github.com/REChain-Network-Solutions/AIPlatform/issues",
        "Documentation": "https://github.com/REChain-Network-Solutions/AIPlatform/blob/main/README.md",
        "Source Code": "https://github.com/REChain-Network-Solutions/AIPlatform",
    },
    packages=find_packages(exclude=["tests*", "examples*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security :: Cryptography",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Natural Language :: Russian",
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: Arabic",
    ],
    package_data={
        "aiplatform": [
            "i18n/translations/*.json",
            "i18n/vocabularies/*.json",
            "examples/*.py",
            "docs/*",
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "sphinx>=4.0.0",
        ],
        "quantum": [
            "qiskit>=0.44.0",
            "qiskit-aer>=0.12.0",
            "qiskit-ibm-runtime>=0.11.0",
        ],
        "vision": [
            "opencv-python>=4.5.0",
            "Pillow>=8.3.0",
            "scikit-image>=0.18.0",
        ],
        "genai": [
            "transformers>=4.21.0",
            "torch>=1.9.0",
            "diffusers>=0.10.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aiplatform=aiplatform.cli:main",
            "aiplatform-demo=aiplatform.examples.platform_demo:main",
            "aiplatform-test=aiplatform.examples.integration_test:main",
        ],
    },
    cmdclass={
        'install': PostInstallCommand,
    },
    keywords=[
        "quantum computing",
        "artificial intelligence",
        "machine learning",
        "computer vision",
        "generative ai",
        "quantum-safe security",
        "federated learning",
        "zero-infrastructure",
        "multimodal ai",
        "post-quantum cryptography",
        "quantum infrastructure",
        "multilingual ai",
        "russian ai",
        "chinese ai",
        "arabic ai",
    ],
    zip_safe=False,
    platforms="any",
)