"""
Config Management Module for AIPlatform

This module provides configuration management for API keys, model settings,
and other application parameters.
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API configuration."""
    openai_key: str = ""
    claude_key: str = ""
    llama_key: str = ""
    gigachat3_key: str = ""
    katya_key: str = ""


@dataclass
class ModelConfig:
    """Model configuration."""
    default_text_model: str = "gigachat3-702b"
    default_embedding_model: str = "text-embedding-ada-002"
    openai_models: list = None
    claude_models: list = None
    llama_models: list = None
    gigachat3_models: list = None
    katya_models: list = None
    
    def __post_init__(self):
        if self.openai_models is None:
            self.openai_models = ["gpt-4", "gpt-3.5-turbo"]
        if self.claude_models is None:
            self.claude_models = ["claude-3-opus", "claude-3-sonnet"]
        if self.llama_models is None:
            self.llama_models = ["llama-2-70b", "llama-2-13b"]
        if self.gigachat3_models is None:
            self.gigachat3_models = ["gigachat3-702b"]
        if self.katya_models is None:
            self.katya_models = ["katya-v1"]


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/aiplatform.log"


@dataclass
class LanguageConfig:
    """Language configuration."""
    default: str = "en"
    supported: list = None
    
    def __post_init__(self):
        if self.supported is None:
            self.supported = ["en", "ru", "zh", "ar"]


@dataclass
class QuantumConfig:
    """Quantum computing configuration."""
    default_qubits: int = 4
    max_qubits: int = 32
    simulator: str = "qiskit"


@dataclass
class VisionConfig:
    """Computer vision configuration."""
    default_confidence: float = 0.5
    max_image_size: int = 10485760  # 10MB
    supported_formats: list = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["jpg", "jpeg", "png", "bmp", "webp"]


@dataclass
class SecurityConfig:
    """Security configuration."""
    did_method: str = "did:ethr"
    key_algorithm: str = "ES256K"


@dataclass
class ProtocolsConfig:
    """Protocols configuration."""
    qmp_port: int = 8080
    timeout: int = 30


@dataclass
class AIPlatformConfig:
    """Main AIPlatform configuration."""
    api_keys: APIConfig = None
    models: ModelConfig = None
    logging: LoggingConfig = None
    language: LanguageConfig = None
    quantum: QuantumConfig = None
    vision: VisionConfig = None
    security: SecurityConfig = None
    protocols: ProtocolsConfig = None
    
    def __post_init__(self):
        if self.api_keys is None:
            self.api_keys = APIConfig()
        if self.models is None:
            self.models = ModelConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.language is None:
            self.language = LanguageConfig()
        if self.quantum is None:
            self.quantum = QuantumConfig()
        if self.vision is None:
            self.vision = VisionConfig()
        if self.security is None:
            self.security = SecurityConfig()
        if self.protocols is None:
            self.protocols = ProtocolsConfig()


class ConfigManager:
    """Configuration manager for AIPlatform."""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = AIPlatformConfig()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Update config with loaded data
                if 'api_keys' in data:
                    self.config.api_keys = APIConfig(**data['api_keys'])
                if 'models' in data:
                    self.config.models = ModelConfig(**data['models'])
                if 'logging' in data:
                    self.config.logging = LoggingConfig(**data['logging'])
                if 'language' in data:
                    self.config.language = LanguageConfig(**data['language'])
                if 'quantum' in data:
                    self.config.quantum = QuantumConfig(**data['quantum'])
                if 'vision' in data:
                    self.config.vision = VisionConfig(**data['vision'])
                if 'security' in data:
                    self.config.security = SecurityConfig(**data['security'])
                if 'protocols' in data:
                    self.config.protocols = ProtocolsConfig(**data['protocols'])
                
                logger.info(f"Configuration loaded from {self.config_file}")
            else:
                logger.info("Config file not found, using defaults")
                self._save_config()  # Create default config
                
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            logger.info("Using default configuration")
    
    def _save_config(self):
        """Save configuration to file."""
        try:
            # Convert config to dict
            config_dict = asdict(self.config)
            
            # Create directory if needed
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error saving config: {e}")
    
    def get_api_key(self, service: str) -> str:
        """Get API key for service."""
        return getattr(self.config.api_keys, f"{service}_key", "")
    
    def set_api_key(self, service: str, key: str):
        """Set API key for service."""
        if hasattr(self.config.api_keys, f"{service}_key"):
            setattr(self.config.api_keys, f"{service}_key", key)
            self._save_config()
            logger.info(f"API key updated for {service}")
        else:
            raise ValueError(f"Unknown service: {service}")
    
    def get_model_config(self, model_type: str) -> str:
        """Get model configuration."""
        return getattr(self.config.models, model_type, "")
    
    def set_model_config(self, model_type: str, value: Any):
        """Set model configuration."""
        if hasattr(self.config.models, model_type):
            setattr(self.config.models, model_type, value)
            self._save_config()
            logger.info(f"Model config updated: {model_type}")
        else:
            raise ValueError(f"Unknown model config: {model_type}")
    
    def list_api_keys(self) -> Dict[str, str]:
        """List all API keys (masked)."""
        keys = {}
        for field in dir(self.config.api_keys):
            if field.endswith('_key'):
                value = getattr(self.config.api_keys, field)
                # Mask the key for security
                if value:
                    masked = value[:4] + "*" * (len(value) - 8) + value[-4:] if len(value) > 8 else "*" * len(value)
                    keys[field.replace('_key', '')] = masked
                else:
                    keys[field.replace('_key', '')] = "Not set"
        return keys
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return status."""
        status = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check API keys
        for service in ['openai', 'claude', 'llama', 'gigachat3', 'katya']:
            key = self.get_api_key(service)
            if not key:
                status['warnings'].append(f"API key not set for {service}")
        
        # Check directories
        log_dir = os.path.dirname(self.config.logging.file)
        if not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception as e:
                status['errors'].append(f"Cannot create log directory: {e}")
                status['valid'] = False
        
        # Check model configurations
        if not self.config.models.default_text_model:
            status['errors'].append("Default text model not specified")
            status['valid'] = False
        
        return status
    
    def export_config(self, export_file: str, include_keys: bool = False):
        """Export configuration to file."""
        try:
            config_dict = asdict(self.config)
            
            if not include_keys:
                # Remove API keys for security
                for key in config_dict['api_keys']:
                    config_dict['api_keys'][key] = ""
            
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration exported to {export_file}")
            
        except Exception as e:
            logger.error(f"Error exporting config: {e}")
            raise
    
    def import_config(self, import_file: str, merge: bool = True):
        """Import configuration from file."""
        try:
            with open(import_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not merge:
                # Replace entire config
                self.config = AIPlatformConfig()
            
            # Update config with imported data
            if 'api_keys' in data:
                for key, value in data['api_keys'].items():
                    if hasattr(self.config.api_keys, key):
                        setattr(self.config.api_keys, key, value)
            
            # Update other sections similarly...
            
            self._save_config()
            logger.info(f"Configuration imported from {import_file}")
            
        except Exception as e:
            logger.error(f"Error importing config: {e}")
            raise
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = AIPlatformConfig()
        self._save_config()
        logger.info("Configuration reset to defaults")


def create_config_manager(config_file: str = "config.json") -> ConfigManager:
    """Create and return a config manager."""
    return ConfigManager(config_file)


if __name__ == "__main__":
    # Test config management
    print("=== Config Management Test ===")
    
    # Create config manager
    config_mgr = create_config_manager()
    
    # Show current API keys
    print("\nCurrent API Keys:")
    keys = config_mgr.list_api_keys()
    for service, key in keys.items():
        print(f"  {service}: {key}")
    
    # Set a test API key
    config_mgr.set_api_key('openai', 'sk-test123456789')
    
    # Validate config
    print("\nConfiguration Validation:")
    status = config_mgr.validate_config()
    print(f"  Valid: {status['valid']}")
    if status['errors']:
        print("  Errors:")
        for error in status['errors']:
            print(f"    - {error}")
    if status['warnings']:
        print("  Warnings:")
        for warning in status['warnings']:
            print(f"    - {warning}")
    
    # Export config
    config_mgr.export_config('test_config.json', include_keys=False)
    print(f"\nConfiguration exported to test_config.json")
    
    print("\n=== Config Management Test Complete ===")
