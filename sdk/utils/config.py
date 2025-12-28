"""
Configuration Management
========================

Centralized configuration for AIPlatform SDK.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import os
import json
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """SDK Configuration."""
    
    # General
    debug: bool = False
    language: str = "en"
    log_level: str = "INFO"
    
    # Quantum
    quantum_backend: str = "aer_simulator"
    quantum_shots: int = 1024
    ibm_api_key: Optional[str] = None
    ibm_instance: Optional[str] = None
    
    # AI/ML
    default_model: str = "gigachat3-702b"
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Security
    security_level: int = 768
    zero_trust_enabled: bool = True
    
    # Network
    qiz_node_id: Optional[str] = None
    qiz_port: int = 8080
    mesh_discovery: bool = True
    
    # Performance
    max_workers: int = 4
    cache_enabled: bool = True
    cache_ttl: int = 300
    
    # Custom settings
    custom: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """
    Configuration manager for SDK.
    
    Supports loading from:
    - Environment variables
    - JSON files
    - YAML files
    - Dictionary
    
    Example:
        >>> config = ConfigManager.load("config.yaml")
        >>> config.get("quantum.backend")
        >>> config.set("debug", True)
    """
    
    ENV_PREFIX = "AIPLATFORM_"
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize config manager.
        
        Args:
            config: Initial configuration
        """
        self._config = config or Config()
        self._overrides: Dict[str, Any] = {}
    
    @classmethod
    def load(cls, path: str) -> 'ConfigManager':
        """
        Load configuration from file.
        
        Args:
            path: Path to config file
            
        Returns:
            ConfigManager instance
        """
        if not os.path.exists(path):
            logger.warning(f"Config file not found: {path}")
            return cls()
        
        with open(path, 'r') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConfigManager':
        """Create from dictionary."""
        config = Config()
        
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                config.custom[key] = value
        
        return cls(config)
    
    @classmethod
    def from_env(cls) -> 'ConfigManager':
        """Load configuration from environment variables."""
        config = Config()
        
        # Map env vars to config
        env_map = {
            "DEBUG": ("debug", lambda x: x.lower() == "true"),
            "LANGUAGE": ("language", str),
            "LOG_LEVEL": ("log_level", str),
            "QUANTUM_BACKEND": ("quantum_backend", str),
            "QUANTUM_SHOTS": ("quantum_shots", int),
            "IBM_API_KEY": ("ibm_api_key", str),
            "OPENAI_API_KEY": ("openai_api_key", str),
            "ANTHROPIC_API_KEY": ("anthropic_api_key", str),
            "SECURITY_LEVEL": ("security_level", int),
            "QIZ_NODE_ID": ("qiz_node_id", str),
            "QIZ_PORT": ("qiz_port", int),
        }
        
        for env_suffix, (attr, converter) in env_map.items():
            env_key = f"{cls.ENV_PREFIX}{env_suffix}"
            value = os.environ.get(env_key)
            
            if value is not None:
                try:
                    setattr(config, attr, converter(value))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid value for {env_key}")
        
        return cls(config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Supports dot notation: "quantum.backend"
        
        Args:
            key: Configuration key
            default: Default value
            
        Returns:
            Configuration value
        """
        # Check overrides first
        if key in self._overrides:
            return self._overrides[key]
        
        # Handle dot notation
        if "." in key:
            parts = key.split(".")
            value = self._config
            
            for part in parts:
                if hasattr(value, part):
                    value = getattr(value, part)
                elif isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return default
            
            return value
        
        # Direct attribute
        if hasattr(self._config, key):
            return getattr(self._config, key)
        
        # Custom settings
        if key in self._config.custom:
            return self._config.custom[key]
        
        return default
    
    def set(self, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        if hasattr(self._config, key):
            setattr(self._config, key, value)
        else:
            self._overrides[key] = value
    
    def override(self, **kwargs):
        """Override multiple values."""
        self._overrides.update(kwargs)
    
    def reset(self):
        """Reset overrides."""
        self._overrides.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {}
        
        for key in dir(self._config):
            if not key.startswith('_'):
                value = getattr(self._config, key)
                if not callable(value):
                    result[key] = value
        
        result.update(self._overrides)
        return result
    
    def save(self, path: str):
        """Save configuration to file."""
        data = self.to_dict()
        
        with open(path, 'w') as f:
            if path.endswith('.yaml') or path.endswith('.yml'):
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)
    
    @property
    def config(self) -> Config:
        """Get underlying Config object."""
        return self._config
    
    def __repr__(self) -> str:
        return f"ConfigManager(debug={self._config.debug})"


# Global config instance
_global_config: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration."""
    global _global_config
    
    if _global_config is None:
        _global_config = ConfigManager.from_env()
    
    return _global_config


def set_config(config: ConfigManager):
    """Set global configuration."""
    global _global_config
    _global_config = config
