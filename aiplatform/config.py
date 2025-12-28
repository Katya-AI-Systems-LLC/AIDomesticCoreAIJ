"""
Configuration module for AIPlatform Quantum Infrastructure Zero SDK

This module provides configuration management for the SDK,
including default settings, file-based configuration, and
runtime parameter handling.
"""

import os
import json
import yaml
from typing import Optional, Dict, Any, Union
from pathlib import Path

class Configuration:
    """
    Configuration management for AIPlatform SDK.
    
    Handles configuration from files, environment variables, and runtime parameters.
    """
    
    # Default configuration
    DEFAULT_CONFIG = {
        "quantum": {
            "backend": "simulator",
            "shots": 1024,
            "optimization_level": 3,
            "timeout": 300
        },
        "qiz": {
            "node_discovery": "quantum_signature",
            "routing_protocol": "qmp",
            "security_level": "high"
        },
        "federated": {
            "protocol": "quantum_secure",
            "aggregation": "secure_multi_party",
            "participation_threshold": 0.8
        },
        "vision": {
            "model": "gigachat-vision-702b",
            "processing_mode": "quantum_accelerated",
            "detection_threshold": 0.7
        },
        "genai": {
            "default_model": "gigachat3-702b",
            "max_tokens": 4096,
            "temperature": 0.7,
            "top_p": 0.9
        },
        "security": {
            "encryption": "kyber",
            "signature": "dilithium",
            "zero_trust": True
        },
        "network": {
            "discovery_timeout": 30,
            "connection_timeout": 60,
            "retry_attempts": 3
        }
    }
    
    def __init__(self, config_file: Optional[str] = None, **kwargs):
        """
        Initialize configuration.
        
        Args:
            config_file (str, optional): Path to configuration file
            **kwargs: Additional configuration parameters
        """
        # Start with default configuration
        self._config = self._deep_copy(self.DEFAULT_CONFIG)
        
        # Load from file if provided
        if config_file:
            self.load_config(config_file)
        
        # Override with provided parameters
        self._update_config(self._config, kwargs)
    
    def load_config(self, config_file: str) -> bool:
        """
        Load configuration from file.
        
        Args:
            config_file (str): Path to configuration file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            # Update configuration
            self._update_config(self._config, file_config)
            return True
            
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_file}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key.
        
        Args:
            key (str): Configuration key (e.g., "quantum.backend")
            default (Any): Default value if key not found
            
        Returns:
            Any: Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value.
        
        Args:
            key (str): Configuration key (e.g., "quantum.backend")
            value (Any): Configuration value
        """
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save(self, config_file: str) -> bool:
        """
        Save configuration to file.
        
        Args:
            config_file (str): Path to configuration file
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self._config, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            raise ValueError(f"Failed to save configuration to {config_file}: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Get configuration as dictionary.
        
        Returns:
            dict: Configuration dictionary
        """
        return self._deep_copy(self._config)
    
    def update(self, config_dict: Dict[str, Any]) -> None:
        """
        Update configuration with dictionary.
        
        Args:
            config_dict (dict): Configuration dictionary to update with
        """
        self._update_config(self._config, config_dict)
    
    def _update_config(self, config: Dict, updates: Dict) -> None:
        """
        Recursively update configuration.
        
        Args:
            config (dict): Configuration to update
            updates (dict): Updates to apply
        """
        for key, value in updates.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                self._update_config(config[key], value)
            else:
                config[key] = value
    
    def _deep_copy(self, obj):
        """
        Create deep copy of object.
        
        Args:
            obj: Object to copy
            
        Returns:
            Copy of object
        """
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set configuration value using bracket notation."""
        self.set(key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if configuration contains key."""
        return self.get(key) is not None

# Global configuration instance
_global_config = None

def get_global_config() -> Configuration:
    """
    Get global configuration instance.
    
    Returns:
        Configuration: Global configuration instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Configuration()
    return _global_config

def set_global_config(config: Union[Configuration, Dict, str]) -> None:
    """
    Set global configuration.
    
    Args:
        config: Configuration instance, dictionary, or file path
    """
    global _global_config
    
    if isinstance(config, str):
        _global_config = Configuration(config)
    elif isinstance(config, dict):
        _global_config = Configuration()
        _global_config.update(config)
    else:
        _global_config = config