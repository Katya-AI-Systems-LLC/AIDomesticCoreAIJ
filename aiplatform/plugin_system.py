"""
Plugin System for AIPlatform

This module provides a dynamic plugin loading system for extending
AIPlatform functionality with custom modules.
"""

import os
import sys
import json
import importlib.util
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Plugin information."""
    name: str
    version: str
    description: str
    author: str
    module_path: str
    enabled: bool = True
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class PluginBase(ABC):
    """Base class for all plugins."""
    
    @property
    @abstractmethod
    def info(self) -> PluginInfo:
        """Return plugin information."""
        pass
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize the plugin."""
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """Cleanup plugin resources."""
        pass
    
    def get_commands(self) -> Dict[str, Callable]:
        """Return available commands from this plugin."""
        return {}
    
    def get_processors(self) -> Dict[str, Callable]:
        """Return available processors from this plugin."""
        return {}


class PluginManager:
    """Plugin manager for loading and managing plugins."""
    
    def __init__(self, plugin_dir: str = "plugins"):
        self.plugin_dir = plugin_dir
        self.plugins: Dict[str, PluginBase] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}
        self.loaded_modules: Dict[str, Any] = {}
        
        # Ensure plugin directory exists
        os.makedirs(plugin_dir, exist_ok=True)
        
        # Load plugins
        self.discover_plugins()
    
    def discover_plugins(self):
        """Discover plugins in plugin directory."""
        logger.info(f"Discovering plugins in {self.plugin_dir}")
        
        for item in os.listdir(self.plugin_dir):
            item_path = os.path.join(self.plugin_dir, item)
            
            # Check for Python files
            if item.endswith('.py') and not item.startswith('__'):
                self._load_plugin_from_file(item_path)
            
            # Check for plugin directories
            elif os.path.isdir(item_path):
                init_file = os.path.join(item_path, '__init__.py')
                if os.path.exists(init_file):
                    self._load_plugin_from_directory(item_path)
    
    def _load_plugin_from_file(self, file_path: str):
        """Load plugin from Python file."""
        try:
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Load module
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for plugin class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, PluginBase) and 
                    attr != PluginBase):
                    
                    plugin_instance = attr()
                    plugin_info = plugin_instance.info
                    
                    self.plugins[plugin_info.name] = plugin_instance
                    self.plugin_info[plugin_info.name] = plugin_info
                    self.loaded_modules[plugin_info.name] = module
                    
                    logger.info(f"Loaded plugin: {plugin_info.name} v{plugin_info.version}")
                    break
                    
        except Exception as e:
            logger.error(f"Error loading plugin from {file_path}: {e}")
    
    def _load_plugin_from_directory(self, dir_path: str):
        """Load plugin from directory."""
        try:
            dir_name = os.path.basename(dir_path)
            
            # Add to path
            if dir_path not in sys.path:
                sys.path.insert(0, dir_path)
            
            # Import as package
            module = importlib.import_module(dir_name)
            
            # Look for plugin class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, PluginBase) and 
                    attr != PluginBase):
                    
                    plugin_instance = attr()
                    plugin_info = plugin_instance.info
                    
                    self.plugins[plugin_info.name] = plugin_instance
                    self.plugin_info[plugin_info.name] = plugin_info
                    self.loaded_modules[plugin_info.name] = module
                    
                    logger.info(f"Loaded plugin: {plugin_info.name} v{plugin_info.version}")
                    break
                    
        except Exception as e:
            logger.error(f"Error loading plugin from {dir_path}: {e}")
    
    def initialize_plugin(self, plugin_name: str, config: Dict[str, Any] = None) -> bool:
        """Initialize a specific plugin."""
        if plugin_name not in self.plugins:
            logger.error(f"Plugin not found: {plugin_name}")
            return False
        
        try:
            plugin = self.plugins[plugin_name]
            
            # Check dependencies
            for dep in plugin.info.dependencies:
                if dep not in self.plugins:
                    logger.error(f"Dependency not found for {plugin_name}: {dep}")
                    return False
            
            # Initialize plugin
            success = plugin.initialize(config)
            
            if success:
                logger.info(f"Plugin initialized: {plugin_name}")
            else:
                logger.error(f"Plugin initialization failed: {plugin_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error initializing plugin {plugin_name}: {e}")
            return False
    
    def initialize_all_plugins(self, config: Dict[str, Any] = None):
        """Initialize all enabled plugins."""
        for plugin_name, plugin_info in self.plugin_info.items():
            if plugin_info.enabled:
                self.initialize_plugin(plugin_name, config)
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginBase]:
        """Get plugin instance."""
        return self.plugins.get(plugin_name)
    
    def list_plugins(self) -> Dict[str, PluginInfo]:
        """List all discovered plugins."""
        return self.plugin_info.copy()
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        if plugin_name in self.plugin_info:
            self.plugin_info[plugin_name].enabled = True
            logger.info(f"Plugin enabled: {plugin_name}")
            return True
        return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        if plugin_name in self.plugin_info:
            self.plugin_info[plugin_name].enabled = False
            logger.info(f"Plugin disabled: {plugin_name}")
            return True
        return False
    
    def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin."""
        if plugin_name in self.plugins:
            try:
                # Cleanup old plugin
                self.plugins[plugin_name].cleanup()
                
                # Remove from collections
                del self.plugins[plugin_name]
                del self.plugin_info[plugin_name]
                del self.loaded_modules[plugin_name]
                
                # Rediscover and reload
                self.discover_plugins()
                
                logger.info(f"Plugin reloaded: {plugin_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error reloading plugin {plugin_name}: {e}")
                return False
        
        return False
    
    def get_all_commands(self) -> Dict[str, Callable]:
        """Get all commands from all plugins."""
        commands = {}
        for plugin_name, plugin in self.plugins.items():
            if self.plugin_info[plugin_name].enabled:
                plugin_commands = plugin.get_commands()
                for cmd_name, cmd_func in plugin_commands.items():
                    commands[f"{plugin_name}:{cmd_name}"] = cmd_func
        return commands
    
    def get_all_processors(self) -> Dict[str, Callable]:
        """Get all processors from all plugins."""
        processors = {}
        for plugin_name, plugin in self.plugins.items():
            if self.plugin_info[plugin_name].enabled:
                plugin_processors = plugin.get_processors()
                for proc_name, proc_func in plugin_processors.items():
                    processors[f"{plugin_name}:{proc_name}"] = proc_func
        return processors
    
    def cleanup_all_plugins(self):
        """Cleanup all plugins."""
        for plugin_name, plugin in self.plugins.items():
            try:
                plugin.cleanup()
                logger.info(f"Plugin cleaned up: {plugin_name}")
            except Exception as e:
                logger.error(f"Error cleaning up plugin {plugin_name}: {e}")
    
    def save_plugin_config(self, config_file: str = "plugin_config.json"):
        """Save plugin configuration."""
        config = {}
        for plugin_name, plugin_info in self.plugin_info.items():
            config[plugin_name] = {
                'enabled': plugin_info.enabled,
                'version': plugin_info.version,
                'dependencies': plugin_info.dependencies
            }
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"Plugin configuration saved to {config_file}")
        except Exception as e:
            logger.error(f"Error saving plugin config: {e}")
    
    def load_plugin_config(self, config_file: str = "plugin_config.json"):
        """Load plugin configuration."""
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                for plugin_name, plugin_config in config.items():
                    if plugin_name in self.plugin_info:
                        self.plugin_info[plugin_name].enabled = plugin_config.get('enabled', True)
                
                logger.info(f"Plugin configuration loaded from {config_file}")
        except Exception as e:
            logger.error(f"Error loading plugin config: {e}")


# Example plugin
class ExamplePlugin(PluginBase):
    """Example plugin for demonstration."""
    
    def __init__(self):
        self._info = PluginInfo(
            name="example",
            version="1.0.0",
            description="Example plugin for demonstration",
            author="AIPlatform Team",
            module_path="example_plugin"
        )
        self.initialized = False
    
    @property
    def info(self) -> PluginInfo:
        return self._info
    
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize example plugin."""
        logger.info("Initializing example plugin")
        self.initialized = True
        return True
    
    def cleanup(self) -> bool:
        """Cleanup example plugin."""
        logger.info("Cleaning up example plugin")
        self.initialized = False
        return True
    
    def get_commands(self) -> Dict[str, Callable]:
        """Return example commands."""
        return {
            'hello': self.hello_command,
            'echo': self.echo_command
        }
    
    def hello_command(self, name: str = "World") -> str:
        """Hello command."""
        return f"Hello, {name}! From example plugin!"
    
    def echo_command(self, text: str) -> str:
        """Echo command."""
        return f"Echo: {text}"


def create_plugin_manager(plugin_dir: str = "plugins") -> PluginManager:
    """Create and return a plugin manager."""
    return PluginManager(plugin_dir)


if __name__ == "__main__":
    # Test plugin system
    print("=== Plugin System Test ===")
    
    # Create plugin manager
    plugin_mgr = create_plugin_manager()
    
    # List plugins
    plugins = plugin_mgr.list_plugins()
    print(f"\nDiscovered plugins: {len(plugins)}")
    for name, info in plugins.items():
        print(f"  {name} v{info.version} - {info.description}")
    
    # Initialize all plugins
    plugin_mgr.initialize_all_plugins()
    
    # Get all commands
    commands = plugin_mgr.get_all_commands()
    print(f"\nAvailable commands: {len(commands)}")
    for cmd_name, cmd_func in commands.items():
        print(f"  {cmd_name}")
    
    # Test example command if available
    if "example:hello" in commands:
        result = commands["example:hello"]("AIPlatform")
        print(f"\nTest command result: {result}")
    
    # Save plugin config
    plugin_mgr.save_plugin_config()
    
    print("\n=== Plugin System Test Complete ===")
