# AIPlatform CLI Features Documentation

## Overview

AIPlatform CLI includes advanced features for enhanced functionality:

- **Batch Processing** - Process multiple prompts concurrently
- **Interactive Mode** - Conversational CLI interface
- **Config Management** - API keys and settings management
- **Plugin System** - Dynamic module loading
- **Performance Monitoring** - Metrics and profiling

## 1. Batch Processing

### Features
- Concurrent processing with ThreadPoolExecutor
- Progress tracking and retry logic
- Export results to JSON/CSV
- Load from files or directories

### Usage
```bash
# Create sample batch file
aiplatform batch create-sample --file my_batch.json

# Process batch
aiplatform batch process --file my_batch.json --max-workers 4

# Export results
aiplatform batch export --progress-file batch_progress.json --output results.json
```

### Batch File Format
```json
{
  "items": [
    {
      "id": "item1",
      "prompt": "Explain quantum computing",
      "model": "gigachat3-702b",
      "parameters": {"max_tokens": 100},
      "output_file": "output/quantum.txt"
    }
  ]
}
```

## 2. Interactive Mode

### Features
- Real-time chat interface
- Command system (/help, /history, /clear)
- Session persistence
- Model switching

### Usage
```bash
# Start interactive mode
aiplatform interactive --model gigachat3-702b --language en

# Interactive commands
/help          # Show help
/history       # Show conversation history
/clear         # Clear history
/model gpt-4   # Switch model
/save chat.json # Save conversation
/exit          # Exit
```

## 3. Config Management

### Features
- API key management
- Model configuration
- Validation and export/import
- Security masking

### Usage
```python
from aiplatform.config_manager import ConfigManager

# Create config manager
config = ConfigManager()

# Set API key
config.set_api_key('openai', 'sk-your-key')

# Get API key
key = config.get_api_key('openai')

# Validate config
status = config.validate_config()

# Export config (without keys)
config.export_config('config_backup.json')

# Import config
config.import_config('config_backup.json')
```

### Config Structure
```json
{
  "api_keys": {
    "openai_key": "",
    "claude_key": "",
    "llama_key": "",
    "gigachat3_key": "",
    "katya_key": ""
  },
  "models": {
    "default_text_model": "gigachat3-702b",
    "default_embedding_model": "text-embedding-ada-002",
    "openai_models": ["gpt-4", "gpt-3.5-turbo"],
    "claude_models": ["claude-3-opus", "claude-3-sonnet"]
  },
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/aiplatform.log"
  },
  "language": {
    "default": "en",
    "supported": ["en", "ru", "zh", "ar"]
  }
}
```

## 4. Plugin System

### Features
- Dynamic plugin loading
- Plugin lifecycle management
- Command and processor registration
- Dependency resolution

### Creating a Plugin
```python
from aiplatform.plugin_system import PluginBase, PluginInfo

class MyPlugin(PluginBase):
    def __init__(self):
        self._info = PluginInfo(
            name="my_plugin",
            version="1.0.0",
            description="My custom plugin",
            author="Your Name",
            module_path="my_plugin"
        )
    
    @property
    def info(self):
        return self._info
    
    def initialize(self, config=None):
        # Initialize plugin
        return True
    
    def cleanup(self):
        # Cleanup resources
        return True
    
    def get_commands(self):
        return {
            'my_command': self.my_command
        }
    
    def my_command(self, arg1):
        return f"Plugin command result: {arg1}"
```

### Plugin Usage
```python
from aiplatform.plugin_system import PluginManager

# Create plugin manager
plugin_mgr = PluginManager("plugins")

# Initialize plugins
plugin_mgr.initialize_all_plugins()

# Get commands
commands = plugin_mgr.get_all_commands()

# Execute plugin command
result = commands["my_plugin:my_command"]("test")
```

## 5. Performance Monitoring

### Features
- Operation timing
- System resource monitoring
- Performance decorators
- Metrics export

### Usage
```python
from aiplatform.performance_monitor import PerformanceMonitor, monitor_performance

# Create monitor
monitor = PerformanceMonitor()

# Start monitoring
monitor.start_monitoring()

# Use decorator
@monitor_performance("my_operation")
def my_function():
    # Your code here
    pass

# Get performance report
report = monitor.get_performance_report()

# Export metrics
monitor.export_metrics("metrics.json")
```

### Performance Report
```json
{
  "report_time": "2024-01-01T12:00:00",
  "monitoring_active": true,
  "total_metrics": 150,
  "total_operations": 5,
  "operations_summary": {
    "my_operation": {
      "calls": 10,
      "avg_time": 0.05,
      "success_rate": 1.0,
      "errors": 0
    }
  },
  "system_status": {
    "cpu_avg": 25.5,
    "memory_avg": 45.2
  }
}
```

## Installation Requirements

```bash
pip install psutil  # For performance monitoring
```

## Integration Examples

### Complete Workflow
```python
from aiplatform.config_manager import ConfigManager
from aiplatform.plugin_system import PluginManager
from aiplatform.performance_monitor import PerformanceMonitor

# Setup
config = ConfigManager()
plugin_mgr = PluginManager()
monitor = PerformanceMonitor()

# Configure
config.set_api_key('openai', 'your-key')
monitor.start_monitoring()

# Load and initialize plugins
plugin_mgr.initialize_all_plugins()

# Process with monitoring
@monitor_performance("batch_process")
def process_batch():
    # Your batch processing logic
    pass

# Get results
report = monitor.get_performance_report()
metrics = monitor.get_metrics_summary()
```

## Troubleshooting

### Common Issues

1. **psutil not installed**
   ```bash
   pip install psutil
   ```

2. **Plugin not loading**
   - Check plugin inherits from PluginBase
   - Verify plugin file location
   - Check for syntax errors

3. **Config file not found**
   - Config will be created automatically
   - Check file permissions

4. **Performance monitoring not working**
   - Ensure psutil is installed
   - Check system permissions

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

1. **Config Management**
   - Use environment variables for sensitive data
   - Export config without API keys
   - Validate config before use

2. **Plugin Development**
   - Implement proper cleanup
   - Handle initialization errors
   - Document dependencies

3. **Performance Monitoring**
   - Use decorators for automatic tracking
   - Monitor critical operations
   - Export metrics regularly

4. **Batch Processing**
   - Use appropriate worker count
   - Implement retry logic
   - Monitor progress

## API Reference

### ConfigManager
- `get_api_key(service)` - Get API key
- `set_api_key(service, key)` - Set API key
- `validate_config()` - Validate configuration
- `export_config(filename, include_keys=False)` - Export config
- `import_config(filename, merge=True)` - Import config

### PluginManager
- `discover_plugins()` - Discover plugins
- `initialize_plugin(name, config=None)` - Initialize plugin
- `get_plugin(name)` - Get plugin instance
- `get_all_commands()` - Get all plugin commands
- `reload_plugin(name)` - Reload plugin

### PerformanceMonitor
- `start_monitoring()` - Start monitoring
- `stop_monitoring()` - Stop monitoring
- `add_metric(name, value, unit, tags=None)` - Add metric
- `record_operation(name, duration, success=True)` - Record operation
- `get_performance_report()` - Get performance report

## Contributing

When adding new features:

1. Follow existing code patterns
2. Add comprehensive tests
3. Update documentation
4. Consider performance impact
5. Ensure backward compatibility

## License

Apache 2.0 License
