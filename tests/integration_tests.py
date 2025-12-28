#!/usr/bin/env python3
"""
Integration Tests for AIPlatform CLI Features

This module provides comprehensive integration tests for all new CLI features:
- Batch Processing
- Interactive Mode  
- Config Management
- Plugin System
- Performance Monitoring
"""

import os
import sys
import json
import time
import tempfile
import unittest
from typing import Dict, Any
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules to test
from aiplatform.config_manager import ConfigManager, create_config_manager
from aiplatform.plugin_system import PluginManager, create_plugin_manager, ExamplePlugin
from aiplatform.performance_monitor import PerformanceMonitor, monitor_performance, get_performance_report

# Test batch processing
sys.path.insert(0, os.path.dirname(__file__))
try:
    from batch import BatchProcessor, create_batch_config
except ImportError:
    BatchProcessor = None
    create_batch_config = None

# Test interactive mode
try:
    from interactive import InteractiveChat
except ImportError:
    InteractiveChat = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestConfigManagement(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.json")
        self.config_mgr = ConfigManager(self.config_file)
    
    def tearDown(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_creation(self):
        """Test config file creation."""
        self.assertTrue(os.path.exists(self.config_file))
        
        with open(self.config_file, 'r') as f:
            data = json.load(f)
        
        self.assertIn('api_keys', data)
        self.assertIn('models', data)
        self.assertIn('logging', data)
    
    def test_api_key_management(self):
        """Test API key management."""
        # Set API key
        self.config_mgr.set_api_key('openai', 'sk-test123456789')
        
        # Get API key
        key = self.config_mgr.get_api_key('openai')
        self.assertEqual(key, 'sk-test123456789')
        
        # List API keys (masked)
        keys = self.config_mgr.list_api_keys()
        self.assertIn('openai', keys)
        self.assertTrue(keys['openai'].endswith('6789'))
    
    def test_config_validation(self):
        """Test configuration validation."""
        status = self.config_mgr.validate_config()
        
        self.assertIn('valid', status)
        self.assertIn('errors', status)
        self.assertIn('warnings', status)
        
        # Should have warnings for missing API keys
        self.assertTrue(len(status['warnings']) > 0)
    
    def test_config_export_import(self):
        """Test config export and import."""
        # Set some values
        self.config_mgr.set_api_key('openai', 'sk-test123')
        
        # Export config
        export_file = os.path.join(self.temp_dir, 'export.json')
        self.config_mgr.export_config(export_file, include_keys=False)
        
        self.assertTrue(os.path.exists(export_file))
        
        # Import config
        new_config_mgr = ConfigManager(os.path.join(self.temp_dir, 'new_config.json'))
        new_config_mgr.import_config(export_file)
        
        # Verify import (keys should be empty due to include_keys=False)
        self.assertEqual(new_config_mgr.get_api_key('openai'), '')


class TestPluginSystem(unittest.TestCase):
    """Test plugin system."""
    
    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.plugin_mgr = PluginManager(self.temp_dir)
        
        # Create example plugin file
        plugin_file = os.path.join(self.temp_dir, 'example_plugin.py')
        with open(plugin_file, 'w') as f:
            f.write("""
from aiplatform.plugin_system import PluginBase, PluginInfo

class TestPlugin(PluginBase):
    def __init__(self):
        self._info = PluginInfo(
            name="test_plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test",
            module_path="test_plugin"
        )
        self.initialized = False
    
    @property
    def info(self):
        return self._info
    
    def initialize(self, config=None):
        self.initialized = True
        return True
    
    def cleanup(self):
        self.initialized = False
        return True
    
    def get_commands(self):
        return {
            'test_cmd': lambda: "test_result"
        }
""")
    
    def tearDown(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_plugin_discovery(self):
        """Test plugin discovery."""
        # Force rediscovery
        self.plugin_mgr.discover_plugins()
        
        plugins = self.plugin_mgr.list_plugins()
        self.assertIn('test_plugin', plugins)
        
        info = plugins['test_plugin']
        self.assertEqual(info.name, 'test_plugin')
        self.assertEqual(info.version, '1.0.0')
    
    def test_plugin_initialization(self):
        """Test plugin initialization."""
        # Initialize plugin
        success = self.plugin_mgr.initialize_plugin('test_plugin')
        self.assertTrue(success)
        
        plugin = self.plugin_mgr.get_plugin('test_plugin')
        self.assertTrue(plugin.initialized)
    
    def test_plugin_commands(self):
        """Test plugin commands."""
        # Initialize plugin
        self.plugin_mgr.initialize_plugin('test_plugin')
        
        # Get commands
        commands = self.plugin_mgr.get_all_commands()
        self.assertIn('test_plugin:test_cmd', commands)
        
        # Execute command
        result = commands['test_plugin:test_cmd']()
        self.assertEqual(result, "test_result")


class TestPerformanceMonitoring(unittest.TestCase):
    """Test performance monitoring."""
    
    def setUp(self):
        """Setup test environment."""
        self.monitor = PerformanceMonitor()
    
    def tearDown(self):
        """Cleanup test environment."""
        self.monitor.stop_monitoring()
    
    def test_metric_recording(self):
        """Test metric recording."""
        # Add test metric
        self.monitor.add_metric('test_metric', 42.0, 'units')
        
        self.assertEqual(len(self.monitor.metrics), 1)
        metric = self.monitor.metrics[0]
        self.assertEqual(metric.name, 'test_metric')
        self.assertEqual(metric.value, 42.0)
        self.assertEqual(metric.unit, 'units')
    
    def test_operation_recording(self):
        """Test operation recording."""
        # Record operation
        self.monitor.record_operation('test_op', 0.1, True)
        
        summary = self.monitor.get_metrics_summary('test_op')
        self.assertEqual(summary['total_calls'], 1)
        self.assertEqual(summary['avg_time'], 0.1)
        self.assertEqual(summary['success_rate'], 1.0)
    
    def test_performance_decorator(self):
        """Test performance decorator."""
        @monitor_performance('decorated_test')
        def test_function():
            time.sleep(0.01)
            return "result"
        
        # Call function
        result = test_function()
        self.assertEqual(result, "result")
        
        # Check metrics
        summary = self.monitor.get_metrics_summary('decorated_test')
        self.assertEqual(summary['total_calls'], 1)
        self.assertGreater(summary['avg_time'], 0.01)
    
    def test_system_monitoring(self):
        """Test system monitoring."""
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Wait for some metrics
        time.sleep(2)
        
        # Check system metrics
        cpu_metrics = [m for m in self.monitor.metrics if m.name == 'system_cpu_usage']
        self.assertGreater(len(cpu_metrics), 0)
        
        # Stop monitoring
        self.monitor.stop_monitoring()


@unittest.skipIf(BatchProcessor is None, "Batch processing not available")
class TestBatchProcessing(unittest.TestCase):
    """Test batch processing."""
    
    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test batch file
        self.batch_file = os.path.join(self.temp_dir, 'test_batch.json')
        batch_data = {
            "items": [
                {
                    "id": "item1",
                    "prompt": "Test prompt 1",
                    "model": "test_model",
                    "parameters": {"max_tokens": 50}
                },
                {
                    "id": "item2", 
                    "prompt": "Test prompt 2",
                    "model": "test_model",
                    "parameters": {"max_tokens": 50}
                }
            ]
        }
        
        with open(self.batch_file, 'w') as f:
            json.dump(batch_data, f)
    
    def tearDown(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_batch_loading(self):
        """Test batch file loading."""
        config = create_batch_config(max_workers=2)
        processor = BatchProcessor(config)
        
        items = processor.load_from_file(self.batch_file)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0].id, 'item1')
        self.assertEqual(items[1].id, 'item2')
    
    def test_batch_processing(self):
        """Test batch processing."""
        config = create_batch_config(max_workers=2)
        processor = BatchProcessor(config)
        
        # Load items
        processor.load_from_file(self.batch_file)
        
        # Define processing function
        def process_prompt(prompt, model, **kwargs):
            return f"Processed: {prompt}"
        
        # Process batch
        results = processor.process_batch(process_prompt)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].status, 'completed')
        self.assertEqual(results[1].status, 'completed')
        
        # Check summary
        summary = processor.get_summary()
        self.assertEqual(summary['total_items'], 2)
        self.assertEqual(summary['completed'], 2)
        self.assertEqual(summary['failed'], 0)


@unittest.skipIf(InteractiveChat is None, "Interactive mode not available")
class TestInteractiveMode(unittest.TestCase):
    """Test interactive mode."""
    
    def test_chat_creation(self):
        """Test chat creation."""
        chat = InteractiveChat('test_model')
        
        self.assertEqual(chat.model, 'test_model')
        self.assertEqual(chat.language, 'en')
        self.assertTrue(chat.running)
        self.assertEqual(len(chat.history), 0)
    
    def test_help_command(self):
        """Test help command."""
        chat = InteractiveChat()
        
        # This should not raise an exception
        chat.show_help()
    
    def test_history_management(self):
        """Test history management."""
        chat = InteractiveChat()
        
        # Add messages
        chat._add_to_history('user', 'test message')
        chat._add_to_history('assistant', 'test response')
        
        self.assertEqual(len(chat.history), 2)
        
        # Clear history
        chat.clear_history()
        self.assertEqual(len(chat.history), 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for all features."""
    
    def setUp(self):
        """Setup test environment."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_with_performance_monitoring(self):
        """Test config management with performance monitoring."""
        # Create config manager
        config_file = os.path.join(self.temp_dir, 'integration_config.json')
        config_mgr = ConfigManager(config_file)
        
        # Create performance monitor
        monitor = PerformanceMonitor()
        
        # Monitor config operations
        @monitor_performance('config_set_api_key')
        def set_api_key():
            config_mgr.set_api_key('openai', 'sk-test123')
        
        @monitor_performance('config_get_api_key') 
        def get_api_key():
            return config_mgr.get_api_key('openai')
        
        # Execute operations
        set_api_key()
        key = get_api_key()
        
        # Verify results
        self.assertEqual(key, 'sk-test123')
        
        # Check performance metrics
        summary = monitor.get_metrics_summary('config_set_api_key')
        self.assertEqual(summary['total_calls'], 1)
        self.assertGreater(summary['avg_time'], 0)
    
    def test_plugin_with_config(self):
        """Test plugin system with config management."""
        # Create config manager
        config_file = os.path.join(self.temp_dir, 'plugin_config.json')
        config_mgr = ConfigManager(config_file)
        
        # Create plugin manager
        plugin_mgr = PluginManager(self.temp_dir)
        
        # Create test plugin that uses config
        plugin_file = os.path.join(self.temp_dir, 'config_plugin.py')
        with open(plugin_file, 'w') as f:
            f.write("""
from aiplatform.plugin_system import PluginBase, PluginInfo

class ConfigPlugin(PluginBase):
    def __init__(self):
        self._info = PluginInfo(
            name="config_plugin",
            version="1.0.0",
            description="Config test plugin",
            author="Test",
            module_path="config_plugin"
        )
        self.config = None
    
    @property
    def info(self):
        return self._info
    
    def initialize(self, config=None):
        self.config = config
        return True
    
    def cleanup(self):
        self.config = None
        return True
    
    def get_config_value(self, key):
        return self.config.get(key, None) if self.config else None
""")
        
        # Rediscover plugins
        plugin_mgr.discover_plugins()
        
        # Initialize plugin with config
        test_config = {'test_key': 'test_value'}
        success = plugin_mgr.initialize_plugin('config_plugin', test_config)
        
        self.assertTrue(success)
        
        plugin = plugin_mgr.get_plugin('config_plugin')
        self.assertEqual(plugin.get_config_value('test_key'), 'test_value')


def run_integration_tests():
    """Run all integration tests."""
    print("=== AIPlatform Integration Tests ===")
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestConfigManagement,
        TestPluginSystem, 
        TestPerformanceMonitoring,
        TestBatchProcessing,
        TestInteractiveMode,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n=== Test Summary ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    print(f"\nOverall result: {'PASSED' if success else 'FAILED'}")
    
    return success


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
