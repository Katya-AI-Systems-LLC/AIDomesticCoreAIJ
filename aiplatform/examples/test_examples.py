"""
Test Script for AIPlatform SDK Examples

This script verifies that all examples in the AIPlatform SDK work correctly
and produce expected results across all supported languages.
"""

import sys
import os
import unittest
from typing import Dict, Any
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import example modules
try:
    from aiplatform.examples.quantum_ai_hybrid_example import (
        QuantumClassicalHybridAI, HybridAIResult
    )
    QUANTUM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Quantum example not available: {e}")
    QUANTUM_AVAILABLE = False

try:
    from aiplatform.examples.vision_demo import VisionDemo, VisionResult
    VISION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Vision example not available: {e}")
    VISION_AVAILABLE = False

try:
    from aiplatform.examples.multimodal_ai_example import (
        MultimodalAIDemo, MultimodalResult
    )
    MULTIMODAL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Multimodal example not available: {e}")
    MULTIMODAL_AVAILABLE = False


class TestAIPlatformExamples(unittest.TestCase):
    """Test cases for AIPlatform SDK examples."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.languages = ['en', 'ru', 'zh', 'ar']
        self.test_data = {
            'text': "This is a test for multimodal AI processing.",
            'audio_duration': 5.0,
            'video_frames': 10,
            'spatial_points': 100
        }
    
    def test_quantum_hybrid_ai_initialization(self):
        """Test quantum-classical hybrid AI initialization."""
        if not QUANTUM_AVAILABLE:
            self.skipTest("Quantum example not available")
        
        for language in self.languages:
            with self.subTest(language=language):
                try:
                    hybrid_ai = QuantumClassicalHybridAI(language=language)
                    self.assertIsNotNone(hybrid_ai)
                    self.assertEqual(hybrid_ai.language, language)
                except Exception as e:
                    self.fail(f"Quantum hybrid AI initialization failed for {language}: {e}")
    
    def test_quantum_hybrid_network_setup(self):
        """Test quantum-classical hybrid network setup."""
        if not QUANTUM_AVAILABLE:
            self.skipTest("Quantum example not available")
        
        hybrid_ai = QuantumClassicalHybridAI(language='en')
        node_ids = hybrid_ai.setup_hybrid_training(num_nodes=2)
        
        self.assertIsInstance(node_ids, list)
        self.assertEqual(len(node_ids), 2)
        self.assertTrue(all(isinstance(node_id, str) for node_id in node_ids))
    
    def test_quantum_hybrid_training(self):
        """Test quantum-classical hybrid training."""
        if not QUANTUM_AVAILABLE:
            self.skipTest("Quantum example not available")
        
        hybrid_ai = QuantumClassicalHybridAI(language='en')
        node_ids = hybrid_ai.setup_hybrid_training(num_nodes=2)
        result = hybrid_ai.train_hybrid_model(node_ids, epochs=1)
        
        self.assertIsInstance(result, HybridAIResult)
        self.assertGreaterEqual(result.overall_accuracy, 0.0)
        self.assertGreaterEqual(result.processing_time, 0.0)
        self.assertIsNotNone(result.quantum_results)
        self.assertIsNotNone(result.classical_results)
    
    def test_vision_demo_initialization(self):
        """Test vision demo initialization."""
        if not VISION_AVAILABLE:
            self.skipTest("Vision example not available")
        
        for language in self.languages:
            with self.subTest(language=language):
                try:
                    vision_demo = VisionDemo(language=language)
                    self.assertIsNotNone(vision_demo)
                    self.assertEqual(vision_demo.language, language)
                except Exception as e:
                    self.fail(f"Vision demo initialization failed for {language}: {e}")
    
    def test_vision_object_detection(self):
        """Test vision object detection."""
        if not VISION_AVAILABLE:
            self.skipTest("Vision example not available")
        
        vision_demo = VisionDemo(language='en')
        result = vision_demo.run_object_detection_demo(platform='web')
        
        self.assertIsInstance(result, VisionResult)
        self.assertIsNotNone(result.objects_detected)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertGreaterEqual(result.processing_time, 0.0)
        self.assertEqual(result.platform, 'web')
    
    def test_vision_face_recognition(self):
        """Test vision face recognition."""
        if not VISION_AVAILABLE:
            self.skipTest("Vision example not available")
        
        vision_demo = VisionDemo(language='en')
        result = vision_demo.run_face_recognition_demo(platform='linux')
        
        self.assertIsInstance(result, VisionResult)
        self.assertIsNotNone(result.faces_recognized)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertGreaterEqual(result.processing_time, 0.0)
        self.assertEqual(result.platform, 'linux')
    
    def test_vision_gesture_recognition(self):
        """Test vision gesture recognition."""
        if not VISION_AVAILABLE:
            self.skipTest("Vision example not available")
        
        vision_demo = VisionDemo(language='en')
        result = vision_demo.run_gesture_recognition_demo(platform='katyaos')
        
        self.assertIsInstance(result, VisionResult)
        self.assertIsNotNone(result.gestures_detected)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertGreaterEqual(result.processing_time, 0.0)
        self.assertEqual(result.platform, 'katyaos')
    
    def test_vision_slam(self):
        """Test vision SLAM processing."""
        if not VISION_AVAILABLE:
            self.skipTest("Vision example not available")
        
        vision_demo = VisionDemo(language='en')
        result = vision_demo.run_slam_demo(platform='linux')
        
        self.assertIsInstance(result, VisionResult)
        self.assertIsNotNone(result.slam_map)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertGreaterEqual(result.processing_time, 0.0)
        self.assertEqual(result.platform, 'linux')
    
    def test_multimodal_ai_initialization(self):
        """Test multimodal AI initialization."""
        if not MULTIMODAL_AVAILABLE:
            self.skipTest("Multimodal example not available")
        
        for language in self.languages:
            with self.subTest(language=language):
                try:
                    multimodal_demo = MultimodalAIDemo(language=language)
                    self.assertIsNotNone(multimodal_demo)
                    self.assertEqual(multimodal_demo.language, language)
                except Exception as e:
                    self.fail(f"Multimodal AI initialization failed for {language}: {e}")
    
    def test_multimodal_text_processing(self):
        """Test multimodal text processing."""
        if not MULTIMODAL_AVAILABLE:
            self.skipTest("Multimodal example not available")
        
        multimodal_demo = MultimodalAIDemo(language='en')
        result = multimodal_demo.process_text_data(self.test_data['text'])
        
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(result.get('confidence', 0), 0.0)
        self.assertIn('sentiment', result)
        self.assertIn('entities', result)
    
    def test_multimodal_audio_processing(self):
        """Test multimodal audio processing."""
        if not MULTIMODAL_AVAILABLE:
            self.skipTest("Multimodal example not available")
        
        multimodal_demo = MultimodalAIDemo(language='en')
        result = multimodal_demo.process_audio_data(self.test_data['audio_duration'])
        
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(result.get('confidence', 0), 0.0)
        self.assertIn('duration', result)
        self.assertIn('transcription', result)
    
    def test_multimodal_video_processing(self):
        """Test multimodal video processing."""
        if not MULTIMODAL_AVAILABLE:
            self.skipTest("Multimodal example not available")
        
        multimodal_demo = MultimodalAIDemo(language='en')
        result = multimodal_demo.process_video_data(self.test_data['video_frames'])
        
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(result.get('confidence', 0), 0.0)
        self.assertIn('frame_count', result)
        self.assertIn('objects_detected', result)
    
    def test_multimodal_3d_processing(self):
        """Test multimodal 3D processing."""
        if not MULTIMODAL_AVAILABLE:
            self.skipTest("Multimodal example not available")
        
        multimodal_demo = MultimodalAIDemo(language='en')
        result = multimodal_demo.process_3d_spatial_data(self.test_data['spatial_points'])
        
        self.assertIsInstance(result, dict)
        self.assertGreaterEqual(result.get('confidence', 0), 0.0)
        self.assertIn('point_count', result)
        self.assertIn('objects', result)
    
    def test_multimodal_integrated_analysis(self):
        """Test integrated multimodal analysis."""
        if not MULTIMODAL_AVAILABLE:
            self.skipTest("Multimodal example not available")
        
        multimodal_demo = MultimodalAIDemo(language='en')
        result = multimodal_demo.run_integrated_multimodal_analysis()
        
        self.assertIsInstance(result, MultimodalResult)
        self.assertIsNotNone(result.text_analysis)
        self.assertIsNotNone(result.audio_analysis)
        self.assertIsNotNone(result.video_analysis)
        self.assertIsNotNone(result.spatial_analysis)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertGreaterEqual(result.processing_time, 0.0)
    
    def test_cross_platform_vision(self):
        """Test cross-platform vision capabilities."""
        if not VISION_AVAILABLE:
            self.skipTest("Vision example not available")
        
        vision_demo = VisionDemo(language='en')
        results = vision_demo.run_cross_platform_demo()
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        # Check that we have results for different platforms
        self.assertTrue(any('web' in key for key in results.keys()))
        self.assertTrue(any('linux' in key for key in results.keys()))
        self.assertTrue(any('katyaos' in key for key in results.keys()))
    
    def test_multilingual_support(self):
        """Test multilingual support across all examples."""
        languages_tested = 0
        
        # Test quantum example
        if QUANTUM_AVAILABLE:
            for language in self.languages:
                try:
                    hybrid_ai = QuantumClassicalHybridAI(language=language)
                    node_ids = hybrid_ai.setup_hybrid_training(num_nodes=1)
                    result = hybrid_ai.train_hybrid_model(node_ids, epochs=1)
                    self.assertIsInstance(result, HybridAIResult)
                    languages_tested += 1
                except Exception as e:
                    self.fail(f"Quantum example failed for {language}: {e}")
        
        # Test vision example
        if VISION_AVAILABLE:
            for language in self.languages:
                try:
                    vision_demo = VisionDemo(language=language)
                    result = vision_demo.run_object_detection_demo()
                    self.assertIsInstance(result, VisionResult)
                    languages_tested += 1
                except Exception as e:
                    self.fail(f"Vision example failed for {language}: {e}")
        
        # Test multimodal example
        if MULTIMODAL_AVAILABLE:
            for language in self.languages:
                try:
                    multimodal_demo = MultimodalAIDemo(language=language)
                    result = multimodal_demo.run_integrated_multimodal_analysis()
                    self.assertIsInstance(result, MultimodalResult)
                    languages_tested += 1
                except Exception as e:
                    self.fail(f"Multimodal example failed for {language}: {e}")
        
        # Ensure we tested at least some languages
        self.assertGreater(languages_tested, 0, "No languages were successfully tested")
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        if not MULTIMODAL_AVAILABLE:
            self.skipTest("Multimodal example not available")
        
        multimodal_demo = MultimodalAIDemo(language='en')
        result = multimodal_demo.run_integrated_multimodal_analysis()
        
        # Check that performance metrics are collected
        self.assertGreaterEqual(result.processing_time, 0.0)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertIsNotNone(result.languages_supported)
        self.assertIn('en', result.languages_supported)


def run_example_tests():
    """Run all example tests."""
    print("=" * 60)
    print("AIPLATFORM SDK EXAMPLES TEST SUITE")
    print("=" * 60)
    print()
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAIPlatformExamples)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print()
    
    if result.failures:
        print("FAILURES:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
        print()
    
    if result.errors:
        print("ERRORS:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
        print()
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_example_tests()
    sys.exit(0 if success else 1)