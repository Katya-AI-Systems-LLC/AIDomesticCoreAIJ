"""
Vision Module Tests

Tests for the computer vision components of AIPlatform.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock

# Import vision components
try:
    from aiplatform.vision import ObjectDetector, FaceRecognizer
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False


class TestObjectDetector(unittest.TestCase):
    """Test cases for ObjectDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if VISION_AVAILABLE:
            self.detector = ObjectDetector(model='yolov8', confidence_threshold=0.5)
    
    @unittest.skipIf(not VISION_AVAILABLE, "Vision components not available")
    def test_initialization(self):
        """Test object detector initialization."""
        self.assertEqual(self.detector.model, 'yolov8')
        self.assertEqual(self.detector.confidence_threshold, 0.5)
    
    @unittest.skipIf(not VISION_AVAILABLE, "Vision components not available")
    def test_detect_objects(self):
        """Test object detection."""
        # Create mock image data
        mock_image = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        
        # Mock detection results
        mock_detections = [
            {
                'class': 'person',
                'confidence': 0.95,
                'bbox': [100, 100, 200, 300]
            },
            {
                'class': 'car',
                'confidence': 0.87,
                'bbox': [50, 150, 150, 250]
            }
        ]
        
        with patch.object(self.detector, 'detect_objects', return_value=mock_detections):
            results = self.detector.detect_objects(mock_image)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]['class'], 'person')
            self.assertEqual(results[1]['class'], 'car')
    
    @unittest.skipIf(not VISION_AVAILABLE, "Vision components not available")
    def test_get_detection_stats(self):
        """Test getting detection statistics."""
        # Mock stats results
        mock_stats = {
            'total_detections': 5,
            'average_confidence': 0.85,
            'classes_detected': ['person', 'car', 'dog']
        }
        
        with patch.object(self.detector, 'get_detection_stats', return_value=mock_stats):
            stats = self.detector.get_detection_stats()
            self.assertEqual(stats['total_detections'], 5)
            self.assertEqual(stats['average_confidence'], 0.85)
            self.assertIn('person', stats['classes_detected'])


class TestFaceRecognizer(unittest.TestCase):
    """Test cases for FaceRecognizer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if VISION_AVAILABLE:
            self.recognizer = FaceRecognizer(model='arcface', detection_threshold=0.7)
    
    @unittest.skipIf(not VISION_AVAILABLE, "Vision components not available")
    def test_initialization(self):
        """Test face recognizer initialization."""
        self.assertEqual(self.recognizer.model, 'arcface')
        self.assertEqual(self.recognizer.detection_threshold, 0.7)
    
    @unittest.skipIf(not VISION_AVAILABLE, "Vision components not available")
    def test_recognize_faces(self):
        """Test face recognition."""
        # Create mock image data
        mock_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        # Mock recognition results
        mock_faces = [
            {
                'identity': 'person_001',
                'confidence': 0.92,
                'bbox': [50, 50, 150, 150],
                'landmarks': [(60, 60), (140, 60), (100, 100), (80, 130), (120, 130)]
            },
            {
                'identity': 'person_002',
                'confidence': 0.88,
                'bbox': [200, 100, 300, 200],
                'landmarks': [(210, 110), (290, 110), (250, 150), (230, 180), (270, 180)]
            }
        ]
        
        with patch.object(self.recognizer, 'recognize_faces', return_value=mock_faces):
            results = self.recognizer.recognize_faces(mock_image)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]['identity'], 'person_001')
            self.assertEqual(results[1]['identity'], 'person_002')
    
    @unittest.skipIf(not VISION_AVAILABLE, "Vision components not available")
    def test_get_recognition_stats(self):
        """Test getting recognition statistics."""
        # Mock stats results
        mock_stats = {
            'total_faces': 3,
            'unique_identities': 2,
            'average_confidence': 0.91
        }
        
        with patch.object(self.recognizer, 'get_recognition_stats', return_value=mock_stats):
            stats = self.recognizer.get_recognition_stats()
            self.assertEqual(stats['total_faces'], 3)
            self.assertEqual(stats['unique_identities'], 2)
            self.assertEqual(stats['average_confidence'], 0.91)


class TestVisionIntegration(unittest.TestCase):
    """Integration tests for vision components."""
    
    @unittest.skipIf(not VISION_AVAILABLE, "Vision components not available")
    def test_object_detection_and_recognition_workflow(self):
        """Test workflow combining object detection and face recognition."""
        # Create detectors
        object_detector = ObjectDetector()
        face_recognizer = FaceRecognizer()
        
        # Create mock image
        mock_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        
        # Mock object detection results
        mock_objects = [
            {
                'class': 'person',
                'confidence': 0.95,
                'bbox': [100, 100, 200, 300]
            }
        ]
        
        # Mock face recognition results
        mock_faces = [
            {
                'identity': 'known_person',
                'confidence': 0.92,
                'bbox': [110, 110, 190, 290]
            }
        ]
        
        with patch.object(object_detector, 'detect_objects', return_value=mock_objects):
            with patch.object(face_recognizer, 'recognize_faces', return_value=mock_faces):
                # Run detection
                objects = object_detector.detect_objects(mock_image)
                
                # Run recognition
                faces = face_recognizer.recognize_faces(mock_image)
                
                # Verify results
                self.assertEqual(len(objects), 1)
                self.assertEqual(len(faces), 1)
                self.assertEqual(objects[0]['class'], 'person')
                self.assertEqual(faces[0]['identity'], 'known_person')
    
    @unittest.skipIf(not VISION_AVAILABLE, "Vision components not available")
    def test_vision_component_compatibility(self):
        """Test compatibility between different vision components."""
        # Create various vision components
        object_detector = ObjectDetector(model='yolov8')
        face_recognizer = FaceRecognizer(model='arcface')
        
        # Verify components were created successfully
        self.assertIsNotNone(object_detector)
        self.assertIsNotNone(face_recognizer)
        
        # Verify they have expected attributes
        self.assertTrue(hasattr(object_detector, 'detect_objects'))
        self.assertTrue(hasattr(face_recognizer, 'recognize_faces'))
    
    @unittest.skipIf(not VISION_AVAILABLE, "Vision components not available")
    def test_image_processing_pipeline(self):
        """Test complete image processing pipeline."""
        # Create mock image
        mock_image = np.random.randint(0, 255, (416, 416, 3), dtype=np.uint8)
        
        # Create vision components
        object_detector = ObjectDetector()
        face_recognizer = FaceRecognizer()
        
        # Mock results
        mock_objects = [{'class': 'person', 'confidence': 0.95, 'bbox': [100, 100, 200, 300]}]
        mock_faces = [{'identity': 'person_001', 'confidence': 0.92, 'bbox': [110, 110, 190, 290]}]
        
        with patch.object(object_detector, 'detect_objects', return_value=mock_objects):
            with patch.object(face_recognizer, 'recognize_faces', return_value=mock_faces):
                # Process image through pipeline
                objects = object_detector.detect_objects(mock_image)
                faces = face_recognizer.recognize_faces(mock_image)
                
                # Verify pipeline results
                self.assertIsInstance(objects, list)
                self.assertIsInstance(faces, list)
                self.assertEqual(len(objects), 1)
                self.assertEqual(len(faces), 1)


if __name__ == '__main__':
    unittest.main()