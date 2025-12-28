"""
AI & Vision Lab Example for AIPlatform SDK

This example demonstrates computer vision and big data capabilities with multilingual support.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiplatform.vision import (
    create_object_detector,
    create_face_recognizer,
    create_gesture_recognizer,
    create_slam_system,
    create_video_processor,
    create_big_data_pipeline,
    create_streaming_analytics,
    create_multimodal_processor
)
import numpy as np


def object_detection_example(language='en'):
    """Demonstrate object detection."""
    print(f"=== {translate('object_detection_example', language) or 'Object Detection Example'} ===")
    
    # Create object detector
    detector = create_object_detector('yolo', language=language)
    
    # Simulate image detection
    detections = detector.detect(None)  # None as placeholder for actual image
    
    print(f"Detected {len(detections)} objects:")
    for i, detection in enumerate(detections[:3]):  # Show first 3 detections
        print(f"  {i+1}. {detection['class']} (confidence: {detection['confidence']:.2f})")
    print()


def face_recognition_example(language='en'):
    """Demonstrate face recognition."""
    print(f"=== {translate('face_recognition_example', language) or 'Face Recognition Example'} ===")
    
    # Create face recognizer
    recognizer = create_face_recognizer(language=language)
    
    # Add known faces
    alice_encoding = np.random.random(128).tolist()
    bob_encoding = np.random.random(128).tolist()
    
    recognizer.add_face("Alice", alice_encoding)
    recognizer.add_face("Bob", bob_encoding)
    
    # Recognize a face
    test_encoding = np.array(alice_encoding) + np.random.normal(0, 0.1, 128)  # Slightly perturbed
    recognized = recognizer.recognize(test_encoding.tolist())
    
    print(f"Recognized face: {recognized}")
    print()


def gesture_recognition_example(language='en'):
    """Demonstrate gesture recognition."""
    print(f"=== {translate('gesture_recognition_example', language) or 'Gesture Recognition Example'} ===")
    
    # Create gesture recognizer
    gesture_recognizer = create_gesture_recognizer(language=language)
    
    # Simulate hand landmarks
    hand_landmarks = [(np.random.random(), np.random.random()) for _ in range(21)]
    
    # Recognize gesture
    gesture = gesture_recognizer.recognize_gesture(hand_landmarks)
    
    print(f"Recognized gesture: {gesture}")
    print()


def slam_example(language='en'):
    """Demonstrate SLAM system."""
    print(f"=== {translate('slam_example', language) or 'SLAM Example'} ===")
    
    # Create SLAM system
    slam = create_slam_system(language=language)
    
    # Simulate sensor updates
    for i in range(5):
        sensor_data = {
            'movement': [0.1, 0.05, 0.02]  # x, y, theta movement
        }
        result = slam.update(sensor_data)
        print(f"Update {i+1}: Pose = [{result['pose'][0]:.2f}, {result['pose'][1]:.2f}, {result['pose'][2]:.2f}]")
    
    print(f"Total map points: {result['map_points']}")
    print()


def video_processing_example(language='en'):
    """Demonstrate video processing."""
    print(f"=== {translate('video_processing_example', language) or 'Video Processing Example'} ===")
    
    # Create video processor
    processor = create_video_processor(language=language)
    
    # Process frames
    for i in range(3):
        frame_data = f"frame_{i}"
        result = processor.process_frame(frame_data)
        print(f"Processed frame {i}: {result}")
    
    # Extract features
    features = processor.extract_features("video.mp4")
    print(f"Extracted {len(features)} features")
    print()


def big_data_pipeline_example(language='en'):
    """Demonstrate big data pipeline."""
    print(f"=== {translate('big_data_pipeline_example', language) or 'Big Data Pipeline Example'} ===")
    
    # Create big data pipeline
    pipeline = create_big_data_pipeline(language=language)
    
    # Add processing stages
    def stage1(data):
        return {"processed": True, "stage": 1, "value": data * 2}
    
    def stage2(data):
        return {"processed": True, "stage": 2, "value": data["value"] + 10}
    
    pipeline.add_stage("multiply_by_2", stage1)
    pipeline.add_stage("add_10", stage2)
    
    # Process batch
    data_batch = [1, 2, 3, 4, 5]
    processed_batch = pipeline.process_batch(data_batch)
    
    print(f"Original batch: {data_batch}")
    print(f"Processed batch: {[item['value'] for item in processed_batch]}")
    
    # Get statistics
    stats = pipeline.get_statistics()
    print(f"Pipeline statistics: {stats}")
    print()


def streaming_analytics_example(language='en'):
    """Demonstrate streaming analytics."""
    print(f"=== {translate('streaming_analytics_example', language) or 'Streaming Analytics Example'} ===")
    
    # Create streaming analytics
    analytics = create_streaming_analytics(window_size=100, language=language)
    
    # Add data points
    for i in range(20):
        data_point = np.random.random() * 100
        analytics.add_data_point(data_point)
    
    # Calculate metrics
    metrics = analytics.calculate_metrics()
    print(f"Analytics metrics: {metrics}")
    print()


def multimodal_processing_example(language='en'):
    """Demonstrate multimodal processing."""
    print(f"=== {translate('multimodal_processing_example', language) or 'Multimodal Processing Example'} ===")
    
    # Create multimodal processor
    processor = create_multimodal_processor(language=language)
    
    # Process multimodal input
    inputs = {
        'text': "Hello, world!",
        'audio': b"audio_data_placeholder",
        'video': b"video_data_placeholder",
        'image': b"image_data_placeholder"
    }
    
    result = processor.process_multimodal_input(inputs)
    print(f"Multimodal processing result: {result}")
    print()


def translate(key, language):
    """Simple translation function for example titles."""
    translations = {
        'object_detection_example': {
            'ru': 'Пример обнаружения объектов',
            'zh': '物体检测示例',
            'ar': 'مثال اكتشاف الكائنات'
        },
        'face_recognition_example': {
            'ru': 'Пример распознавания лиц',
            'zh': '人脸识别示例',
            'ar': 'مثال التعرف على الوجه'
        },
        'gesture_recognition_example': {
            'ru': 'Пример распознавания жестов',
            'zh': '手势识别示例',
            'ar': 'مثال التعرف على الإيماءات'
        },
        'slam_example': {
            'ru': 'Пример SLAM',
            'zh': 'SLAM示例',
            'ar': 'مثال SLAM'
        },
        'video_processing_example': {
            'ru': 'Пример обработки видео',
            'zh': '视频处理示例',
            'ar': 'مثال معالجة الفيديو'
        },
        'big_data_pipeline_example': {
            'ru': 'Пример конвейера больших данных',
            'zh': '大数据管道示例',
            'ar': 'مثال خط أنابيب البيانات الكبيرة'
        },
        'streaming_analytics_example': {
            'ru': 'Пример потоковой аналитики',
            'zh': '流分析示例',
            'ar': 'مثال التحليلات البثية'
        },
        'multimodal_processing_example': {
            'ru': 'Пример мультимодальной обработки',
            'zh': '多模态处理示例',
            'ar': 'مثال المعالجة متعددة الوسائط'
        }
    }
    
    if key in translations and language in translations[key]:
        return translations[key][language]
    return None


def main():
    """Run all vision examples."""
    languages = ['en', 'ru', 'zh', 'ar']
    
    for language in languages:
        print(f"\n{'='*50}")
        print(f"VISION EXAMPLES - {language.upper()}")
        print(f"{'='*50}\n")
        
        try:
            object_detection_example(language)
            face_recognition_example(language)
            gesture_recognition_example(language)
            slam_example(language)
            video_processing_example(language)
            big_data_pipeline_example(language)
            streaming_analytics_example(language)
            multimodal_processing_example(language)
        except Exception as e:
            print(f"Error in {language} examples: {e}")


if __name__ == "__main__":
    main()