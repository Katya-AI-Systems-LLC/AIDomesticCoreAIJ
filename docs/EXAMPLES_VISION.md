# Vision Lab Examples

## Overview

This document provides examples of computer vision applications using the REChain AI & Vision Lab module. These examples demonstrate object detection, facial recognition, gesture recognition, 3D computer vision, and integration with quantum-enhanced processing.

## Example 1: Real-time Object Detection

### Problem Description
Detecting and tracking multiple objects in a real-time video stream using the GigaChat Vision 702B model.

### Implementation

```python
from aiplatform.vision import ObjectDetector
from aiplatform.vision.tracking import MultiObjectTracker
import cv2

# Initialize object detector with GigaChat Vision model
detector = ObjectDetector(
    model="gigachat-vision-702b",
    confidence_threshold=0.7,
    device="cuda"  # Use GPU if available
)

# Initialize multi-object tracker
tracker = MultiObjectTracker(
    max_age=30,
    min_hits=3,
    iou_threshold=0.3
)

# Open video stream
cap = cv2.VideoCapture(0)  # Use webcam
# cap = cv2.VideoCapture("path/to/video.mp4")  # Use video file

# Process video frames
frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects in the frame
    detections = detector.detect_objects(
        frame,
        classes=["person", "car", "bicycle", "dog"]
    )
    
    # Update tracker with detections
    tracked_objects = tracker.update(detections)
    
    # Draw detections and tracks
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_name = detection['class']
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw tracked objects
    for track_id, track in tracked_objects.items():
        x1, y1, x2, y2 = track['bbox']
        # Draw tracking ID
        cv2.putText(frame, f"ID: {track_id}", (x1, y2+20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Display frame
    cv2.imshow('Object Detection and Tracking', frame)
    
    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    frame_count += 1
    if frame_count % 30 == 0:  # Print FPS every 30 frames
        print(f"Processed {frame_count} frames")

# Clean up
cap.release()
cv2.destroyAllWindows()

# Print tracking statistics
print(f"Total frames processed: {frame_count}")
print(f"Unique objects tracked: {len(tracker.get_all_tracks())}")
```

## Example 2: Facial Recognition and Emotion Analysis

### Problem Description
Recognizing faces and analyzing emotions in images and video streams.

### Implementation

```python
from aiplatform.vision import FaceRecognizer
from aiplatform.vision.emotion import EmotionAnalyzer
import cv2
import numpy as np

# Initialize face recognizer
face_recognizer = FaceRecognizer(
    model="gigachat-face-702b",
    detection_threshold=0.8
)

# Initialize emotion analyzer
emotion_analyzer = EmotionAnalyzer(
    model="gigachat-emotion-702b"
)

# Load known faces database
known_faces = face_recognizer.load_database("data/known_faces/")

def process_image(image_path):
    """Process a single image for face recognition and emotion analysis."""
    
    # Load image
    image = cv2.imread(image_path)
    
    # Detect faces
    faces = face_recognizer.detect_faces(image)
    
    results = []
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        face_image = image[y1:y2, x1:x2]
        
        # Recognize face
        identity = face_recognizer.recognize_face(face_image)
        
        # Analyze emotion
        emotion = emotion_analyzer.analyze_emotion(face_image)
        
        # Draw results
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw identity
        if identity['match']:
            label = f"{identity['name']} ({identity['confidence']:.2f})"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)
        
        cv2.putText(image, label, (x1, y1-30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw emotion
        emotion_label = f"{emotion['emotion']} ({emotion['confidence']:.2f})"
        cv2.putText(image, emotion_label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        results.append({
            'bbox': face['bbox'],
            'identity': identity,
            'emotion': emotion
        })
    
    return image, results

def process_video(video_path):
    """Process a video stream for face recognition and emotion analysis."""
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))
    
    frame_count = 0
    emotion_stats = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 5th frame for performance
        if frame_count % 5 == 0:
            # Detect faces
            faces = face_recognizer.detect_faces(frame)
            
            for face in faces:
                x1, y1, x2, y2 = face['bbox']
                face_image = frame[y1:y2, x1:x2]
                
                # Recognize face
                identity = face_recognizer.recognize_face(face_image)
                
                # Analyze emotion
                emotion = emotion_analyzer.analyze_emotion(face_image)
                
                # Update emotion statistics
                emotion_name = emotion['emotion']
                emotion_stats[emotion_name] = emotion_stats.get(emotion_name, 0) + 1
                
                # Draw results
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw identity
                if identity['match']:
                    label = f"{identity['name']}"
                else:
                    label = "Unknown"
                
                cv2.putText(frame, label, (x1, y1-30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Draw emotion
                emotion_label = f"{emotion['emotion']}"
                cv2.putText(frame, emotion_label, (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Write frame to output video
        out.write(frame)
        
        frame_count += 1
        
        # Print progress
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    # Clean up
    cap.release()
    out.release()
    
    # Print emotion statistics
    print("\nEmotion Statistics:")
    total_emotions = sum(emotion_stats.values())
    for emotion, count in sorted(emotion_stats.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_emotions) * 100
        print(f"{emotion}: {count} ({percentage:.1f}%)")
    
    return emotion_stats

# Example usage
if __name__ == "__main__":
    # Process a single image
    result_image, results = process_image("data/test_image.jpg")
    cv2.imwrite("result_image.jpg", result_image)
    print(f"Found {len(results)} faces in the image")
    
    # Process a video
    emotion_stats = process_video("data/test_video.mp4")
    print("Video processing completed")
```

## Example 3: Gesture Recognition for Human-Computer Interaction

### Problem Description
Recognizing hand gestures for controlling computer applications and games.

### Implementation

```python
from aiplatform.vision import GestureRecognizer
from aiplatform.vision.tracking import HandTracker
import cv2
import numpy as np

# Initialize gesture recognizer
gesture_recognizer = GestureRecognizer(
    model="gigachat-gesture-702b",
    confidence_threshold=0.7
)

# Initialize hand tracker
hand_tracker = HandTracker(
    max_hands=2,
    detection_confidence=0.7,
    tracking_confidence=0.5
)

class GestureController:
    def __init__(self):
        self.gesture_history = []
        self.gesture_cooldown = 0
        self.actions = {
            'thumbs_up': self.volume_up,
            'thumbs_down': self.volume_down,
            'peace': self.play_pause,
            'fist': self.stop,
            'open_palm': self.next_track,
            'point_up': self.previous_track,
            'okay': self.mute
        }
    
    def volume_up(self):
        print("Volume Up")
        # In a real application, this would control system volume
        # import pyautogui
        # pyautogui.press('volumeup')
    
    def volume_down(self):
        print("Volume Down")
        # import pyautogui
        # pyautogui.press('volumedown')
    
    def play_pause(self):
        print("Play/Pause")
        # import pyautogui
        # pyautogui.press('space')
    
    def stop(self):
        print("Stop")
        # import pyautogui
        # pyautogui.press('s')
    
    def next_track(self):
        print("Next Track")
        # import pyautogui
        # pyautogui.press('right')
    
    def previous_track(self):
        print("Previous Track")
        # import pyautogui
        # pyautogui.press('left')
    
    def mute(self):
        print("Mute")
        # import pyautogui
        # pyautogui.press('m')
    
    def process_gesture(self, gesture_name):
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            return
        
        if gesture_name in self.actions:
            self.actions[gesture_name]()
            self.gesture_cooldown = 15  # 15 frames cooldown

def process_gesture_stream():
    """Process real-time gesture recognition from webcam."""
    
    # Initialize components
    controller = GestureController()
    cap = cv2.VideoCapture(0)
    
    # Set camera properties for better gesture recognition
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Gesture Recognition Started. Press 'q' to quit.")
    print("Recognized gestures: thumbs_up, thumbs_down, peace, fist, open_palm, point_up, okay")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Detect hands
        hand_landmarks = hand_tracker.process_frame(frame)
        
        # Process each detected hand
        for landmarks in hand_landmarks:
            # Recognize gesture from hand landmarks
            gesture = gesture_recognizer.recognize_gesture(landmarks)
            
            if gesture and gesture['confidence'] > 0.7:
                # Get bounding box for drawing
                h, w, _ = frame.shape
                x_coords = [lm.x for lm in landmarks.landmark]
                y_coords = [lm.y for lm in landmarks.landmark]
                x1, x2 = int(min(x_coords) * w), int(max(x_coords) * w)
                y1, y2 = int(min(y_coords) * h), int(max(y_coords) * h)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw gesture name and confidence
                label = f"{gesture['name']}: {gesture['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Process gesture action
                controller.process_gesture(gesture['name'])
                
                # Draw landmarks
                hand_tracker.draw_landmarks(frame, landmarks)
        
        # Display frame
        cv2.imshow('Gesture Recognition', frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames")

# Example usage
if __name__ == "__main__":
    process_gesture_stream()
```

## Example 4: 3D Computer Vision and SLAM

### Problem Description
Implementing Simultaneous Localization and Mapping (SLAM) with 3D reconstruction capabilities.

### Implementation

```python
from aiplatform.vision import SLAMSystem
from aiplatform.vision.reconstruction import PointCloudReconstructor
import cv2
import numpy as np
import open3d as o3d

class AdvancedSLAM:
    def __init__(self, config=None):
        self.slam_system = SLAMSystem(
            feature_extractor="orb",
            matcher="bf",
            max_features=2000
        )
        
        self.reconstructor = PointCloudReconstructor(
            method="structure_from_motion",
            resolution="high"
        )
        
        self.trajectory = []
        self.point_cloud = o3d.geometry.PointCloud()
        
    def process_video_stream(self, video_path, output_path=None):
        """Process video stream for SLAM and 3D reconstruction."""
        
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        keyframe_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame for keyframes
            if frame_count % 5 == 0:
                # Convert to grayscale for feature detection
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Process frame through SLAM system
                pose, landmarks = self.slam_system.process_frame(
                    gray_frame, frame_count
                )
                
                if pose is not None:
                    # Add pose to trajectory
                    self.trajectory.append(pose)
                    
                    # If this is a keyframe, add to reconstruction
                    if self.slam_system.is_keyframe(frame_count):
                        keyframe_count += 1
                        
                        # Extract 3D points from landmarks
                        points_3d = self.reconstructor.extract_3d_points(
                            frame, landmarks, pose
                        )
                        
                        # Add points to point cloud
                        if len(points_3d) > 0:
                            self.point_cloud.points.extend(
                                o3d.utility.Vector3dVector(points_3d)
                            )
                        
                        print(f"Keyframe {keyframe_count} processed")
                
                # Visualize current frame with features
                if landmarks is not None:
                    for landmark in landmarks:
                        x, y = int(landmark[0]), int(landmark[1])
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
                
                # Draw trajectory info
                cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Keyframes: {keyframe_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('SLAM Processing', frame)
                
                # Save frame if output path specified
                if output_path:
                    cv2.imwrite(f"{output_path}/frame_{frame_count:04d}.jpg", frame)
            
            # Break on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"SLAM processing completed:")
        print(f"Total frames: {frame_count}")
        print(f"Keyframes: {keyframe_count}")
        print(f"Trajectory points: {len(self.trajectory)}")
        print(f"3D points: {len(self.point_cloud.points)}")
        
        return self.trajectory, self.point_cloud
    
    def visualize_3d_reconstruction(self):
        """Visualize the 3D reconstruction results."""
        
        if len(self.point_cloud.points) == 0:
            print("No 3D points to visualize")
            return
        
        # Estimate normals for better visualization
        self.point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )
        
        # Visualize point cloud
        o3d.visualization.draw_geometries([self.point_cloud])
    
    def save_results(self, output_dir):
        """Save SLAM results to files."""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save trajectory
        trajectory_path = os.path.join(output_dir, "trajectory.txt")
        with open(trajectory_path, 'w') as f:
            for i, pose in enumerate(self.trajectory):
                f.write(f"{i} " + " ".join(map(str, pose.flatten())) + "\n")
        
        # Save point cloud
        pointcloud_path = os.path.join(output_dir, "pointcloud.ply")
        o3d.io.write_point_cloud(pointcloud_path, self.point_cloud)
        
        print(f"Results saved to {output_dir}")

# Example usage
def main():
    # Initialize advanced SLAM system
    slam = AdvancedSLAM()
    
    # Process video stream
    video_path = "data/slam_video.mp4"
    output_dir = "slam_results"
    
    trajectory, point_cloud = slam.process_video_stream(
        video_path, output_dir
    )
    
    # Save results
    slam.save_results(output_dir)
    
    # Visualize 3D reconstruction
    slam.visualize_3d_reconstruction()

if __name__ == "__main__":
    main()
```

## Example 5: Quantum-Enhanced Computer Vision

### Problem Description
Using quantum computing to enhance computer vision algorithms for improved performance and accuracy.

### Implementation

```python
from aiplatform.vision import QuantumVisionProcessor
from aiplatform.quantum.ml import QuantumFeatureMap
import cv2
import numpy as np

class QuantumEnhancedVision:
    def __init__(self):
        # Initialize quantum vision processor
        self.quantum_processor = QuantumVisionProcessor(
            backend="simulator",  # or "ibm_nighthawk" for real quantum hardware
            num_qubits=8
        )
        
        # Initialize quantum feature map for enhanced feature extraction
        self.quantum_feature_map = QuantumFeatureMap(
            feature_dimension=64,  # Number of classical features
            num_qubits=12,
            feature_map_type="pauli_expansion",
            entanglement="full"
        )
        
        # Classical computer vision components
        self.classifier = None  # Will be initialized during training
    
    def extract_quantum_features(self, image):
        """Extract quantum-enhanced features from image."""
        
        # Extract classical features first (e.g., HOG, SIFT, etc.)
        classical_features = self.extract_classical_features(image)
        
        # Transform features using quantum feature map
        quantum_features = self.quantum_feature_map.transform(
            classical_features.reshape(1, -1)
        )
        
        return quantum_features.flatten()
    
    def extract_classical_features(self, image):
        """Extract classical computer vision features."""
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Resize to standard size
        resized = cv2.resize(gray, (64, 64))
        
        # Extract HOG features
        hog = cv2.HOGDescriptor((64, 64), (16, 16), (8, 8), (8, 8), 9)
        features = hog.compute(resized)
        
        return features.flatten()
    
    def quantum_optimized_matching(self, template_features, image_features):
        """Use quantum algorithms for optimized feature matching."""
        
        # Use quantum kernel methods for similarity computation
        similarity = self.quantum_processor.compute_quantum_kernel(
            template_features.reshape(1, -1),
            image_features.reshape(1, -1)
        )
        
        return similarity[0, 0]
    
    def quantum_enhanced_object_detection(self, image, templates):
        """Perform object detection using quantum-enhanced features."""
        
        results = []
        
        # Extract quantum features from input image
        image_features = self.extract_quantum_features(image)
        
        # Process each template
        for template_name, template_image in templates.items():
            # Extract quantum features from template
            template_features = self.extract_quantum_features(template_image)
            
            # Compute similarity using quantum optimization
            similarity = self.quantum_optimized_matching(
                template_features, image_features
            )
            
            # Apply quantum threshold optimization
            threshold = self.quantum_processor.optimize_threshold(
                similarity, confidence_level=0.95
            )
            
            if similarity > threshold:
                # Find object location using quantum search
                bbox = self.quantum_processor.quantum_search_object(
                    image, template_image, similarity
                )
                
                results.append({
                    'class': template_name,
                    'confidence': float(similarity),
                    'bbox': bbox,
                    'quantum_advantage': True
                })
        
        return results
    
    def train_quantum_classifier(self, training_data, labels):
        """Train a quantum-enhanced classifier."""
        
        # Extract quantum features for all training samples
        quantum_features = []
        for image in training_data:
            features = self.extract_quantum_features(image)
            quantum_features.append(features)
        
        quantum_features = np.array(quantum_features)
        
        # Train quantum classifier
        self.classifier = self.quantum_processor.train_quantum_classifier(
            quantum_features, labels,
            algorithm="variational_classifier",
            num_epochs=100
        )
        
        return self.classifier
    
    def predict_quantum_class(self, image):
        """Predict class using quantum-enhanced classifier."""
        
        if self.classifier is None:
            raise ValueError("Classifier not trained yet")
        
        # Extract quantum features
        features = self.extract_quantum_features(image)
        
        # Make prediction using quantum classifier
        prediction = self.classifier.predict(
            features.reshape(1, -1)
        )
        
        # Get quantum confidence
        confidence = self.classifier.get_quantum_confidence(
            features.reshape(1, -1)
        )
        
        return {
            'class': prediction[0],
            'confidence': float(confidence),
            'quantum_metrics': self.classifier.get_quantum_metrics()
        }

# Example usage
def main():
    # Initialize quantum-enhanced vision system
    q_vision = QuantumEnhancedVision()
    
    # Load templates for object detection
    templates = {
        'cat': cv2.imread('templates/cat.jpg'),
        'dog': cv2.imread('templates/dog.jpg'),
        'car': cv2.imread('templates/car.jpg')
    }
    
    # Process test image
    test_image = cv2.imread('test_images/test_scene.jpg')
    
    # Perform quantum-enhanced object detection
    detections = q_vision.quantum_enhanced_object_detection(
        test_image, templates
    )
    
    print("Quantum-Enhanced Object Detections:")
    for detection in detections:
        print(f"  {detection['class']}: {detection['confidence']:.4f}")
        print(f"    Bounding Box: {detection['bbox']}")
        print(f"    Quantum Advantage: {detection['quantum_advantage']}")
    
    # Example of training quantum classifier
    # (This would typically use a larger dataset)
    training_images = [
        cv2.imread('training/cat_001.jpg'),
        cv2.imread('training/dog_001.jpg'),
        cv2.imread('training/car_001.jpg')
    ]
    training_labels = ['cat', 'dog', 'car']
    
    # Train quantum classifier
    classifier = q_vision.train_quantum_classifier(
        training_images, training_labels
    )
    
    # Make quantum prediction
    test_sample = cv2.imread('test_images/unknown_object.jpg')
    prediction = q_vision.predict_quantum_class(test_sample)
    
    print(f"\nQuantum Classification Result:")
    print(f"  Predicted Class: {prediction['class']}")
    print(f"  Confidence: {prediction['confidence']:.4f}")
    print(f"  Quantum Metrics: {prediction['quantum_metrics']}")

if __name__ == "__main__":
    main()
```

## Conclusion

These vision lab examples demonstrate the comprehensive capabilities of the REChain AI & Vision Lab module, including:

1. **Real-time object detection and tracking** with advanced models
2. **Facial recognition and emotion analysis** for human-computer interaction
3. **Gesture recognition** for intuitive control interfaces
4. **3D computer vision and SLAM** for spatial understanding
5. **Quantum-enhanced computer vision** for improved performance and accuracy

Each example showcases different aspects of modern computer vision technology, from basic object detection to advanced quantum-enhanced processing. The modular design of the SDK allows developers to easily integrate these capabilities into their applications while leveraging the power of quantum computing for enhanced performance.