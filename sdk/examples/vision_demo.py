"""
Vision Demo Example
===================

Demonstrates computer vision capabilities including
object detection, face recognition, gesture recognition, and SLAM.
"""

import asyncio
import numpy as np

# SDK imports
from sdk.vision import (
    ObjectDetector,
    FaceRecognizer,
    GestureRecognizer,
    VideoStreamProcessor,
    Vision3DEngine,
    SLAMProcessor,
    WebXRIntegration
)


def object_detection_demo():
    """
    Demo: Object detection with YOLO.
    """
    print("=" * 60)
    print("Object Detection Demo")
    print("=" * 60)
    
    # Create detector
    detector = ObjectDetector(model="yolov8", confidence_threshold=0.5)
    
    # Create sample image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect objects
    result = detector.detect(image)
    
    print(f"\nImage size: {result.image_size}")
    print(f"Model: {result.model_name}")
    print(f"Inference time: {result.inference_time_ms:.2f}ms")
    print(f"Detections found: {len(result.detections)}")
    
    for det in result.detections[:5]:
        print(f"  - {det.class_name}: {det.confidence:.2f} at {det.bbox}")


def face_recognition_demo():
    """
    Demo: Face detection and recognition.
    """
    print("\n" + "=" * 60)
    print("Face Recognition Demo")
    print("=" * 60)
    
    # Create recognizer
    recognizer = FaceRecognizer()
    
    # Create sample image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Detect faces
    faces = recognizer.detect_faces(image)
    
    print(f"\nFaces detected: {len(faces)}")
    
    for i, face in enumerate(faces):
        print(f"\nFace {i + 1}:")
        print(f"  Bounding box: {face.bbox}")
        print(f"  Confidence: {face.confidence:.2f}")
        
        if face.landmarks:
            print(f"  Landmarks: {list(face.landmarks.keys())}")
        
        # Analyze face
        analysis = recognizer.analyze_face(image, face)
        print(f"  Age: {analysis.age}")
        print(f"  Gender: {analysis.gender}")
        print(f"  Emotion: {analysis.emotion}")
        
        # Register face
        if face.embedding is not None:
            recognizer.register_face(f"person_{i}", face.embedding, {"name": f"Person {i}"})
    
    print(f"\nRegistered faces: {recognizer.get_registered_faces()}")


def gesture_recognition_demo():
    """
    Demo: Hand gesture recognition.
    """
    print("\n" + "=" * 60)
    print("Gesture Recognition Demo")
    print("=" * 60)
    
    # Create recognizer
    recognizer = GestureRecognizer()
    
    # Create sample frames
    for frame_num in range(5):
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Detect gestures
        gestures = recognizer.detect_gestures(image)
        
        print(f"\nFrame {frame_num + 1}:")
        for g in gestures:
            print(f"  Gesture: {g.gesture_name} ({g.confidence:.2f})")
            recognizer.track_gesture(g)
    
    # Get gesture sequence
    sequence = recognizer.get_gesture_sequence()
    print(f"\nGesture sequence: {sequence}")


def video_processing_demo():
    """
    Demo: Video stream processing.
    """
    print("\n" + "=" * 60)
    print("Video Stream Processing Demo")
    print("=" * 60)
    
    # Create processor
    processor = VideoStreamProcessor(target_fps=30)
    
    # Add object detector to pipeline
    detector = ObjectDetector()
    processor.add_pipeline(lambda frame: detector.detect(frame))
    
    print(f"Pipeline processors: {len(processor._pipeline)}")
    print(f"Target FPS: {processor.target_fps}")
    
    # Simulate processing a few frames
    print("\nProcessing simulated frames...")
    
    for i in range(5):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        result = processor._process_frame(frame)
        
        print(f"  Frame {i + 1}: {len(result.detections)} detections, "
              f"{result.processing_time_ms:.2f}ms")
    
    print(f"\nStatistics: {processor.get_statistics()}")


def vision_3d_demo():
    """
    Demo: 3D vision and depth estimation.
    """
    print("\n" + "=" * 60)
    print("3D Vision Demo")
    print("=" * 60)
    
    # Create 3D engine
    engine = Vision3DEngine()
    
    # Create sample image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Estimate depth
    print("\nEstimating depth...")
    depth = engine.estimate_depth(image)
    
    print(f"Depth map shape: {depth.shape}")
    print(f"Depth range: {depth.min():.2f} - {depth.max():.2f} meters")
    
    # Generate point cloud
    print("\nGenerating point cloud...")
    cloud = engine.generate_point_cloud(image, depth)
    
    print(f"Point cloud size: {len(cloud.points)} points")
    
    if cloud.colors is not None:
        print(f"Colors available: Yes")
    
    # Compute normals
    print("\nComputing normals...")
    cloud_with_normals = engine.compute_normals(cloud, k_neighbors=5)
    
    if cloud_with_normals.normals is not None:
        print(f"Normals computed: {len(cloud_with_normals.normals)}")


def slam_demo():
    """
    Demo: Visual SLAM.
    """
    print("\n" + "=" * 60)
    print("Visual SLAM Demo")
    print("=" * 60)
    
    # Create SLAM processor
    slam = SLAMProcessor(method="orb")
    
    # Initialize with first frame
    first_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\nInitializing SLAM...")
    success = slam.initialize(first_frame)
    print(f"Initialization: {'Success' if success else 'Failed'}")
    
    # Process frames
    print("\nProcessing frames...")
    for i in range(10):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        pose = slam.process_frame(frame, timestamp=i * 0.033)
        
        if pose:
            print(f"  Frame {i + 1}: Position = [{pose.position[0]:.3f}, "
                  f"{pose.position[1]:.3f}, {pose.position[2]:.3f}]")
    
    # Get trajectory
    trajectory = slam.get_trajectory()
    print(f"\nTrajectory points: {len(trajectory)}")
    
    # Get map points
    map_points = slam.get_map_points()
    print(f"Map points: {len(map_points)}")
    
    print(f"\nSLAM statistics: {slam.get_statistics()}")


async def webxr_demo():
    """
    Demo: WebXR integration.
    """
    print("\n" + "=" * 60)
    print("WebXR Integration Demo")
    print("=" * 60)
    
    from sdk.vision.webxr import XRSessionMode, XRReferenceSpace
    
    # Create WebXR integration
    xr = WebXRIntegration()
    
    # Check support
    ar_supported = await xr.is_supported(XRSessionMode.IMMERSIVE_AR)
    vr_supported = await xr.is_supported(XRSessionMode.IMMERSIVE_VR)
    
    print(f"\nAR supported: {ar_supported}")
    print(f"VR supported: {vr_supported}")
    
    # Start AR session
    print("\nStarting AR session...")
    success = await xr.request_session(XRSessionMode.IMMERSIVE_AR)
    print(f"Session started: {success}")
    
    # Get viewer pose
    pose = xr.get_viewer_pose()
    if pose:
        print(f"Viewer position: {pose.position}")
    
    # Create anchor
    from sdk.vision.webxr import XRPose
    anchor_pose = XRPose(
        position=np.array([0, 0, -1]),
        orientation=np.array([0, 0, 0, 1]),
        timestamp=0
    )
    
    anchor = await xr.create_anchor(anchor_pose, {"type": "marker"})
    print(f"Created anchor: {anchor.anchor_id}")
    
    # Get input sources
    inputs = xr.get_input_sources()
    print(f"Input sources: {len(inputs)}")
    
    # End session
    await xr.end_session()
    print("Session ended")
    
    print(f"\nXR statistics: {xr.get_statistics()}")


async def main():
    """Run all vision demos."""
    print("\n" + "=" * 60)
    print("AIPlatform SDK - Vision Demo")
    print("=" * 60)
    
    object_detection_demo()
    face_recognition_demo()
    gesture_recognition_demo()
    video_processing_demo()
    vision_3d_demo()
    slam_demo()
    await webxr_demo()
    
    print("\n" + "=" * 60)
    print("All vision demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
