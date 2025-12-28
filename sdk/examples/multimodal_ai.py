"""
Multimodal AI Example
=====================

Demonstrates multimodal processing with text, audio, video, and 3D data.
"""

import asyncio
import numpy as np

# SDK imports
from sdk.multimodal import (
    MultimodalProcessor,
    TextProcessor,
    AudioProcessor,
    VideoProcessor,
    Spatial3DProcessor,
    GigaChat3Client
)
from sdk.multimodal.processor import ModalityType, ModalityInput


def text_processing_demo():
    """
    Demo: Advanced text processing.
    """
    print("=" * 60)
    print("Text Processing Demo")
    print("=" * 60)
    
    processor = TextProcessor()
    
    # Analyze text
    text = """
    The AIPlatform SDK represents a breakthrough in quantum-AI integration.
    This revolutionary technology combines IBM Quantum computing with 
    advanced machine learning to solve previously intractable problems.
    The system supports multiple languages and provides excellent performance.
    """
    
    print("\nAnalyzing text...")
    result = processor.analyze(text)
    
    print(f"\nLanguage: {result.language}")
    print(f"Sentiment: {result.sentiment} ({result.sentiment_score:.2f})")
    print(f"Keywords: {result.keywords[:5]}")
    print(f"Entities: {len(result.entities)}")
    
    for entity in result.entities[:3]:
        print(f"  - {entity['type']}: {entity['text']}")
    
    if result.summary:
        print(f"\nSummary: {result.summary[:200]}...")
    
    # Similarity comparison
    text2 = "Quantum computing and AI are transforming technology."
    similarity = processor.similarity(text, text2)
    print(f"\nSimilarity with related text: {similarity:.4f}")


def audio_processing_demo():
    """
    Demo: Audio processing and speech recognition.
    """
    print("\n" + "=" * 60)
    print("Audio Processing Demo")
    print("=" * 60)
    
    processor = AudioProcessor(model="katya-speech")
    
    # Create sample audio (5 seconds at 16kHz)
    sample_rate = 16000
    duration = 5
    audio = np.random.randn(sample_rate * duration).astype(np.float32) * 0.1
    
    print("\nAnalyzing audio...")
    result = processor.analyze(audio, sample_rate)
    
    print(f"\nDuration: {result.duration_seconds:.2f}s")
    print(f"Sample rate: {result.sample_rate}Hz")
    print(f"Channels: {result.channels}")
    print(f"Language: {result.language}")
    print(f"Speakers: {result.speaker_count}")
    
    if result.transcript:
        print(f"Transcript: {result.transcript}")
    
    # Transcribe with timestamps
    print("\nTranscribing with timestamps...")
    segments = processor.transcribe_with_timestamps(audio, sample_rate)
    
    for seg in segments[:3]:
        print(f"  [{seg.start_time:.2f}s - {seg.end_time:.2f}s] "
              f"{seg.speaker_id}: {seg.text}")
    
    # Extract features
    print("\nExtracting audio features...")
    features = processor.extract_features(audio, sample_rate)
    
    for name, feat in features.items():
        print(f"  {name}: shape {feat.shape}")


def video_processing_demo():
    """
    Demo: Video analysis.
    """
    print("\n" + "=" * 60)
    print("Video Processing Demo")
    print("=" * 60)
    
    processor = VideoProcessor()
    
    # Create sample video frames (30 frames at 30fps = 1 second)
    fps = 30
    num_frames = 30
    frames = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(num_frames)
    ]
    
    print("\nAnalyzing video...")
    result = processor.analyze(frames, fps)
    
    print(f"\nDuration: {result.duration_seconds:.2f}s")
    print(f"FPS: {result.fps}")
    print(f"Resolution: {result.resolution}")
    print(f"Frame count: {result.frame_count}")
    print(f"Scene changes: {len(result.scene_changes)}")
    print(f"Actions detected: {len(result.actions)}")
    
    for action in result.actions[:3]:
        print(f"  - {action['action']} ({action['start_time']:.2f}s - "
              f"{action['end_time']:.2f}s): {action['confidence']:.2f}")
    
    print(f"\nSummary: {result.summary}")
    
    # Extract keyframes
    keyframes = processor.extract_keyframes(frames, num_keyframes=5)
    print(f"\nExtracted {len(keyframes)} keyframes")
    
    # Get scenes
    scenes = processor.get_scenes(frames, fps)
    print(f"Detected {len(scenes)} scenes")


def spatial_3d_demo():
    """
    Demo: 3D spatial processing.
    """
    print("\n" + "=" * 60)
    print("Spatial 3D Processing Demo")
    print("=" * 60)
    
    processor = Spatial3DProcessor()
    
    # Create sample point cloud
    num_points = 5000
    points = np.random.randn(num_points, 3).astype(np.float32)
    points[:, 2] = np.abs(points[:, 2])  # Make z positive
    
    colors = np.random.randint(0, 255, (num_points, 3), dtype=np.uint8)
    
    print("\nAnalyzing 3D scene...")
    scene = processor.analyze_scene(points, colors)
    
    print(f"\nScene bounds: {scene.bounds[0]} to {scene.bounds[1]}")
    print(f"Scene center: {scene.center}")
    print(f"Objects detected: {len(scene.objects)}")
    
    for obj in scene.objects[:5]:
        print(f"  - {obj.class_name} at {obj.position} "
              f"(confidence: {obj.confidence:.2f})")
    
    # Compute spatial relations
    print("\nComputing spatial relations...")
    relations = processor.compute_spatial_relations(scene.objects[:5])
    
    for rel in relations[:5]:
        print(f"  {rel['subject']} is {rel['relation']} {rel['object']} "
              f"(distance: {rel['distance']:.2f})")
    
    # Segment scene
    print("\nSegmenting scene...")
    segments = processor.segment_scene(points)
    
    for name, indices in segments.items():
        print(f"  {name}: {len(indices)} points")


async def gigachat_demo():
    """
    Demo: GigaChat3-702B multimodal AI.
    """
    print("\n" + "=" * 60)
    print("GigaChat3-702B Demo")
    print("=" * 60)
    
    from sdk.multimodal.gigachat import GigaChatMessage, GigaChatRole
    
    client = GigaChat3Client()
    client.set_system_prompt(
        "You are a helpful AI assistant specialized in quantum computing and AI."
    )
    
    # Text chat
    print("\nText conversation:")
    
    messages = [
        GigaChatMessage(GigaChatRole.USER, "Hello! What can you help me with?")
    ]
    
    response = await client.chat(messages)
    print(f"User: {messages[0].content}")
    print(f"GigaChat: {response.content}")
    print(f"Latency: {response.latency_ms:.2f}ms")
    
    # Image understanding
    print("\n\nImage understanding:")
    
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    messages = [
        GigaChatMessage(
            GigaChatRole.USER,
            "What do you see in this image?",
            images=[image]
        )
    ]
    
    response = await client.chat(messages)
    print(f"User: [Image] What do you see?")
    print(f"GigaChat: {response.content}")
    
    # Get embeddings
    print("\n\nGenerating embeddings:")
    
    text_emb = await client.embed("Quantum computing is fascinating")
    image_emb = await client.embed_image(image)
    
    print(f"Text embedding shape: {text_emb.shape}")
    print(f"Image embedding shape: {image_emb.shape}")
    
    # Usage stats
    print(f"\nUsage stats: {client.get_usage_stats()}")


async def multimodal_fusion_demo():
    """
    Demo: Multimodal fusion processing.
    """
    print("\n" + "=" * 60)
    print("Multimodal Fusion Demo")
    print("=" * 60)
    
    processor = MultimodalProcessor(
        model="gigachat3-702b",
        fusion_method="attention"
    )
    
    # Create multimodal inputs
    text = "Describe what you see and hear in this scene"
    image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    audio = np.random.randn(16000).astype(np.float32)  # 1 second
    
    inputs = [
        ModalityInput(ModalityType.TEXT, text),
        ModalityInput(ModalityType.IMAGE, image),
        ModalityInput(ModalityType.AUDIO, audio)
    ]
    
    print("\nProcessing multimodal inputs...")
    print(f"  - Text: '{text[:50]}...'")
    print(f"  - Image: {image.shape}")
    print(f"  - Audio: {len(audio)} samples")
    
    result = processor.process(inputs)
    
    print(f"\nResults:")
    print(f"  Modalities processed: {list(result.embeddings.keys())}")
    print(f"  Fused embedding shape: {result.fused_embedding.shape}")
    print(f"  Confidence: {result.confidence:.4f}")
    print(f"  Processing time: {result.processing_time_ms:.2f}ms")
    
    if result.predictions:
        print(f"\nPredictions:")
        for key, value in result.predictions.items():
            if isinstance(value, str):
                print(f"  {key}: {value[:100]}...")
            else:
                print(f"  {key}: {value}")


async def main():
    """Run all multimodal demos."""
    print("\n" + "=" * 60)
    print("AIPlatform SDK - Multimodal AI Demo")
    print("=" * 60)
    
    text_processing_demo()
    audio_processing_demo()
    video_processing_demo()
    spatial_3d_demo()
    await gigachat_demo()
    await multimodal_fusion_demo()
    
    print("\n" + "=" * 60)
    print("All multimodal demos completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
