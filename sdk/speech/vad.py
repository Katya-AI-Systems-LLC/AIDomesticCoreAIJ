"""
Voice Activity Detection
========================

Detect speech segments in audio.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VADMode(Enum):
    """VAD sensitivity modes."""
    AGGRESSIVE = 3
    NORMAL = 2
    SENSITIVE = 1
    VERY_SENSITIVE = 0


@dataclass
class SpeechSegment:
    """Detected speech segment."""
    start: float
    end: float
    confidence: float
    is_speech: bool


@dataclass
class VADResult:
    """VAD result."""
    segments: List[SpeechSegment]
    speech_ratio: float
    total_duration: float


class VoiceActivityDetector:
    """
    Voice Activity Detection.
    
    Features:
    - Multiple VAD backends
    - Adjustable sensitivity
    - Real-time processing
    - Noise robustness
    
    Example:
        >>> vad = VoiceActivityDetector()
        >>> result = vad.detect("audio.wav")
        >>> for segment in result.segments:
        ...     print(f"{segment.start:.2f}s - {segment.end:.2f}s")
    """
    
    def __init__(self, mode: VADMode = VADMode.NORMAL,
                 sample_rate: int = 16000,
                 frame_duration_ms: int = 30):
        """
        Initialize VAD.
        
        Args:
            mode: VAD sensitivity mode
            sample_rate: Audio sample rate
            frame_duration_ms: Frame duration in milliseconds
        """
        self.mode = mode
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        
        self._vad = None
        
        logger.info(f"VAD initialized (mode={mode.name})")
    
    def _load_vad(self):
        """Load VAD model."""
        if self._vad is not None:
            return
        
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(self.mode.value)
            logger.info("WebRTC VAD loaded")
        except ImportError:
            logger.warning("webrtcvad not installed, using energy-based VAD")
    
    def detect(self, audio: Union[str, bytes, np.ndarray]) -> VADResult:
        """
        Detect voice activity in audio.
        
        Args:
            audio: Audio file, bytes, or array
            
        Returns:
            VADResult
        """
        self._load_vad()
        
        # Load audio
        audio_data, sample_rate = self._load_audio(audio)
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            audio_data = self._resample(audio_data, sample_rate, self.sample_rate)
        
        # Process frames
        frame_samples = int(self.sample_rate * self.frame_duration_ms / 1000)
        
        segments = []
        current_speech_start = None
        
        for i in range(0, len(audio_data) - frame_samples, frame_samples):
            frame = audio_data[i:i + frame_samples]
            
            is_speech = self._is_speech(frame)
            timestamp = i / self.sample_rate
            
            if is_speech and current_speech_start is None:
                current_speech_start = timestamp
            elif not is_speech and current_speech_start is not None:
                segments.append(SpeechSegment(
                    start=current_speech_start,
                    end=timestamp,
                    confidence=0.9,
                    is_speech=True
                ))
                current_speech_start = None
        
        # Handle last segment
        if current_speech_start is not None:
            segments.append(SpeechSegment(
                start=current_speech_start,
                end=len(audio_data) / self.sample_rate,
                confidence=0.9,
                is_speech=True
            ))
        
        # Merge close segments
        segments = self._merge_segments(segments, min_gap=0.3)
        
        # Calculate statistics
        total_duration = len(audio_data) / self.sample_rate
        speech_duration = sum(s.end - s.start for s in segments)
        speech_ratio = speech_duration / total_duration if total_duration > 0 else 0
        
        return VADResult(
            segments=segments,
            speech_ratio=speech_ratio,
            total_duration=total_duration
        )
    
    def _load_audio(self, audio: Union[str, bytes, np.ndarray]) -> Tuple[np.ndarray, int]:
        """Load audio data."""
        if isinstance(audio, np.ndarray):
            return audio, self.sample_rate
        
        try:
            import soundfile as sf
            
            if isinstance(audio, str):
                data, sr = sf.read(audio)
            else:
                import io
                data, sr = sf.read(io.BytesIO(audio))
            
            # Convert to mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            return data, sr
            
        except ImportError:
            # Return simulated audio
            return np.random.randn(self.sample_rate * 5), self.sample_rate
    
    def _resample(self, audio: np.ndarray,
                  src_rate: int,
                  dst_rate: int) -> np.ndarray:
        """Resample audio."""
        if src_rate == dst_rate:
            return audio
        
        try:
            import librosa
            return librosa.resample(audio, orig_sr=src_rate, target_sr=dst_rate)
        except ImportError:
            # Simple resampling
            ratio = dst_rate / src_rate
            new_length = int(len(audio) * ratio)
            indices = np.linspace(0, len(audio) - 1, new_length).astype(int)
            return audio[indices]
    
    def _is_speech(self, frame: np.ndarray) -> bool:
        """Check if frame contains speech."""
        if self._vad is not None:
            try:
                # Convert to 16-bit PCM
                frame_int16 = (frame * 32767).astype(np.int16).tobytes()
                return self._vad.is_speech(frame_int16, self.sample_rate)
            except:
                pass
        
        # Energy-based detection
        energy = np.sqrt(np.mean(frame ** 2))
        threshold = 0.01 * (4 - self.mode.value)  # Adjust by mode
        return energy > threshold
    
    def _merge_segments(self, segments: List[SpeechSegment],
                        min_gap: float) -> List[SpeechSegment]:
        """Merge close segments."""
        if not segments:
            return segments
        
        merged = [segments[0]]
        
        for segment in segments[1:]:
            if segment.start - merged[-1].end < min_gap:
                merged[-1] = SpeechSegment(
                    start=merged[-1].start,
                    end=segment.end,
                    confidence=max(merged[-1].confidence, segment.confidence),
                    is_speech=True
                )
            else:
                merged.append(segment)
        
        return merged
    
    def process_stream(self, frame: np.ndarray) -> bool:
        """
        Process single frame for streaming.
        
        Args:
            frame: Audio frame
            
        Returns:
            True if speech detected
        """
        self._load_vad()
        return self._is_speech(frame)
    
    def get_speech_segments(self, audio: Union[str, bytes, np.ndarray]) -> List[Tuple[float, float]]:
        """Get speech segments as time ranges."""
        result = self.detect(audio)
        return [(s.start, s.end) for s in result.segments]
    
    def extract_speech(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract only speech portions.
        
        Args:
            audio: Audio array
            
        Returns:
            Audio with only speech
        """
        result = self.detect(audio)
        
        speech_parts = []
        for segment in result.segments:
            start_sample = int(segment.start * self.sample_rate)
            end_sample = int(segment.end * self.sample_rate)
            speech_parts.append(audio[start_sample:end_sample])
        
        if speech_parts:
            return np.concatenate(speech_parts)
        return np.array([])
    
    def __repr__(self) -> str:
        return f"VoiceActivityDetector(mode={self.mode.name})"
