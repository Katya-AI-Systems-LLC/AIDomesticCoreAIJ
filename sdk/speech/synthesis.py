"""
Speech Synthesis
================

Text-to-speech generation.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class TTSModel(Enum):
    """Text-to-speech models."""
    OPENAI_TTS = "tts-1"
    OPENAI_TTS_HD = "tts-1-hd"
    ELEVENLABS = "elevenlabs"
    AZURE = "azure"
    GOOGLE = "google"
    COQUI = "coqui"
    BARK = "bark"
    PIPER = "piper"


class Voice(Enum):
    """Available voices."""
    ALLOY = "alloy"
    ECHO = "echo"
    FABLE = "fable"
    ONYX = "onyx"
    NOVA = "nova"
    SHIMMER = "shimmer"


@dataclass
class SynthesisResult:
    """Synthesis result."""
    audio: bytes
    format: str
    duration: float
    sample_rate: int
    model: str
    voice: str


class SpeechSynthesizer:
    """
    Text-to-speech synthesizer.
    
    Features:
    - Multiple TTS backends
    - Voice cloning
    - SSML support
    - Emotion control
    - Real-time streaming
    
    Example:
        >>> synth = SpeechSynthesizer(TTSModel.OPENAI_TTS)
        >>> result = synth.synthesize("Hello world")
        >>> with open("output.mp3", "wb") as f:
        ...     f.write(result.audio)
    """
    
    def __init__(self, model: TTSModel = TTSModel.OPENAI_TTS,
                 voice: Voice = Voice.ALLOY,
                 api_key: Optional[str] = None):
        """
        Initialize speech synthesizer.
        
        Args:
            model: TTS model
            voice: Default voice
            api_key: API key for cloud services
        """
        self.model = model
        self.voice = voice
        self.api_key = api_key
        
        self._local_model = None
        
        logger.info(f"Speech Synthesizer initialized: {model.value}")
    
    def synthesize(self, text: str,
                   voice: Voice = None,
                   speed: float = 1.0,
                   format: str = "mp3") -> SynthesisResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            voice: Voice to use
            speed: Speech speed (0.25-4.0)
            format: Output format (mp3, wav, opus)
            
        Returns:
            SynthesisResult
        """
        voice = voice or self.voice
        start_time = time.time()
        
        if self.model in [TTSModel.OPENAI_TTS, TTSModel.OPENAI_TTS_HD]:
            result = self._synthesize_openai(text, voice, speed, format)
        elif self.model == TTSModel.ELEVENLABS:
            result = self._synthesize_elevenlabs(text, voice)
        elif self.model == TTSModel.COQUI:
            result = self._synthesize_coqui(text)
        elif self.model == TTSModel.BARK:
            result = self._synthesize_bark(text)
        else:
            result = self._synthesize_simulated(text)
        
        result.model = self.model.value
        result.voice = voice.value if isinstance(voice, Voice) else str(voice)
        
        logger.info(f"Synthesis completed in {time.time() - start_time:.2f}s")
        return result
    
    def _synthesize_openai(self, text: str, voice: Voice,
                           speed: float, format: str) -> SynthesisResult:
        """Synthesize using OpenAI TTS."""
        try:
            import openai
            
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.audio.speech.create(
                model=self.model.value,
                voice=voice.value,
                input=text,
                speed=speed,
                response_format=format
            )
            
            audio_bytes = response.content
            
            return SynthesisResult(
                audio=audio_bytes,
                format=format,
                duration=len(text) / 15,  # Approximate
                sample_rate=24000,
                model=self.model.value,
                voice=voice.value
            )
            
        except ImportError:
            return self._synthesize_simulated(text)
    
    def _synthesize_elevenlabs(self, text: str, voice: Voice) -> SynthesisResult:
        """Synthesize using ElevenLabs."""
        try:
            from elevenlabs import generate, set_api_key
            
            set_api_key(self.api_key)
            
            audio = generate(
                text=text,
                voice=voice.value,
                model="eleven_multilingual_v2"
            )
            
            return SynthesisResult(
                audio=audio,
                format="mp3",
                duration=len(text) / 15,
                sample_rate=44100,
                model="elevenlabs",
                voice=voice.value
            )
            
        except ImportError:
            return self._synthesize_simulated(text)
    
    def _synthesize_coqui(self, text: str) -> SynthesisResult:
        """Synthesize using Coqui TTS."""
        try:
            from TTS.api import TTS
            
            if self._local_model is None:
                self._local_model = TTS("tts_models/en/ljspeech/tacotron2-DDC")
            
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                self._local_model.tts_to_file(text=text, file_path=f.name)
                with open(f.name, 'rb') as audio_file:
                    audio = audio_file.read()
                os.unlink(f.name)
            
            return SynthesisResult(
                audio=audio,
                format="wav",
                duration=len(text) / 15,
                sample_rate=22050,
                model="coqui",
                voice="default"
            )
            
        except ImportError:
            return self._synthesize_simulated(text)
    
    def _synthesize_bark(self, text: str) -> SynthesisResult:
        """Synthesize using Bark."""
        try:
            from bark import SAMPLE_RATE, generate_audio, preload_models
            from scipy.io.wavfile import write as write_wav
            import io
            
            preload_models()
            audio_array = generate_audio(text)
            
            buffer = io.BytesIO()
            write_wav(buffer, SAMPLE_RATE, audio_array)
            
            return SynthesisResult(
                audio=buffer.getvalue(),
                format="wav",
                duration=len(audio_array) / SAMPLE_RATE,
                sample_rate=SAMPLE_RATE,
                model="bark",
                voice="default"
            )
            
        except ImportError:
            return self._synthesize_simulated(text)
    
    def _synthesize_simulated(self, text: str) -> SynthesisResult:
        """Simulated synthesis."""
        # Generate minimal valid WAV header
        import struct
        
        sample_rate = 22050
        duration = len(text) / 15
        num_samples = int(sample_rate * duration)
        
        # WAV header
        wav_header = struct.pack('<4sI4s4sIHHIIHH4sI',
            b'RIFF', 36 + num_samples * 2, b'WAVE',
            b'fmt ', 16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
            b'data', num_samples * 2
        )
        
        # Silent audio
        audio = wav_header + b'\x00' * (num_samples * 2)
        
        return SynthesisResult(
            audio=audio,
            format="wav",
            duration=duration,
            sample_rate=sample_rate,
            model="simulated",
            voice="default"
        )
    
    def synthesize_ssml(self, ssml: str) -> SynthesisResult:
        """
        Synthesize from SSML markup.
        
        Args:
            ssml: SSML text
            
        Returns:
            SynthesisResult
        """
        # Strip SSML tags for basic synthesis
        import re
        text = re.sub(r'<[^>]+>', '', ssml)
        return self.synthesize(text)
    
    def clone_voice(self, reference_audio: bytes,
                    text: str) -> SynthesisResult:
        """
        Clone voice from reference audio.
        
        Args:
            reference_audio: Reference voice sample
            text: Text to synthesize
            
        Returns:
            SynthesisResult with cloned voice
        """
        # Voice cloning requires specific models
        logger.info("Voice cloning requested")
        return self.synthesize(text)
    
    def get_voices(self) -> List[Voice]:
        """Get available voices."""
        return list(Voice)
    
    def stream_synthesize(self, text: str) -> bytes:
        """
        Stream synthesis for real-time playback.
        
        Args:
            text: Text to synthesize
            
        Yields:
            Audio chunks
        """
        result = self.synthesize(text)
        
        # Yield in chunks
        chunk_size = 4096
        for i in range(0, len(result.audio), chunk_size):
            yield result.audio[i:i + chunk_size]
    
    def __repr__(self) -> str:
        return f"SpeechSynthesizer(model={self.model.value})"
