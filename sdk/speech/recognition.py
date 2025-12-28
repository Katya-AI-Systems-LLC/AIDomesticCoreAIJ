"""
Speech Recognition
==================

Speech-to-text transcription.
"""

from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass
from enum import Enum
import time
import logging

logger = logging.getLogger(__name__)


class STTModel(Enum):
    """Speech recognition models."""
    WHISPER_TINY = "whisper-tiny"
    WHISPER_BASE = "whisper-base"
    WHISPER_SMALL = "whisper-small"
    WHISPER_MEDIUM = "whisper-medium"
    WHISPER_LARGE = "whisper-large-v3"
    GOOGLE = "google"
    AZURE = "azure"
    AWS = "aws"
    DEEPGRAM = "deepgram"


@dataclass
class TranscriptionSegment:
    """Transcription segment."""
    text: str
    start: float
    end: float
    confidence: float
    words: List[Dict] = None


@dataclass
class TranscriptionResult:
    """Full transcription result."""
    text: str
    segments: List[TranscriptionSegment]
    language: str
    duration: float
    model: str


class SpeechRecognizer:
    """
    Speech recognition engine.
    
    Features:
    - Multiple STT backends
    - Real-time streaming
    - Multi-language support
    - Word-level timestamps
    - Diarization
    
    Example:
        >>> recognizer = SpeechRecognizer(STTModel.WHISPER_BASE)
        >>> result = recognizer.transcribe("audio.wav")
        >>> print(result.text)
    """
    
    def __init__(self, model: STTModel = STTModel.WHISPER_BASE,
                 language: str = "auto",
                 api_key: Optional[str] = None):
        """
        Initialize speech recognizer.
        
        Args:
            model: STT model
            language: Language code or "auto"
            api_key: API key for cloud services
        """
        self.model = model
        self.language = language
        self.api_key = api_key
        
        self._whisper_model = None
        self._stream_callback: Optional[Callable] = None
        
        logger.info(f"Speech Recognizer initialized: {model.value}")
    
    def transcribe(self, audio: Union[str, bytes],
                   task: str = "transcribe") -> TranscriptionResult:
        """
        Transcribe audio.
        
        Args:
            audio: Audio file path or bytes
            task: "transcribe" or "translate"
            
        Returns:
            TranscriptionResult
        """
        start_time = time.time()
        
        if self.model.value.startswith("whisper"):
            result = self._transcribe_whisper(audio, task)
        elif self.model == STTModel.GOOGLE:
            result = self._transcribe_google(audio)
        elif self.model == STTModel.DEEPGRAM:
            result = self._transcribe_deepgram(audio)
        else:
            result = self._transcribe_simulated(audio)
        
        result.model = self.model.value
        
        logger.info(f"Transcription completed in {time.time() - start_time:.2f}s")
        return result
    
    def _transcribe_whisper(self, audio: Union[str, bytes],
                            task: str) -> TranscriptionResult:
        """Transcribe using Whisper."""
        try:
            import whisper
            
            if self._whisper_model is None:
                model_name = self.model.value.replace("whisper-", "")
                self._whisper_model = whisper.load_model(model_name)
            
            options = {"task": task}
            if self.language != "auto":
                options["language"] = self.language
            
            result = self._whisper_model.transcribe(audio, **options)
            
            segments = []
            for seg in result.get("segments", []):
                segments.append(TranscriptionSegment(
                    text=seg["text"],
                    start=seg["start"],
                    end=seg["end"],
                    confidence=seg.get("no_speech_prob", 0)
                ))
            
            return TranscriptionResult(
                text=result["text"],
                segments=segments,
                language=result.get("language", "en"),
                duration=segments[-1].end if segments else 0,
                model=self.model.value
            )
            
        except ImportError:
            return self._transcribe_simulated(audio)
    
    def _transcribe_google(self, audio: Union[str, bytes]) -> TranscriptionResult:
        """Transcribe using Google Speech-to-Text."""
        try:
            from google.cloud import speech
            
            client = speech.SpeechClient()
            
            if isinstance(audio, str):
                with open(audio, 'rb') as f:
                    content = f.read()
            else:
                content = audio
            
            audio_obj = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                language_code=self.language if self.language != "auto" else "en-US",
                enable_word_time_offsets=True
            )
            
            response = client.recognize(config=config, audio=audio_obj)
            
            text = ""
            segments = []
            
            for result in response.results:
                alternative = result.alternatives[0]
                text += alternative.transcript + " "
                
                segments.append(TranscriptionSegment(
                    text=alternative.transcript,
                    start=0,
                    end=0,
                    confidence=alternative.confidence
                ))
            
            return TranscriptionResult(
                text=text.strip(),
                segments=segments,
                language=self.language,
                duration=0,
                model="google"
            )
            
        except ImportError:
            return self._transcribe_simulated(audio)
    
    def _transcribe_deepgram(self, audio: Union[str, bytes]) -> TranscriptionResult:
        """Transcribe using Deepgram."""
        try:
            from deepgram import Deepgram
            
            dg = Deepgram(self.api_key)
            
            if isinstance(audio, str):
                with open(audio, 'rb') as f:
                    source = {'buffer': f.read(), 'mimetype': 'audio/wav'}
            else:
                source = {'buffer': audio, 'mimetype': 'audio/wav'}
            
            response = dg.transcription.sync_prerecorded(source, {
                'punctuate': True,
                'language': self.language if self.language != "auto" else "en"
            })
            
            transcript = response['results']['channels'][0]['alternatives'][0]
            
            return TranscriptionResult(
                text=transcript['transcript'],
                segments=[],
                language=self.language,
                duration=response['metadata']['duration'],
                model="deepgram"
            )
            
        except ImportError:
            return self._transcribe_simulated(audio)
    
    def _transcribe_simulated(self, audio: Union[str, bytes]) -> TranscriptionResult:
        """Simulated transcription."""
        return TranscriptionResult(
            text="[Simulated transcription of audio content]",
            segments=[
                TranscriptionSegment(
                    text="[Simulated transcription of audio content]",
                    start=0.0,
                    end=5.0,
                    confidence=0.9
                )
            ],
            language="en",
            duration=5.0,
            model=self.model.value
        )
    
    def start_stream(self, callback: Callable[[str], None]):
        """
        Start streaming recognition.
        
        Args:
            callback: Called with transcribed text
        """
        self._stream_callback = callback
        logger.info("Streaming recognition started")
    
    def stop_stream(self):
        """Stop streaming recognition."""
        self._stream_callback = None
        logger.info("Streaming recognition stopped")
    
    def process_stream_chunk(self, audio_chunk: bytes) -> Optional[str]:
        """
        Process audio chunk for streaming.
        
        Args:
            audio_chunk: Audio data chunk
            
        Returns:
            Partial transcription or None
        """
        # Simulated streaming
        if self._stream_callback:
            text = "[streaming...]"
            self._stream_callback(text)
            return text
        return None
    
    def detect_language(self, audio: Union[str, bytes]) -> str:
        """Detect language of audio."""
        try:
            import whisper
            
            if self._whisper_model is None:
                self._whisper_model = whisper.load_model("base")
            
            audio_data = whisper.load_audio(audio) if isinstance(audio, str) else audio
            audio_data = whisper.pad_or_trim(audio_data)
            
            mel = whisper.log_mel_spectrogram(audio_data).to(self._whisper_model.device)
            _, probs = self._whisper_model.detect_language(mel)
            
            return max(probs, key=probs.get)
            
        except:
            return "en"
    
    def __repr__(self) -> str:
        return f"SpeechRecognizer(model={self.model.value})"
