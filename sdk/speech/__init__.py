"""
Speech Module
=============

Speech recognition and synthesis.

Features:
- Speech-to-text (STT)
- Text-to-speech (TTS)
- Speaker recognition
- Voice activity detection
- Real-time transcription
"""

from .recognition import SpeechRecognizer
from .synthesis import SpeechSynthesizer
from .speaker import SpeakerRecognition
from .vad import VoiceActivityDetector

__all__ = [
    "SpeechRecognizer",
    "SpeechSynthesizer",
    "SpeakerRecognition",
    "VoiceActivityDetector"
]
