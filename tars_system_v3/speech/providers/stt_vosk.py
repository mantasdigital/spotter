"""
Speech-to-Text using Vosk.

Offline speech recognition using Vosk models. Suitable for Raspberry Pi
and supports multiple languages including English and Lithuanian.
"""

import json
import wave
from typing import Optional
from pathlib import Path

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False

from hardware.interfaces import ISTT


class VoskSTT(ISTT):
    """
    Vosk-based Speech-to-Text provider.

    Provides offline speech recognition using Vosk models.
    Supports multiple languages through different model files.
    """

    def __init__(
        self,
        model_path_en: str = "models/vosk-model-small-en-us-0.15",
        model_path_lt: Optional[str] = None,
        sample_rate: int = 16000
    ):
        """
        Initialize Vosk STT.

        Args:
            model_path_en: Path to English Vosk model
            model_path_lt: Path to Lithuanian Vosk model (optional)
            sample_rate: Audio sample rate in Hz

        Raises:
            ImportError: If Vosk is not installed
            RuntimeError: If model files cannot be loaded
        """
        if not VOSK_AVAILABLE:
            raise ImportError(
                "Vosk not installed. Install with: pip install vosk"
            )

        self.sample_rate = sample_rate
        self.current_language = "en"

        # Load English model
        if not Path(model_path_en).exists():
            raise RuntimeError(f"English Vosk model not found at {model_path_en}")

        self.model_en = Model(model_path_en)
        self.recognizer_en = KaldiRecognizer(self.model_en, sample_rate)

        # Load Lithuanian model if provided
        self.model_lt = None
        self.recognizer_lt = None
        if model_path_lt:
            if Path(model_path_lt).exists():
                self.model_lt = Model(model_path_lt)
                self.recognizer_lt = KaldiRecognizer(self.model_lt, sample_rate)
            else:
                print(f"Warning: Lithuanian model not found at {model_path_lt}")

    def transcribe(self, audio_data: bytes, language: str = "en") -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_data: Raw audio data as bytes (PCM, 16-bit, mono)
            language: Language code ("en" or "lt")

        Returns:
            str: Transcribed text, empty string if no speech detected
        """
        # Select appropriate recognizer
        recognizer = self._get_recognizer(language)
        if recognizer is None:
            return ""

        # Reset recognizer
        recognizer.Reset()

        # Process audio
        if recognizer.AcceptWaveform(audio_data):
            result = json.loads(recognizer.Result())
        else:
            result = json.loads(recognizer.FinalResult())

        # Extract text
        text = result.get("text", "").strip()
        return text

    def transcribe_wav(self, wav_path: str, language: str = "en") -> str:
        """
        Transcribe WAV file to text.

        Args:
            wav_path: Path to WAV audio file
            language: Language code ("en" or "lt")

        Returns:
            str: Transcribed text, empty string if no speech detected

        Raises:
            FileNotFoundError: If WAV file doesn't exist
        """
        if not Path(wav_path).exists():
            raise FileNotFoundError(f"WAV file not found: {wav_path}")

        # Select appropriate recognizer
        recognizer = self._get_recognizer(language)
        if recognizer is None:
            return ""

        # Reset recognizer
        recognizer.Reset()

        # Open and process WAV file
        with wave.open(wav_path, "rb") as wf:
            # Verify format
            if wf.getnchannels() != 1:
                print("Warning: Audio must be mono")
                return ""

            if wf.getsampwidth() != 2:
                print("Warning: Audio must be 16-bit")
                return ""

            if wf.getframerate() != self.sample_rate:
                print(f"Warning: Sample rate must be {self.sample_rate}Hz")
                return ""

            # Process audio in chunks
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break

                recognizer.AcceptWaveform(data)

            # Get final result
            result = json.loads(recognizer.FinalResult())
            text = result.get("text", "").strip()
            return text

    def set_language(self, language: str):
        """
        Set recognition language.

        Args:
            language: Language code ("en" or "lt")
        """
        if language not in ["en", "lt"]:
            print(f"Warning: Unsupported language '{language}', using English")
            language = "en"

        if language == "lt" and self.recognizer_lt is None:
            # Vosk LT model not available, but wav2vec2 handles Lithuanian STT
            # so this is not a problem - silently fall back to English for Vosk
            language = "en"

        self.current_language = language

    def _get_recognizer(self, language: str):
        """
        Get recognizer for specified language.

        Args:
            language: Language code ("en" or "lt")

        Returns:
            KaldiRecognizer instance, or None if not available
        """
        if language == "en":
            return self.recognizer_en
        elif language == "lt":
            return self.recognizer_lt
        else:
            return self.recognizer_en


class MockSTT(ISTT):
    """
    Mock STT for testing without Vosk.

    Returns simulated transcriptions.
    """

    def __init__(self):
        """Initialize mock STT."""
        self.current_language = "en"
        self.transcriptions = [
            "hello",
            "move forward",
            "turn left",
            "stop"
        ]
        self.call_count = 0

    def transcribe(self, audio_data: bytes, language: str = "en") -> str:
        """Return mock transcription."""
        result = self.transcriptions[self.call_count % len(self.transcriptions)]
        self.call_count += 1
        return result

    def transcribe_wav(self, wav_path: str, language: str = "en") -> str:
        """Return mock transcription."""
        return self.transcribe(b"", language)

    def set_language(self, language: str):
        """Set language (no-op for mock)."""
        self.current_language = language
