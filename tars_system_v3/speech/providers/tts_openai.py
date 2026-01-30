"""
Text-to-Speech using OpenAI TTS.

Cloud-based high-quality speech synthesis using OpenAI's TTS API.
"""

import os
import subprocess
import tempfile
import threading
from typing import Optional
from pathlib import Path

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from hardware.interfaces import ITTS


class OpenAITTS(ITTS):
    """
    OpenAI TTS provider.

    Provides high-quality speech synthesis using OpenAI's TTS API.
    Supports multiple voices and languages.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice: str = "alloy",
        model: str = "tts-1",
        speed: float = 1.0,
        volume: float = 2.0  # Volume multiplier (1.0 = normal, 2.0 = 2x louder)
    ):
        """
        Initialize OpenAI TTS.

        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            voice: Voice name (alloy, echo, fable, onyx, nova, shimmer)
            model: Model name (tts-1 or tts-1-hd)
            speed: Speech speed (0.25 to 4.0)
            volume: Volume multiplier (1.0 = normal, 2.0 = 2x louder, etc.)

        Raises:
            ImportError: If openai package is not installed
            ValueError: If API key is not provided
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed. Install with: pip install openai"
            )

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        self.client = OpenAI(api_key=self.api_key)
        self.voice = voice
        self.model = model
        self.speed = speed
        self.volume = volume

        self._speaking = False
        self._speaking_lock = threading.Lock()
        self._playback_process: Optional[subprocess.Popen] = None

    def speak(self, text: str, language: str = "en") -> bool:
        """
        Speak text using TTS engine (blocking).

        Uses streaming for faster time-to-first-audio when text is long.

        Args:
            text: Text to speak
            language: Language code ("en", "lt", etc.) - OpenAI TTS auto-detects

        Returns:
            bool: True on success, False on failure
        """
        if not text.strip():
            return False

        try:
            with self._speaking_lock:
                self._speaking = True

            # Use streaming for faster playback start
            # Short text: regular method is fine
            # Long text: stream to pipe for faster start
            if len(text) > 100:
                return self._speak_streaming(text)

            # Regular method for short text
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                response = self.client.audio.speech.create(
                    model=self.model,
                    voice=self.voice,
                    input=text,
                    speed=self.speed
                )

                # Save to file
                response.stream_to_file(temp_path)

                # Play audio
                self._play_audio(temp_path)

                return True

            finally:
                # Clean up temp file
                if Path(temp_path).exists():
                    Path(temp_path).unlink()

                with self._speaking_lock:
                    self._speaking = False

        except Exception as e:
            print(f"OpenAI TTS error: {e}")
            with self._speaking_lock:
                self._speaking = False
            return False

    def _speak_streaming(self, text: str) -> bool:
        """
        Speak using streaming for faster time-to-first-audio.

        Streams audio directly to ffplay without waiting for full download.

        Args:
            text: Text to speak

        Returns:
            bool: True on success
        """
        try:
            # Create streaming response
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text,
                speed=self.speed,
                response_format="mp3"
            )

            # Try streaming to ffplay via pipe
            try:
                self._playback_process = subprocess.Popen(
                    [
                        "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                        "-af", f"volume={self.volume}",
                        "-i", "pipe:0"
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                # Stream chunks to ffplay
                for chunk in response.iter_bytes(chunk_size=4096):
                    if self._playback_process.stdin:
                        self._playback_process.stdin.write(chunk)

                if self._playback_process.stdin:
                    self._playback_process.stdin.close()

                self._playback_process.wait()
                self._playback_process = None

                return True

            except FileNotFoundError:
                # ffplay not available, fall back to file-based method
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
                    temp_path = temp_file.name

                response.stream_to_file(temp_path)
                self._play_audio(temp_path)

                if Path(temp_path).exists():
                    Path(temp_path).unlink()

                return True

        except Exception as e:
            print(f"OpenAI TTS streaming error: {e}")
            return False

        finally:
            with self._speaking_lock:
                self._speaking = False

    def speak_async(self, text: str, language: str = "en") -> bool:
        """
        Speak text asynchronously (non-blocking).

        Args:
            text: Text to speak
            language: Language code ("en", "lt", etc.)

        Returns:
            bool: True if started successfully
        """
        if not text.strip():
            return False

        # Start speaking in background thread
        thread = threading.Thread(
            target=self.speak,
            args=(text, language),
            daemon=True
        )
        thread.start()

        return True

    def is_speaking(self) -> bool:
        """
        Check if TTS is currently speaking.

        Returns:
            bool: True if speech is in progress
        """
        with self._speaking_lock:
            return self._speaking

    def stop_speaking(self):
        """Stop current speech immediately."""
        with self._speaking_lock:
            if self._playback_process:
                self._playback_process.terminate()
                self._playback_process.wait(timeout=1.0)
                self._playback_process = None
            self._speaking = False

    def _play_audio(self, audio_path: str):
        """
        Play audio file using ffplay, mpg123, or aplay with volume control.

        Args:
            audio_path: Path to audio file
        """
        # Try ffplay first (best volume control with -af filter)
        try:
            self._playback_process = subprocess.Popen(
                [
                    "ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet",
                    "-af", f"volume={self.volume}",
                    audio_path
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self._playback_process.wait()
            self._playback_process = None
            return
        except FileNotFoundError:
            pass

        # Try mpg123 with volume scale (-f flag, 32768 = 100%)
        try:
            # Scale factor: 32768 = 100%, so multiply by volume
            scale = int(32768 * self.volume)
            self._playback_process = subprocess.Popen(
                ["mpg123", "-q", "-f", str(scale), audio_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self._playback_process.wait()
            self._playback_process = None
            return
        except FileNotFoundError:
            pass

        # Try aplay for WAV files (no volume control, but works)
        try:
            self._playback_process = subprocess.Popen(
                ["aplay", "-q", audio_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self._playback_process.wait()
            self._playback_process = None
            return
        except FileNotFoundError:
            pass

        raise RuntimeError(
            "No audio player found. Install ffplay, mpg123, or aplay."
        )


class MockTTS(ITTS):
    """
    Mock TTS for testing.

    Simulates speech without actually producing audio.
    """

    def __init__(self):
        """Initialize mock TTS."""
        self._speaking = False
        self._speaking_lock = threading.Lock()

    def speak(self, text: str, language: str = "en") -> bool:
        """Simulate speaking (blocking)."""
        import time

        if not text.strip():
            return False

        with self._speaking_lock:
            self._speaking = True

        # Simulate speech duration (0.1s per word)
        words = len(text.split())
        duration = words * 0.1

        time.sleep(duration)

        with self._speaking_lock:
            self._speaking = False

        return True

    def speak_async(self, text: str, language: str = "en") -> bool:
        """Simulate speaking asynchronously."""
        thread = threading.Thread(
            target=self.speak,
            args=(text, language),
            daemon=True
        )
        thread.start()
        return True

    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        with self._speaking_lock:
            return self._speaking

    def stop_speaking(self):
        """Stop speaking."""
        with self._speaking_lock:
            self._speaking = False
