"""
Text-to-Speech using Piper.

Offline neural TTS using Piper models. Lightweight and suitable for
Raspberry Pi, supports multiple languages.
"""

import subprocess
import tempfile
import threading
from typing import Optional
from pathlib import Path

from hardware.interfaces import ITTS


class PiperTTS(ITTS):
    """
    Piper TTS provider.

    Provides offline neural text-to-speech using Piper.
    Efficient enough to run on Raspberry Pi.
    """

    def __init__(
        self,
        piper_path: str = "piper",
        model_path_en: str = "models/en_US-lessac-medium.onnx",
        model_path_lt: Optional[str] = None,
        speaker_id: int = 0
    ):
        """
        Initialize Piper TTS.

        Args:
            piper_path: Path to piper executable
            model_path_en: Path to English Piper model (.onnx)
            model_path_lt: Path to Lithuanian Piper model (.onnx, optional)
            speaker_id: Speaker ID for multi-speaker models

        Raises:
            RuntimeError: If Piper is not found or models are missing
        """
        self.piper_path = piper_path
        self.model_path_en = model_path_en
        self.model_path_lt = model_path_lt
        self.speaker_id = speaker_id

        # Verify piper is installed
        try:
            subprocess.run(
                [piper_path, "--version"],
                capture_output=True,
                check=True
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                f"Piper not found at '{piper_path}'. "
                "Install from: https://github.com/rhasspy/piper"
            )

        # Verify English model exists
        if not Path(model_path_en).exists():
            raise RuntimeError(f"English Piper model not found at {model_path_en}")

        # Check Lithuanian model
        if model_path_lt and not Path(model_path_lt).exists():
            print(f"Warning: Lithuanian model not found at {model_path_lt}")
            self.model_path_lt = None

        self._speaking = False
        self._speaking_lock = threading.Lock()
        self._playback_process: Optional[subprocess.Popen] = None

    def speak(self, text: str, language: str = "en") -> bool:
        """
        Speak text using TTS engine (blocking).

        Args:
            text: Text to speak
            language: Language code ("en" or "lt")

        Returns:
            bool: True on success, False on failure
        """
        if not text.strip():
            return False

        # Select model based on language
        model_path = self._get_model_path(language)
        if not model_path:
            return False

        try:
            with self._speaking_lock:
                self._speaking = True

            # Generate speech to temporary WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # Run piper
                cmd = [
                    self.piper_path,
                    "--model", model_path,
                    "--output_file", temp_path
                ]

                if self.speaker_id > 0:
                    cmd.extend(["--speaker", str(self.speaker_id)])

                process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )

                # Send text to piper
                process.communicate(input=text.encode('utf-8'))

                if process.returncode != 0:
                    print(f"Piper TTS failed with code {process.returncode}")
                    return False

                # Play the audio
                self._play_audio(temp_path)

                return True

            finally:
                # Clean up temp file
                if Path(temp_path).exists():
                    Path(temp_path).unlink()

                with self._speaking_lock:
                    self._speaking = False

        except Exception as e:
            print(f"Piper TTS error: {e}")
            with self._speaking_lock:
                self._speaking = False
            return False

    def speak_async(self, text: str, language: str = "en") -> bool:
        """
        Speak text asynchronously (non-blocking).

        Args:
            text: Text to speak
            language: Language code ("en" or "lt")

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

    def _get_model_path(self, language: str) -> Optional[str]:
        """
        Get model path for specified language.

        Args:
            language: Language code ("en" or "lt")

        Returns:
            Path to model file, or None if not available
        """
        if language == "en":
            return self.model_path_en
        elif language == "lt":
            if self.model_path_lt:
                return self.model_path_lt
            else:
                print("Warning: Lithuanian model not available, using English")
                return self.model_path_en
        else:
            return self.model_path_en

    def _play_audio(self, audio_path: str):
        """
        Play audio file using aplay or ffplay.

        Args:
            audio_path: Path to WAV audio file
        """
        # Try aplay first (standard for Raspberry Pi)
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

        # Try ffplay as fallback
        try:
            self._playback_process = subprocess.Popen(
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self._playback_process.wait()
            self._playback_process = None
            return
        except FileNotFoundError:
            pass

        raise RuntimeError(
            "No audio player found. Install aplay or ffplay."
        )
