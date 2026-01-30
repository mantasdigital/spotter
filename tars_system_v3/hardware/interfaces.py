"""
Hardware Interface Definitions.

Abstract base classes for all hardware dependencies, enabling dependency injection
and mock implementations for testing without physical hardware.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import numpy as np


class IRobotCar(ABC):
    """
    Abstract interface for robot car hardware control.

    Defines all methods needed to control the PiCar-X robot, including motors,
    servos, and sensors. Implementations can be real hardware or mocks for testing.
    """

    @abstractmethod
    def forward(self, speed: int):
        """
        Move forward at given speed.

        Args:
            speed: Speed value 0-100 (percentage of max speed)
        """
        pass

    @abstractmethod
    def backward(self, speed: int):
        """
        Move backward at given speed.

        Args:
            speed: Speed value 0-100 (percentage of max speed)
        """
        pass

    @abstractmethod
    def stop(self):
        """Stop all motors immediately."""
        pass

    @abstractmethod
    def set_dir_servo_angle(self, angle: int):
        """
        Set steering servo angle.

        Args:
            angle: Angle in degrees, typically -30 to 30
                  (negative = right, positive = left)
        """
        pass

    @abstractmethod
    def set_cam_pan_angle(self, angle: int):
        """
        Set camera pan servo angle.

        Args:
            angle: Pan angle in degrees, typically -90 to 90
        """
        pass

    @abstractmethod
    def set_cam_tilt_angle(self, angle: int):
        """
        Set camera tilt servo angle.

        Args:
            angle: Tilt angle in degrees, typically -35 to 65
        """
        pass

    @abstractmethod
    def get_distance(self) -> float:
        """
        Read ultrasonic distance sensor.

        Returns:
            float: Distance in centimeters, -1 if no obstacle detected
        """
        pass

    @abstractmethod
    def get_grayscale_data(self) -> List[int]:
        """
        Read grayscale line-following sensors.

        Returns:
            List of 3 integers representing sensor values
        """
        pass

    @abstractmethod
    def get_cliff_status(self, grayscale_values: List[int]) -> bool:
        """
        Check if cliff is detected.

        Args:
            grayscale_values: Current grayscale sensor readings

        Returns:
            bool: True if cliff detected (danger), False if safe
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset robot to initial state (center servos, stop motors)."""
        pass

    @abstractmethod
    def close(self):
        """Clean shutdown of hardware resources."""
        pass


class ICamera(ABC):
    """
    Abstract interface for camera.

    Provides frame capture functionality with start/stop lifecycle management.
    """

    @abstractmethod
    def capture_frame(self) -> np.ndarray:
        """
        Capture current frame from camera.

        Returns:
            np.ndarray: Frame as numpy array with shape (height, width, 3) in RGB format

        Raises:
            RuntimeError: If camera is not started
        """
        pass

    @abstractmethod
    def start(self):
        """
        Start camera capture.

        Must be called before capture_frame().
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Stop camera capture and release resources.
        """
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """
        Check if camera is currently active.

        Returns:
            bool: True if camera is started and capturing
        """
        pass


class ISTT(ABC):
    """
    Abstract interface for Speech-To-Text.

    Provides text transcription from audio input.
    """

    @abstractmethod
    def transcribe(self, audio_data: bytes, language: str = "en") -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_data: Raw audio data as bytes
            language: Language code ("en-us", "lt-lt", etc.)

        Returns:
            str: Transcribed text, empty string if no speech detected
        """
        pass

    @abstractmethod
    def transcribe_wav(self, wav_path: str, language: str = "en") -> str:
        """
        Transcribe WAV file to text.

        Args:
            wav_path: Path to WAV audio file
            language: Language code ("en-us", "lt-lt", etc.)

        Returns:
            str: Transcribed text, empty string if no speech detected

        Raises:
            FileNotFoundError: If WAV file doesn't exist
        """
        pass

    @abstractmethod
    def set_language(self, language: str):
        """
        Set recognition language.

        Args:
            language: Language code ("en-us", "lt-lt", etc.)
        """
        pass


class ITTS(ABC):
    """
    Abstract interface for Text-To-Speech.

    Provides speech synthesis from text.
    """

    @abstractmethod
    def speak(self, text: str, language: str = "en") -> bool:
        """
        Speak text using TTS engine.

        Args:
            text: Text to speak
            language: Language code ("en", "lt", etc.)

        Returns:
            bool: True on success, False on failure
        """
        pass

    @abstractmethod
    def speak_async(self, text: str, language: str = "en") -> bool:
        """
        Speak text asynchronously (non-blocking).

        Args:
            text: Text to speak
            language: Language code ("en", "lt", etc.)

        Returns:
            bool: True if started successfully
        """
        pass

    @abstractmethod
    def is_speaking(self) -> bool:
        """
        Check if TTS is currently speaking.

        Returns:
            bool: True if speech is in progress
        """
        pass

    @abstractmethod
    def stop_speaking(self):
        """Stop current speech immediately."""
        pass


class ILLMProvider(ABC):
    """
    Abstract interface for Large Language Model provider.

    Provides chat completion functionality.
    """

    @abstractmethod
    def chat(self, messages: List[dict], stream: bool = False, **kwargs) -> any:
        """
        Send chat completion request to LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters (temperature, max_tokens, etc.)

        Returns:
            Response object (format depends on provider and stream setting)

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant"},
            ...     {"role": "user", "content": "Hello"}
            ... ]
            >>> response = llm.chat(messages, stream=False)
        """
        pass

    @abstractmethod
    def extract_text(self, response: any) -> str:
        """
        Extract text content from LLM response.

        Args:
            response: Response object from chat()

        Returns:
            str: Extracted text content
        """
        pass


class IFaceDetector(ABC):
    """
    Abstract interface for face detection.

    Provides face detection in images using various backends (Haar, DNN, etc.).
    """

    @abstractmethod
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame.

        Args:
            frame: Image frame as numpy array (BGR or RGB)

        Returns:
            List of face bounding boxes as (x, y, width, height) tuples
        """
        pass

    @abstractmethod
    def detect_hands(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect hands/open palms in frame (for stop gesture).

        Args:
            frame: Image frame as numpy array (BGR or RGB)

        Returns:
            List of hand bounding boxes as (x, y, width, height) tuples
        """
        pass


class IVisualMemory(ABC):
    """
    Abstract interface for visual memory storage.

    Manages storage and retrieval of visual observations with tagging.
    """

    @abstractmethod
    def add_visual(self, image: np.ndarray, description: str, tags: List[str], label: Optional[str] = None):
        """
        Add a visual memory.

        Args:
            image: Image frame as numpy array
            description: Text description of the scene
            tags: List of tag strings
            label: Optional human-provided label
        """
        pass

    @abstractmethod
    def find_by_tags(self, tags: List[str], max_results: int = 5) -> List[dict]:
        """
        Find visual memories by tags.

        Args:
            tags: List of tags to search for
            max_results: Maximum number of results to return

        Returns:
            List of visual memory dicts
        """
        pass

    @abstractmethod
    def save(self):
        """Persist visual memory to disk."""
        pass

    @abstractmethod
    def load(self):
        """Load visual memory from disk."""
        pass
