"""
Mock Hardware Implementations.

Provides mock implementations of all hardware interfaces for testing without
physical robot hardware.
"""

import logging
import time
import threading
from typing import List, Optional, Tuple
import numpy as np

from hardware.interfaces import (
    IRobotCar, ICamera, ISTT, ITTS, ILLMProvider, IFaceDetector, IVisualMemory
)


class MockRobotCar(IRobotCar):
    """
    Mock robot car for testing without hardware.

    Simulates robot behavior and tracks all commands for test assertions.
    """

    def __init__(self):
        """Initialize mock robot with default state."""
        self.logger = logging.getLogger(__name__)
        self.speed = 0
        self.direction_angle = 0
        self.pan_angle = 0
        self.tilt_angle = 0
        self.simulated_distance = 100.0
        self.simulated_grayscale = [500, 500, 500]
        self.command_history: List[Tuple[str, dict]] = []

    def _log_command(self, command: str, **kwargs):
        """Record command for test verification."""
        self.command_history.append((command, kwargs))
        self.logger.info(f"[MOCK] {command}: {kwargs}")

    def forward(self, speed: int):
        """Simulate forward movement."""
        self.speed = speed
        self._log_command("forward", speed=speed)

    def backward(self, speed: int):
        """Simulate backward movement."""
        self.speed = -speed
        self._log_command("backward", speed=speed)

    def stop(self):
        """Simulate stop."""
        self.speed = 0
        self._log_command("stop")

    def set_dir_servo_angle(self, angle: int):
        """Simulate steering."""
        self.direction_angle = max(-30, min(30, angle))
        self._log_command("set_dir_servo_angle", angle=angle)

    def set_cam_pan_angle(self, angle: int):
        """Simulate camera pan."""
        self.pan_angle = max(-90, min(90, angle))
        self._log_command("set_cam_pan_angle", angle=angle)

    def set_cam_tilt_angle(self, angle: int):
        """Simulate camera tilt."""
        self.tilt_angle = max(-35, min(65, angle))
        self._log_command("set_cam_tilt_angle", angle=angle)

    def get_distance(self) -> float:
        """Return simulated distance."""
        return self.simulated_distance

    def get_grayscale_data(self) -> List[int]:
        """Return simulated grayscale sensor data."""
        return list(self.simulated_grayscale)

    def get_cliff_status(self, grayscale_values: List[int]) -> bool:
        """
        Simulate cliff detection.

        Returns True if cliff detected (values below threshold = danger).
        Lower grayscale values indicate cliff edge (less reflection).
        """
        cliff_threshold = 500  # Values below this indicate cliff
        for val in grayscale_values:
            if val < cliff_threshold:
                return True
        return False

    def reset(self):
        """Reset to default state."""
        self.direction_angle = 0
        self.pan_angle = 0
        self.tilt_angle = 0
        self.speed = 0
        self._log_command("reset")

    def close(self):
        """Clean shutdown."""
        self.stop()
        self._log_command("close")

    # Test helper methods
    def set_simulated_distance(self, distance_cm: float):
        """Set simulated ultrasonic distance for testing."""
        self.simulated_distance = distance_cm

    def set_simulated_grayscale(self, values: List[int]):
        """Set simulated grayscale sensor values for testing."""
        self.simulated_grayscale = values

    def get_command_history(self) -> List[Tuple[str, dict]]:
        """Get history of commands for test assertions."""
        return list(self.command_history)

    def clear_command_history(self):
        """Clear command history."""
        self.command_history = []


class MockCamera(ICamera):
    """
    Mock camera returning test images.

    Can inject specific frames for testing vision algorithms.
    """

    def __init__(self, width: int = 640, height: int = 480):
        """
        Initialize mock camera.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self.logger = logging.getLogger(__name__)
        self.width = width
        self.height = height
        self.is_active_flag = False
        self.injected_frame: Optional[np.ndarray] = None
        self.injected_faces: List[Tuple[int, int, int, int]] = []

    def capture_frame(self) -> np.ndarray:
        """
        Return mock frame or injected test frame.

        Returns:
            np.ndarray: Frame as RGB image
        """
        if not self.is_active_flag:
            raise RuntimeError("Camera not started")

        if self.injected_frame is not None:
            self.logger.info("[MOCK] Returning injected frame")
            return self.injected_frame

        # Return blank frame
        self.logger.info("[MOCK] Capturing blank frame")
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def start(self):
        """Start mock camera."""
        self.is_active_flag = True
        self.logger.info("[MOCK] Camera started")

    def stop(self):
        """Stop mock camera."""
        self.is_active_flag = False
        self.logger.info("[MOCK] Camera stopped")

    def is_active(self) -> bool:
        """Check if camera is active."""
        return self.is_active_flag

    # Test helper methods
    def inject_frame(self, frame: np.ndarray):
        """Inject a specific frame for testing."""
        self.injected_frame = frame

    def inject_face(self, x: int, y: int, width: int, height: int):
        """
        Inject a face bounding box that will be detected.

        Args:
            x: Face center x coordinate
            y: Face center y coordinate
            width: Face width
            height: Face height
        """
        self.injected_faces.append((x, y, width, height))

    def get_injected_faces(self) -> List[Tuple[int, int, int, int]]:
        """Get injected face bounding boxes."""
        return list(self.injected_faces)


class MockSTT(ISTT):
    """
    Mock Speech-To-Text for testing.

    Returns canned responses or can be programmed with specific transcriptions.
    """

    def __init__(self):
        """Initialize mock STT."""
        self.logger = logging.getLogger(__name__)
        self.canned_responses = [
            "tars move forward",
            "tars what do you see",
            "tars stop",
            "tars roam mode"
        ]
        self.response_idx = 0
        self.current_language = "en"

    def transcribe(self, audio_data: bytes, language: str = "en") -> str:
        """
        Return canned response.

        Args:
            audio_data: Ignored in mock
            language: Language code

        Returns:
            str: Next canned response in sequence
        """
        response = self.canned_responses[self.response_idx % len(self.canned_responses)]
        self.response_idx += 1
        self.logger.info(f"[MOCK] STT ({language}): {response}")
        return response

    def transcribe_wav(self, wav_path: str, language: str = "en") -> str:
        """Return canned response for WAV file."""
        return self.transcribe(b"", language)

    def set_language(self, language: str):
        """Set recognition language."""
        self.current_language = language
        self.logger.info(f"[MOCK] STT language set to {language}")

    # Test helper methods
    def set_responses(self, responses: List[str]):
        """Set canned responses for testing."""
        self.canned_responses = responses
        self.response_idx = 0


class MockTTS(ITTS):
    """
    Mock Text-To-Speech for testing.

    Logs all speech requests for verification.
    """

    def __init__(self):
        """Initialize mock TTS."""
        self.logger = logging.getLogger(__name__)
        self._is_speaking_flag = False
        self.speech_history: List[Tuple[str, str]] = []  # (text, language)

    def speak(self, text: str, language: str = "en") -> bool:
        """
        Log speech request.

        Args:
            text: Text to speak
            language: Language code

        Returns:
            bool: Always True
        """
        self.speech_history.append((text, language))
        self.logger.info(f"[MOCK] TTS ({language}): {text}")
        return True

    def speak_async(self, text: str, language: str = "en") -> bool:
        """Log async speech request."""
        self._is_speaking_flag = True
        self.speak(text, language)
        # Simulate async completion
        threading.Timer(0.1, lambda: setattr(self, '_is_speaking_flag', False)).start()
        return True

    def is_speaking(self) -> bool:
        """Check if speaking."""
        return self._is_speaking_flag

    def stop_speaking(self):
        """Stop speaking."""
        self._is_speaking_flag = False
        self.logger.info("[MOCK] TTS stopped")

    # Test helper methods
    def get_speech_history(self) -> List[Tuple[str, str]]:
        """Get history of speech requests."""
        return list(self.speech_history)

    def clear_speech_history(self):
        """Clear speech history."""
        self.speech_history = []


class MockLLMProvider(ILLMProvider):
    """
    Mock LLM provider for testing.

    Returns pre-programmed responses or simple echo responses.
    """

    def __init__(self):
        """Initialize mock LLM."""
        self.logger = logging.getLogger(__name__)
        self.responses: dict = {}
        self.call_history: List[List[dict]] = []

    def chat(self, messages: List[dict], stream: bool = False, **kwargs) -> dict:
        """
        Return mock response.

        Args:
            messages: Message list
            stream: Ignored in mock
            **kwargs: Ignored

        Returns:
            dict: Mock response object
        """
        self.call_history.append(messages)

        # Extract last user message
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        # Check for programmed response
        response_text = self.responses.get(user_msg, f"Mock response to: {user_msg}")

        self.logger.info(f"[MOCK] LLM response: {response_text}")

        return {
            "choices": [{
                "message": {
                    "content": response_text
                }
            }]
        }

    def extract_text(self, response: dict) -> str:
        """Extract text from mock response."""
        try:
            return response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            return ""

    # Test helper methods
    def set_response(self, user_input: str, response: str):
        """Program a specific response for testing."""
        self.responses[user_input] = response

    def get_call_history(self) -> List[List[dict]]:
        """Get history of LLM calls."""
        return list(self.call_history)


class MockFaceDetector(IFaceDetector):
    """
    Mock face detector for testing.

    Returns pre-programmed face/hand detections.
    """

    def __init__(self):
        """Initialize mock face detector."""
        self.logger = logging.getLogger(__name__)
        self.programmed_faces: List[Tuple[int, int, int, int]] = []
        self.programmed_hands: List[Tuple[int, int, int, int]] = []

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Return programmed face detections.

        Args:
            frame: Ignored in mock

        Returns:
            List of (x, y, width, height) tuples
        """
        self.logger.info(f"[MOCK] Detected {len(self.programmed_faces)} faces")
        return list(self.programmed_faces)

    def detect_hands(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Return programmed hand detections.

        Args:
            frame: Ignored in mock

        Returns:
            List of (x, y, width, height) tuples
        """
        self.logger.info(f"[MOCK] Detected {len(self.programmed_hands)} hands")
        return list(self.programmed_hands)

    # Test helper methods
    def set_faces(self, faces: List[Tuple[int, int, int, int]]):
        """Program face detections for testing."""
        self.programmed_faces = faces

    def set_hands(self, hands: List[Tuple[int, int, int, int]]):
        """Program hand detections for testing."""
        self.programmed_hands = hands


class MockVisualMemory(IVisualMemory):
    """
    Mock visual memory for testing.

    Stores visual observations in memory (not persisted).
    """

    def __init__(self):
        """Initialize mock visual memory."""
        self.logger = logging.getLogger(__name__)
        self.memories: List[dict] = []

    def add_visual(self, image: np.ndarray, description: str, tags: List[str], label: Optional[str] = None):
        """
        Add visual memory.

        Args:
            image: Image array
            description: Scene description
            tags: Tag list
            label: Optional label
        """
        memory = {
            "timestamp": time.time(),
            "description": description,
            "tags": tags,
            "label": label,
            "image_shape": image.shape
        }
        self.memories.append(memory)
        self.logger.info(f"[MOCK] Added visual memory: {description}")

    def find_by_tags(self, tags: List[str], max_results: int = 5) -> List[dict]:
        """
        Find memories by tags.

        Args:
            tags: Tags to search for
            max_results: Max results to return

        Returns:
            List of matching memory dicts
        """
        results = []
        for mem in self.memories:
            mem_tags = set(mem["tags"])
            search_tags = set(tags)
            if mem_tags & search_tags:  # Intersection
                results.append(mem)
                if len(results) >= max_results:
                    break

        self.logger.info(f"[MOCK] Found {len(results)} memories for tags: {tags}")
        return results

    def save(self):
        """Mock save (no-op)."""
        self.logger.info("[MOCK] Visual memory saved")

    def load(self):
        """Mock load (no-op)."""
        self.logger.info("[MOCK] Visual memory loaded")

    # Test helper methods
    def get_all_memories(self) -> List[dict]:
        """Get all stored memories."""
        return list(self.memories)

    def clear(self):
        """Clear all memories."""
        self.memories = []
