"""
Face and hand detection implementation using OpenCV.

Provides face detection using Haar cascades with fallback support for
DNN-based detection. Also includes hand detection for stop gestures.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

from hardware.interfaces import IFaceDetector


class HaarFaceDetector(IFaceDetector):
    """
    Face and hand detector using OpenCV Haar cascades.

    Provides efficient face and hand detection using pre-trained Haar
    cascade classifiers. Suitable for real-time detection on Raspberry Pi.
    """

    def __init__(
        self,
        face_cascade_path: Optional[str] = None,
        hand_cascade_path: Optional[str] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: Tuple[int, int] = (30, 30),
        min_hand_size: Tuple[int, int] = (30, 30)
    ):
        """
        Initialize face and hand detector.

        Args:
            face_cascade_path: Path to Haar cascade XML for face detection.
                             If None, uses OpenCV's default frontal face cascade.
            hand_cascade_path: Path to Haar cascade XML for hand detection.
                             If None, hand detection will not be available.
            scale_factor: Scale factor for image pyramid (1.1 = 10% reduction per level)
            min_neighbors: Minimum number of neighbor rectangles for detection
            min_face_size: Minimum face size (width, height) in pixels
            min_hand_size: Minimum hand size (width, height) in pixels
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size
        self.min_hand_size = min_hand_size

        # Load face cascade
        if face_cascade_path:
            self.face_cascade = cv2.CascadeClassifier(face_cascade_path)
        else:
            # Use OpenCV's built-in frontal face cascade
            # Try different methods to find the cascade file
            cascade_file = None

            # Method 1: cv2.data (OpenCV 4.x)
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                test_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if Path(test_file).exists():
                    cascade_file = test_file

            # Method 2: System paths (try these first as they're most reliable)
            if not cascade_file:
                for path in [
                    '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                    '/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml',
                    '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
                    '/usr/local/share/opencv/haarcascades/haarcascade_frontalface_default.xml',
                ]:
                    if Path(path).exists():
                        cascade_file = path
                        break

            # Method 3: Try pkg_resources as last resort
            if not cascade_file:
                try:
                    import pkg_resources
                    test_file = pkg_resources.resource_filename(
                        'cv2',
                        'data/haarcascade_frontalface_default.xml'
                    )
                    if Path(test_file).exists():
                        cascade_file = test_file
                except:
                    pass

            if not cascade_file or not Path(cascade_file).exists():
                raise RuntimeError(
                    "Cannot find haarcascade_frontalface_default.xml. "
                    "Install with: sudo apt install opencv-data\n"
                    "Or specify face_cascade_path manually."
                )

            self.face_cascade = cv2.CascadeClassifier(cascade_file)

        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load face cascade from {face_cascade_path or cascade_file or 'default'}")

        # Load hand cascade if provided
        self.hand_cascade = None
        if hand_cascade_path:
            self.hand_cascade = cv2.CascadeClassifier(hand_cascade_path)
            if self.hand_cascade.empty():
                print(f"Warning: Failed to load hand cascade from {hand_cascade_path}")
                self.hand_cascade = None

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in frame.

        Args:
            frame: Image frame as numpy array (BGR or RGB)

        Returns:
            List of face bounding boxes as (x, y, width, height) tuples
        """
        # Convert to grayscale for detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size
        )

        # Convert to list of tuples
        return [tuple(face) for face in faces]

    def detect_hands(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect hands/open palms in frame (for stop gesture).

        Args:
            frame: Image frame as numpy array (BGR or RGB)

        Returns:
            List of hand bounding boxes as (x, y, width, height) tuples
        """
        if self.hand_cascade is None:
            # Hand detection not available
            return []

        # Convert to grayscale for detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect hands
        hands = self.hand_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_hand_size
        )

        # Convert to list of tuples
        return [tuple(hand) for hand in hands]

    def get_largest_face(self, faces: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the largest detected face from a list.

        Args:
            faces: List of face bounding boxes

        Returns:
            Largest face bounding box, or None if list is empty
        """
        if not faces:
            return None

        # Find face with largest area
        largest = max(faces, key=lambda f: f[2] * f[3])
        return largest

    def get_face_center(self, face: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Calculate center point of face bounding box.

        Args:
            face: Face bounding box (x, y, width, height)

        Returns:
            Tuple of (center_x, center_y)
        """
        x, y, w, h = face
        center_x = x + w / 2.0
        center_y = y + h / 2.0
        return (center_x, center_y)


class MockFaceDetector(IFaceDetector):
    """
    Mock face detector for testing without OpenCV.

    Returns simulated detections for testing purposes.
    """

    def __init__(self):
        """Initialize mock detector."""
        self.face_enabled = True
        self.hand_enabled = False

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Mock face detection.

        Returns a simulated face in the center of the frame.
        """
        if not self.face_enabled:
            return []

        h, w = frame.shape[:2]
        # Simulate a face in the center
        face_w = w // 4
        face_h = h // 4
        face_x = (w - face_w) // 2
        face_y = (h - face_h) // 2

        return [(face_x, face_y, face_w, face_h)]

    def detect_hands(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Mock hand detection.

        Returns empty list (no hands detected).
        """
        if not self.hand_enabled:
            return []

        h, w = frame.shape[:2]
        # Simulate a hand in the upper right
        hand_w = w // 6
        hand_h = h // 6
        hand_x = w - hand_w - 20
        hand_y = 20

        return [(hand_x, hand_y, hand_w, hand_h)]
