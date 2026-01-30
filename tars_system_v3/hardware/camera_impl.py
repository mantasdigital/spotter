"""
Camera Implementation for Picamera2.

Wraps Picamera2 to implement the ICamera interface.
"""

import numpy as np
from typing import Optional

from hardware.interfaces import ICamera


class Picamera2Camera(ICamera):
    """
    Real Raspberry Pi camera implementation using Picamera2.

    Wraps Picamera2 library to provide frame capture functionality.
    """

    def __init__(self, picamera2_instance):
        """
        Initialize camera wrapper.

        Args:
            picamera2_instance: Picamera2 instance to wrap

        Example:
            >>> from picamera2 import Picamera2
            >>> picam2 = Picamera2()
            >>> camera = Picamera2Camera(picam2)
        """
        self.camera = picamera2_instance
        self._active = False

    def capture_frame(self) -> np.ndarray:
        """
        Capture current frame from camera.

        Returns:
            np.ndarray: Frame as numpy array (height, width, 3) in RGB format

        Raises:
            RuntimeError: If camera is not started
        """
        if not self._active:
            raise RuntimeError("Camera not started. Call start() first.")

        # Capture array returns RGB format
        frame = self.camera.capture_array()
        return frame

    def start(self):
        """
        Start camera capture.

        Configures and starts the camera for continuous capture.
        """
        if not self._active:
            # Configure camera for preview (continuous capture)
            config = self.camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            self._active = True

    def stop(self):
        """
        Stop camera capture and release resources.
        """
        if self._active:
            self.camera.stop()
            self._active = False

    def is_active(self) -> bool:
        """
        Check if camera is currently active.

        Returns:
            bool: True if camera is started and capturing
        """
        return self._active
