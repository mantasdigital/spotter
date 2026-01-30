"""
Hardware subsystem for TARS.

Provides hardware abstractions and implementations for robot car and camera.
"""

from hardware.interfaces import IRobotCar, ICamera, ISTT, ITTS, ILLMProvider, IFaceDetector, IVisualMemory
from hardware.picarx_impl import PicarXRobotCar
from hardware.camera_impl import Picamera2Camera
from hardware.mock_hardware import MockRobotCar, MockCamera

__all__ = [
    'IRobotCar',
    'ICamera',
    'ISTT',
    'ITTS',
    'ILLMProvider',
    'IFaceDetector',
    'IVisualMemory',
    'PicarXRobotCar',
    'Picamera2Camera',
    'MockRobotCar',
    'MockCamera',
]
