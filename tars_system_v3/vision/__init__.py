"""
Vision subsystem for TARS.

Provides face detection, scene analysis, visual memory, and gesture detection.
"""

from vision.face_detector import HaarFaceDetector, MockFaceDetector
from vision.scene_analyzer import SceneAnalyzer, MockSceneAnalyzer
from vision.visual_memory import VisualMemory, MockVisualMemory
from vision.gesture_detector import GestureDetector

__all__ = [
    'HaarFaceDetector',
    'MockFaceDetector',
    'SceneAnalyzer',
    'MockSceneAnalyzer',
    'VisualMemory',
    'MockVisualMemory',
    'GestureDetector',
]
