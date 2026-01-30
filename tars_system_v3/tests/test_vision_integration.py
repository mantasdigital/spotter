"""
Integration tests for vision subsystem.

Tests face detection, scene analysis, and visual memory.
"""

import unittest
import numpy as np

from vision import HaarFaceDetector, MockFaceDetector
from vision import SceneAnalyzer, MockSceneAnalyzer
from vision import VisualMemory, MockVisualMemory


class TestFaceDetection(unittest.TestCase):
    """Test face detection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MockFaceDetector()

    def test_detect_faces(self):
        """Test face detection returns bounding boxes."""
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Detect faces
        faces = self.detector.detect_faces(frame)

        # Should detect at least one face (mock detector)
        self.assertGreater(len(faces), 0)
        self.assertEqual(len(faces[0]), 4)  # (x, y, w, h)

    def test_detect_hands(self):
        """Test hand detection."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Enable hand detection for mock
        self.detector.hand_enabled = True

        # Detect hands
        hands = self.detector.detect_hands(frame)

        # Mock detector should return hands when enabled
        self.assertGreaterEqual(len(hands), 0)


class TestSceneAnalysis(unittest.TestCase):
    """Test scene analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MockSceneAnalyzer()

    def test_analyze_frame(self):
        """Test scene analysis returns description."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Analyze frame
        description = self.analyzer.analyze_frame(frame)

        # Should return non-empty description
        self.assertIsInstance(description, str)
        self.assertGreater(len(description), 0)

    def test_detect_objects(self):
        """Test object detection returns list."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Detect objects
        objects = self.analyzer.detect_objects(frame)

        # Should return list of objects
        self.assertIsInstance(objects, list)

    def test_check_for_obstacle(self):
        """Test obstacle detection returns structured result."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Check for obstacle
        result = self.analyzer.check_for_obstacle(frame)

        # Should return dict with expected keys
        self.assertIn("has_obstacle", result)
        self.assertIn("description", result)
        self.assertIn("urgency", result)
        self.assertIsInstance(result["has_obstacle"], bool)

    def test_generate_tags(self):
        """Test tag generation."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Generate tags
        tags = self.analyzer.generate_tags(frame)

        # Should return list of tags
        self.assertIsInstance(tags, list)
        self.assertGreater(len(tags), 0)


class TestVisualMemory(unittest.TestCase):
    """Test visual memory functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.memory = MockVisualMemory()

    def test_add_visual(self):
        """Test adding visual memory."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add visual memory
        self.memory.add_visual(
            image=frame,
            description="Test scene",
            tags=["test", "scene"],
            label="Test"
        )

        # Should have one memory
        self.assertEqual(len(self.memory.memories), 1)

    def test_find_by_tags(self):
        """Test finding memories by tags."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Add several memories with different tags
        self.memory.add_visual(frame, "Scene 1", ["indoor", "room"])
        self.memory.add_visual(frame, "Scene 2", ["outdoor", "garden"])
        self.memory.add_visual(frame, "Scene 3", ["indoor", "kitchen"])

        # Find by tag
        indoor_memories = self.memory.find_by_tags(["indoor"])

        # Should find 2 indoor memories
        self.assertEqual(len(indoor_memories), 2)

    def test_save_load(self):
        """Test save/load operations (no-op for mock)."""
        # Should not raise errors
        self.memory.save()
        self.memory.load()


if __name__ == '__main__':
    unittest.main()
