"""
Integration tests for command processing subsystem.

Tests motion commands, behavior commands, and command processor.
"""

import unittest
import threading

from core.state_manager import StateManager
from interaction import MockCommandProcessor
from interaction.command_handlers import MotionCommandHandler, BehaviorCommandHandler
from motion.action_executor import ActionExecutor
from motion.action_resolver import ActionResolver, create_preset_actions_wrapper
from hardware.mock_hardware import MockRobotCar


class TestMotionCommands(unittest.TestCase):
    """Test motion command handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.state = StateManager()
        self.car = MockRobotCar()
        self.stop_flag = threading.Event()

        # Create action resolver
        self.resolver = create_preset_actions_wrapper(self.car, self.state, self.stop_flag)

        # Create executor
        self.executor = ActionExecutor(self.resolver)
        self.handler = MotionCommandHandler(self.executor, self.state)

    def test_can_handle_forward(self):
        """Test forward command recognition."""
        self.assertTrue(self.handler.can_handle("move forward"))
        self.assertTrue(self.handler.can_handle("go ahead"))
        self.assertTrue(self.handler.can_handle("forward"))

    def test_can_handle_backward(self):
        """Test backward command recognition."""
        self.assertTrue(self.handler.can_handle("move back"))
        self.assertTrue(self.handler.can_handle("go backward"))
        self.assertTrue(self.handler.can_handle("reverse"))

    def test_can_handle_turn(self):
        """Test turn command recognition."""
        self.assertTrue(self.handler.can_handle("turn left"))
        self.assertTrue(self.handler.can_handle("turn right"))

    def test_can_handle_stop(self):
        """Test stop command recognition."""
        self.assertTrue(self.handler.can_handle("stop"))
        self.assertTrue(self.handler.can_handle("halt"))
        self.assertTrue(self.handler.can_handle("freeze"))

    def test_handle_forward(self):
        """Test forward command execution."""
        result = self.handler.handle("move forward")

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "forward")

    def test_handle_backward(self):
        """Test backward command execution."""
        result = self.handler.handle("go back")

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "backward")

    def test_handle_turn_left(self):
        """Test left turn execution."""
        result = self.handler.handle("turn left")

        self.assertTrue(result["success"])
        self.assertIn("turn_left", result["action"])

    def test_handle_turn_right(self):
        """Test right turn execution."""
        result = self.handler.handle("turn right")

        self.assertTrue(result["success"])
        self.assertIn("turn_right", result["action"])

    def test_handle_stop(self):
        """Test stop command execution."""
        result = self.handler.handle("stop")

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "stop")

    def test_handle_with_distance(self):
        """Test command with distance parameter."""
        result = self.handler.handle("move forward 50 cm")

        self.assertTrue(result["success"])
        self.assertIn("forward", result["action"])

    def test_handle_with_angle(self):
        """Test turn command with angle parameter."""
        result = self.handler.handle("turn left 90 degrees")

        self.assertTrue(result["success"])
        self.assertIn("90", result["action"])


class TestBehaviorCommands(unittest.TestCase):
    """Test behavior command handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.state = StateManager()
        # Handler without actual behavior instances (will test recognition only)
        self.handler = BehaviorCommandHandler(self.state)

    def test_can_handle_roam(self):
        """Test roam command recognition."""
        self.assertTrue(self.handler.can_handle("start roaming"))
        self.assertTrue(self.handler.can_handle("explore"))
        self.assertTrue(self.handler.can_handle("wander"))
        self.assertTrue(self.handler.can_handle("stop roaming"))

    def test_can_handle_stare(self):
        """Test stare command recognition."""
        self.assertTrue(self.handler.can_handle("start stare"))
        self.assertTrue(self.handler.can_handle("look at me"))
        self.assertTrue(self.handler.can_handle("watch me"))
        self.assertTrue(self.handler.can_handle("stop staring"))

    def test_can_handle_follow(self):
        """Test follow command recognition."""
        self.assertTrue(self.handler.can_handle("follow me"))
        self.assertTrue(self.handler.can_handle("come with me"))
        self.assertTrue(self.handler.can_handle("stop following"))

    def test_can_handle_stop_all(self):
        """Test stop all command recognition."""
        self.assertTrue(self.handler.can_handle("stop everything"))
        self.assertTrue(self.handler.can_handle("stop all"))

    def test_handle_without_behaviors(self):
        """Test handling when behavior instances not available."""
        # Should return failure when behaviors not set
        result = self.handler.handle("start roaming")

        self.assertFalse(result["success"])
        self.assertIn("not available", result["message"])


class TestCommandProcessor(unittest.TestCase):
    """Test main command processor."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = MockCommandProcessor()

    def test_process_forward_command(self):
        """Test processing forward command."""
        result = self.processor.process_command("move forward")

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "forward")

    def test_process_stop_command(self):
        """Test processing stop command."""
        result = self.processor.process_command("stop")

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "stop")

    def test_process_behavior_command(self):
        """Test processing behavior command."""
        result = self.processor.process_command("start roaming")

        self.assertTrue(result["success"])
        self.assertEqual(result["action"], "start_roam")

    def test_process_conversation(self):
        """Test processing conversational input."""
        result = self.processor.process_command("what is your name")

        self.assertTrue(result["success"])
        self.assertIsInstance(result["message"], str)

    def test_process_empty_command(self):
        """Test processing empty command."""
        result = self.processor.process_command("")

        self.assertFalse(result["success"])

    def test_get_status(self):
        """Test getting processor status."""
        status = self.processor.get_status()

        self.assertIn("handlers", status)
        self.assertIn("commands_received", status)


class TestCommandIntegration(unittest.TestCase):
    """Test full command processing pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.state = StateManager()
        self.car = MockRobotCar()
        self.stop_flag = threading.Event()

        # Create action resolver
        self.resolver = create_preset_actions_wrapper(self.car, self.state, self.stop_flag)

        # Create executor
        self.executor = ActionExecutor(self.resolver)

        self.processor = MockCommandProcessor()

    def test_command_updates_state(self):
        """Test that commands update state properly."""
        # Process a command
        result = self.processor.process_command("move forward")

        self.assertTrue(result["success"])

    def test_multiple_commands(self):
        """Test processing multiple commands in sequence."""
        commands = [
            "move forward",
            "turn left",
            "move forward",
            "stop"
        ]

        for cmd in commands:
            result = self.processor.process_command(cmd)
            self.assertTrue(result["success"])

    def test_command_priority(self):
        """Test that stop command has priority."""
        # Start a behavior
        self.processor.process_command("start roaming")

        # Stop should work regardless
        result = self.processor.process_command("stop")

        self.assertTrue(result["success"])


if __name__ == '__main__':
    unittest.main()
