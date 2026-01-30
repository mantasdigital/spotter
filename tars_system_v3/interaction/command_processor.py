"""
Command Processor.

Main command processing pipeline that coordinates multiple command handlers
and processes voice commands using pattern matching.
"""

from typing import List, Dict, Any, Optional

from core.state_manager import StateManager
from interaction.command_handlers.motion_commands import MotionCommandHandler
from interaction.command_handlers.behavior_commands import BehaviorCommandHandler


class CommandProcessor:
    """
    Main command processor.

    Coordinates multiple command handlers to process voice commands using
    pattern matching. Complex queries are handled by ConversationAgent in main.py.
    """

    def __init__(
        self,
        state: StateManager,
        executor: Optional[Any] = None
    ):
        """
        Initialize command processor.

        Args:
            state: State manager
            executor: Optional ActionExecutor for clearing accumulated actions
        """
        self.state = state
        self.executor = executor
        self.handlers: List[Any] = []

    def add_handler(self, handler):
        """
        Add a command handler.

        Args:
            handler: Command handler instance with can_handle() and handle() methods
        """
        self.handlers.append(handler)

    def process_command(self, command: str, image_data: str = None, clear_actions: bool = True) -> Dict[str, Any]:
        """
        Process a voice command.

        Tries pattern-based handlers first, then falls back to LLM if available.

        Args:
            command: Voice command text
            image_data: Optional base64-encoded image for vision queries
            clear_actions: If True, clear accumulated actions before processing.
                          Set to False when processing sub-commands in a multi-command sequence.

        Returns:
            Dict with keys:
                - success (bool): Whether command was processed successfully
                - action (str): Action that was taken
                - message (str): Response message
                - handler (str): Which handler processed the command
        """
        if not command or not command.strip():
            return {
                "success": False,
                "action": "empty",
                "message": "No command received",
                "handler": "none"
            }

        # Clean up command
        command = command.strip()

        # Clear accumulated actions at start of new command for macro recording
        # Only clear if this is a new command sequence, not a sub-command
        # AND skip clearing for macro-related commands (they need previous actions)
        if clear_actions and self.executor and hasattr(self.executor, 'clear_accumulated_actions'):
            cmd_lower = command.lower()
            is_macro_command = any(kw in cmd_lower for kw in [
                'save macro', 'same macro', 'run macro', 'one macro', 'won macro',
                'macro ', 'repeat macro', 'stop macro', 'delete macro', 'forget macro'
            ])
            if not is_macro_command:
                self.executor.clear_accumulated_actions()

        # Record command in state
        self.state.interaction.record_voice_command(command)

        # Check for compound commands with "and" or "then"
        # Examples: "go forward 100 and back 20", "turn left then move forward"
        compound_separators = [" and ", " then ", ", and ", ", then "]
        is_compound = any(sep in command.lower() for sep in compound_separators)

        if is_compound:
            # Split into multiple commands and execute sequentially
            # Try each separator
            parts = []
            for sep in compound_separators:
                if sep in command.lower():
                    parts = command.lower().split(sep)
                    break

            if len(parts) > 1:
                print(f"[COMPOUND] Detected {len(parts)} commands: {parts}")
                results = []
                for i, part in enumerate(parts):
                    part = part.strip()
                    if not part:
                        continue

                    print(f"[COMPOUND] Executing part {i+1}/{len(parts)}: {part}")

                    # Process each part individually
                    result = self._process_single_command(part, image_data)
                    results.append(result)

                    # If any part fails, stop
                    if not result.get("success"):
                        print(f"[COMPOUND] Part {i+1} failed, stopping")
                        break

                # Return combined result
                messages = [r.get("message", "") for r in results if r.get("message")]
                return {
                    "success": all(r.get("success") for r in results),
                    "action": "compound",
                    "message": " ".join(messages) if messages else "Executed compound command",
                    "handler": "CompoundCommand",
                    "parts": results
                }

        # Single command - process normally
        return self._process_single_command(command, image_data)

    def _process_single_command(self, command: str, image_data: str = None) -> Dict[str, Any]:
        """
        Process a single command (non-compound).

        Args:
            command: Command text
            image_data: Optional base64-encoded image

        Returns:
            Processing result dict
        """
        # Try each handler in order
        for handler in self.handlers:
            if handler.can_handle(command):
                try:
                    result = handler.handle(command)
                    result["handler"] = handler.__class__.__name__
                    return result
                except Exception as e:
                    print(f"Handler error: {e}")
                    continue

        # No handler matched - return failure so main.py can use ConversationAgent
        return {
            "success": False,
            "action": "unknown",
            "message": "No pattern handler matched",
            "handler": "none"
        }


    def get_status(self) -> Dict[str, Any]:
        """
        Get command processor status.

        Returns:
            Dict with status information
        """
        return {
            "handlers": [h.__class__.__name__ for h in self.handlers],
            "last_command": self.state.interaction.last_voice_cmd,
            "last_command_time": self.state.interaction.last_voice_cmd_time
        }


class SimpleCommandRouter:
    """
    Simple command router without LLM dependency.

    Routes commands to appropriate handlers based on pattern matching only.
    Useful for offline or lightweight deployments.
    """

    def __init__(self):
        """Initialize simple router."""
        self.handlers = []

    def add_handler(self, handler):
        """Add a command handler."""
        self.handlers.append(handler)

    def route(self, command: str) -> Dict[str, Any]:
        """
        Route command to appropriate handler.

        Args:
            command: Voice command text

        Returns:
            Processing result dict
        """
        if not command or not command.strip():
            return {
                "success": False,
                "action": "empty",
                "message": "No command"
            }

        command = command.strip()

        # Try each handler
        for handler in self.handlers:
            if handler.can_handle(command):
                try:
                    return handler.handle(command)
                except Exception as e:
                    print(f"Handler error: {e}")
                    continue

        # No match
        return {
            "success": False,
            "action": "unknown",
            "message": "Command not recognized"
        }


class MockCommandProcessor:
    """
    Mock command processor for testing.

    Simulates command processing without executing actions.
    """

    def __init__(self):
        """Initialize mock processor."""
        self.commands_received = []

    def process_command(self, command: str) -> Dict[str, Any]:
        """Mock process command."""
        # Check for empty command
        if not command or not command.strip():
            return {
                "success": False,
                "action": "empty",
                "message": "No command received",
                "handler": "none"
            }

        self.commands_received.append(command)

        # Simple mock responses
        if "forward" in command.lower():
            return {
                "success": True,
                "action": "forward",
                "message": "Moving forward",
                "handler": "MockMotionHandler"
            }
        elif "stop" in command.lower():
            return {
                "success": True,
                "action": "stop",
                "message": "Stopping",
                "handler": "MockMotionHandler"
            }
        elif "roam" in command.lower():
            return {
                "success": True,
                "action": "start_roam",
                "message": "Starting to explore",
                "handler": "MockBehaviorHandler"
            }
        else:
            return {
                "success": True,
                "action": "conversation",
                "message": "I understand",
                "handler": "MockLLM"
            }

    def get_status(self) -> Dict[str, Any]:
        """Get mock status."""
        return {
            "commands_received": len(self.commands_received),
            "handlers": ["MockMotionHandler", "MockBehaviorHandler"],
            "llm_available": True
        }
