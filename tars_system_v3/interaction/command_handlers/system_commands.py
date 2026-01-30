"""
System Command Handler.

Handles system-level commands like memory management, settings, etc.
"""

from typing import Dict, Any


class SystemCommandHandler:
    """
    Handles system commands.

    Commands:
    - Memory management (clear memory, forget everything, etc.)
    - Language switching with variations
    """

    def __init__(self, state, language_manager, conversation_memory):
        """
        Initialize system command handler.

        Args:
            state: StateManager instance
            language_manager: LanguageManager instance
            conversation_memory: ConversationMemory instance
        """
        self.state = state
        self.language_manager = language_manager
        self.conversation_memory = conversation_memory

    def can_handle(self, command: str, **kwargs) -> bool:
        """
        Check if this handler can process the command.

        Args:
            command: Command text
            **kwargs: Additional context

        Returns:
            bool: True if can handle
        """
        cmd_lower = command.lower().strip()

        # Memory commands
        memory_patterns = [
            "clear memory", "delete memory", "wipe memory",
            "forget everything", "reset memory", "erase memory",
            # Lithuanian variants
            "išvalikatmenti", "isvalyk atminti", "išvalyk atmintį"
        ]

        # Language switch patterns (Lithuanian) - including Vosk misrecognitions from v58
        lithuanian_patterns = [
            # Explicit patterns
            "speak lithuanian", "talk lithuanian", "talk in lithuanian",
            "switch to lithuanian", "switch to a lithuanian",
            "switched to a lithuanian", "switch to lithuanians",
            "switch to lithuania", "switched to lithuanian",
            "switched to lithuanians", "switch the lithuanian",
            "switch to lithuanian and", "switched to lithuanian and",
            # Vosk misrecognitions (from v58)
            "switch to live waning", "switch to live way",
            "switch to live way inin", "switch to look way in",
            "switched to live way in", "switch to live way the and",
            "switch to live for anyone", "switch to live way neon",
            # Additional Vosk misrecognitions
            "switch to live for any", "switch to live for any and",
            "switch to live", "switch to lith", "switch to lit",
            "switch to leave", "switch to life", "switch to liv",
            "switched to live", "switch to living", "switch to live in",
            "switch to live away", "switch to live weigh",
            "switch to live alien", "switch to live anyone",
            "switch live", "switch to the lithuanian",
            # Lithuanian language patterns
            "kalbek lietuviskai", "kalbėk lietuviškai",
            "pakeisk į lietuvių", "lietuvių kalba",
        ]

        # English switch patterns
        english_patterns = [
            "speak english", "talk english", "talk in english",
            "switch to english", "switch to english language",
            "switched to english"
        ]

        return (
            any(pattern in cmd_lower for pattern in memory_patterns) or
            any(pattern in cmd_lower for pattern in lithuanian_patterns) or
            any(pattern in cmd_lower for pattern in english_patterns)
        )

    def handle(self, command: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the command.

        Args:
            command: Command text
            **kwargs: Additional context

        Returns:
            dict: Result with success, action, message, and handler
        """
        cmd_lower = command.lower().strip()

        # Memory commands
        memory_patterns = [
            "clear memory", "delete memory", "wipe memory",
            "forget everything", "reset memory", "erase memory",
            # Lithuanian variants
            "išvalikatmenti", "isvalyk atminti", "išvalyk atmintį"
        ]

        if any(pattern in cmd_lower for pattern in memory_patterns):
            # Clear conversation memory (use clear() method, not clear_memory())
            self.conversation_memory.clear()

            return {
                "success": True,
                "action": "clear_memory",
                "message": "Memory wiped. I don't know who you are anymore.",
                "handler": "SystemCommandHandler"
            }

        # Lithuanian language switch - including Vosk misrecognitions from v58
        lithuanian_patterns = [
            # Explicit patterns
            "speak lithuanian", "talk lithuanian", "talk in lithuanian",
            "switch to lithuanian", "switch to a lithuanian",
            "switched to a lithuanian", "switch to lithuanians",
            "switch to lithuania", "switched to lithuanian",
            "switched to lithuanians", "switch the lithuanian",
            "switch to lithuanian and", "switched to lithuanian and",
            # Vosk misrecognitions (from v58)
            "switch to live waning", "switch to live way",
            "switch to live way inin", "switch to look way in",
            "switched to live way in", "switch to live way the and",
            "switch to live for anyone", "switch to live way neon",
            # Additional Vosk misrecognitions
            "switch to live for any", "switch to live for any and",
            "switch to live", "switch to lith", "switch to lit",
            "switch to leave", "switch to life", "switch to liv",
            "switched to live", "switch to living", "switch to live in",
            "switch to live away", "switch to live weigh",
            "switch to live alien", "switch to live anyone",
            "switch live", "switch to the lithuanian",
            # Lithuanian language patterns
            "kalbek lietuviskai", "kalbėk lietuviškai",
            "pakeisk į lietuvių", "lietuvių kalba",
        ]

        if any(pattern in cmd_lower for pattern in lithuanian_patterns):
            try:
                self.language_manager.switch_language("lt")
                msg = self.language_manager.get_confirmation_message("lt")
                return {
                    "success": True,
                    "action": "switch_language",
                    "message": msg,
                    "handler": "SystemCommandHandler"
                }
            except Exception as e:
                return {
                    "success": False,
                    "action": "switch_language",
                    "message": f"Failed to switch language: {e}",
                    "handler": "SystemCommandHandler"
                }

        # English language switch
        english_patterns = [
            "speak english", "talk english", "talk in english",
            "switch to english", "switch to english language",
            "switched to english"
        ]

        if any(pattern in cmd_lower for pattern in english_patterns):
            try:
                self.language_manager.switch_language("en")
                msg = self.language_manager.get_confirmation_message("en")
                return {
                    "success": True,
                    "action": "switch_language",
                    "message": msg,
                    "handler": "SystemCommandHandler"
                }
            except Exception as e:
                return {
                    "success": False,
                    "action": "switch_language",
                    "message": f"Failed to switch language: {e}",
                    "handler": "SystemCommandHandler"
                }

        # Should not reach here if can_handle() works correctly
        return {
            "success": False,
            "message": "Unknown system command",
            "handler": "SystemCommandHandler"
        }
