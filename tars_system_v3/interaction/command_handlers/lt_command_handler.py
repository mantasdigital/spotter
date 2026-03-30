"""
Lithuanian Fuzzy Command Handler.

Catches Lithuanian voice commands using fuzzy matching before they
fall through to English-only pattern handlers or the LLM.

This handler should be added with HIGH priority (early in the handler chain)
so Lithuanian commands are caught even when ASR output is imperfect.
"""

import re
from typing import Dict, Any, Optional

from speech.lt_fuzzy import (
    fuzzy_match_lt_command,
    normalize_lt,
    extract_lt_number,
    is_lithuanian_text,
)


class LithuanianCommandHandler:
    """
    Fuzzy-matching handler for Lithuanian voice commands.

    Intercepts Lithuanian commands that would otherwise be missed by
    English regex patterns. Uses Levenshtein distance matching against
    a canonical Lithuanian command dictionary.

    Must be placed BEFORE MotionCommandHandler and BehaviorCommandHandler
    in the handler chain.
    """

    # Response messages in Lithuanian
    MESSAGES_LT = {
        "forward":      "Vaziuoju pirmyn",
        "backward":     "Vaziuoju atgal",
        "turn_left":    "Suku i kaire",
        "turn_right":   "Suku i desine",
        "stop":         "Sustojau",
        "turn_around":  "Apsisuku",
        "spin":         "Sukuosi",
        "dance":        "Soku!",
        "wiggle":       "Krakausi!",
        "head_left":    "Ziuriu i kaire",
        "head_right":   "Ziuriu i desine",
        "head_center":  "Ziuriu tiesiai",
        "head_up":      "Ziuriu aukstyn",
        "head_down":    "Ziuriu zemyn",
        "start_roam":   "Pradedu tyrineti",
        "stop_roam":    "Nustojau tyrineti",
        "start_stare":  "Dabar stebiu tave",
        "start_follow": "Dabar seku paskui tave",
        "stop_follow":  "Nustojau sekti",
        "clear_memory": "Atmintis istrinta. Nebezinau kas tu esi.",
    }

    def __init__(
        self,
        executor=None,
        state=None,
        language_manager=None,
        roam_behavior=None,
        stare_behavior=None,
        follow_behavior=None,
        threshold: float = 0.65,
    ):
        """
        Initialize Lithuanian command handler.

        Args:
            executor: ActionExecutor for motion commands
            state: StateManager
            language_manager: LanguageManager for current language check
            roam_behavior: RoamBehavior instance
            stare_behavior: StareBehavior instance
            follow_behavior: FollowBehavior instance
            threshold: Fuzzy matching threshold (0.0-1.0)
        """
        self.executor = executor
        self.state = state
        self.language_manager = language_manager
        self.roam = roam_behavior
        self.stare = stare_behavior
        self.follow = follow_behavior
        self.threshold = threshold

    def can_handle(self, command: str) -> bool:
        """
        Check if this handler can process the command.

        Only activates when:
        1. Current language is Lithuanian, OR
        2. Text appears to be Lithuanian (diacritics/markers detected)

        Then checks if fuzzy matching finds a Lithuanian command.
        """
        # Check if we're in Lithuanian mode or text looks Lithuanian
        is_lt_mode = False
        if self.language_manager:
            try:
                is_lt_mode = self.language_manager.get_current_language() == "lt"
            except Exception:
                pass

        if not is_lt_mode and not is_lithuanian_text(command):
            return False

        # Try fuzzy match
        match = fuzzy_match_lt_command(command, threshold=self.threshold)
        return match is not None

    def handle(self, command: str) -> Dict[str, Any]:
        """
        Handle Lithuanian command using fuzzy matching.

        Returns:
            Dict with success, action, message keys
        """
        match = fuzzy_match_lt_command(command, threshold=self.threshold)

        if not match:
            return {
                "success": False,
                "action": "unknown",
                "message": "Nesupratau komandos",
            }

        canonical, category, action_key, score = match
        print(f"[LT-FUZZY] Matched '{command}' → {canonical} ({action_key}) score={score:.2f}")

        if category == "motion":
            return self._handle_motion(command, action_key)
        elif category == "behavior":
            return self._handle_behavior(action_key)
        elif category == "system":
            return self._handle_system(action_key)
        elif category == "vision":
            # Translate to English equivalent so downstream vision pipeline works
            # "ka matai" → "what do you see" which the LLM vision handler recognizes
            return {
                "success": False,
                "action": "vision_query",
                "message": "",
                "translated_command": "what do you see",
            }
        elif category == "web":
            # Let web handler deal with this
            return {"success": False, "action": "web_passthrough", "message": ""}

        return {
            "success": False,
            "action": "unknown",
            "message": "Nesupratau komandos",
        }

    def _handle_motion(self, command: str, action_key: str) -> Dict[str, Any]:
        """Execute a motion command."""
        if not self.executor:
            return {
                "success": False,
                "action": action_key,
                "message": "Judesiu sistema nepasiekiama",
            }

        # Extract distance/angle from command if present
        number = extract_lt_number(command)

        # Build action string
        if action_key in ("forward", "backward") and number:
            action = f"{action_key}_{int(number)}cm"
        elif action_key in ("turn_left", "turn_right") and number:
            action = f"{action_key}_{int(number)}deg"
        else:
            action = action_key

        # Stop all behaviors before motion (like the English handler does)
        if self.state and action_key == "stop":
            self.state.behavior.global_stop.set()
            # Clear after brief delay
            import threading
            import time
            def clear():
                time.sleep(0.5)
                self.state.behavior.global_stop.clear()
            threading.Thread(target=clear, daemon=True).start()

        success = self.executor.execute_action(action)

        message = self.MESSAGES_LT.get(action_key, "Atlikta")
        if number and action_key in ("forward", "backward"):
            message += f" {int(number)} cm"
        elif number and action_key in ("turn_left", "turn_right"):
            message += f" {int(number)} laipsniu"

        return {
            "success": success,
            "action": action,
            "message": message,
        }

    def _handle_behavior(self, action_key: str) -> Dict[str, Any]:
        """Execute a behavior command."""
        if action_key == "start_roam":
            if not self.roam:
                return {"success": False, "action": action_key, "message": "Klajojimas nepasiekiamas"}
            self._stop_all_behaviors()
            self.roam.start()
            return {"success": True, "action": action_key, "message": self.MESSAGES_LT.get(action_key, "Atlikta")}

        elif action_key == "stop_roam":
            if not self.roam:
                return {"success": False, "action": action_key, "message": "Klajojimas nepasiekiamas"}
            self.roam.stop()
            return {"success": True, "action": action_key, "message": self.MESSAGES_LT.get(action_key, "Atlikta")}

        elif action_key == "start_stare":
            if not self.stare:
                return {"success": False, "action": action_key, "message": "Stebejimas nepasiekiamas"}
            self._stop_all_behaviors()
            self.stare.start()
            return {"success": True, "action": action_key, "message": self.MESSAGES_LT.get(action_key, "Atlikta")}

        elif action_key == "start_follow":
            if not self.follow:
                return {"success": False, "action": action_key, "message": "Sekimas nepasiekiamas"}
            self._stop_all_behaviors()
            self.follow.start()
            return {"success": True, "action": action_key, "message": self.MESSAGES_LT.get(action_key, "Atlikta")}

        elif action_key == "stop_follow":
            if not self.follow:
                return {"success": False, "action": action_key, "message": "Sekimas nepasiekiamas"}
            self.follow.stop()
            return {"success": True, "action": action_key, "message": self.MESSAGES_LT.get(action_key, "Atlikta")}

        return {"success": False, "action": action_key, "message": "Neatpazinta komanda"}

    def _handle_system(self, action_key: str) -> Dict[str, Any]:
        """Handle system commands (language switch, memory clear)."""
        if action_key == "switch_lt":
            if self.language_manager:
                self.language_manager.switch_language("lt")
            return {
                "success": True,
                "action": "switch_language",
                "message": "Gerai, dabar kalbesiu lietuviskai.",
            }

        elif action_key == "switch_en":
            if self.language_manager:
                self.language_manager.switch_language("en")
            return {
                "success": True,
                "action": "switch_language",
                "message": "Switched to English",
            }

        elif action_key == "clear_memory":
            return {
                "success": True,
                "action": "clear_memory",
                "message": self.MESSAGES_LT.get(action_key, "Atlikta"),
            }

        return {"success": False, "action": action_key, "message": "Neatpazinta komanda"}

    def _stop_all_behaviors(self):
        """Stop all active behaviors."""
        if self.roam:
            self.roam.stop()
        if self.stare:
            self.stare.stop()
        if self.follow:
            self.follow.stop()
        if self.state:
            self.state.behavior.stop_all_behaviors()
