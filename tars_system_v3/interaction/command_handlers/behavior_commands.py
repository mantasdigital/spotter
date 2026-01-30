"""
Behavior Command Handlers.

Handles voice commands for autonomous behaviors (roaming, following, staring).
"""

import re
from typing import Dict, Any, Optional

from core.state_manager import StateManager
from behaviors import RoamBehavior, StareBehavior, FollowBehavior


class BehaviorCommandHandler:
    """
    Handler for behavior-related voice commands.

    Parses commands to start/stop autonomous behaviors like roaming,
    face tracking, and person following.
    """

    # Command patterns for roaming (English)
    ROAM_START_PATTERNS = [
        r'\b(start |begin )?roam(ing)?\b',
        r'\broam mode\b',
        r'\bexplor(e|ing)\b',              # Matches both "explore" and "exploring"
        r'\bexplor(e|ing) mode\b',         # Matches "explore mode" and "exploring mode"
        r'\bstart explor(e|ing)\b',        # Matches "start explore" and "start exploring"
        r'\bwander\b',
        r'\b(go )?patrol\b',
        r'\bwatch (the )?perimeter\b',
        r'\bprobe\b',
        r'\bscout( around)?\b',
        r'\blook around\b',
    ]

    ROAM_STOP_PATTERNS = [
        r'\bstop roam(ing)?\b',
        r'\bend roam(ing)?\b',
        r'\bstop exploring\b',
        r'\bstop explore\b',
        r'\bexit roam\b',
        r'\bexit explore\b',
        r'\bcome back\b',
        r'\breturn\b',
    ]

    # Lithuanian roam patterns (from v58)
    ROAM_START_PATTERNS_LT = [
        r'\bklajok\b',
        r'\bklajoti\b',
        r'\bišeik pasivaikščioti\b',
        r'\bisieik pasivaik(s|š)(c|č)ioti\b',  # ASR variants
        r'\bpasižvalgyti\b',
        r'\bpasiž(i|y)urėk\b',
        # Additional Lithuanian roam commands
        r'\blakstyk\b',                         # "lakstyk" - run around
        r'\bbegiok\b',                          # "begiok" - run
    ]

    ROAM_STOP_PATTERNS_LT = [
        r'\bbaik klajoti\b',
        r'\bsustok klajoti\b',
        r'\bstok\b',
        r'\bnustok klajoti\b',
        r'\bgrįžk\b',
    ]

    # Command patterns for face tracking
    STARE_START_PATTERNS = [
        r'\b(start |begin )?stare\b',
        r'\bstare at me\b',
        r'\bstare at you\b',
        r'\b(start |begin )?track(ing)? (my )?face\b',
        r'\blook at me\b',
        r'\blook at you\b',
        r'\bwatch me\b',
        r'\bwatch you\b',
        # Lithuanian variants (žiūrėk į mane = look at me)
        r'\bžiūrėk į mane\b',
        r'\bžiaurėja kimone\b',  # wav2vec2 misrecognition variant
        r'\bžurekimone\b',  # LT command identifier variant
        r'\bžureki mone\b',  # LT variant with space
        r'\bzurekimone\b',  # ASCII fallback
        # Additional Lithuanian variants for "žiūrėk į mane" (stare at me)
        r'\bžuriekimone\b',  # ASR variant without diacritics
        r'\bžiūriekįmone\b',  # ASR variant joined
        r'\bziurekimone\b',  # ASCII variant
        r'\bziureki mone\b',  # ASCII with space
        r'\bšiaurekimone\b',  # ASR misrecognition variant
        r'\bsiaurekimone\b',  # ASCII fallback for šiaurekimone
    ]

    STARE_STOP_PATTERNS = [
        r'\bstop staring\b',
        r'\bstop tracking\b',
        r'\blook away\b',
        r'\bstop watching\b',
    ]

    # Command patterns for following
    FOLLOW_START_PATTERNS = [
        r'\b(start |begin )?follow(ing)? me\b',
        r'\bfollow you\b',
        r'\bfollow my head\b',
        r'\bcome with me\b',
        r'\bfollow\b',
        # Lithuanian variants (sek mane = follow me, sek paskui mane = follow behind me)
        r'\bsek mane\b',           # "sek mane" - follow me
        r'\bsekmane\b',            # ASR variant without space
        r'\bsetmane\b',            # ASR misrecognition variant
        r'\bsec mane\b',           # ASR misrecognition variant
        r'\bpasvec man\b',         # ASR misrecognition variant
        r'\bsek paskui mane\b',    # "sek paskui mane" - follow behind me
        r'\bsekpaskuimane\b',      # ASR variant without spaces
        r'\beik paskui mane\b',    # "eik paskui mane" - go behind me
        r'\beikpaskuimane\b',      # ASR variant without spaces
        # Lithuanian "ateik pas mane" (come to me) variants - ASR misrecognitions
        r'\bateik pas mane\b',     # "ateik pas mane" - come to me (correct)
        r'\batekpasmane\b',        # ASR joined variant
        r'\batek pas mane\b',      # ASR variant
        r'\batiek pas mane\b',     # ASR variant
        r'\batiekpasmane\b',       # ASR joined variant
        r'\batekpas mane\b',       # ASR variant
        r'\bateikpasmane\b',       # ASR joined variant
        r'\bateikpas mane\b',      # ASR variant
        r'\bateik pasmane\b',      # ASR variant
        r'\batekpas mane\b',       # ASR variant (duplicate check)
        r'\batek pasmane\b',       # ASR variant
        r'\batiekpas mane\b',      # ASR variant
        r'\batiek pasmane\b',      # ASR variant
        # Lithuanian "ateik čia" (come here) variants
        r'\bateik čia\b',          # "ateik čia" - come here (correct)
        r'\bateikčia\b',           # ASR joined variant
        r'\bateikcia\b',           # ASCII fallback joined
        r'\bateik cia\b',          # ASCII fallback with space
    ]

    FOLLOW_STOP_PATTERNS = [
        r'\bstop following\b',
        r'\bstay (there|here)\b',
        r'\bstop\b',
        # Lithuanian stop variants
        r'\bnustok sekti\b',       # "nustok sekti" - stop following
        r'\bnesek manęs\b',        # "nesek manęs" - don't follow me
        r'\bnesek manes\b',        # ASR variant without diacritics
        r'\bstovėk\b',             # "stovėk" - stay/stand
        r'\bstovek\b',             # ASR variant without diacritics
        r'\blik čia\b',            # "lik čia" - stay here
        r'\blik cia\b',            # ASR variant without diacritics
    ]

    # General stop patterns
    STOP_ALL_PATTERNS = [
        r'\bstop everything\b',
        r'\bstop all\b',
        r'\bcancel all\b',
    ]

    # Language-specific messages
    MESSAGES = {
        "en": {
            "start_roam": "Starting to explore",
            "stop_roam": "Stopped exploring",
            "start_stare": "Watching you now",
            "stop_stare": "Stopped watching",
            "start_follow": "Following you now",
            "stop_follow": "Stopped following",
            "stop_all": "All behaviors stopped",
            "roam_unavailable": "Roaming not available",
            "stare_unavailable": "Face tracking not available",
            "follow_unavailable": "Following not available",
        },
        "lt": {
            "start_roam": "Pradedu tyrinėti",
            "stop_roam": "Nustojau tyrinėti",
            "start_stare": "Dabar stebiu tave",
            "stop_stare": "Nustojau stebėti",
            "start_follow": "Dabar seku paskui tave",
            "stop_follow": "Nustojau sekti",
            "stop_all": "Visi elgsenos režimai sustabdyti",
            "roam_unavailable": "Klajojimas neprieinamas",
            "stare_unavailable": "Veido sekimas neprieinamas",
            "follow_unavailable": "Sekimas neprieinamas",
        }
    }

    def __init__(
        self,
        state: StateManager,
        roam_behavior: Optional[RoamBehavior] = None,
        stare_behavior: Optional[StareBehavior] = None,
        follow_behavior: Optional[FollowBehavior] = None,
        language_manager=None
    ):
        """
        Initialize behavior command handler.

        Args:
            state: State manager
            roam_behavior: Roaming behavior instance
            stare_behavior: Stare/face tracking behavior instance
            follow_behavior: Follow behavior instance
            language_manager: Optional language manager for localized messages
        """
        self.state = state
        self.roam = roam_behavior
        self.stare = stare_behavior
        self.follow = follow_behavior
        self.language_manager = language_manager

    def _get_message(self, key: str) -> str:
        """Get localized message based on current language."""
        lang = "en"
        if self.language_manager:
            try:
                lang = self.language_manager.get_current_language()
            except Exception:
                pass
        messages = self.MESSAGES.get(lang, self.MESSAGES["en"])
        return messages.get(key, self.MESSAGES["en"].get(key, key))

    def can_handle(self, command: str) -> bool:
        """
        Check if this handler can process the command.

        Args:
            command: Voice command text

        Returns:
            bool: True if command is a behavior command
        """
        command_lower = command.lower()

        all_patterns = (
            self.ROAM_START_PATTERNS + self.ROAM_STOP_PATTERNS +
            self.ROAM_START_PATTERNS_LT + self.ROAM_STOP_PATTERNS_LT +
            self.STARE_START_PATTERNS + self.STARE_STOP_PATTERNS +
            self.FOLLOW_START_PATTERNS + self.FOLLOW_STOP_PATTERNS +
            self.STOP_ALL_PATTERNS
        )

        for pattern in all_patterns:
            if re.search(pattern, command_lower):
                return True

        return False

    def handle(self, command: str) -> Dict[str, Any]:
        """
        Handle behavior command.

        Args:
            command: Voice command text

        Returns:
            Dict with keys:
                - success (bool): Whether command was handled successfully
                - action (str): Action taken
                - message (str): Response message
        """
        command_lower = command.lower()

        # Check for stop all first
        if self._matches_any(command_lower, self.STOP_ALL_PATTERNS):
            return self._handle_stop_all()

        # Check for roaming commands (English and Lithuanian)
        if self._matches_any(command_lower, self.ROAM_START_PATTERNS + self.ROAM_START_PATTERNS_LT):
            return self._handle_start_roam()

        if self._matches_any(command_lower, self.ROAM_STOP_PATTERNS + self.ROAM_STOP_PATTERNS_LT):
            return self._handle_stop_roam()

        # Check for stare commands
        if self._matches_any(command_lower, self.STARE_START_PATTERNS):
            return self._handle_start_stare()

        if self._matches_any(command_lower, self.STARE_STOP_PATTERNS):
            return self._handle_stop_stare()

        # Check for follow commands
        if self._matches_any(command_lower, self.FOLLOW_START_PATTERNS):
            return self._handle_start_follow()

        if self._matches_any(command_lower, self.FOLLOW_STOP_PATTERNS):
            return self._handle_stop_follow()

        return {
            "success": False,
            "action": "unknown",
            "message": "Behavior command not recognized"
        }

    def _matches_any(self, text: str, patterns: list) -> bool:
        """Check if text matches any pattern in list."""
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

    def _handle_start_roam(self) -> Dict[str, Any]:
        """Handle start roaming command."""
        if self.roam is None:
            return {
                "success": False,
                "action": "start_roam",
                "message": self._get_message("roam_unavailable")
            }

        # Stop other behaviors first
        self._stop_all_behaviors()

        # Start roaming
        self.roam.start()

        return {
            "success": True,
            "action": "start_roam",
            "message": self._get_message("start_roam")
        }

    def _handle_stop_roam(self) -> Dict[str, Any]:
        """Handle stop roaming command."""
        if self.roam is None:
            return {
                "success": False,
                "action": "stop_roam",
                "message": self._get_message("roam_unavailable")
            }

        self.roam.stop()

        return {
            "success": True,
            "action": "stop_roam",
            "message": self._get_message("stop_roam")
        }

    def _handle_start_stare(self) -> Dict[str, Any]:
        """Handle start stare/face tracking command."""
        if self.stare is None:
            return {
                "success": False,
                "action": "start_stare",
                "message": self._get_message("stare_unavailable")
            }

        # Stop other behaviors first
        self._stop_all_behaviors()

        # Start face tracking
        self.stare.start()

        return {
            "success": True,
            "action": "start_stare",
            "message": self._get_message("start_stare")
        }

    def _handle_stop_stare(self) -> Dict[str, Any]:
        """Handle stop stare command."""
        if self.stare is None:
            return {
                "success": False,
                "action": "stop_stare",
                "message": self._get_message("stare_unavailable")
            }

        self.stare.stop()

        return {
            "success": True,
            "action": "stop_stare",
            "message": self._get_message("stop_stare")
        }

    def _handle_start_follow(self) -> Dict[str, Any]:
        """Handle start following command."""
        if self.follow is None:
            return {
                "success": False,
                "action": "start_follow",
                "message": self._get_message("follow_unavailable")
            }

        # Stop other behaviors first
        self._stop_all_behaviors()

        # Start following
        self.follow.start()

        return {
            "success": True,
            "action": "start_follow",
            "message": self._get_message("start_follow")
        }

    def _handle_stop_follow(self) -> Dict[str, Any]:
        """Handle stop following command."""
        if self.follow is None:
            return {
                "success": False,
                "action": "stop_follow",
                "message": self._get_message("follow_unavailable")
            }

        self.follow.stop()

        return {
            "success": True,
            "action": "stop_follow",
            "message": self._get_message("stop_follow")
        }

    def _handle_stop_all(self) -> Dict[str, Any]:
        """Handle stop all behaviors command."""
        self._stop_all_behaviors()

        return {
            "success": True,
            "action": "stop_all",
            "message": self._get_message("stop_all")
        }

    def _stop_all_behaviors(self):
        """Stop all active behaviors."""
        if self.roam:
            self.roam.stop()

        if self.stare:
            self.stare.stop()

        if self.follow:
            self.follow.stop()

        self.state.behavior.stop_all_behaviors()
