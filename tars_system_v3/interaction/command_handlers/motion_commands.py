"""
Motion Command Handlers.

Handles voice commands for basic robot motion (forward, backward, turn, stop).
"""

import re
from typing import Optional, Dict, Any

from motion.action_executor import ActionExecutor
from core.state_manager import StateManager
from utils.music_player import stop_dance_music


class MotionCommandHandler:
    """
    Handler for motion-related voice commands.

    Parses commands like "move forward", "turn left", "stop", etc.
    and executes appropriate motor actions.
    """

    # Command patterns - including Vosk misrecognitions
    FORWARD_PATTERNS = [
        r'\b(move |go |drive )?forward\b',
        r'\b(move |go |drive )?ahead\b',
        r'\bstraight\b',
        # Vosk misrecognitions: "go for" instead of "go forward"
        r'\bgo for\s+\d',  # "go for 100" or "go for one hundred"
        r'\bgo for\s+(one|two|three|four|five|six|seven|eight|nine|ten)',  # word numbers
        r'\bgo for\s+(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)',
        r'\bgo for\s+a?\s*hundred',
    ]

    BACKWARD_PATTERNS = [
        r'\b(move |go |drive )?back(ward)?s?\b',  # Matches back, backward, backwards
        r'\bgo back\b',
        r'\bgo backwards\b',
        r'\breverse\b',
        # Vosk misrecognitions
        r'\bback\s+\d',  # "back 20"
        r'\bback\s+(one|two|three|four|five|six|seven|eight|nine|ten)',
        r'\bback\s+(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)',
    ]

    # Word to number mapping for speech recognition
    # Includes common STT misrecognitions: "to/too" for "two", "for" for "four"
    WORD_TO_NUMBER = {
        'zero': 0, 'one': 1, 'two': 2, 'to': 2, 'too': 2,  # STT often hears "to" as "two"
        'three': 3, 'four': 4, 'for': 4,  # STT often hears "for" as "four"
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
        'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
        'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60,
        'seventy': 70, 'eighty': 80, 'ninety': 90, 'hundred': 100,
    }

    LEFT_PATTERNS = [
        r'\bturn left\b',
        r'\bgo left\b',
        r'\bleft turn\b',
    ]

    RIGHT_PATTERNS = [
        r'\bturn right\b',
        r'\bgo right\b',
        r'\bright turn\b',
    ]

    STOP_PATTERNS = [
        r'\bstop\b',
        r'\bhalt\b',
        r'\bfreeze\b',
        r'\bwait\b',
    ]

    # Fun action patterns
    DANCE_PATTERNS = [
        r'\bdance\b',
        r'\bdo a dance\b',
        r'\bstart dancing\b',
        r'\bšok\b',              # Lithuanian "šok" - dance
        r'\bsok\b',              # ASCII fallback for "šok"
        r'\bpašok\b',            # Lithuanian "pašok" - dance (imperative)
        r'\bpasok\b',            # ASCII fallback for "pašok"
    ]

    WIGGLE_PATTERNS = [
        r'\bwiggle\b',
        r'\bshake\b',
        r'\bdo a wiggle\b',
    ]

    # Turn around = 180 degree turn (different from spin!)
    TURN_AROUND_PATTERNS = [
        r'\bturn around\b',
        r'\bu turn\b',
        r'\bu-turn\b',
        r'\b180\b',
        r'\bturn back\b',
    ]

    SPIN_PATTERNS = [
        r'\bspin around\b',
        r'\bspin in place\b',
        r'\brotate\b',
        r'\bdo a spin\b',
        r'\bspin\b',
    ]

    SPIN_LEFT_PATTERNS = [
        r'\bspin left\b',
        r'\brotate left\b',
        r'\bturn around left\b',
    ]

    SPIN_RIGHT_PATTERNS = [
        r'\bspin right\b',
        r'\brotate right\b',
        r'\bturn around right\b',
    ]

    # Camera/head movement patterns
    LOOK_LEFT_PATTERNS = [
        r'\blook left\b',
        r'\blook to the left\b',
        r'\bhead left\b',
        r'\bturn head left\b',
        r'\bpan left\b',
    ]

    LOOK_RIGHT_PATTERNS = [
        r'\blook right\b',
        r'\blook to the right\b',
        r'\bhead right\b',
        r'\bturn head right\b',
        r'\bpan right\b',
    ]

    LOOK_CENTER_PATTERNS = [
        r'\blook center\b',
        r'\blook straight\b',
        r'\bhead center\b',
        r'\breset head\b',
        r'\blook forward\b',
        r'\bcenter head\b',
    ]

    LOOK_UP_PATTERNS = [
        r'\blook up\b',
        r'\bhead up\b',
        r'\btilt up\b',
    ]

    LOOK_DOWN_PATTERNS = [
        r'\blook down\b',
        r'\bhead down\b',
        r'\btilt down\b',
    ]

    SCAN_PATTERNS = [
        r'\bscan around\b',
        r'\blook around\b',
        r'\bhead scan\b',
        r'\bscan the area\b',
    ]

    def __init__(
        self,
        executor: ActionExecutor,
        state: StateManager,
        default_speed: int = 30,
        default_turn_angle: int = 30
    ):
        """
        Initialize motion command handler.

        Args:
            executor: Action executor for motion commands
            state: State manager
            default_speed: Default speed for movements (0-100)
            default_turn_angle: Default turn angle in degrees
        """
        self.executor = executor
        self.state = state
        self.default_speed = default_speed
        self.default_turn_angle = default_turn_angle

    def can_handle(self, command: str) -> bool:
        """
        Check if this handler can process the command.

        Args:
            command: Voice command text

        Returns:
            bool: True if command is a motion command
        """
        command_lower = command.lower()

        all_patterns = (
            self.FORWARD_PATTERNS +
            self.BACKWARD_PATTERNS +
            self.LEFT_PATTERNS +
            self.RIGHT_PATTERNS +
            self.STOP_PATTERNS +
            self.DANCE_PATTERNS +
            self.WIGGLE_PATTERNS +
            self.TURN_AROUND_PATTERNS +
            self.SPIN_PATTERNS +
            self.SPIN_LEFT_PATTERNS +
            self.SPIN_RIGHT_PATTERNS +
            self.LOOK_LEFT_PATTERNS +
            self.LOOK_RIGHT_PATTERNS +
            self.LOOK_CENTER_PATTERNS +
            self.LOOK_UP_PATTERNS +
            self.LOOK_DOWN_PATTERNS +
            self.SCAN_PATTERNS
        )

        for pattern in all_patterns:
            if re.search(pattern, command_lower):
                return True

        return False

    def handle(self, command: str) -> Dict[str, Any]:
        """
        Handle motion command.

        Args:
            command: Voice command text

        Returns:
            Dict with keys:
                - success (bool): Whether command was handled successfully
                - action (str): Action taken
                - message (str): Response message
        """
        command_lower = command.lower()

        # Check for stop first (highest priority)
        if self._matches_any(command_lower, self.STOP_PATTERNS):
            return self._handle_stop()

        # Check for fun actions
        if self._matches_any(command_lower, self.DANCE_PATTERNS):
            return self._handle_dance()

        if self._matches_any(command_lower, self.WIGGLE_PATTERNS):
            return self._handle_wiggle()

        # Check for turn around (180 degrees) BEFORE spin
        if self._matches_any(command_lower, self.TURN_AROUND_PATTERNS):
            return self._handle_turn_around()

        # Check for spinning (check specific directions before general spin)
        if self._matches_any(command_lower, self.SPIN_LEFT_PATTERNS):
            return self._handle_spin_left()

        if self._matches_any(command_lower, self.SPIN_RIGHT_PATTERNS):
            return self._handle_spin_right()

        if self._matches_any(command_lower, self.SPIN_PATTERNS):
            return self._handle_spin()

        # Check for camera/head movements
        if self._matches_any(command_lower, self.LOOK_LEFT_PATTERNS):
            return self._handle_look_left()

        if self._matches_any(command_lower, self.LOOK_RIGHT_PATTERNS):
            return self._handle_look_right()

        if self._matches_any(command_lower, self.LOOK_CENTER_PATTERNS):
            return self._handle_look_center()

        if self._matches_any(command_lower, self.LOOK_UP_PATTERNS):
            return self._handle_look_up()

        if self._matches_any(command_lower, self.LOOK_DOWN_PATTERNS):
            return self._handle_look_down()

        if self._matches_any(command_lower, self.SCAN_PATTERNS):
            return self._handle_scan()

        # Check for directional movements
        if self._matches_any(command_lower, self.FORWARD_PATTERNS):
            return self._handle_forward(command_lower)

        if self._matches_any(command_lower, self.BACKWARD_PATTERNS):
            return self._handle_backward(command_lower)

        if self._matches_any(command_lower, self.LEFT_PATTERNS):
            return self._handle_turn_left(command_lower)

        if self._matches_any(command_lower, self.RIGHT_PATTERNS):
            return self._handle_turn_right(command_lower)

        return {
            "success": False,
            "action": "unknown",
            "message": "Motion command not recognized"
        }

    def _matches_any(self, text: str, patterns: list) -> bool:
        """Check if text matches any pattern in list."""
        for pattern in patterns:
            if re.search(pattern, text):
                return True
        return False

    def _extract_distance(self, command: str) -> Optional[float]:
        """
        Extract distance from command text.

        Looks for patterns like "5 cm", "10 centimeters", "one hundred", etc.
        Handles both numeric and word-based numbers (like v58).
        """
        # First try: number followed by distance unit
        match = re.search(r'(\d+)\s*(cm|centimeter|centimetre|inch|meter|metre)', command)
        if match:
            value = float(match.group(1))
            unit = match.group(2)

            # Convert to cm
            if 'meter' in unit or 'metre' in unit:
                value *= 100
            elif 'inch' in unit:
                value *= 2.54

            return value

        # Second try: bare number (like "go for 100", "back 20")
        match = re.search(r'\b(\d+)\b', command)
        if match:
            return float(match.group(1))

        # Third try: word numbers (like "one hundred", "twenty")
        return self._parse_word_number(command)

    def _parse_word_number(self, command: str) -> Optional[float]:
        """
        Parse word numbers from command (e.g., "one hundred", "twenty").

        Handles combinations like "one hundred", "twenty five", etc.
        """
        command_lower = command.lower()
        total = 0
        found_number = False

        # Split by spaces and look for number words
        words = command_lower.split()

        for i, word in enumerate(words):
            if word in self.WORD_TO_NUMBER:
                num = self.WORD_TO_NUMBER[word]
                found_number = True

                # Handle "hundred" multiplier
                if word == 'hundred':
                    if total == 0:
                        total = 100
                    else:
                        total *= 100
                else:
                    total += num

        return total if found_number else None

    def _extract_angle(self, command: str) -> Optional[int]:
        """
        Extract turn angle from command text.

        Looks for patterns like "90 degrees", "45 deg", etc.
        """
        match = re.search(r'(\d+)\s*(degree|deg)', command)
        if match:
            return int(match.group(1))

        return None

    def _handle_forward(self, command: str) -> Dict[str, Any]:
        """Handle forward movement command."""
        distance_cm = self._extract_distance(command)

        if distance_cm:
            # Execute forward motion for specific distance
            action = f"forward_{int(distance_cm)}cm"
            success = self.executor.execute_action(action)
        else:
            # Default forward motion
            action = "forward"
            success = self.executor.execute_action(action)

        return {
            "success": success,
            "action": action,
            "message": f"Moving forward{f' {int(distance_cm)} cm' if distance_cm else ''}"
        }

    def _handle_backward(self, command: str) -> Dict[str, Any]:
        """Handle backward movement command."""
        distance_cm = self._extract_distance(command)

        if distance_cm:
            action = f"backward_{int(distance_cm)}cm"
            success = self.executor.execute_action(action)
        else:
            action = "backward"
            success = self.executor.execute_action(action)

        return {
            "success": success,
            "action": action,
            "message": f"Moving backward{f' {int(distance_cm)} cm' if distance_cm else ''}"
        }

    def _handle_turn_left(self, command: str) -> Dict[str, Any]:
        """Handle left turn command."""
        angle = self._extract_angle(command) or self.default_turn_angle

        action = f"turn_left_{angle}deg"
        success = self.executor.execute_action(action)

        return {
            "success": success,
            "action": action,
            "message": f"Turning left {angle} degrees"
        }

    def _handle_turn_right(self, command: str) -> Dict[str, Any]:
        """Handle right turn command."""
        angle = self._extract_angle(command) or self.default_turn_angle

        action = f"turn_right_{angle}deg"
        success = self.executor.execute_action(action)

        return {
            "success": success,
            "action": action,
            "message": f"Turning right {angle} degrees"
        }

    def _handle_stop(self) -> Dict[str, Any]:
        """Handle stop command."""
        # Set global stop flag to stop all behaviors (roam, stare, follow)
        self.state.behavior.global_stop.set()

        # Stop any playing dance music immediately
        stop_dance_music()

        # Stop any running macro repetition
        if hasattr(self.executor, 'macro_store') and self.executor.macro_store:
            self.executor.macro_store.repeat_stop_event.set()
            self.executor.macro_store.voice_interrupt = True
            print("[STOP] Macro repetition stopped")

        # Stop motors
        success = self.executor.execute_action("stop")

        # Clear global stop after a brief moment so behaviors can restart later
        import threading
        def clear_global_stop():
            import time
            time.sleep(0.5)
            self.state.behavior.global_stop.clear()
            # Also clear macro interrupt after stop
            if hasattr(self.executor, 'macro_store') and self.executor.macro_store:
                self.executor.macro_store.voice_interrupt = False
        threading.Thread(target=clear_global_stop, daemon=True).start()

        return {
            "success": success,
            "action": "stop",
            "message": "Stopping"
        }

    def _handle_dance(self) -> Dict[str, Any]:
        """Handle dance command."""
        success = self.executor.execute_action("dance")

        return {
            "success": success,
            "action": "dance",
            "message": "Dancing!"
        }

    def _handle_wiggle(self) -> Dict[str, Any]:
        """Handle wiggle command."""
        success = self.executor.execute_action("wiggle")

        return {
            "success": success,
            "action": "wiggle",
            "message": "Wiggling"
        }

    def _handle_spin(self) -> Dict[str, Any]:
        """Handle general spin command (no direction specified)."""
        success = self.executor.execute_action("spin")

        return {
            "success": success,
            "action": "spin",
            "message": "Spinning around"
        }

    def _handle_spin_left(self) -> Dict[str, Any]:
        """Handle spin left command."""
        success = self.executor.execute_action("spin_left")

        return {
            "success": success,
            "action": "spin_left",
            "message": "Spinning left"
        }

    def _handle_spin_right(self) -> Dict[str, Any]:
        """Handle spin right command."""
        success = self.executor.execute_action("spin_right")

        return {
            "success": success,
            "action": "spin_right",
            "message": "Spinning right"
        }

    def _handle_turn_around(self) -> Dict[str, Any]:
        """Handle turn around (180 degree) command."""
        success = self.executor.execute_action("turn_around")

        return {
            "success": success,
            "action": "turn_around",
            "message": "Turning around"
        }

    def _handle_look_left(self) -> Dict[str, Any]:
        """Handle look left command."""
        success = self.executor.execute_action("head_left")

        return {
            "success": success,
            "action": "head_left",
            "message": "Looking left"
        }

    def _handle_look_right(self) -> Dict[str, Any]:
        """Handle look right command."""
        success = self.executor.execute_action("head_right")

        return {
            "success": success,
            "action": "head_right",
            "message": "Looking right"
        }

    def _handle_look_center(self) -> Dict[str, Any]:
        """Handle look center/reset head command."""
        success = self.executor.execute_action("head_center")

        return {
            "success": success,
            "action": "head_center",
            "message": "Looking straight ahead"
        }

    def _handle_look_up(self) -> Dict[str, Any]:
        """Handle look up command."""
        success = self.executor.execute_action("head_up")

        return {
            "success": success,
            "action": "head_up",
            "message": "Looking up"
        }

    def _handle_look_down(self) -> Dict[str, Any]:
        """Handle look down command."""
        success = self.executor.execute_action("head_down")

        return {
            "success": success,
            "action": "head_down",
            "message": "Looking down"
        }

    def _handle_scan(self) -> Dict[str, Any]:
        """Handle scan around/look around command."""
        success = self.executor.execute_action("head_scan")

        return {
            "success": success,
            "action": "head_scan",
            "message": "Scanning the area"
        }
