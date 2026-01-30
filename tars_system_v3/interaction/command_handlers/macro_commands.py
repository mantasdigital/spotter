"""
Macro Command Handlers.

Handles voice commands for macro execution and repetition.
Includes all phrase variations from v58 for maximum compatibility.
"""

import re
import threading
from typing import Dict, Any, Optional

from memory.macro_store import MacroStore


class MacroCommandHandler:
    """
    Handler for macro-related voice commands.

    Supports:
    - Running macros once: "run macro dance", "macro wiggle"
    - Repeating macros: "repeat macro dance 10 times", "macro dance every 5 seconds 10 times"
    - Stopping macros: "stop macro"

    Includes extensive phrase variations from v58 for STT error tolerance.
    """

    # Word to number mapping for STT that outputs words instead of digits
    WORD_TO_NUMBER = {
        'zero': 0, 'one': 1, 'two': 2, 'to': 2, 'too': 2,
        'three': 3, 'four': 4, 'for': 4, 'five': 5, 'six': 6,
        'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
        'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
        'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80,
        'ninety': 90, 'hundred': 100,
    }

    # All variations from v58 for "macro" command detection
    # Includes STT misrecognitions: "one" for "run", "won" for "run", "macros" for "macro"
    MACRO_MARKERS = [
        "run macros ", "run macro",  # "macros" = STT mishearing, check plural first
        "one macro", "won macro",  # "one"/"won" = STT mishearing "run"
        "macro ", "repeat macro", "let's repeat macro",
        "rep beat macro macro", "reepete macro", "read beat macro",
        "rabbit macro", "where beat macro", "on rep macro",
        "be a macro", "bee macro", "rep beat macro ", "be macro"
    ]

    STOP_MACRO_PATTERNS = [
        r'\bstop\s+macro\b',
        r'\bstopmacro\b',
        r'\bstop\s+repeating\b',
        r'\bcancel\s+macro\b'
    ]

    SAVE_MACRO_PATTERNS = [
        r'\bsave\s+macro\b',
        r'\bsame\s+macro\b',  # STT misrecognition of "save macro"
        # Note: "remember that as X" is handled by LLM via macro_save intent
        # This pattern is for explicit "save macro X" commands
    ]

    DELETE_MACRO_PATTERNS = [
        r'\bdelete\s+macro\b',
        r'\bremove\s+macro\b',
        r'\bforget\s+macro\b',
        r'\bclear\s+macro\b'
    ]

    def __init__(self, macro_store: MacroStore):
        """
        Initialize macro command handler.

        Args:
            macro_store: MacroStore instance for macro management
        """
        self.macros = macro_store
        self._repeat_thread: Optional[threading.Thread] = None
        self._repeat_stop_event = threading.Event()

    def can_handle(self, command: str) -> bool:
        """
        Check if this handler can process the command.

        Args:
            command: Voice command text

        Returns:
            bool: True if command is macro-related
        """
        cmd_lower = command.lower()

        # Check stop patterns
        for pattern in self.STOP_MACRO_PATTERNS:
            if re.search(pattern, cmd_lower):
                return True

        # Check save patterns
        for pattern in self.SAVE_MACRO_PATTERNS:
            if re.search(pattern, cmd_lower):
                return True

        # Check delete patterns
        for pattern in self.DELETE_MACRO_PATTERNS:
            if re.search(pattern, cmd_lower):
                return True

        # Check macro markers
        for marker in self.MACRO_MARKERS:
            if cmd_lower.startswith(marker):
                return True

        # Check for "then" sequences (complex macro scripts)
        if " then " in cmd_lower and "macro" in cmd_lower:
            return True

        return False

    def _parse_number_from_text(self, text: str) -> Optional[int]:
        """
        Parse a number from text, supporting both digits and word numbers.

        Args:
            text: Text containing a number (e.g., "5", "twenty", "four five")

        Returns:
            int: Parsed number, or None if no number found
        """
        # First try: extract digits
        digits = re.findall(r'\d+', text)
        if digits:
            return int(digits[0])

        # Second try: word numbers
        words = text.lower().split()
        total = 0
        found = False

        for word in words:
            if word in self.WORD_TO_NUMBER:
                num = self.WORD_TO_NUMBER[word]
                found = True
                if word == 'hundred':
                    total = total * 100 if total > 0 else 100
                else:
                    total += num

        return total if found else None

    def handle(self, command: str) -> Dict[str, Any]:
        """
        Handle macro command.

        Args:
            command: Voice command text

        Returns:
            Dict with keys:
                - success (bool): Whether command was handled successfully
                - action (str): Action taken
                - message (str): Response message
        """
        cmd_lower = command.lower()
        cmd_norm = re.sub(r"[^\w\s:]", " ", cmd_lower)
        cmd_norm = re.sub(r"\s+", " ", cmd_norm).strip()

        # Check for stop macro command
        for pattern in self.STOP_MACRO_PATTERNS:
            if re.search(pattern, cmd_norm):
                return self._handle_stop_macro()

        # Check for save macro command
        for pattern in self.SAVE_MACRO_PATTERNS:
            if re.search(pattern, cmd_norm):
                return self._handle_save_macro(cmd_norm, pattern)

        # Check for delete macro command
        for pattern in self.DELETE_MACRO_PATTERNS:
            if re.search(pattern, cmd_norm):
                return self._handle_delete_macro(cmd_norm, pattern)

        # Check for complex sequences with "then"
        if " then " in cmd_norm and "macro" in cmd_norm:
            return self._handle_macro_sequence(cmd_norm)

        # Check for macro commands
        for marker in self.MACRO_MARKERS:
            if cmd_norm.startswith(marker):
                return self._handle_macro_command(cmd_norm, marker)

        return {
            "success": False,
            "action": "unknown",
            "message": "Macro command not recognized"
        }

    def _handle_stop_macro(self) -> Dict[str, Any]:
        """Stop any running macro repetition."""
        self._repeat_stop_event.set()

        return {
            "success": True,
            "action": "stop_macro",
            "message": "Stopped repeating macro"
        }

    def _handle_macro_command(self, cmd_norm: str, marker: str) -> Dict[str, Any]:
        """
        Parse and execute macro command.

        Supports patterns like:
        - "macro dance" - run once
        - "macro dance 10 times" - repeat 10 times (5 sec interval)
        - "macro dance every 5 seconds 10 times" - repeat with custom interval
        """
        # Remove the marker prefix
        rest = cmd_norm.replace(marker, "", 1).strip()

        # Extract macro name (first word only - letters, digits, hyphens, underscores)
        # This handles cases like "simple four five times" where STT garbles numbers
        macro_match = re.match(r'^([a-zA-Z][a-zA-Z0-9_\-]*)', rest)
        if macro_match:
            macro_name = macro_match.group(1)
        else:
            # Fallback: find earliest keyword position
            macro_part = rest
            earliest_pos = len(rest)
            for keyword in [" for ", " every ", " times", " time "]:
                pos = rest.find(keyword)
                if pos != -1 and pos < earliest_pos:
                    earliest_pos = pos
            macro_name = rest[:earliest_pos].strip().split()[0] if earliest_pos > 0 else rest.split()[0] if rest else ""
        if not macro_name:
            return {
                "success": False,
                "action": "macro_parse_error",
                "message": "No macro name specified"
            }

        # Parse repetition parameters
        times = 20  # default
        interval = 5.0  # default

        # Check for "N times" pattern (supports digits and word numbers)
        # Match patterns like "5 times", "five times", "ten times"
        # Note: "for" is NOT included here because "for X times" is common phrase
        times_match = re.search(r'(\d+)\s*times\b', rest)
        if times_match:
            times = int(times_match.group(1))
        else:
            # Try word numbers - look for number words before "times"
            # Don't include "for" since "for ten times" is common (for = preposition, not four)
            times_word_match = re.search(r'\b((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)(?:\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred))*)\s+times\b', rest)
            if times_word_match:
                parsed = self._parse_number_from_text(times_word_match.group(1))
                if parsed and parsed > 0:
                    times = parsed

        # Check for "every N seconds/minutes" pattern (supports digits and word numbers)
        # Match "every 20 seconds", "every twenty seconds", etc.
        # Use word boundary to avoid matching "s" in "times"
        interval_match = re.search(r'every\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?|seconds?|secs?)\b', rest)
        if interval_match:
            interval = float(interval_match.group(1))
            unit = interval_match.group(2)
            if unit and unit.startswith('m'):
                interval = interval * 60.0
        else:
            # Try word numbers for interval - look for number words between "every" and unit
            interval_word_match = re.search(r'every\s+((?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)(?:\s+(?:one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety))*)\s*(minutes?|mins?|seconds?|secs?)\b', rest)
            if interval_word_match:
                parsed = self._parse_number_from_text(interval_word_match.group(1))
                if parsed and parsed > 0:
                    interval = float(parsed)
                    unit = interval_word_match.group(2)
                    if unit and unit.startswith('m'):
                        interval = interval * 60.0

        # Debug output
        print(f"[MACRO-CMD] Parsed: name='{macro_name}', times={times}, interval={interval}s")

        # Check if macro exists
        if not self.macros.get_macro(macro_name):
            return {
                "success": False,
                "action": "macro_not_found",
                "message": f"Macro '{macro_name}' not found"
            }

        # Decide whether to repeat or run once
        has_repeat_keywords = any(kw in rest for kw in ["times", "repeat", "every"])

        if has_repeat_keywords and times > 1:
            # Run in background thread
            return self._start_repeat_macro(macro_name, times, interval)
        else:
            # Run once
            return self._run_macro_once(macro_name)

    def _run_macro_once(self, macro_name: str) -> Dict[str, Any]:
        """Run a macro a single time."""
        success = self.macros.run_macro(macro_name)

        if success:
            return {
                "success": True,
                "action": "run_macro",
                "message": f"Running macro {macro_name}"
            }
        else:
            return {
                "success": False,
                "action": "macro_failed",
                "message": f"Failed to run macro {macro_name}"
            }

    def _start_repeat_macro(self, macro_name: str, times: int, interval: float) -> Dict[str, Any]:
        """Start repeating a macro in background thread."""
        # Stop any existing repeat
        self._repeat_stop_event.set()
        if self._repeat_thread and self._repeat_thread.is_alive():
            self._repeat_thread.join(timeout=1.0)

        # Clear stop event
        self._repeat_stop_event.clear()

        # Start new repeat thread
        def repeat_worker():
            self.macros.repeat_macro(macro_name, times=times, interval_sec=interval)

        self._repeat_thread = threading.Thread(target=repeat_worker, daemon=True)
        self._repeat_thread.start()

        return {
            "success": True,
            "action": "repeat_macro",
            "message": f"Repeating macro {macro_name} {times} times every {interval} seconds"
        }

    def _handle_save_macro(self, cmd_norm: str, pattern: str) -> Dict[str, Any]:
        """
        Handle save macro command.

        Extracts macro name from pattern:
        - "save macro dance"
        - "same macro dance" (STT misrecognition)

        Note: "remember that as X" is handled by LLM via macro_save intent
        """
        # Extract name after "save macro" or "same macro"
        name = cmd_norm.replace("save macro", "", 1).replace("same macro", "", 1).strip()

        if not name:
            return {
                "success": False,
                "action": "save_macro",
                "message": "What should I call this macro?"
            }

        success = self.macros.save_macro(name)

        if success:
            return {
                "success": True,
                "action": "save_macro",
                "message": f"Macro '{name}' saved",
                "macro_name": name
            }
        else:
            return {
                "success": False,
                "action": "save_macro",
                "message": "No recent actions to save as macro",
                "macro_name": name
            }

    def _handle_delete_macro(self, cmd_norm: str, pattern: str) -> Dict[str, Any]:
        """
        Handle delete macro command.

        Extracts macro name from patterns like:
        - "delete macro dance"
        - "forget macro wiggle"
        """
        # Extract name after the pattern
        for kw in ["delete macro", "remove macro", "forget macro", "clear macro"]:
            if kw in cmd_norm:
                name = cmd_norm.replace(kw, "", 1).strip()
                break
        else:
            name = ""

        if not name:
            return {
                "success": False,
                "action": "delete_macro",
                "message": "Which macro should I delete?"
            }

        success = self.macros.delete_macro(name)

        if success:
            return {
                "success": True,
                "action": "delete_macro",
                "message": f"Macro '{name}' deleted",
                "macro_name": name
            }
        else:
            return {
                "success": False,
                "action": "delete_macro",
                "message": f"Macro '{name}' not found",
                "macro_name": name
            }

    def _handle_macro_sequence(self, cmd_norm: str) -> Dict[str, Any]:
        """
        Handle complex macro sequences with 'then'.

        Examples:
        - "run macro dance then repeat macro wiggle 5 times"
        - "macro dance then sleep 2 seconds then macro wave 3 times"
        - "repeat macro x 2 times then repeat macro y every 1 minute for 10 times"
        """
        # The entire command is the script
        script = cmd_norm

        try:
            # Run the sequence (blocking)
            self.macros.run_macro_sequence(script)

            return {
                "success": True,
                "action": "macro_sequence",
                "message": "Executed macro sequence",
                "script": script
            }
        except Exception as e:
            return {
                "success": False,
                "action": "macro_sequence",
                "message": f"Error executing sequence: {e}",
                "script": script
            }
