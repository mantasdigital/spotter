"""
Macro Recording and Playback System.

Allows recording action sequences as named macros and playing them back,
including support for repeated playback with timing control.
"""

import json
import os
import time
import re
import threading
import logging
from typing import Optional, Dict, Callable


class MacroStore:
    """
    Manages macro recording and playback.

    Macros are saved action sequences that can be replayed on command.
    Supports:
    - Simple one-shot playback
    - Repeated playback with intervals
    - Macro scripts with multiple macros and delays

    Example:
        >>> macros = MacroStore("/path/to/macros.json")
        >>> # Record actions
        >>> macros.set_last_actions("forward(50), shake_head()")
        >>> macros.save_macro("dance")
        >>> # Play back
        >>> macros.run_macro("dance")  # Executes: forward(50), shake_head()
    """

    def __init__(
        self,
        file_path: str,
        executor: Optional[Callable[[str, bool], None]] = None
    ):
        """
        Initialize macro store.

        Args:
            file_path: Path to JSON file for macro persistence
            executor: Optional callback function for executing actions
                     Should have signature: executor(actions_text, speak=False)
        """
        self.file_path = file_path
        self.executor = executor
        self.logger = logging.getLogger(__name__)

        self.macros: Dict[str, str] = self._load_macros()
        self.last_actions_line: Optional[str] = None

        # Control flags for repeat macro
        self.repeat_stop_event = threading.Event()
        self.voice_interrupt = False

    def _load_macros(self) -> Dict[str, str]:
        """
        Load macros from JSON file.

        Returns:
            Dict mapping macro names to action strings
        """
        if not os.path.exists(self.file_path):
            self.logger.info(f"[MACRO] No existing macro file at {self.file_path}")
            return {}

        try:
            with open(self.file_path, "r") as f:
                macros = json.load(f)
            self.logger.info(f"[MACRO] Loaded {len(macros)} macros")
            return macros
        except Exception as e:
            self.logger.error(f"[MACRO] Failed to load macros: {e}")
            return {}

    def save_macros(self):
        """Persist macros to JSON file."""
        try:
            with open(self.file_path, "w") as f:
                json.dump(self.macros, f, indent=2)
            self.logger.info(f"[MACRO] Saved {len(self.macros)} macros to disk")
        except Exception as e:
            self.logger.error(f"[MACRO] Failed to save macros: {e}")

    def set_last_actions(self, actions_line: str):
        """
        Set the last executed actions for potential macro recording.

        Args:
            actions_line: Action string (e.g., "forward(50), shake_head()")

        Example:
            >>> macros.set_last_actions("forward(50), turn_left()")
            >>> macros.save_macro("left_turn")
        """
        self.last_actions_line = actions_line

    def save_macro(self, name: str) -> bool:
        """
        Save last actions as a named macro.

        Args:
            name: Macro name (will be lowercased)

        Returns:
            bool: True if saved successfully, False if no actions to save

        Example:
            >>> macros.set_last_actions("shake_head(), wave_hands()")
            >>> macros.save_macro("greet")
        """
        if not self.last_actions_line:
            self.logger.warning("[MACRO] No actions to save as macro")
            return False

        key = name.strip().lower()
        self.macros[key] = self.last_actions_line
        self.save_macros()

        self.logger.info(f"[MACRO] Saved macro '{key}': {self.last_actions_line}")
        return True

    def run_macro(self, name: str) -> bool:
        """
        Execute a saved macro.

        Args:
            name: Macro name to run

        Returns:
            bool: True if macro exists and was executed

        Example:
            >>> macros.run_macro("greet")  # Runs: shake_head(), wave_hands()
        """
        key = name.strip().lower()
        actions_line = self.macros.get(key)

        if not actions_line:
            self.logger.warning(f"[MACRO] No macro found with name '{key}'")
            return False

        print(f"[MACRO] Running macro '{key}': {actions_line}")

        if self.executor:
            fake_text = f"ACTIONS: {actions_line}"
            print(f"[MACRO] Calling executor with: {fake_text}")
            result = self.executor(fake_text)
            print(f"[MACRO] Executor returned: {result}")
        else:
            print("[MACRO] No executor configured, cannot run macro")

        return True

    def repeat_macro(
        self,
        name: str,
        times: int = 20,
        interval_sec: float = 5.0
    ) -> bool:
        """
        Run a macro multiple times with interval between runs.

        Can be interrupted by voice_interrupt flag or repeat_stop_event.

        Args:
            name: Macro name to repeat
            times: Number of times to run
            interval_sec: Seconds to wait between runs

        Returns:
            bool: True if macro exists (even if interrupted)

        Example:
            >>> # Run "dance" macro 10 times, 3 seconds apart
            >>> macros.repeat_macro("dance", times=10, interval_sec=3.0)
        """
        key = name.strip().lower()
        actions_line = self.macros.get(key)

        if not actions_line:
            self.logger.warning(f"[MACRO] No macro found with name '{key}'")
            return False

        self.logger.info(
            f"[MACRO] Repeating macro '{key}' {times} times every {interval_sec} seconds"
        )

        # Clear stop signals
        self.repeat_stop_event.clear()
        self.voice_interrupt = False

        fake_text = f"ACTIONS: {actions_line}"

        for i in range(times):
            if self.repeat_stop_event.is_set() or self.voice_interrupt:
                print(f"[MACRO] Repeat macro '{key}' stopped after {i} runs due to interrupt")
                break

            print(f"[MACRO] Macro '{key}' run {i+1}/{times}")

            if self.executor:
                self.executor(fake_text)

            # Sleep in small chunks so stop reacts quickly
            if i < times - 1:
                remaining = interval_sec
                while remaining > 0 and not self.repeat_stop_event.is_set() and not self.voice_interrupt:
                    step = min(0.1, remaining)
                    time.sleep(step)
                    remaining -= step

        return True

    def run_macro_sequence(self, script: str):
        """
        Run a macro script with multiple segments.

        Supported segment types:
        - "macro simple" - run macro once
        - "macro z 10 times" - repeat macro 10 times
        - "macro z every 5 seconds 2 times" - repeat with custom interval
        - "sleep 3 seconds" - pause between segments

        Segments are separated by "then".

        Args:
            script: Macro script string

        Example:
            >>> macros.run_macro_sequence(
            ...     "macro dance then sleep 2 seconds then macro wave 3 times"
            ... )
        """
        segments = [
            s.strip()
            for s in re.split(r"\bthen\b", script, flags=re.IGNORECASE)
            if s.strip()
        ]

        self.logger.info(f"[MACRO] Running script with {len(segments)} segments")

        for seg in segments:
            self._run_macro_segment(seg)

    def _run_macro_segment(self, seg: str):
        """
        Execute a single macro script segment.

        Args:
            seg: Segment string (e.g., "macro simple", "sleep 3 seconds")
        """
        lower = seg.lower()

        # 1) Sleep segment: "sleep 3 seconds" / "sleep 1000 ms" / "sleep 2 minutes"
        m = re.search(
            r"sleep\s+(\d+(\.\d+)?)\s*(minutes?|mins?|seconds?|secs?|s|ms|milliseconds?)?",
            lower
        )
        if m:
            val = float(m.group(1))
            unit = m.group(3) or "s"
            if unit.startswith("ms"):
                delay = val / 1000.0
            elif unit.startswith("m") and not unit.startswith("ms"):
                delay = val * 60.0  # minutes to seconds
            else:
                delay = val

            self.logger.info(f"[MACRO] Sleeping for {delay} seconds")
            time.sleep(delay)
            return

        # 2) Macro with interval and times: "macro z every 5 seconds 2 times" or "every 5 minutes"
        m = re.search(
            r"macro\s+([a-zA-Z0-9_\-]+)\s+every\s+(\d+(\.\d+)?)\s*(minutes?|mins?|m|seconds?|secs?|s)\s+(\d+)\s*times?",
            lower,
        )
        if m:
            name = m.group(1)
            interval = float(m.group(2))
            unit = m.group(4)
            # Convert minutes to seconds
            if unit and unit.startswith("m") and not unit.startswith("ms"):
                interval = interval * 60.0
            times = int(m.group(5))

            self.logger.info(
                f"[MACRO] Script: macro '{name}' every {interval}s, {times} times"
            )
            self.repeat_macro(name, times=times, interval_sec=interval)
            return

        # 3) Macro with times: "macro z 10 times"
        m = re.search(r"macro\s+([a-zA-Z0-9_\-]+)\s+(\d+)\s*times?", lower)
        if m:
            name = m.group(1)
            times = int(m.group(2))

            self.logger.info(f"[MACRO] Script: macro '{name}' {times} times")
            self.repeat_macro(name, times=times, interval_sec=5.0)
            return

        # 4) Simple macro: "macro simple"
        m = re.search(r"macro\s+([a-zA-Z0-9_\-]+)", lower)
        if m:
            name = m.group(1)

            self.logger.info(f"[MACRO] Script: single macro '{name}'")
            self.run_macro(name)
            return

        # 5) Action segment: any other command like "forward 50", "turn left"
        # Pass to executor as an action
        if self.executor:
            self.logger.info(f"[MACRO] Script: executing action '{seg}'")
            fake_text = f"ACTIONS: {seg}"
            self.executor(fake_text)
            return

        self.logger.warning(f"[MACRO] Could not parse macro segment: '{seg}'")

    def get_macro(self, name: str) -> Optional[str]:
        """
        Get macro actions by name.

        Args:
            name: Macro name

        Returns:
            str: Action string, or None if not found
        """
        key = name.strip().lower()
        return self.macros.get(key)

    def list_macros(self) -> Dict[str, str]:
        """
        Get all saved macros.

        Returns:
            Dict mapping macro names to action strings
        """
        return dict(self.macros)

    def delete_macro(self, name: str) -> bool:
        """
        Delete a saved macro.

        Args:
            name: Macro name to delete

        Returns:
            bool: True if macro was deleted, False if not found
        """
        key = name.strip().lower()
        if key in self.macros:
            del self.macros[key]
            self.save_macros()
            self.logger.info(f"[MACRO] Deleted macro '{key}'")
            return True

        self.logger.warning(f"[MACRO] No macro found with name '{key}'")
        return False

    def clear_all(self):
        """Clear all macros and delete file."""
        self.macros = {}

        if os.path.exists(self.file_path):
            try:
                os.remove(self.file_path)
                self.logger.info(f"[MACRO] Deleted {self.file_path}")
            except Exception as e:
                self.logger.error(f"[MACRO] Failed to delete {self.file_path}: {e}")

        self.logger.info("[MACRO] All macros cleared")

    def is_recording(self) -> bool:
        """
        Check if there are pending actions for macro recording.

        Returns:
            bool: True if last_actions_line is set
        """
        return self.last_actions_line is not None
