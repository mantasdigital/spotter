"""
Auto-Roam Watchdog.

Automatically starts roaming mode after a period of inactivity.
Ported from v58 implementation.
"""

import time
import threading
from typing import Optional, Callable

from core.state_manager import StateManager


class AutoRoamWatchdog:
    """
    Watchdog that starts roaming after inactivity.

    Monitors user activity and automatically starts roaming
    when the robot has been idle for a configurable period.
    """

    DEFAULT_INACTIVITY_SEC = 60.0  # 1 minute like v58

    def __init__(
        self,
        state: StateManager,
        start_roam_callback: Callable[[], None],
        inactivity_threshold_sec: float = DEFAULT_INACTIVITY_SEC,
        speak_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize auto-roam watchdog.

        Args:
            state: State manager
            start_roam_callback: Callback to start roaming mode
            inactivity_threshold_sec: Seconds of inactivity before auto-roam
            speak_callback: Optional callback to announce auto-roam
        """
        self.state = state
        self.start_roam = start_roam_callback
        self.inactivity_threshold = inactivity_threshold_sec
        self.speak_callback = speak_callback

        self._stop_event = threading.Event()
        self._watchdog_thread: Optional[threading.Thread] = None
        self._enabled = False

    def start(self):
        """Start the auto-roam watchdog."""
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            print("[AUTO-ROAM] Watchdog already running")
            return

        self._enabled = True
        self._stop_event.clear()

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="AutoRoamWatchdog",
            daemon=True
        )
        self._watchdog_thread.start()
        print(f"[AUTO-ROAM] Watchdog started (threshold: {self.inactivity_threshold}s)")

    def stop(self):
        """Stop the auto-roam watchdog."""
        self._enabled = False
        self._stop_event.set()

        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=2.0)

        print("[AUTO-ROAM] Watchdog stopped")

    def enable(self):
        """Enable auto-roam functionality."""
        self._enabled = True
        print("[AUTO-ROAM] Auto-roam enabled")

    def disable(self):
        """Disable auto-roam functionality (watchdog thread keeps running but won't trigger)."""
        self._enabled = False
        print("[AUTO-ROAM] Auto-roam disabled")

    def is_enabled(self) -> bool:
        """Check if auto-roam is enabled."""
        return self._enabled

    def set_threshold(self, seconds: float):
        """
        Set inactivity threshold.

        Args:
            seconds: Seconds of inactivity before auto-roam triggers
        """
        self.inactivity_threshold = max(10.0, seconds)  # Minimum 10 seconds
        print(f"[AUTO-ROAM] Threshold set to {self.inactivity_threshold}s")

    def _watchdog_loop(self):
        """
        Main watchdog loop.

        Checks for inactivity every 5 seconds and triggers roam when threshold is reached.
        """
        while not self._stop_event.is_set():
            try:
                # Sleep between checks
                time.sleep(5.0)

                # Skip if disabled
                if not self._enabled:
                    continue

                # Check inactivity duration
                idle_for = self.state.interaction.seconds_since_activity()

                # Not idle long enough
                if idle_for < self.inactivity_threshold:
                    continue

                # Skip if any behavior is already active
                if self._should_skip_auto_roam():
                    continue

                # Trigger auto-roam
                print(f"[AUTO-ROAM] Idle for {int(idle_for)}s, starting roam mode")

                # Update activity time to prevent immediate re-trigger
                self.state.interaction.update_activity()

                # Announce auto-roam
                if self.speak_callback:
                    try:
                        self.speak_callback(
                            "I have been idle for a while. Time to explore."
                        )
                    except Exception as e:
                        print(f"[AUTO-ROAM] Speech error: {e}")

                # Start roaming
                try:
                    self.start_roam()
                except Exception as e:
                    print(f"[AUTO-ROAM] Failed to start roam: {e}")

            except Exception as e:
                print(f"[AUTO-ROAM] Watchdog error: {e}")
                time.sleep(5.0)

    def _should_skip_auto_roam(self) -> bool:
        """
        Check if auto-roam should be skipped.

        Returns:
            bool: True if auto-roam should be skipped
        """
        # Skip if repeating macro is active
        if not self.state.behavior.repeat_macro_stop.is_set():
            return True

        # Skip if following
        if self.state.behavior.following.is_set():
            return True

        # Skip if face tracking (staring)
        if self.state.behavior.face_tracking.is_set():
            return True

        # Skip if already roaming
        if self.state.behavior.roaming.is_set():
            return True

        # Skip if alarm active
        if self.state.behavior.alarm_active:
            return True

        # Skip if global stop is set
        if self.state.behavior.global_stop.is_set():
            return True

        return False
