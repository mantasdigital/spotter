"""
TARS State Manager.

Centralizes all state management that was previously scattered across 41+ global
variables in the v58 implementation. Provides thread-safe access to all system state.
"""

import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


@dataclass
class HardwareState:
    """
    Hardware and device instances.

    Stores references to physical hardware abstractions.
    """

    car_instance: Optional[Any] = None
    vac_instance: Optional[Any] = None
    camera_instance: Optional[Any] = None
    face_cascade: Optional[Any] = None


@dataclass
class TTSState:
    """
    Text-to-speech coordination state.

    Manages speech synthesis state to prevent overlapping speech and
    control speech timing during roam mode.

    CRITICAL: Both roam observations and command responses must use
    this state to coordinate and avoid voice overlap.
    """

    is_speaking: bool = False
    speaking_source: str = ""  # "roam", "command", or ""
    speaking_start_time: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_roam_spoken: Optional[str] = None
    last_roam_spoken_time: float = 0.0
    last_roam_text: Optional[str] = None
    roam_speech_min_interval: float = 6.0

    # Speech timeout for auto-clearing stuck speaking flag
    speech_timeout: float = 30.0  # Max speech duration before auto-clear

    def start_speaking(self, source: str = "unknown") -> bool:
        """
        Attempt to start speaking. Returns True if allowed, False if blocked.

        Args:
            source: Who is trying to speak ("roam", "command", etc.)

        Returns:
            bool: True if speaking can start, False if another speech is active
        """
        with self.lock:
            # Auto-clear if timed out (stuck speaking flag)
            if self.is_speaking:
                elapsed = time.time() - self.speaking_start_time
                if elapsed > self.speech_timeout:
                    print(f"[TTS-STATE] Auto-clearing stuck speaking flag (was: {self.speaking_source})")
                    self.is_speaking = False
                    self.speaking_source = ""

            if self.is_speaking:
                return False

            self.is_speaking = True
            self.speaking_source = source
            self.speaking_start_time = time.time()
            return True

    def stop_speaking(self):
        """Mark that speaking has finished."""
        with self.lock:
            self.is_speaking = False
            self.speaking_source = ""

    def is_currently_speaking(self) -> bool:
        """
        Check if TTS is currently speaking (with timeout auto-clear).

        Returns:
            bool: True if speaking is in progress
        """
        with self.lock:
            if not self.is_speaking:
                return False

            # Auto-clear if timed out
            elapsed = time.time() - self.speaking_start_time
            if elapsed > self.speech_timeout:
                self.is_speaking = False
                self.speaking_source = ""
                return False

            return True

    def get_speaking_source(self) -> str:
        """Get who is currently speaking."""
        with self.lock:
            return self.speaking_source if self.is_speaking else ""

    def wait_for_speech_end(self, timeout: float = 5.0) -> bool:
        """
        Wait for current speech to end (blocking).

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            bool: True if speech ended, False if timed out
        """
        start = time.time()
        while self.is_currently_speaking():
            if time.time() - start > timeout:
                return False
            time.sleep(0.1)
        return True

    def mark_speaking_start(self):
        """Mark that TTS has started speaking (legacy compatibility)."""
        self.start_speaking("legacy")

    def mark_speaking_end(self):
        """Mark that TTS has finished speaking (legacy compatibility)."""
        self.stop_speaking()

    def can_speak_roam_update(self) -> bool:
        """
        Check if enough time has passed to speak another roam update.

        Returns:
            bool: True if enough time has elapsed since last roam speech
        """
        with self.lock:
            elapsed = time.time() - self.last_roam_spoken_time
            return elapsed >= self.roam_speech_min_interval

    def record_roam_speech(self, text: str):
        """
        Record that a roam update was spoken.

        Args:
            text: The text that was spoken
        """
        with self.lock:
            self.last_roam_spoken = text
            self.last_roam_spoken_time = time.time()


@dataclass
class BehaviorState:
    """
    Active behavior modes.

    Uses threading.Event for mode flags to allow threads to wait/signal.
    """

    roaming: threading.Event = field(default_factory=threading.Event)
    following: threading.Event = field(default_factory=threading.Event)
    face_tracking: threading.Event = field(default_factory=threading.Event)
    global_stop: threading.Event = field(default_factory=threading.Event)
    voice_interrupt: bool = False
    alarm_active: bool = False
    repeat_macro_stop: threading.Event = field(default_factory=threading.Event)

    def stop_all_behaviors(self):
        """
        Emergency stop - clear all active behavior modes.

        Used for safety stops and mode transitions.
        """
        self.roaming.clear()
        self.following.clear()
        self.face_tracking.clear()
        self.repeat_macro_stop.set()

    def is_any_behavior_active(self) -> bool:
        """
        Check if any behavior mode is currently active.

        Returns:
            bool: True if roaming, following, or face_tracking is active
        """
        return (
            self.roaming.is_set() or
            self.following.is_set() or
            self.face_tracking.is_set()
        )


@dataclass
class RoamState:
    """
    Roaming mode state.

    Tracks robot position, path history, and obstacle avoidance state
    during autonomous roaming.
    """

    # Position tracking (odometry)
    x_m: float = 0.0
    y_m: float = 0.0
    heading_deg: float = 0.0

    # Path history: list of (timestamp, x, y, heading)
    path: List[Tuple[float, float, float, float]] = field(default_factory=list)

    # Obstacle avoidance
    forward_block_count: int = 0
    last_clear_time: float = 0.0
    blocked_headings: List[Tuple[float, float]] = field(default_factory=list)

    # Spatial memory: (grid_x, grid_y, heading_bucket) -> visit_count
    visited_grid: Dict[Tuple[int, int, int], int] = field(default_factory=dict)

    # Roam thread
    roam_thread: Optional[threading.Thread] = None

    # State dict for additional dynamic state
    state_dict: Dict[str, Any] = field(default_factory=lambda: {
        "start_time": 0.0,
        "last_obs_time": 0.0,
        "obs_count": 0,
    })

    def reset(self):
        """Reset roam state for a new roaming session."""
        self.x_m = 0.0
        self.y_m = 0.0
        self.heading_deg = 0.0
        self.path = []
        self.forward_block_count = 0
        self.last_clear_time = 0.0
        self.blocked_headings = []
        self.state_dict = {
            "start_time": time.time(),
            "last_obs_time": 0.0,
            "obs_count": 0,
        }

    def integrate_motion(self, distance_cm: float, turn_deg: float = 0.0):
        """
        Update odometry based on motion.

        Args:
            distance_cm: Distance traveled in cm (positive = forward)
            turn_deg: Change in heading in degrees (positive = left)
        """
        import math

        # Update heading
        self.heading_deg = (self.heading_deg + turn_deg) % 360.0

        # Update position
        heading_rad = math.radians(self.heading_deg)
        distance_m = distance_cm / 100.0

        self.x_m += distance_m * math.cos(heading_rad)
        self.y_m += distance_m * math.sin(heading_rad)

        # Record in path history
        self.path.append((time.time(), self.x_m, self.y_m, self.heading_deg))

    def get_grid_cell(self, cell_size_cm: float = 25.0, heading_bucket_deg: float = 45.0) -> Tuple[int, int, int]:
        """
        Get current grid cell coordinates for visit tracking.

        Args:
            cell_size_cm: Size of grid cells in cm
            heading_bucket_deg: Size of heading buckets in degrees

        Returns:
            Tuple of (grid_x, grid_y, heading_bucket)
        """
        cell_size_m = cell_size_cm / 100.0
        grid_x = int(self.x_m / cell_size_m)
        grid_y = int(self.y_m / cell_size_m)
        heading_bucket = int(self.heading_deg / heading_bucket_deg)

        return (grid_x, grid_y, heading_bucket)

    def mark_visited(self, cell_size_cm: float = 25.0, heading_bucket_deg: float = 45.0):
        """
        Mark current location as visited.

        Args:
            cell_size_cm: Size of grid cells in cm
            heading_bucket_deg: Size of heading buckets in degrees
        """
        cell = self.get_grid_cell(cell_size_cm, heading_bucket_deg)
        self.visited_grid[cell] = self.visited_grid.get(cell, 0) + 1

    def get_visit_count(self, cell_size_cm: float = 25.0, heading_bucket_deg: float = 45.0) -> int:
        """
        Get visit count for current location.

        Args:
            cell_size_cm: Size of grid cells in cm
            heading_bucket_deg: Size of heading buckets in degrees

        Returns:
            int: Number of times this cell has been visited
        """
        cell = self.get_grid_cell(cell_size_cm, heading_bucket_deg)
        return self.visited_grid.get(cell, 0)


@dataclass
class StareState:
    """
    Face tracking (stare mode) state.

    Tracks camera servo angles and last detected face position.
    """

    x_angle: float = 0.0
    y_angle: float = 0.0
    last_face: Dict[str, Any] = field(default_factory=lambda: {
        "x": 0.0,
        "y": 0.0,
        "width": 0,
        "height": 0,
        "seen": False,
        "last_seen_time": 0.0
    })

    def update_servo_angles(self, pan: float, tilt: float):
        """
        Update camera servo angles.

        Args:
            pan: Pan angle in degrees
            tilt: Tilt angle in degrees
        """
        self.x_angle = pan
        self.y_angle = tilt

    def update_face_detection(self, x: float, y: float, width: int, height: int):
        """
        Update with newly detected face.

        Args:
            x: Face center x coordinate
            y: Face center y coordinate
            width: Face bounding box width
            height: Face bounding box height
        """
        self.last_face = {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "seen": True,
            "last_seen_time": time.time()
        }

    def mark_face_lost(self):
        """Mark that no face is currently detected."""
        self.last_face["seen"] = False


@dataclass
class LanguageState:
    """
    Language configuration state.

    Tracks current language and STT/TTS instances for each language.
    """

    current_language: str = "en"
    lt_asr_instance: Optional[Any] = None

    def switch_language(self, lang: str, supported_langs: set):
        """
        Switch to a different language.

        Args:
            lang: Language code ("en" or "lt")
            supported_langs: Set of supported language codes

        Raises:
            ValueError: If language is not supported
        """
        if lang not in supported_langs:
            raise ValueError(f"Unsupported language: {lang}")
        self.current_language = lang


@dataclass
class InteractionState:
    """
    User interaction tracking.

    Tracks recent voice commands and executed actions.
    """

    last_voice_cmd: Optional[str] = None
    last_voice_cmd_time: float = 0.0
    last_activity_time: float = field(default_factory=time.time)
    executed_actions_this_turn: List[str] = field(default_factory=list)

    # Command processing flag - set when processing a command to skip roam observations
    command_processing: bool = False
    command_processing_start: float = 0.0
    command_processing_timeout: float = 10.0  # Auto-clear after 10 seconds

    def start_command_processing(self):
        """Mark that we're processing a command - roam should skip observations."""
        self.command_processing = True
        self.command_processing_start = time.time()

    def end_command_processing(self):
        """Mark that command processing is complete."""
        self.command_processing = False

    def is_command_processing(self) -> bool:
        """
        Check if currently processing a command.

        Auto-clears after timeout to prevent stuck state.

        Returns:
            bool: True if processing a command
        """
        if not self.command_processing:
            return False

        # Auto-clear if timed out
        if time.time() - self.command_processing_start > self.command_processing_timeout:
            self.command_processing = False
            return False

        return True

    def record_voice_command(self, command: str):
        """
        Record a voice command.

        Args:
            command: The voice command text
        """
        self.last_voice_cmd = command
        self.last_voice_cmd_time = time.time()
        self.last_activity_time = time.time()

    def track_action(self, description: str):
        """
        Track an executed action.

        Args:
            description: Description of the action executed
        """
        self.executed_actions_this_turn.append(description)

    def get_and_clear_actions(self) -> List[str]:
        """
        Get executed actions and clear for next turn.

        Returns:
            List of action descriptions from this turn
        """
        actions = list(self.executed_actions_this_turn)
        self.executed_actions_this_turn = []
        return actions

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity_time = time.time()

    def seconds_since_activity(self) -> float:
        """
        Get seconds since last activity.

        Returns:
            float: Seconds elapsed since last activity
        """
        return time.time() - self.last_activity_time


class StateManager:
    """
    Central state manager for TARS system.

    Replaces all global variables with structured state objects.
    Provides thread-safe access to all state.

    Example:
        >>> state = StateManager()
        >>> state.behavior.roaming.set()  # Start roaming
        >>> state.roam.integrate_motion(distance_cm=50)
        >>> state.interaction.track_action("forward")
    """

    def __init__(self):
        """Initialize all state containers."""
        self.hardware = HardwareState()
        self.tts = TTSState()
        self.behavior = BehaviorState()
        self.roam = RoamState()
        self.stare = StareState()
        self.language = LanguageState()
        self.interaction = InteractionState()

        self._lock = threading.Lock()

    def reset_behavior_state(self):
        """
        Reset all behavior flags (for emergency stop).

        Thread-safe method to stop all active behaviors.
        """
        with self._lock:
            self.behavior.stop_all_behaviors()

    def update_activity(self):
        """
        Update last activity timestamp.

        Should be called whenever user interaction occurs.
        """
        with self._lock:
            self.interaction.update_activity()

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of current state for debugging.

        Returns:
            Dict containing key state information
        """
        with self._lock:
            return {
                "roaming": self.behavior.roaming.is_set(),
                "following": self.behavior.following.is_set(),
                "face_tracking": self.behavior.face_tracking.is_set(),
                "language": self.language.current_language,
                "position": (self.roam.x_m, self.roam.y_m, self.roam.heading_deg),
                "seconds_since_activity": self.interaction.seconds_since_activity(),
            }
