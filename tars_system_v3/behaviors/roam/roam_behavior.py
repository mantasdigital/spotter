"""
Roam Behavior Implementation.

Autonomous exploration behavior with obstacle avoidance, spatial memory,
and curiosity-driven navigation.

Implements v58-style 6-level hierarchical obstacle avoidance:
1. Forward clear - move forward normally
2. Side escape - simple turn when first blocked
3. Stuck - more aggressive turn after repeated blocks
4. Very stuck - backward + turn
5. Hard escape - longer backward + larger turn
6. Ultra stuck - full reverse + random exploration
"""

import time
import random
import threading
import math
import statistics
from typing import Optional, Tuple, List, Callable, Any

from hardware.interfaces import IRobotCar, ICamera
from core.state_manager import StateManager
from motion.action_executor import ActionExecutor
from vision.scene_analyzer import SceneAnalyzer


class RoamBehavior:
    """
    Autonomous roaming behavior.

    Explores environment while avoiding obstacles, tracking visited areas,
    and generating observations about interesting scenes.

    Uses v58-style 6-level hierarchical obstacle avoidance for robust navigation.
    """

    # Obstacle avoidance thresholds (v58-style, more conservative)
    DIST_CLEAR = 55.0       # Path clear threshold (cm)
    DIST_CAUTION = 48.0     # Slow down threshold (cm) - increased from 45
    DIST_BLOCKED = 38.0     # Stop and turn threshold (cm) - increased from 35
    DIST_CRITICAL = 20.0    # Emergency backup threshold (cm) - increased from 18

    # Soft object detection (trash bags, fabric, etc. that absorb ultrasonic)
    SOFT_OBJECT_JUMP_CM = 50.0    # Sudden distance increase might indicate soft object
    SOFT_OBJECT_CONSECUTIVE = 3   # Consecutive suspicious readings before treating as obstacle

    # Escape escalation thresholds (number of consecutive blocks)
    SIDE_ESCAPE_BLOCKS = 1      # Simple turn
    STUCK_BLOCKS = 3            # More aggressive turn
    VERY_STUCK_BLOCKS = 5       # Backward + turn
    HARD_ESCAPE_BLOCKS = 8      # Longer backward + larger turn
    ULTRA_STUCK_BLOCKS = 12     # Full reverse + random exploration

    # Timing constants
    BLOCKED_HEADING_DECAY_SEC = 20.0   # How long to remember blocked headings
    PATH_REPETITION_WINDOW_SEC = 60.0  # Window for detecting path repetition
    CLEAR_TIMEOUT_SEC = 5.0            # Seconds of clear path to start reducing block count
    CLEAR_COUNT_TO_RESET = 10          # Number of consecutive clear readings to fully reset block count

    # Hidden obstacle detection (obstacles not visible to ultrasonic sensor)
    STUCK_FORWARD_COUNT_THRESHOLD = 15  # Forward commands without obstacles
    STUCK_PROBE_BACKUP_TIME = 0.5       # Seconds to backup when probing
    STUCK_PROBE_TURN_TIME = 0.4         # Seconds to turn during probe

    # Exploration variety - don't go straight forever on clear paths
    WANDER_TRIGGER_COUNT = 8            # After this many straight forwards, maybe turn
    WANDER_CHANCE = 0.4                 # 40% chance to wander when triggered
    WANDER_ANGLES = [30, -30, 45, -45, 60, -60]  # Possible wander turn angles

    # Roam timeout (v58-style)
    MAX_ROAM_DURATION_SEC = 600.0       # 10 minutes max roaming time

    # Curiosity detection tags (v58-style)
    CURIOUS_TAGS = {
        "person", "face", "human",
        "pet", "animal", "cat", "dog",
        "gnome", "figurine", "toy",
        "mushroom", "creature", "plush",
    }

    def __init__(
        self,
        car: IRobotCar,
        camera: ICamera,
        state: StateManager,
        action_executor: ActionExecutor,
        scene_analyzer: Optional[SceneAnalyzer] = None,
        visual_memory: Optional[Any] = None,
        gesture_detector: Optional[Any] = None,
        forward_speed: int = 30,
        turn_speed: int = 25,
        obstacle_distance_cm: float = 20.0,
        cliff_check_enabled: bool = True,
        observation_interval_sec: float = 6.0,
        speak_callback: Optional[Callable[[str], None]] = None,
        max_roam_duration_sec: Optional[float] = None,
        curiosity_enabled: bool = True,
        on_timeout_callback: Optional[Callable[[], None]] = None,
        on_curiosity_callback: Optional[Callable[[str, Any], None]] = None,
        language_getter: Optional[Callable[[], str]] = None
    ):
        """
        Initialize roam behavior.

        Args:
            car: Robot car hardware interface
            camera: Camera interface
            state: State manager
            action_executor: Motion action executor
            scene_analyzer: Optional scene analyzer for observations
            visual_memory: Optional VisualMemory for saving observations
            gesture_detector: Optional GestureDetector for hand-in-face stop
            forward_speed: Speed for forward movement (0-100)
            turn_speed: Speed for turning (0-100)
            obstacle_distance_cm: Distance threshold for obstacle detection
            cliff_check_enabled: Whether to check for cliffs
            observation_interval_sec: How often to speak observations (default 6s like v58)
            speak_callback: Optional callback to speak observations (tts.speak)
            max_roam_duration_sec: Maximum roaming time before auto-stop (default 10 min)
            curiosity_enabled: Whether to stop when interesting objects are detected
            on_timeout_callback: Callback when roam times out
            on_curiosity_callback: Callback when curiosity is triggered (description, frame)
            language_getter: Optional callback to get current language code ("en", "lt")
        """
        self.car = car
        self.camera = camera
        self.state = state
        self.executor = action_executor
        self.scene_analyzer = scene_analyzer
        self.visual_memory = visual_memory
        self.gesture_detector = gesture_detector

        self.forward_speed = forward_speed
        self.turn_speed = turn_speed
        self.obstacle_distance_cm = obstacle_distance_cm
        self.cliff_check_enabled = cliff_check_enabled
        self.observation_interval_sec = observation_interval_sec
        self.speak_callback = speak_callback
        self.max_roam_duration_sec = max_roam_duration_sec if max_roam_duration_sec is not None else self.MAX_ROAM_DURATION_SEC
        self.curiosity_enabled = curiosity_enabled
        self.on_timeout_callback = on_timeout_callback
        self.on_curiosity_callback = on_curiosity_callback
        self.language_getter = language_getter

        self._stop_event = threading.Event()
        self._last_observation_time = 0.0
        self._roam_start_time = 0.0

        # Path repetition tracking (v58-style)
        self._recent_positions: List[Tuple[float, float, float, float]] = []  # (time, x, y, heading)
        self._last_escape_direction = 0  # Last escape turn direction (for alternation)

        # Escape state flag - disables gesture detection during escapes
        self._escaping = False

        # Cliff detection state - require consecutive readings to avoid false positives
        self._consecutive_cliff_count = 0
        self._last_cliff_time = 0.0

        # Hidden obstacle detection (obstacles not visible to sensor)
        self._clear_forward_count = 0    # Consecutive forwards without obstacles

        # Soft object detection (erratic ultrasonic readings)
        self._recent_distances: List[float] = []  # Last few distance readings
        self._soft_object_suspect_count = 0  # Consecutive suspicious readings

        # Exploration variety - track straight movement for wander triggers
        self._straight_forward_count = 0  # Consecutive forwards without turning

        # Non-blocking observation (prevent LLM from blocking roam loop)
        self._observation_in_progress = False
        self._observation_thread: Optional[threading.Thread] = None

        # Voice overlap prevention - use global TTS state for coordination
        # (local tracking removed - now uses state.tts for global coordination)
        self._last_speak_time = 0.0
        self._min_speak_interval = 3.0  # Minimum seconds between speech

        # Curiosity tracking (v58-style)
        self._curious_target_seen = False
        self._curious_last_snapshot_time = 0.0

    def start(self):
        """
        Start roaming behavior in background thread.

        Returns immediately, roaming runs in daemon thread.
        """
        if self.state.behavior.roaming.is_set():
            print("[ROAM] Roaming already active")
            return

        # Reset roam state
        self.state.roam.reset()

        # Reset local tracking
        self._recent_positions = []
        self._last_escape_direction = 0
        self._last_observation_time = 0.0
        self._clear_forward_count = 0
        self._observation_in_progress = False
        self._step_count = 0
        self._roam_start_time = time.time()
        self._recent_distances = []  # Reset soft object tracking
        self._soft_object_suspect_count = 0
        self._straight_forward_count = 0  # Reset wander tracking

        # Reset voice timing (global TTS state managed centrally)
        self._last_speak_time = 0.0

        # Reset curiosity tracking (v58-style)
        self._curious_target_seen = False
        self._curious_last_snapshot_time = 0.0

        # Reset gesture detector debounce
        if self.gesture_detector:
            self.gesture_detector.reset_debounce()

        # Set roaming flag
        self.state.behavior.roaming.set()
        self._stop_event.clear()

        # Start roam thread
        thread = threading.Thread(
            target=self._roam_loop,
            name="RoamThread",
            daemon=True
        )
        thread.start()
        self.state.roam.roam_thread = thread

        print("[ROAM] Roaming started with v58-style obstacle avoidance")

    def stop(self):
        """Stop roaming behavior."""
        if not self.state.behavior.roaming.is_set():
            return

        # Signal stop
        self._stop_event.set()
        self.state.behavior.roaming.clear()

        # Stop motors
        self.car.stop()

        print("[ROAM] Roaming stopped")

    def _roam_loop(self):
        """
        Main roaming loop.

        Continuously explores environment until stopped.
        Includes timeout check (v58-style: 10 minutes max).
        """
        try:
            while not self._stop_event.is_set() and self.state.behavior.roaming.is_set():
                # Check for global stop
                if self.state.behavior.global_stop.is_set():
                    break

                # Check for roam timeout (v58-style)
                elapsed = time.time() - self._roam_start_time
                if elapsed > self.max_roam_duration_sec:
                    print(f"[ROAM] Timeout reached ({int(elapsed)}s), auto-stopping")
                    # Get current language for appropriate message
                    timeout_msg = "Roam mode timeout. I am bored now."
                    if self.language_getter:
                        try:
                            if self.language_getter() == "lt":
                                timeout_msg = "Klajojimo režimo laikas baigėsi. Man nuobodu."
                        except Exception:
                            pass
                    # Use force=True for important timeout message
                    self._safe_speak(timeout_msg, force=True)
                    if self.on_timeout_callback:
                        try:
                            self.on_timeout_callback()
                        except Exception as e:
                            print(f"[ROAM] Timeout callback error: {e}")
                    break

                # Execute one roam step
                self._roam_step()

                # Small delay between iterations
                time.sleep(0.1)

        except Exception as e:
            print(f"Roam error: {e}")

        finally:
            # Ensure motors stop
            self.car.stop()
            self.state.behavior.roaming.clear()

    def _roam_step(self):
        """
        Execute one step of roaming behavior.

        Uses v58-style hierarchical obstacle avoidance with 6 escalation levels.
        """
        now = time.time()

        # Clean up old blocked headings (decay after BLOCKED_HEADING_DECAY_SEC)
        self.state.roam.blocked_headings = [
            (h, t) for h, t in self.state.roam.blocked_headings
            if now - t < self.BLOCKED_HEADING_DECAY_SEC
        ]

        # Clean up old position history (keep only PATH_REPETITION_WINDOW_SEC)
        self._recent_positions = [
            p for p in self._recent_positions
            if now - p[0] < self.PATH_REPETITION_WINDOW_SEC
        ]

        # Check for obstacles FIRST (before gesture detection)
        # This allows us to pass distance to gesture detector to avoid false positives
        distance = self.car.get_distance()

        # Handle invalid distance readings (None, negative, very high)
        # CRITICAL: Invalid readings often mean sensor is TOO CLOSE to obstacle
        # (ultrasonic can't read when object is < ~2cm away)
        distance_valid = True
        if distance is None:
            distance = -1.0
            distance_valid = False
        else:
            try:
                distance = float(distance)
                # Negative or very small values = sensor error (too close)
                # Very high values (>400cm) = no echo (open space or error)
                if distance < 0 or distance > 400:
                    distance_valid = False
            except (TypeError, ValueError):
                distance = -1.0
                distance_valid = False

        # Track consecutive invalid readings to detect "stuck against obstacle"
        if not hasattr(self, '_invalid_distance_count'):
            self._invalid_distance_count = 0

        if not distance_valid:
            self._invalid_distance_count += 1
        else:
            self._invalid_distance_count = 0

        # Debug: Log distance reading (every 3rd step)
        if not hasattr(self, '_step_count'):
            self._step_count = 0
        self._step_count += 1
        if self._step_count % 3 == 0:
            valid_str = "OK" if distance_valid else "INVALID"
            print(f"[ROAM] dist={distance:.0f}cm ({valid_str}) step={self._step_count}", flush=True)

        # Check for hand-in-face stop gesture (skip if currently escaping obstacle)
        # Pass distance to gesture detector so it can disable low-variance detection
        # when robot is close to obstacle (low variance is expected near walls/floors)
        if self.gesture_detector and self.camera.is_active() and not self._escaping:
            try:
                frame = self.camera.capture_frame()
                if frame is not None:
                    # Handle XBGR8888 format
                    if len(frame.shape) == 3 and frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                    # Pass distance to gesture detector for context-aware detection
                    self.gesture_detector.set_obstacle_distance(distance)
                    if self.gesture_detector.check_and_handle(frame, "roam"):
                        print("[ROAM] Gesture stop triggered!", flush=True)
                        self.car.stop()
                        self.stop()
                        return
            except Exception as e:
                print(f"[ROAM] Gesture check error: {e}", flush=True)

        # Check for cliff if enabled (with consecutive reading requirement)
        if self.cliff_check_enabled:
            grayscale = self.car.get_grayscale_data()

            # Debug: Log grayscale values occasionally
            if self._step_count % 10 == 0 and grayscale:
                print(f"[ROAM] Grayscale: {grayscale}", flush=True)

            if grayscale and self.car.get_cliff_status(grayscale):
                self._consecutive_cliff_count += 1

                # Immediate response on first detection (cliff is dangerous!)
                # But add cooldown - don't trigger again within 2 seconds
                if now - self._last_cliff_time > 2.0:
                    print(f"[ROAM] CLIFF DETECTED! Sensors: {grayscale} (count={self._consecutive_cliff_count})")
                    self._last_cliff_time = now
                    self._consecutive_cliff_count = 0
                    self._handle_cliff()
                    return
                else:
                    # In cooldown but still seeing cliff - keep count but don't trigger
                    if self._consecutive_cliff_count > 3:
                        print(f"[ROAM] Cliff persists during cooldown: {grayscale}")
                        self._consecutive_cliff_count = 0
            else:
                # No cliff - reset consecutive count
                self._consecutive_cliff_count = 0

        # CRITICAL: Invalid distance readings likely mean we're TOO CLOSE to obstacle
        # Treat as emergency backup situation after 2+ consecutive invalid readings
        if not distance_valid and self._invalid_distance_count >= 2:
            print(f"[ROAM] SENSOR ERROR (dist={distance:.0f}cm) - likely too close, backing up!", flush=True)
            self._handle_emergency_backup()
            self._clear_forward_count = 0
            self._recent_distances = []  # Reset soft object tracking
            return

        # SOFT OBJECT DETECTION: Trash bags, fabric, etc. that absorb ultrasonic pulses
        # These give erratic readings or sudden distance "jumps" (object absorbs pulse → appears far)
        if distance_valid:
            # Track recent distances (keep last 5)
            self._recent_distances.append(distance)
            if len(self._recent_distances) > 5:
                self._recent_distances.pop(0)

            # Check for suspicious patterns that indicate soft object
            is_soft_object_suspect = False

            if len(self._recent_distances) >= 3:
                # Pattern 1: Sudden jump - previous readings were close, now suddenly far
                # This happens when soft object absorbs the ultrasonic pulse
                prev_readings = self._recent_distances[-3:-1]  # 2nd and 3rd most recent
                prev_avg = sum(prev_readings) / len(prev_readings)
                if prev_avg < self.DIST_BLOCKED and distance > prev_avg + self.SOFT_OBJECT_JUMP_CM:
                    print(f"[ROAM] Soft object suspect: jump from {prev_avg:.0f}cm to {distance:.0f}cm", flush=True)
                    is_soft_object_suspect = True

                # Pattern 2: High variance in recent readings (erratic = soft surface)
                if len(self._recent_distances) >= 4:
                    variance = statistics.variance(self._recent_distances)
                    if variance > 400:  # High variance threshold
                        recent_min = min(self._recent_distances)
                        if recent_min < self.DIST_CAUTION:  # Only if some readings were close
                            print(f"[ROAM] Soft object suspect: high variance={variance:.0f}, min={recent_min:.0f}cm", flush=True)
                            is_soft_object_suspect = True

            if is_soft_object_suspect:
                self._soft_object_suspect_count += 1
                if self._soft_object_suspect_count >= self.SOFT_OBJECT_CONSECUTIVE:
                    print(f"[ROAM] SOFT OBJECT DETECTED after {self._soft_object_suspect_count} suspicious readings!", flush=True)
                    self._soft_object_suspect_count = 0
                    self._recent_distances = []
                    # Treat as blocked obstacle
                    self._handle_obstacle_hierarchical(self.DIST_BLOCKED)
                    self._clear_forward_count = 0
                    return
            else:
                # Reset if reading seems normal
                self._soft_object_suspect_count = 0

        # Hierarchical obstacle avoidance (v58-style 6 levels)
        if distance_valid and distance < self.DIST_CRITICAL:
            # Level 6: Ultra stuck - critical distance, emergency backup
            print(f"[ROAM] CRITICAL at {distance:.0f}cm - backup!", flush=True)
            self._handle_emergency_backup()
            self._clear_forward_count = 0
        elif distance_valid and distance < self.DIST_BLOCKED:
            # Obstacle detected - escalate based on block count
            print(f"[ROAM] BLOCKED at {distance:.0f}cm - escaping", flush=True)
            self._handle_obstacle_hierarchical(distance)
            self._clear_forward_count = 0
            self._consecutive_clear_count = 0  # Reset clear count when blocked
            self._straight_forward_count = 0   # Reset wander count after obstacle
        elif distance_valid and distance < self.DIST_CAUTION:
            # Caution zone - slow down
            if self._step_count % 10 == 0:
                print(f"[ROAM] CAUTION at {distance:.0f}cm - slow", flush=True)
            self._move_forward(speed_factor=0.6)
            # Caution zone doesn't count as "clear" - reset consecutive clear count
            self._consecutive_clear_count = 0
            self._clear_forward_count += 1
        else:
            # Path clear - move forward normally
            self._move_forward()
            self._clear_forward_count += 1
            self._straight_forward_count += 1

            # EXPLORATION VARIETY: Don't go straight forever - occasionally wander
            if self._straight_forward_count >= self.WANDER_TRIGGER_COUNT:
                if random.random() < self.WANDER_CHANCE:
                    # Time to explore a different direction!
                    wander_angle = random.choice(self.WANDER_ANGLES)
                    print(f"[ROAM] Curiosity wander! Turning {wander_angle} deg to explore", flush=True)
                    self._execute_turn(wander_angle)
                    self._straight_forward_count = 0  # Reset after wander
                else:
                    # Didn't wander this time, but reset counter to check again later
                    self._straight_forward_count = self.WANDER_TRIGGER_COUNT // 2

            # Track consecutive clear readings for gradual block count reduction
            if not hasattr(self, '_consecutive_clear_count'):
                self._consecutive_clear_count = 0
            self._consecutive_clear_count += 1

            # Gradually reduce block count after sustained clear path (not instant reset)
            if self._consecutive_clear_count >= self.CLEAR_COUNT_TO_RESET:
                # Fully reset after many clear readings
                if self.state.roam.forward_block_count > 0:
                    self.state.roam.forward_block_count = 0
                    print("[ROAM] Block count reset after sustained clear path")
            elif self._consecutive_clear_count >= 5 and now - self.state.roam.last_clear_time > self.CLEAR_TIMEOUT_SEC:
                # Reduce by 1 after some clear readings
                self.state.roam.forward_block_count = max(0, self.state.roam.forward_block_count - 1)

            self.state.roam.last_clear_time = now

        # Check for hidden obstacle (stuck on something sensor can't see)
        if self._clear_forward_count >= self.STUCK_FORWARD_COUNT_THRESHOLD:
            print(f"[ROAM] Hidden obstacle check: {self._clear_forward_count} clear forwards")
            self._probe_hidden_obstacle()

        # Update spatial memory
        self.state.roam.mark_visited()

        # Track position for path repetition detection
        self._recent_positions.append((
            now,
            self.state.roam.x_m,
            self.state.roam.y_m,
            self.state.roam.heading_deg
        ))

        # Periodic observations and memory saving (timer-based like v58)
        if (now - self._last_observation_time) >= self.observation_interval_sec:
            self._capture_observation()
            self._last_observation_time = now

    def _safe_speak(self, text: str, force: bool = False) -> bool:
        """
        Speak text without overlapping with other speech.

        Uses GLOBAL TTS state to coordinate with command responses.
        Prevents voice overlapping by checking if ANY TTS is busy
        (including command responses) and enforcing minimum interval.

        CRITICAL: Command speech ALWAYS has priority over roam observations.
        Even with force=True, roam speech is blocked if a command is being processed.

        Args:
            text: Text to speak
            force: If True, skip overlap check (for important messages like timeout)
                   Note: force=True does NOT override command priority

        Returns:
            bool: True if speech was initiated, False if skipped
        """
        if not self.speak_callback or not text:
            return False

        now = time.time()

        # PRIORITY CHECK: Command processing ALWAYS blocks roam speech
        # Even force=True cannot override - commands take absolute priority
        if self.state.interaction.is_command_processing():
            print(f"[ROAM] Skipping speech (command processing): {text[:50]}...")
            return False

        # Check if command response is currently speaking - never talk over it
        if self.state.tts.is_currently_speaking():
            source = self.state.tts.get_speaking_source()
            if source == "command":
                print(f"[ROAM] Skipping speech (command speaking): {text[:50]}...")
                return False
            # If roam is speaking, skip unless forced (important messages)
            if not force:
                print(f"[ROAM] Skipping speech (TTS busy with {source}): {text[:50]}...")
                return False

        # Check minimum interval between roam speech (unless forced)
        if not force and (now - self._last_speak_time) < self._min_speak_interval:
            print(f"[ROAM] Skipping speech (too soon): {text[:50]}...")
            return False

        # Try to acquire global TTS lock
        if not self.state.tts.start_speaking("roam"):
            print(f"[ROAM] Skipping speech (couldn't acquire TTS lock): {text[:50]}...")
            return False

        self._last_speak_time = now

        # Speak in background to not block roam loop
        def speak_thread():
            try:
                # Double-check command priority before speaking
                # (command may have started while we were acquiring lock)
                if self.state.interaction.is_command_processing():
                    print(f"[ROAM] Aborting speech (command started): {text[:50]}...")
                    return
                self.speak_callback(text)
            except Exception as e:
                print(f"[ROAM] TTS error: {e}")
            finally:
                # Release global TTS lock
                self.state.tts.stop_speaking()

        thread = threading.Thread(target=speak_thread, daemon=True)
        thread.start()
        return True

    def _move_forward(self, speed_factor: float = 1.0):
        """
        Move forward and update odometry.

        Args:
            speed_factor: Multiplier for speed (0.0-1.0), e.g., 0.6 for caution zone
        """
        speed = int(self.forward_speed * speed_factor)
        speed = max(10, min(100, speed))  # Clamp to valid range

        # Log every movement for debugging
        if self._step_count % 5 == 0:
            print(f"[ROAM] FWD spd={speed}", flush=True)

        # Move forward for a short duration
        self.car.forward(speed)
        time.sleep(0.3)

        # Update odometry (rough estimate: 10cm per 0.3s at speed 30)
        distance_cm = (speed / 100.0) * 33.0  # Rough calibration
        self.state.roam.integrate_motion(distance_cm)

    def _handle_obstacle_hierarchical(self, distance: float):
        """
        Handle obstacle with v58-style hierarchical escalation.

        Escalation levels based on block count:
        1. Side escape (1-2 blocks): Simple turn
        2. Stuck (3-4 blocks): More aggressive turn
        3. Very stuck (5-7 blocks): Backward + turn
        4. Hard escape (8-11 blocks): Longer backward + larger turn
        5. Ultra stuck (12+ blocks): Full reverse + random exploration

        Args:
            distance: Distance to obstacle in cm
        """
        self._escaping = True  # Disable gesture detection during escape
        self.car.stop()

        # Increment block count
        self.state.roam.forward_block_count += 1
        block_count = self.state.roam.forward_block_count

        # Record blocked heading
        self.state.roam.blocked_headings.append(
            (self.state.roam.heading_deg, time.time())
        )

        print(f"[ROAM] Obstacle at {distance:.1f}cm, block count: {block_count}")

        # Check for path repetition (same area visited recently)
        is_repeating = self._check_path_repetition()
        if is_repeating:
            # Boost escalation level if we're repeating paths
            block_count = max(block_count, self.VERY_STUCK_BLOCKS)
            print("[ROAM] Path repetition detected - escalating escape")

        # Hierarchical escape based on block count
        if block_count >= self.ULTRA_STUCK_BLOCKS:
            # Level 6: Ultra stuck - full reverse + random exploration
            print(f"[ROAM] Level 6: Ultra stuck ({block_count} blocks)")
            self._escape_ultra_stuck()
        elif block_count >= self.HARD_ESCAPE_BLOCKS:
            # Level 5: Hard escape - longer backward + larger turn
            print(f"[ROAM] Level 5: Hard escape ({block_count} blocks)")
            self._escape_hard()
        elif block_count >= self.VERY_STUCK_BLOCKS:
            # Level 4: Very stuck - backward + turn
            print(f"[ROAM] Level 4: Very stuck ({block_count} blocks)")
            self._escape_very_stuck()
        elif block_count >= self.STUCK_BLOCKS:
            # Level 3: Stuck - more aggressive turn
            print(f"[ROAM] Level 3: Stuck ({block_count} blocks)")
            self._escape_stuck()
        else:
            # Level 2: Side escape - simple turn
            print(f"[ROAM] Level 2: Side escape ({block_count} blocks)")
            self._escape_side()

        # Re-enable gesture detection after escape completes
        self._escaping = False

    def _handle_cliff(self):
        """Handle cliff detection - back up aggressively and turn."""
        self._escaping = True
        print("[ROAM] CLIFF! Emergency backup...")

        # STOP immediately
        self.car.stop()
        time.sleep(0.1)  # Brief pause

        # Back up aggressively (cliff is very dangerous!)
        self.car.backward(self.forward_speed)  # Use forward speed for faster backup
        time.sleep(1.2)  # Increased from 0.8 for safety
        self.car.stop()

        # Update odometry
        self.state.roam.integrate_motion(-40.0)  # More backup distance

        # Increment block count significantly (cliff is very serious)
        self.state.roam.forward_block_count += 3

        # Clear recent distances to avoid false soft object detection after cliff
        self._recent_distances = []

        # Turn away from cliff with large angle
        turn_angle = random.choice([120, -120, 150, -150])
        self._execute_turn(turn_angle)

        self._escaping = False

    def _handle_emergency_backup(self):
        """
        Emergency backup when at critical distance.

        Used when obstacle is very close (< DIST_CRITICAL).
        """
        self._escaping = True

        # Back up immediately (longer duration like v58: 1.0-1.5s)
        self.car.backward(self.forward_speed)
        time.sleep(1.0)  # Increased from 0.6
        self.car.stop()

        # Update odometry
        self.state.roam.integrate_motion(-35.0)

        # Record blocked heading
        self.state.roam.blocked_headings.append(
            (self.state.roam.heading_deg, time.time())
        )

        # Increment block count significantly
        self.state.roam.forward_block_count += 3

        # Turn away
        turn_angle = self._choose_turn_direction()
        self._execute_turn(turn_angle)

        self._escaping = False

    def _escape_side(self):
        """
        Level 2: Simple side escape.

        Back up slightly then turn to avoid obstacle, alternating direction.
        """
        # Back up first (v58-style - always backup before turning)
        self.car.backward(self.turn_speed)
        time.sleep(0.5)  # Short backup
        self.car.stop()
        self.state.roam.integrate_motion(-15.0)

        # Alternate turn direction to avoid getting stuck in loops
        self._last_escape_direction = -self._last_escape_direction
        if self._last_escape_direction == 0:
            self._last_escape_direction = random.choice([1, -1])

        turn_angle = 60 * self._last_escape_direction  # 60 degrees
        self._execute_turn(turn_angle)

    def _escape_stuck(self):
        """
        Level 3: Stuck escape.

        Back up then more aggressive turn when repeatedly blocked.
        """
        # Back up first (longer than Level 2)
        self.car.backward(self.turn_speed)
        time.sleep(0.7)
        self.car.stop()
        self.state.roam.integrate_motion(-20.0)

        # Use smart turn direction based on blocked headings
        turn_angle = self._choose_turn_direction()
        # Make it at least 90 degrees
        if abs(turn_angle) < 90:
            turn_angle = 90 if turn_angle > 0 else -90

        self._execute_turn(turn_angle)

    def _escape_very_stuck(self):
        """
        Level 4: Very stuck escape.

        Longer back up before turning.
        """
        # Back up first (longer duration)
        self.car.backward(self.forward_speed)
        time.sleep(1.0)  # Increased from 0.4
        self.car.stop()
        self.state.roam.integrate_motion(-30.0)

        # Then turn more aggressively
        turn_angle = self._choose_turn_direction()
        # Make it at least 120 degrees
        if abs(turn_angle) < 120:
            turn_angle = 120 if turn_angle > 0 else -120

        self._execute_turn(turn_angle)

    def _escape_hard(self):
        """
        Level 5: Hard escape.

        Longer backward movement + larger turn.
        """
        # Back up more (extended duration)
        self.car.backward(self.forward_speed)
        time.sleep(1.2)  # Increased from 0.7
        self.car.stop()
        self.state.roam.integrate_motion(-40.0)

        # Turn significantly
        turn_angle = self._choose_turn_direction()
        # Make it at least 150 degrees
        if abs(turn_angle) < 150:
            turn_angle = 150 if turn_angle > 0 else -150

        self._execute_turn(turn_angle)

        # Clear some blocked headings to allow retrying after major turn
        if len(self.state.roam.blocked_headings) > 3:
            self.state.roam.blocked_headings = self.state.roam.blocked_headings[-3:]

    def _escape_ultra_stuck(self):
        """
        Level 6: Ultra stuck escape.

        Full reverse + random exploration. Resets state for fresh start.
        """
        print("[ROAM] Ultra stuck - full reset escape")

        # Full reverse (extended duration)
        self.car.backward(self.forward_speed)
        time.sleep(1.5)  # Increased from 1.0
        self.car.stop()
        self.state.roam.integrate_motion(-50.0)

        # Random large turn
        turn_angle = random.choice([160, -160, 180])
        self._execute_turn(turn_angle)

        # Reset escape state for fresh start
        self.state.roam.forward_block_count = 0
        self.state.roam.blocked_headings = []
        self._last_escape_direction = 0

    def _probe_hidden_obstacle(self):
        """
        Probe for hidden obstacle when many forwards without sensor detection.

        This handles obstacles not visible to ultrasonic sensor (e.g., keyboard
        blocking wheel from the side, low objects, etc.).

        Strategy:
        1. Back up a bit
        2. Turn to side
        3. Try forward again
        4. If still stuck after probe, escalate to full escape
        """
        print("[ROAM] Probing for hidden obstacle...")

        # Reset counter
        self._clear_forward_count = 0

        # Back up first
        self.car.backward(self.forward_speed)
        time.sleep(self.STUCK_PROBE_BACKUP_TIME)
        self.car.stop()

        # Update odometry
        self.state.roam.integrate_motion(-15.0)

        # Turn to side (alternate direction)
        self._last_escape_direction = -self._last_escape_direction
        if self._last_escape_direction == 0:
            self._last_escape_direction = random.choice([1, -1])

        if self._last_escape_direction > 0:
            self.car.set_dir_servo_angle(25)  # Turn left
        else:
            self.car.set_dir_servo_angle(-25)  # Turn right

        self.car.forward(self.turn_speed)
        time.sleep(self.STUCK_PROBE_TURN_TIME)
        self.car.stop()
        self.car.set_dir_servo_angle(0)

        # Update odometry with turn
        self.state.roam.integrate_motion(10.0, 45 * self._last_escape_direction)

        # Increment block count to escalate if this keeps happening
        self.state.roam.forward_block_count += 1

        print(f"[ROAM] Hidden obstacle probe complete, block count now: {self.state.roam.forward_block_count}")

    def _check_path_repetition(self) -> bool:
        """
        Check if robot is repeating the same path.

        Detects if we've been in similar positions recently (within 25cm and 45 degrees).

        Returns:
            bool: True if path repetition detected
        """
        if len(self._recent_positions) < 10:
            return False

        # Current position
        cur_x = self.state.roam.x_m
        cur_y = self.state.roam.y_m
        cur_heading = self.state.roam.heading_deg

        # Check against older positions (exclude last 5 to avoid immediate detection)
        similar_count = 0
        for t, x, y, heading in self._recent_positions[:-5]:
            # Distance check (within 25cm = 0.25m)
            dist = math.sqrt((cur_x - x) ** 2 + (cur_y - y) ** 2)
            if dist < 0.25:
                # Heading check (within 45 degrees)
                heading_diff = abs(cur_heading - heading)
                heading_diff = min(heading_diff, 360 - heading_diff)
                if heading_diff < 45:
                    similar_count += 1

        # If we've been in similar position 3+ times, we're repeating
        return similar_count >= 3

    def _choose_turn_direction(self) -> int:
        """
        Choose which direction to turn.

        Uses visit history and blocked headings to prefer unexplored directions.

        Returns:
            Turn angle in degrees (positive = left, negative = right)
        """
        # Get visit counts for potential headings
        candidates = [90, -90, 135, -135, 180]  # Turn angles to consider

        # Score each candidate
        scores = []
        for turn_angle in candidates:
            new_heading = (self.state.roam.heading_deg + turn_angle) % 360.0

            # Check if this heading was recently blocked
            blocked = False
            for blocked_heading, _ in self.state.roam.blocked_headings:
                angle_diff = abs(new_heading - blocked_heading)
                if angle_diff < 45 or angle_diff > 315:  # Within 45 degrees
                    blocked = True
                    break

            if blocked:
                score = -100  # Heavily penalize blocked directions
            else:
                # Prefer less-visited directions
                # (This is simplified - could check actual grid cells)
                visit_count = self.state.roam.get_visit_count()
                score = -visit_count + random.random() * 10  # Add randomness

            scores.append((turn_angle, score))

        # Choose highest scoring direction
        best_turn = max(scores, key=lambda x: x[1])[0]
        return best_turn

    def _execute_turn(self, turn_angle: int):
        """
        Execute a turn.

        Args:
            turn_angle: Angle to turn in degrees (positive = left, negative = right)
        """
        # Determine turn direction
        if turn_angle > 0:
            # Turn left
            self.car.set_dir_servo_angle(30)  # Max left
        else:
            # Turn right
            self.car.set_dir_servo_angle(-30)  # Max right

        # Move forward while turning
        self.car.forward(self.turn_speed)

        # Calculate turn duration based on angle
        # Rough calibration: 90 degrees takes ~0.5s
        duration = abs(turn_angle) / 180.0

        time.sleep(duration)

        # Stop and center steering
        self.car.stop()
        self.car.set_dir_servo_angle(0)

        # Update odometry
        self.state.roam.integrate_motion(
            distance_cm=10.0,  # Small forward motion during turn
            turn_deg=turn_angle
        )

    def _capture_observation(self):
        """
        Start non-blocking observation capture.

        Spawns a background thread to capture and analyze scene,
        so it doesn't block the roam loop.
        """
        # Skip if observation already in progress
        if self._observation_in_progress:
            return

        # Skip if command is being processed - don't talk over command responses
        if self.state.interaction.is_command_processing():
            print("[ROAM] Skipping observation - command being processed")
            return

        # Capture frame NOW (fast) before starting background analysis
        try:
            if not self.camera.is_active():
                return

            frame = self.camera.capture_frame()
            if frame is None:
                return

            # Handle XBGR8888 format (4 channels)
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]

            # Get current position for tagging
            grid_cell = self.state.roam.get_grid_cell()
            obs_num = self.state.roam.state_dict.get("obs_count", 0) + 1

            # Mark observation in progress
            self._observation_in_progress = True

            # Start background thread for LLM analysis
            self._observation_thread = threading.Thread(
                target=self._analyze_observation_background,
                args=(frame.copy(), grid_cell, obs_num),
                daemon=True
            )
            self._observation_thread.start()

        except Exception as e:
            print(f"[ROAM] Observation capture error: {e}")
            self._observation_in_progress = False

    def _analyze_observation_background(self, frame, grid_cell, obs_num):
        """
        Analyze scene in background thread (non-blocking).

        Includes v58-style curiosity detection for interesting objects.

        Args:
            frame: Captured image frame
            grid_cell: Grid cell position when captured
            obs_num: Observation number
        """
        try:
            description = None

            # Analyze scene if scene analyzer available (this is the slow LLM call)
            if self.scene_analyzer:
                try:
                    # Get current language for appropriate prompt
                    current_lang = "en"
                    if self.language_getter:
                        try:
                            current_lang = self.language_getter()
                        except Exception:
                            pass

                    # Use language-appropriate prompt
                    if current_lang == "lt":
                        prompt = "Vienu sakiniu apibūdink ką matai ir pridėk sąmojingą pastabą. Atsakyk lietuviškai."
                    else:
                        prompt = "In one sentence, describe what you see and add a witty remark."

                    description = self.scene_analyzer.analyze_frame(frame, prompt=prompt)
                except Exception as analyze_err:
                    print(f"[ROAM] Scene analysis error: {analyze_err}")
                    description = "Exploring environment" if not self.language_getter or self.language_getter() != "lt" else "Tyrinėju aplinką"

            # Store as last observation
            self.state.roam.state_dict["last_obs_time"] = time.time()
            self.state.roam.state_dict["obs_count"] = obs_num
            self.state.roam.state_dict["last_observation"] = description

            print(f"[ROAM] Observation: {description}")

            # Save to visual memory with auto-generated tags
            tags = []
            if self.visual_memory and description:
                try:
                    tags = ["roam", "exploration"]
                    tags.append(f"pos_{grid_cell[0]}_{grid_cell[1]}")
                    tags.append(f"observation_{obs_num}")

                    # Extract key words from description for tags
                    if description:
                        desc_lower = description.lower()
                        for keyword in ["person", "face", "wall", "door", "table", "chair",
                                       "floor", "window", "light", "object", "room",
                                       "cat", "dog", "pet", "gnome", "toy", "figurine"]:
                            if keyword in desc_lower:
                                tags.append(keyword)

                    self.visual_memory.add_visual(
                        image=frame,
                        description=description,
                        tags=tags,
                        save_image=True
                    )
                    print(f"[ROAM] Saved to visual memory with tags: {tags}")

                except Exception as mem_err:
                    print(f"[ROAM] Visual memory save error: {mem_err}")

            # Skip speaking if command started during analysis
            if self.state.interaction.is_command_processing():
                print(f"[ROAM] Skipping observation speech (command started during analysis)")
                return

            # Check for curiosity (v58-style)
            if self.curiosity_enabled and not self._curious_target_seen:
                is_curious = self._is_curious_scene(tags, description)
                if is_curious:
                    self._handle_curiosity(frame, description, tags)
                    return  # Skip normal observation speech

            # Speak observation using safe speak (prevents overlapping)
            if description:
                self._safe_speak(description)

        except Exception as e:
            print(f"[ROAM] Observation error: {e}")
        finally:
            self._observation_in_progress = False

    def _is_curious_scene(self, tags: List[str], description: Optional[str]) -> bool:
        """
        Check if scene contains something interesting (v58-style curiosity).

        Args:
            tags: List of detected tags
            description: Scene description

        Returns:
            bool: True if scene contains curious/interesting objects
        """
        # Check tags
        for tag in tags or []:
            if tag.lower() in self.CURIOUS_TAGS:
                return True

        # Check description text
        if description:
            desc_lower = description.lower()
            for keyword in self.CURIOUS_TAGS:
                if keyword in desc_lower:
                    return True

        return False

    def _handle_curiosity(self, frame, description: str, tags: List[str]):
        """
        Handle curiosity detection - interesting object found (v58-style).

        Args:
            frame: Captured image frame
            description: Scene description
            tags: List of detected tags
        """
        self._curious_target_seen = True
        self._curious_last_snapshot_time = time.time()

        # Get current language for appropriate message
        current_lang = "en"
        if self.language_getter:
            try:
                current_lang = self.language_getter()
            except Exception:
                pass

        if current_lang == "lt":
            curious_text = f"Aptikau kažką įdomaus! {description}"
        else:
            curious_text = f"I have detected something interesting! {description}"
        print(f"[ROAM-CURIOUS] {curious_text}")

        # Save curious snapshot
        if self.visual_memory:
            try:
                curious_tags = tags + ["curious", "interesting"]
                self.visual_memory.add_visual(
                    image=frame,
                    description=f"[CURIOUS] {description}",
                    tags=curious_tags,
                    save_image=True
                )
            except Exception as e:
                print(f"[ROAM] Curious save error: {e}")

        # Speak about the discovery (force=True for important curiosity detection)
        self._safe_speak(curious_text, force=True)

        # Trigger curiosity callback if set
        if self.on_curiosity_callback:
            try:
                self.on_curiosity_callback(description, frame)
            except Exception as e:
                print(f"[ROAM] Curiosity callback error: {e}")

    def get_status(self) -> dict:
        """
        Get roaming status summary.

        Returns:
            Dict with roaming statistics including obstacle avoidance info
        """
        # Determine current escape level based on block count
        block_count = self.state.roam.forward_block_count
        if block_count >= self.ULTRA_STUCK_BLOCKS:
            escape_level = "ultra_stuck"
        elif block_count >= self.HARD_ESCAPE_BLOCKS:
            escape_level = "hard_escape"
        elif block_count >= self.VERY_STUCK_BLOCKS:
            escape_level = "very_stuck"
        elif block_count >= self.STUCK_BLOCKS:
            escape_level = "stuck"
        elif block_count >= self.SIDE_ESCAPE_BLOCKS:
            escape_level = "side_escape"
        else:
            escape_level = "clear"

        # Calculate elapsed time
        elapsed_sec = 0.0
        if self._roam_start_time > 0:
            elapsed_sec = time.time() - self._roam_start_time

        return {
            "active": self.state.behavior.roaming.is_set(),
            "position": (self.state.roam.x_m, self.state.roam.y_m),
            "heading": self.state.roam.heading_deg,
            "path_length": len(self.state.roam.path),
            "forward_blocks": block_count,
            "escape_level": escape_level,
            "blocked_headings": len(self.state.roam.blocked_headings),
            "visited_cells": len(self.state.roam.visited_grid),
            "observations": self.state.roam.state_dict.get("obs_count", 0),
            "path_repetition_history": len(self._recent_positions),
            # v58-style additions
            "elapsed_sec": elapsed_sec,
            "max_duration_sec": self.max_roam_duration_sec,
            "time_remaining_sec": max(0, self.max_roam_duration_sec - elapsed_sec),
            "curiosity_enabled": self.curiosity_enabled,
            "curious_target_seen": self._curious_target_seen,
        }
