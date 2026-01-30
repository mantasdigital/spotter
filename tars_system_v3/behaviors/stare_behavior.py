"""
Stare (Face Tracking) Behavior Implementation.

Tracks faces with camera servos, maintaining eye contact by keeping
the detected face centered in the camera frame.

Uses Vilib-style control math from v58 for smooth, responsive tracking.
Includes hand-in-face gesture detection to stop behavior.
"""

import time
import threading
import statistics
import cv2
from typing import Optional, Tuple, Any

from hardware.interfaces import IRobotCar, ICamera, IFaceDetector
from core.state_manager import StateManager


def _clamp_number(value: float, min_val: float, max_val: float) -> float:
    """Clamp a number between min and max values."""
    return max(min_val, min(max_val, value))


class CascadeLoader:
    """
    Shared cascade loader with background loading.

    Loads OpenCV cascades in background threads to avoid blocking startup.
    Cascades are shared between StareBehavior and FollowBehavior.
    """

    _instance = None
    _lock = threading.Lock()

    # Cascade paths
    CASCADE_PATHS = {
        "face": [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        ],
        "profile": [
            "/usr/share/opencv4/haarcascades/haarcascade_profileface.xml",
            "/usr/share/opencv/haarcascades/haarcascade_profileface.xml",
        ],
        "upperbody": [
            "/usr/share/opencv4/haarcascades/haarcascade_upperbody.xml",
            "/usr/share/opencv/haarcascades/haarcascade_upperbody.xml",
        ],
    }

    def __new__(cls):
        """Singleton pattern - only one loader instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize cascade loader (only once due to singleton)."""
        if self._initialized:
            return

        self._initialized = True
        self._cascades = {}
        self._loading_complete = threading.Event()
        self._load_errors = []

        # Start background loading immediately
        self._load_thread = threading.Thread(target=self._load_all_cascades, daemon=True)
        self._load_thread.start()

    def _load_all_cascades(self):
        """Load all cascades in background thread."""
        for name, paths in self.CASCADE_PATHS.items():
            cascade = self._load_cascade(paths, name)
            self._cascades[name] = cascade

        self._loading_complete.set()

    def _load_cascade(self, paths: list, name: str):
        """Load a cascade from list of possible paths."""
        for path in paths:
            try:
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    print(f"[CASCADE] Loaded {name} from {path}")
                    return cascade
            except Exception as e:
                continue

        self._load_errors.append(name)
        print(f"[CASCADE] Warning: Could not load {name}")
        return None

    def get_cascade(self, name: str, timeout: float = 2.0):
        """
        Get a loaded cascade by name.

        Args:
            name: Cascade name ("face", "profile", "upperbody")
            timeout: Max seconds to wait for loading

        Returns:
            CascadeClassifier or None if not available
        """
        # Wait for loading to complete (with timeout)
        self._loading_complete.wait(timeout=timeout)
        return self._cascades.get(name)

    def is_ready(self) -> bool:
        """Check if all cascades are loaded."""
        return self._loading_complete.is_set()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        """Wait for cascades to be ready."""
        return self._loading_complete.wait(timeout=timeout)


# Global cascade loader instance - starts loading immediately on import
_cascade_loader = CascadeLoader()


class StareBehavior:
    """
    Face tracking behavior using Vilib-style control (from v58).

    Continuously tracks detected faces by adjusting camera pan/tilt servos
    to keep the face centered in the frame.
    """

    # Servo angle limits (same as v58)
    PAN_MIN = -35
    PAN_MAX = 35
    TILT_MIN = -35
    TILT_MAX = 35

    # Control parameters (from v58, tuned for faster response)
    SMOOTH_ALPHA = 0.4  # Exponential smoothing factor (higher = faster response)
    SMOOTH_ALPHA_TILT = 0.5  # Even faster for vertical tracking
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    # Pan/Tilt speed multipliers (higher = faster tracking)
    PAN_SPEED_MULT = 12.0   # Horizontal tracking speed
    TILT_SPEED_MULT = 15.0  # Vertical tracking speed (faster than horizontal)

    # Face detection parameters (from v58)
    SCALE_FACTOR = 1.05
    MIN_NEIGHBORS = 4
    MIN_FACE_SIZE = (60, 60)

    def __init__(
        self,
        car: IRobotCar,
        camera: ICamera,
        face_detector: IFaceDetector,
        state: StateManager,
        gesture_detector: Optional[Any] = None,
        update_rate_hz: float = 20.0  # Faster update rate for smoother tracking
    ):
        """
        Initialize stare behavior.

        Args:
            car: Robot car hardware interface
            camera: Camera interface
            face_detector: Face detection interface
            state: State manager
            gesture_detector: Optional GestureDetector for hand-in-face stop
            update_rate_hz: Update rate for face tracking loop
        """
        self.car = car
        self.camera = camera
        self.face_detector = face_detector
        self.state = state
        self.gesture_detector = gesture_detector
        self.update_interval = 1.0 / update_rate_hz

        # Smoothed face position
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None

        # Current servo angles
        self._pan_angle = 0.0
        self._tilt_angle = 0.0

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Use shared cascade loader (non-blocking, loads in background)
        self._face_cascade = None
        self._init_cascade()

    def _init_cascade(self):
        """Get face cascade from shared loader (non-blocking)."""
        # Use shared loader - cascade is already loading in background
        self._face_cascade = _cascade_loader.get_cascade("face", timeout=0.1)
        if self._face_cascade:
            print("[STARE] Using shared face cascade")
        else:
            print("[STARE] Face cascade loading in background...")

    def start(self):
        """
        Start face tracking behavior in background thread.

        Returns immediately, tracking runs in daemon thread.
        """
        if self.state.behavior.face_tracking.is_set():
            print("[STARE] Already in stare mode.")
            return

        # Ensure cascade is loaded (lazy loading if not ready during init)
        if self._face_cascade is None:
            self._face_cascade = _cascade_loader.get_cascade("face", timeout=2.0)
            if self._face_cascade:
                print("[STARE] Face cascade now ready")

        # Ensure camera is started
        if not self.camera.is_active():
            self.camera.start()

        # Reset camera to center position
        self._pan_angle = 0.0
        self._tilt_angle = 0.0
        self.car.set_cam_pan_angle(0)
        self.car.set_cam_tilt_angle(0)
        self.state.stare.update_servo_angles(0, 0)

        # Reset smoothing
        self._smooth_x = None
        self._smooth_y = None

        # Reset gesture detector debounce
        if self.gesture_detector:
            self.gesture_detector.reset_debounce()

        # Set tracking flag
        self.state.behavior.face_tracking.set()
        self._stop_event.clear()

        # Start tracking thread
        self._thread = threading.Thread(
            target=self._tracking_loop,
            name="StareThread",
            daemon=True
        )
        self._thread.start()

        print("[STARE] Starting stare-at-you loop.")

    def stop(self):
        """Stop face tracking behavior."""
        if not self.state.behavior.face_tracking.is_set():
            return

        print("[STARE] Stopping stare mode.")

        # Signal stop
        self._stop_event.set()
        self.state.behavior.face_tracking.clear()

        # Wait for thread
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        # Reset camera to center
        self.car.set_cam_pan_angle(0)
        self.car.set_cam_tilt_angle(0)

    def _tracking_loop(self):
        """
        Main face tracking loop using Vilib-style control (from v58).
        """
        try:
            while not self._stop_event.is_set() and self.state.behavior.face_tracking.is_set():
                # Check for global stop
                if self.state.behavior.global_stop.is_set():
                    break

                # Execute one tracking step
                self._tracking_step_vilib()

                # Wait until next update
                time.sleep(self.update_interval)

        except Exception as e:
            print(f"[STARE] Error in stare loop: {e}")

        finally:
            self.state.behavior.face_tracking.clear()
            print("[STARE] Stare mode loop exiting.")

    def _tracking_step_vilib(self):
        """
        Execute one step of face tracking using Vilib-style control (from v58).

        Uses exponential smoothing of face position and incremental servo adjustments.
        Checks for hand-in-face gesture to stop.
        """
        try:
            # Capture frame
            frame = self.camera.capture_frame()
            if frame is None:
                return

            # Handle XBGR8888 format (4 channels) from VoiceActiveCar's camera
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]  # Drop alpha channel

            # Check for hand-in-face stop gesture
            if self.gesture_detector and self.gesture_detector.check_and_handle(frame, "stare"):
                self.stop()
                return

            # Convert to grayscale and equalize histogram (from v58)
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grey = cv2.equalizeHist(grey)

            frame_h, frame_w = grey.shape[:2]

            # Detect faces using cascade or fallback
            if self._face_cascade is not None:
                faces = self._face_cascade.detectMultiScale(
                    grey,
                    scaleFactor=self.SCALE_FACTOR,
                    minNeighbors=self.MIN_NEIGHBORS,
                    minSize=self.MIN_FACE_SIZE,
                )
                faces = list(faces) if len(faces) > 0 else []
            else:
                # Use fallback detector
                faces = self.face_detector.detect_faces(frame)

            if not faces:
                # No face: just wait, no sweeping - matches Vilib behavior
                self.state.stare.mark_face_lost()
                return

            # Pick largest face (from v58)
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            cx = x + w / 2.0
            cy = y + h / 2.0

            # Update state
            self.state.stare.update_face_detection(cx, cy, w, h)

            # Smoothing of input coordinates (from v58)
            if self._smooth_x is None:
                self._smooth_x = cx
                self._smooth_y = cy
            else:
                # Separate smoothing for horizontal and vertical (vertical faster)
                self._smooth_x = (1.0 - self.SMOOTH_ALPHA) * self._smooth_x + self.SMOOTH_ALPHA * cx
                self._smooth_y = (1.0 - self.SMOOTH_ALPHA_TILT) * self._smooth_y + self.SMOOTH_ALPHA_TILT * cy

            # Vilib-equivalent control (from v58) with tuned speed multipliers
            # Pan: positive angle turns camera left, so if face is right of center, we subtract
            self._pan_angle += (self._smooth_x * self.PAN_SPEED_MULT / frame_w) - (self.PAN_SPEED_MULT / 2.0)
            self._pan_angle = _clamp_number(self._pan_angle, self.PAN_MIN, self.PAN_MAX)
            self.car.set_cam_pan_angle(self._pan_angle)

            # Tilt: positive angle tilts camera up, so if face is below center, we add
            # Uses faster TILT_SPEED_MULT for quicker vertical response
            self._tilt_angle -= (self._smooth_y * self.TILT_SPEED_MULT / frame_h) - (self.TILT_SPEED_MULT / 2.0)
            self._tilt_angle = _clamp_number(self._tilt_angle, self.TILT_MIN, self.TILT_MAX)
            self.car.set_cam_tilt_angle(self._tilt_angle)

            # Update state
            self.state.stare.update_servo_angles(self._pan_angle, self._tilt_angle)

        except Exception as e:
            print(f"[STARE] Tracking step error: {e}")

    def get_status(self) -> dict:
        """
        Get face tracking status.

        Returns:
            Dict with tracking status information
        """
        face_data = self.state.stare.last_face

        return {
            "active": self.state.behavior.face_tracking.is_set(),
            "face_detected": face_data["seen"],
            "face_position": (face_data["x"], face_data["y"]),
            "camera_pan": self._pan_angle,
            "camera_tilt": self._tilt_angle,
            "last_seen": face_data["last_seen_time"]
        }


class FollowBehavior:
    """
    Person following behavior using Vilib-style control (from v58).

    Combines face detection with motor control to follow a person,
    maintaining a target distance based on face size.

    Includes obstacle avoidance for safe navigation.
    """

    # Servo angle limits (same as stare)
    PAN_MIN = -35
    PAN_MAX = 35
    TILT_MIN = -35
    TILT_MAX = 35

    # Control parameters (from v58, tuned for faster response)
    SMOOTH_ALPHA = 0.4  # Exponential smoothing factor (higher = faster response)
    SMOOTH_ALPHA_TILT = 0.5  # Even faster for vertical tracking
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480

    # Pan/Tilt speed multipliers (higher = faster tracking)
    PAN_SPEED_MULT = 12.0   # Horizontal tracking speed
    TILT_SPEED_MULT = 15.0  # Vertical tracking speed (faster than horizontal)

    # Distance control parameters (from v58)
    TARGET_FACE_WIDTH = 140.0  # Desired face width in pixels
    DEADBAND = 25.0  # +/- pixels where no forward/backward
    FORWARD_SPEED = 28
    BACKWARD_SPEED = 28
    MIN_MOVE_TIME = 0.20
    MAX_MOVE_TIME = 0.60

    # Face detection parameters (from v58)
    SCALE_FACTOR = 1.05
    MIN_NEIGHBORS = 4
    MIN_FACE_SIZE = (60, 60)

    # Obstacle avoidance parameters (matching roam mode thresholds)
    OBSTACLE_DIST_CRITICAL = 20.0  # Stop immediately (cm) - matches roam
    OBSTACLE_DIST_CAUTION = 38.0   # Slow down zone (cm) - matches roam DIST_BLOCKED
    OBSTACLE_DIST_CLEAR = 55.0     # Safe to move forward (cm)

    # Soft object detection (like roam mode - trash bags, fabric absorb ultrasonic)
    SOFT_OBJECT_JUMP_CM = 50.0     # Sudden distance increase threshold
    SOFT_OBJECT_VARIANCE = 400     # High variance threshold
    SOFT_OBJECT_CONSECUTIVE = 3    # Readings before treating as obstacle

    # Stuck detection parameters (for obstacles not visible to sensor)
    STUCK_FORWARD_COUNT_THRESHOLD = 8   # Forward commands without progress
    STUCK_FACE_WIDTH_TOLERANCE = 15.0   # Pixels - if face width changes less than this, no progress
    STUCK_ESCAPE_BACKUP_TIME = 0.5      # Seconds to backup when stuck
    STUCK_ESCAPE_TURN_TIME = 0.4        # Seconds to turn when stuck

    # Multi-cascade detection parameters
    PROFILE_SCALE_FACTOR = 1.1
    PROFILE_MIN_NEIGHBORS = 3
    PROFILE_MIN_SIZE = (50, 50)

    BODY_SCALE_FACTOR = 1.08
    BODY_MIN_NEIGHBORS = 4
    BODY_MIN_SIZE = (60, 80)  # Bodies are taller than wide

    # Search behavior when target lost
    SEARCH_PATIENCE_FRAMES = 5          # Frames before starting search
    SEARCH_PAN_STEP = 4.0               # Degrees per search step
    SEARCH_MAX_PAN_OFFSET = 30          # Max additional pan from last position
    SEARCH_TIMEOUT_FRAMES = 50          # Give up after this many frames (~2.5 sec)
    SEARCH_TURN_SPEED = 18              # Speed for searching turn

    # Cliff detection parameters (matching roam mode)
    CLIFF_CHECK_ENABLED = True          # Enable cliff detection in follow mode
    CLIFF_BACKUP_TIME = 0.8             # Seconds to backup on cliff detection
    CLIFF_COOLDOWN_SEC = 2.0            # Cooldown between cliff triggers

    def __init__(
        self,
        car: IRobotCar,
        camera: ICamera,
        face_detector: IFaceDetector,
        state: StateManager,
        gesture_detector: Optional[Any] = None,
        target_distance_cm: float = 50.0,  # Not used, kept for compatibility
        follow_speed: int = 28,
        turn_speed: int = 28
    ):
        """
        Initialize follow behavior.

        Args:
            car: Robot car hardware interface
            camera: Camera interface
            face_detector: Face detection interface
            state: State manager
            gesture_detector: Optional GestureDetector for hand-in-face stop
            target_distance_cm: Target distance (not used, uses face width instead)
            follow_speed: Speed for forward movement
            turn_speed: Speed for turning
        """
        self.car = car
        self.camera = camera
        self.face_detector = face_detector
        self.state = state
        self.gesture_detector = gesture_detector

        self.FORWARD_SPEED = follow_speed
        self.BACKWARD_SPEED = turn_speed

        # Smoothed face position
        self._smooth_x: Optional[float] = None
        self._smooth_y: Optional[float] = None

        # Current servo angles
        self._pan_angle = 0.0
        self._tilt_angle = 0.0

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Stuck detection state (for obstacles not visible to sensor)
        self._forward_count = 0           # Count of consecutive forward commands
        self._last_face_width = 0.0       # Face width at start of forward sequence
        self._stuck_escape_direction = 1  # Alternate escape directions

        # Soft object detection (like roam mode)
        self._recent_distances = []       # Last few distance readings
        self._soft_object_suspect_count = 0
        self._invalid_distance_count = 0  # Consecutive invalid readings

        # Multi-cascade detection (face, profile, body)
        self._face_cascade = None         # Frontal face (primary)
        self._profile_cascade = None      # Side face profile (secondary)
        self._upperbody_cascade = None    # Upper body - works from behind (tertiary)

        # Last-seen tracking for search behavior
        self._last_seen_x = None          # Last known X position in frame
        self._last_seen_y = None          # Last known Y position in frame
        self._last_seen_width = None      # Last known target width
        self._last_seen_time = 0.0        # When last seen
        self._last_seen_pan = 0.0         # Camera pan angle when last seen
        self._last_seen_direction = 0     # -1=left, 0=center, 1=right of frame center
        self._frames_since_seen = 0       # Counter for search timeout
        self._last_detection_type = None  # "face", "profile", "body"
        self._search_pan_offset = 0.0     # Current search pan offset

        # Cliff detection state (like roam mode)
        self._consecutive_cliff_count = 0
        self._last_cliff_time = 0.0

        # Load all cascades
        self._init_cascades()

    def _init_cascades(self):
        """Get cascades from shared loader (non-blocking)."""
        # Use shared loader - cascades are already loading in background
        self._face_cascade = _cascade_loader.get_cascade("face", timeout=0.1)
        self._profile_cascade = _cascade_loader.get_cascade("profile", timeout=0.1)
        self._upperbody_cascade = _cascade_loader.get_cascade("upperbody", timeout=0.1)

        if _cascade_loader.is_ready():
            print(f"[FOLLOW] Using shared cascades: face={self._face_cascade is not None}, "
                  f"profile={self._profile_cascade is not None}, body={self._upperbody_cascade is not None}")
        else:
            print("[FOLLOW] Cascades loading in background...")

    def start(self):
        """Start person following behavior."""
        if self.state.behavior.following.is_set():
            print("[FOLLOW] Already in follow mode.")
            return

        # Ensure cascades are loaded (lazy loading if not ready during init)
        if self._face_cascade is None:
            self._face_cascade = _cascade_loader.get_cascade("face", timeout=2.0)
        if self._profile_cascade is None:
            self._profile_cascade = _cascade_loader.get_cascade("profile", timeout=0.5)
        if self._upperbody_cascade is None:
            self._upperbody_cascade = _cascade_loader.get_cascade("upperbody", timeout=0.5)

        if self._face_cascade:
            print(f"[FOLLOW] Cascades ready: face=True, profile={self._profile_cascade is not None}, "
                  f"body={self._upperbody_cascade is not None}")

        # Ensure camera is started
        if not self.camera.is_active():
            self.camera.start()

        # Reset camera to center
        self._pan_angle = 0.0
        self._tilt_angle = 0.0
        self.car.set_cam_pan_angle(0)
        self.car.set_cam_tilt_angle(0)

        # Reset smoothing
        self._smooth_x = None
        self._smooth_y = None

        # Reset stuck detection
        self._forward_count = 0
        self._last_face_width = 0.0

        # Reset soft object detection
        self._recent_distances = []
        self._soft_object_suspect_count = 0
        self._invalid_distance_count = 0

        # Reset last-seen tracking
        self._last_seen_x = None
        self._last_seen_y = None
        self._last_seen_width = None
        self._last_seen_time = 0.0
        self._last_seen_pan = 0.0
        self._last_seen_direction = 0
        self._frames_since_seen = 0
        self._last_detection_type = None
        self._search_pan_offset = 0.0

        # Reset cliff detection state
        self._consecutive_cliff_count = 0
        self._last_cliff_time = 0.0

        # Reset gesture detector debounce
        if self.gesture_detector:
            self.gesture_detector.reset_debounce()

        # Set following flag
        self.state.behavior.following.set()
        self._stop_event.clear()

        # Start follow thread
        self._thread = threading.Thread(
            target=self._follow_loop,
            name="FollowThread",
            daemon=True
        )
        self._thread.start()

        print("[FOLLOW] Starting follow loop.")

    def stop(self):
        """Stop following behavior."""
        if not self.state.behavior.following.is_set():
            return

        print("[FOLLOW] Stopping follow mode.")

        # Signal stop
        self._stop_event.set()
        self.state.behavior.following.clear()

        # Stop motors
        self.car.stop()

        # Wait for thread
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _detect_person(self, grey, frame_w: int, frame_h: int):
        """
        Detect person using cascade priority: face > profile > body.

        Args:
            grey: Grayscale image
            frame_w: Frame width
            frame_h: Frame height

        Returns:
            tuple: (detected, detection_type, x, y, w, h)
        """
        # Try frontal face first (best for distance estimation)
        if self._face_cascade is not None:
            faces = self._face_cascade.detectMultiScale(
                grey,
                scaleFactor=self.SCALE_FACTOR,
                minNeighbors=self.MIN_NEIGHBORS,
                minSize=self.MIN_FACE_SIZE,
            )
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
                return (True, "face", x, y, w, h)

        # Try profile face (side view) - check both orientations
        if self._profile_cascade is not None:
            # Normal orientation (left profile)
            profiles = self._profile_cascade.detectMultiScale(
                grey,
                scaleFactor=self.PROFILE_SCALE_FACTOR,
                minNeighbors=self.PROFILE_MIN_NEIGHBORS,
                minSize=self.PROFILE_MIN_SIZE,
            )
            if len(profiles) > 0:
                x, y, w, h = max(profiles, key=lambda r: r[2] * r[3])
                return (True, "profile", x, y, w, h)

            # Flipped (right profile)
            grey_flipped = cv2.flip(grey, 1)
            profiles_r = self._profile_cascade.detectMultiScale(
                grey_flipped,
                scaleFactor=self.PROFILE_SCALE_FACTOR,
                minNeighbors=self.PROFILE_MIN_NEIGHBORS,
                minSize=self.PROFILE_MIN_SIZE,
            )
            if len(profiles_r) > 0:
                x, y, w, h = max(profiles_r, key=lambda r: r[2] * r[3])
                # Flip x coordinate back
                x = frame_w - x - w
                return (True, "profile", x, y, w, h)

        # Try upper body (works from behind!)
        if self._upperbody_cascade is not None:
            bodies = self._upperbody_cascade.detectMultiScale(
                grey,
                scaleFactor=self.BODY_SCALE_FACTOR,
                minNeighbors=self.BODY_MIN_NEIGHBORS,
                minSize=self.BODY_MIN_SIZE,
            )
            if len(bodies) > 0:
                x, y, w, h = max(bodies, key=lambda r: r[2] * r[3])
                return (True, "body", x, y, w, h)

        return (False, None, 0, 0, 0, 0)

    def _update_last_seen(self, x: float, y: float, w: float, h: float,
                          det_type: str, frame_w: int):
        """
        Update last-seen memory when target is detected.

        Args:
            x, y, w, h: Detection bounding box
            det_type: Detection type ("face", "profile", "body")
            frame_w: Frame width for direction calculation
        """
        cx = x + w / 2.0

        self._last_seen_x = cx
        self._last_seen_y = y + h / 2.0
        self._last_seen_width = w
        self._last_seen_time = time.time()
        self._last_seen_pan = self._pan_angle
        self._last_detection_type = det_type

        # Determine direction relative to frame center
        frame_center = frame_w / 2.0
        if cx < frame_center - 50:
            self._last_seen_direction = -1  # Left
        elif cx > frame_center + 50:
            self._last_seen_direction = 1   # Right
        else:
            self._last_seen_direction = 0   # Center

    def _execute_search_step(self):
        """
        Execute one step of search behavior when target is lost.

        Pans camera toward last-seen direction and tries to re-acquire.
        """
        # Calculate search direction based on last-seen
        if self._last_seen_direction != 0:
            # Pan toward last-seen direction
            target_offset = self._last_seen_direction * self.SEARCH_MAX_PAN_OFFSET

            # Gradually move toward target offset
            if abs(self._search_pan_offset - target_offset) > self.SEARCH_PAN_STEP:
                if self._search_pan_offset < target_offset:
                    self._search_pan_offset += self.SEARCH_PAN_STEP
                else:
                    self._search_pan_offset -= self.SEARCH_PAN_STEP
            else:
                self._search_pan_offset = target_offset

            # Apply search pan offset
            search_pan = self._last_seen_pan + self._search_pan_offset
            search_pan = _clamp_number(search_pan, self.PAN_MIN, self.PAN_MAX)
            self.car.set_cam_pan_angle(search_pan)

            # Log search progress occasionally
            if self._frames_since_seen % 10 == 0:
                print(f"[FOLLOW] Searching... pan={search_pan:.1f}, frames_lost={self._frames_since_seen}")

        # If lost for a while, also turn the robot body toward last-seen direction
        if self._frames_since_seen > self.SEARCH_PATIENCE_FRAMES * 3:
            if self._last_seen_direction != 0:
                # Slow turn toward last-seen direction
                turn_angle = 15 * self._last_seen_direction
                self.car.set_dir_servo_angle(turn_angle)
                self.car.forward(self.SEARCH_TURN_SPEED)
                time.sleep(0.15)
                self.car.stop()
                self.car.set_dir_servo_angle(0)

    def _post_escape_recovery(self):
        """
        After obstacle escape, search for target in last-known direction.
        """
        print(f"[FOLLOW] Post-escape recovery, last direction: {self._last_seen_direction}")

        # Pan camera toward last-seen direction
        if self._last_seen_direction != 0:
            recovery_pan = self._last_seen_pan + (20 * self._last_seen_direction)
            recovery_pan = _clamp_number(recovery_pan, self.PAN_MIN, self.PAN_MAX)
            self.car.set_cam_pan_angle(recovery_pan)
            self._pan_angle = recovery_pan
            time.sleep(0.2)  # Allow camera to settle

        # Reset search state - start searching immediately
        self._frames_since_seen = self.SEARCH_PATIENCE_FRAMES
        self._search_pan_offset = 0.0

    def _follow_loop(self):
        """Main person following loop using Vilib-style control (from v58)."""
        try:
            while not self._stop_event.is_set() and self.state.behavior.following.is_set():
                # Check for global stop
                if self.state.behavior.global_stop.is_set():
                    break

                # Execute one follow step
                self._follow_step_vilib()

                # Small delay
                time.sleep(0.05)

        except Exception as e:
            print(f"[FOLLOW] Error in follow loop: {e}")
            self.car.stop()

        finally:
            self.car.stop()
            self.state.behavior.following.clear()
            print("[FOLLOW] Follow loop exiting.")

    def _follow_step_vilib(self):
        """
        Execute one step of following behavior using Vilib-style control (from v58).

        Includes obstacle avoidance for safe navigation and stuck detection
        for obstacles not visible to the ultrasonic sensor.
        Uses roam-style soft object detection for trash bags, fabric, etc.
        """
        try:
            # Check for obstacles FIRST (safety priority)
            obstacle_distance = self.car.get_distance()

            # Handle invalid distance readings (like roam mode)
            distance_valid = True
            if obstacle_distance is None:
                obstacle_distance = -1.0
                distance_valid = False
            else:
                try:
                    obstacle_distance = float(obstacle_distance)
                    # Negative or very small = sensor error (too close)
                    # Very high (>400cm) = no echo
                    if obstacle_distance < 0 or obstacle_distance > 400:
                        distance_valid = False
                except (TypeError, ValueError):
                    obstacle_distance = -1.0
                    distance_valid = False

            # Track invalid readings (like roam mode)
            if not distance_valid:
                self._invalid_distance_count += 1
            else:
                self._invalid_distance_count = 0

            obstacle_blocking = False

            # Debug: Log distance and grayscale periodically
            if not hasattr(self, '_follow_step_count'):
                self._follow_step_count = 0
            self._follow_step_count += 1
            if self._follow_step_count % 20 == 0:
                valid_str = "OK" if distance_valid else "INVALID"
                print(f"[FOLLOW] Distance: {obstacle_distance:.1f}cm ({valid_str}) step={self._follow_step_count}")
                # Log grayscale sensors for cliff debugging
                try:
                    grayscale = self.car.get_grayscale_data()
                    if grayscale:
                        print(f"[FOLLOW] Grayscale: {grayscale}")
                except Exception:
                    pass

            # Invalid readings likely mean TOO CLOSE (sensor error)
            if not distance_valid and self._invalid_distance_count >= 2:
                print(f"[FOLLOW] SENSOR ERROR - likely too close, blocking forward!")
                obstacle_blocking = True

            # CLIFF DETECTION (like roam mode)
            if self._check_for_cliff():
                self._handle_cliff()
                return  # Return and continue following after cliff avoidance

            # SOFT OBJECT DETECTION (like roam mode)
            if distance_valid:
                self._recent_distances.append(obstacle_distance)
                if len(self._recent_distances) > 5:
                    self._recent_distances.pop(0)

                is_soft_object_suspect = False

                if len(self._recent_distances) >= 3:
                    # Pattern 1: Sudden jump (soft object absorbed pulse)
                    prev_readings = self._recent_distances[-3:-1]
                    prev_avg = sum(prev_readings) / len(prev_readings)
                    if prev_avg < self.OBSTACLE_DIST_CAUTION and obstacle_distance > prev_avg + self.SOFT_OBJECT_JUMP_CM:
                        print(f"[FOLLOW] Soft object suspect: jump from {prev_avg:.0f}cm to {obstacle_distance:.0f}cm")
                        is_soft_object_suspect = True

                    # Pattern 2: High variance (erratic readings)
                    if len(self._recent_distances) >= 4:
                        variance = statistics.variance(self._recent_distances)
                        recent_min = min(self._recent_distances)
                        if variance > self.SOFT_OBJECT_VARIANCE and recent_min < self.OBSTACLE_DIST_CAUTION:
                            print(f"[FOLLOW] Soft object suspect: variance={variance:.0f}, min={recent_min:.0f}cm")
                            is_soft_object_suspect = True

                if is_soft_object_suspect:
                    self._soft_object_suspect_count += 1
                    if self._soft_object_suspect_count >= self.SOFT_OBJECT_CONSECUTIVE:
                        print(f"[FOLLOW] SOFT OBJECT DETECTED - blocking forward!")
                        obstacle_blocking = True
                        self._soft_object_suspect_count = 0
                        self._recent_distances = []
                else:
                    self._soft_object_suspect_count = 0

            # Normal obstacle check
            if distance_valid and obstacle_distance < self.OBSTACLE_DIST_CRITICAL:
                # Critical obstacle - cannot move forward at all
                obstacle_blocking = True
                print(f"[FOLLOW] Critical obstacle at {obstacle_distance:.1f}cm - blocking forward")

            # Capture frame
            frame = self.camera.capture_frame()
            if frame is None:
                self.car.stop()
                return

            # Handle XBGR8888 format (4 channels) from VoiceActiveCar's camera
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]  # Drop alpha channel

            # Check for hand-in-face stop gesture
            if self.gesture_detector and self.gesture_detector.check_and_handle(frame, "follow"):
                self.car.stop()
                self.stop()
                return

            # Convert to grayscale and equalize histogram (from v58)
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grey = cv2.equalizeHist(grey)

            frame_h, frame_w = grey.shape[:2]

            # Multi-cascade person detection: face > profile > body
            detected, det_type, x, y, w, h = self._detect_person(grey, frame_w, frame_h)

            if not detected:
                # Target lost - increment lost counter
                self._frames_since_seen += 1

                if self._frames_since_seen < self.SEARCH_PATIENCE_FRAMES:
                    # Brief loss - keep moving in same direction, don't stop yet
                    pass
                elif self._frames_since_seen < self.SEARCH_TIMEOUT_FRAMES:
                    # Search mode - pan toward last-seen direction
                    self.car.stop()
                    self._execute_search_step()
                else:
                    # Give up - stop
                    self.car.stop()
                    if self._frames_since_seen == self.SEARCH_TIMEOUT_FRAMES:
                        print(f"[FOLLOW] Target lost after {self.SEARCH_TIMEOUT_FRAMES} frames - stopping search")

                # Reset forward tracking when no target
                self._forward_count = 0
                return

            # Target found! Reset search state
            self._frames_since_seen = 0
            self._search_pan_offset = 0.0

            # Update last-seen memory
            self._update_last_seen(x, y, w, h, det_type, frame_w)

            # Log detection type changes
            if det_type != self._last_detection_type and self._last_detection_type is not None:
                print(f"[FOLLOW] Detection changed: {self._last_detection_type} -> {det_type}")

            # Adjust effective width based on detection type
            # Body is larger than face, so scale down for distance estimation
            if det_type == "body":
                effective_w = w * 0.5  # Face would be ~50% of upper body width
            elif det_type == "profile":
                effective_w = w * 0.85  # Profile slightly narrower than frontal
            else:
                effective_w = w

            cx = x + w / 2.0
            cy = y + h / 2.0

            # Smoothing of input coordinates (from v58, with faster tilt)
            if self._smooth_x is None:
                self._smooth_x = cx
                self._smooth_y = cy
            else:
                # Separate smoothing for horizontal and vertical (vertical faster)
                self._smooth_x = (1.0 - self.SMOOTH_ALPHA) * self._smooth_x + self.SMOOTH_ALPHA * cx
                self._smooth_y = (1.0 - self.SMOOTH_ALPHA_TILT) * self._smooth_y + self.SMOOTH_ALPHA_TILT * cy

            # --- Pan/tilt same as stare (Vilib math from v58) with tuned speed ---
            self._pan_angle += (self._smooth_x * self.PAN_SPEED_MULT / frame_w) - (self.PAN_SPEED_MULT / 2.0)
            self._pan_angle = _clamp_number(self._pan_angle, self.PAN_MIN, self.PAN_MAX)
            self.car.set_cam_pan_angle(self._pan_angle)

            # Faster vertical tracking
            self._tilt_angle -= (self._smooth_y * self.TILT_SPEED_MULT / frame_h) - (self.TILT_SPEED_MULT / 2.0)
            self._tilt_angle = _clamp_number(self._tilt_angle, self.TILT_MIN, self.TILT_MAX)
            self.car.set_cam_tilt_angle(self._tilt_angle)

            # --- Forward / backward follow by face size (from v58) ---
            # With obstacle avoidance and stuck detection integration
            # Use effective_w for distance calculation (accounts for body/profile detection)
            err = effective_w - self.TARGET_FACE_WIDTH

            if abs(err) <= self.DEADBAND:
                self.car.stop()
                # In deadband, reset forward tracking
                self._forward_count = 0
            elif err < 0:
                # Face too small => want to move forward
                # Check for stuck condition first (obstacle not visible to sensor)
                if self._forward_count == 0:
                    # Starting new forward sequence, record initial face width
                    self._last_face_width = w

                self._forward_count += 1

                # Check if stuck: many forward commands but face width not increasing
                if self._forward_count >= self.STUCK_FORWARD_COUNT_THRESHOLD:
                    width_change = w - self._last_face_width
                    if width_change < self.STUCK_FACE_WIDTH_TOLERANCE:
                        # We're stuck! Face width didn't increase despite forward commands
                        print(f"[FOLLOW] STUCK detected! {self._forward_count} forwards, width change: {width_change:.1f}px")
                        self._execute_stuck_escape()
                        return

                # Now check for obstacles
                if obstacle_blocking:
                    # Cannot move forward due to obstacle - try to steer around it
                    print(f"[FOLLOW] Want forward but obstacle at {obstacle_distance:.1f}cm - steering around")

                    # Steer in the opposite direction of where the face was last seen
                    # This helps us go around obstacle while keeping face in view
                    if self._last_seen_direction == 0:
                        # Face is center - pick direction based on camera pan
                        steer_dir = -1 if self._pan_angle > 0 else 1
                    else:
                        # Steer opposite to face direction (go around obstacle on the other side)
                        steer_dir = -self._last_seen_direction

                    # Try to sidestep around obstacle while staying close to target
                    steer_angle = 15 * steer_dir
                    self.car.set_dir_servo_angle(steer_angle)
                    self.car.forward(self.FORWARD_SPEED)
                    time.sleep(0.25)
                    self.car.stop()
                    self.car.set_dir_servo_angle(0)

                    # Reset forward tracking - we made a maneuver
                    self._forward_count = 0

                elif obstacle_distance >= 0 and obstacle_distance < self.OBSTACLE_DIST_CAUTION:
                    # Caution zone - move forward slowly with slight steering to avoid obstacle
                    ratio = min(1.0, abs(err) / self.TARGET_FACE_WIDTH)
                    # Reduce speed and time in caution zone
                    speed_factor = obstacle_distance / self.OBSTACLE_DIST_CAUTION
                    speed = int(self.FORWARD_SPEED * max(0.5, speed_factor))
                    move_time = (self.MIN_MOVE_TIME + (self.MAX_MOVE_TIME - self.MIN_MOVE_TIME) * ratio) * 0.5

                    # Slight steer based on face position to help go around obstacle
                    # If face is left, steer right (and vice versa) to go around obstacle
                    if self._last_seen_direction != 0:
                        slight_steer = -5 * self._last_seen_direction  # Very slight steering
                        self.car.set_dir_servo_angle(slight_steer)

                    print(f"[FOLLOW] Cautious forward, w={w:.1f}, dist={obstacle_distance:.1f}cm, dt={move_time:.2f}")
                    self.car.forward(speed)
                    time.sleep(move_time)
                    self.car.stop()
                    self.car.set_dir_servo_angle(0)
                else:
                    # Path clear - move forward normally
                    ratio = min(1.0, abs(err) / self.TARGET_FACE_WIDTH)
                    move_time = self.MIN_MOVE_TIME + (self.MAX_MOVE_TIME - self.MIN_MOVE_TIME) * ratio
                    print(f"[FOLLOW] Forward, w={w:.1f}, dt={move_time:.2f}")
                    self.car.forward(self.FORWARD_SPEED)
                    time.sleep(move_time)
                    self.car.stop()
            else:
                # Face too large => step back (always safe to go backward)
                # Reset forward tracking when going backward
                self._forward_count = 0

                ratio = min(1.0, abs(err) / self.TARGET_FACE_WIDTH)
                move_time = self.MIN_MOVE_TIME + (self.MAX_MOVE_TIME - self.MIN_MOVE_TIME) * ratio
                print(f"[FOLLOW] Backward, w={w:.1f}, dt={move_time:.2f}")
                self.car.backward(self.BACKWARD_SPEED)
                time.sleep(move_time)
                self.car.stop()

        except Exception as e:
            print(f"[FOLLOW] Follow step error: {e}")
            self.car.stop()

    def _execute_stuck_escape(self):
        """
        Execute escape maneuver when stuck on obstacle not visible to sensor.

        Backs up and turns to side, alternating direction each time.
        """
        print(f"[FOLLOW] Executing stuck escape (direction: {'left' if self._stuck_escape_direction > 0 else 'right'})")

        # Reset forward tracking
        self._forward_count = 0
        self._last_face_width = 0.0

        # Back up first
        self.car.backward(self.BACKWARD_SPEED)
        time.sleep(self.STUCK_ESCAPE_BACKUP_TIME)
        self.car.stop()

        # Turn to side (alternating direction)
        if self._stuck_escape_direction > 0:
            self.car.set_dir_servo_angle(25)  # Turn left
        else:
            self.car.set_dir_servo_angle(-25)  # Turn right

        self.car.forward(self.FORWARD_SPEED)
        time.sleep(self.STUCK_ESCAPE_TURN_TIME)
        self.car.stop()
        self.car.set_dir_servo_angle(0)  # Center steering

        # Alternate direction for next escape
        self._stuck_escape_direction = -self._stuck_escape_direction

        print("[FOLLOW] Stuck escape complete")

        # Try to re-acquire target in last-seen direction
        self._post_escape_recovery()

    def _handle_cliff(self):
        """
        Handle cliff detection - back up and try to continue following.

        Unlike roam mode which turns randomly, follow mode backs up and
        tries to maintain visual contact with the target while avoiding the cliff.
        """
        now = time.time()
        print("[FOLLOW] CLIFF DETECTED! Emergency backup...")

        # STOP immediately
        self.car.stop()
        time.sleep(0.1)  # Brief pause

        # Back up from cliff
        self.car.backward(self.BACKWARD_SPEED)
        time.sleep(self.CLIFF_BACKUP_TIME)
        self.car.stop()

        # Update cliff cooldown
        self._last_cliff_time = now
        self._consecutive_cliff_count = 0

        # Clear recent distance readings to avoid false positives
        self._recent_distances = []
        self._soft_object_suspect_count = 0

        # Reset forward tracking
        self._forward_count = 0
        self._last_face_width = 0.0

        # Try side escape - alternate direction each time
        self._stuck_escape_direction = -self._stuck_escape_direction

        # Sidestep: turn slightly while moving to avoid cliff area
        if self._stuck_escape_direction > 0:
            self.car.set_dir_servo_angle(20)  # Turn slightly left
        else:
            self.car.set_dir_servo_angle(-20)  # Turn slightly right

        self.car.forward(self.FORWARD_SPEED)
        time.sleep(0.3)
        self.car.stop()
        self.car.set_dir_servo_angle(0)

        print("[FOLLOW] Cliff avoidance complete, resuming follow")

        # Try to re-acquire target after cliff avoidance
        self._post_escape_recovery()

    def _check_for_cliff(self) -> bool:
        """
        Check grayscale sensors for cliff detection.

        Returns:
            bool: True if cliff detected (should stop forward movement)
        """
        if not self.CLIFF_CHECK_ENABLED:
            return False

        now = time.time()

        # Cooldown check - don't trigger again within CLIFF_COOLDOWN_SEC
        if now - self._last_cliff_time < self.CLIFF_COOLDOWN_SEC:
            return False

        try:
            grayscale = self.car.get_grayscale_data()
            if grayscale and self.car.get_cliff_status(grayscale):
                self._consecutive_cliff_count += 1

                # Immediate response on first detection (cliff is dangerous!)
                if self._consecutive_cliff_count >= 1:
                    print(f"[FOLLOW] Cliff sensors triggered: {grayscale}")
                    return True
            else:
                # No cliff - reset consecutive count
                self._consecutive_cliff_count = 0
        except Exception as e:
            print(f"[FOLLOW] Cliff check error: {e}")

        return False

    def get_status(self) -> dict:
        """Get following status including obstacle avoidance and cliff detection info."""
        # Get current obstacle distance
        try:
            obstacle_dist = self.car.get_distance()
        except Exception:
            obstacle_dist = -1

        # Get cliff sensor status
        cliff_detected = False
        grayscale = None
        try:
            grayscale = self.car.get_grayscale_data()
            if grayscale:
                cliff_detected = self.car.get_cliff_status(grayscale)
        except Exception:
            pass

        return {
            "active": self.state.behavior.following.is_set(),
            "target_face_width": self.TARGET_FACE_WIDTH,
            "camera_pan": self._pan_angle,
            "camera_tilt": self._tilt_angle,
            "obstacle_distance": obstacle_dist,
            "obstacle_blocking": obstacle_dist >= 0 and obstacle_dist < self.OBSTACLE_DIST_CRITICAL,
            "cliff_detected": cliff_detected,
            "grayscale_sensors": grayscale,
            "cliff_check_enabled": self.CLIFF_CHECK_ENABLED,
            "last_seen_direction": self._last_seen_direction,
        }
