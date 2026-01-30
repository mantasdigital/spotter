"""
Gesture Detection for Safety Stop.

Detects hand-in-face gesture to stop stare, follow, and roam modes.
Based on v58's skin-color + contour heuristic approach.
"""

import time
import cv2
import numpy as np
from typing import Optional


class GestureDetector:
    """
    Detects stop gesture (hand/object blocking camera).

    Uses TWO detection methods:
    1. Skin-color detection for bare hands
    2. Low-variance detection for any object blocking camera (gloves, etc.)

    Requires consecutive frames of detection to avoid false positives.
    """

    # Skin color thresholds in HSV
    SKIN_LOWER1 = np.array([0, 30, 70], dtype=np.uint8)
    SKIN_UPPER1 = np.array([20, 180, 255], dtype=np.uint8)
    SKIN_LOWER2 = np.array([160, 30, 70], dtype=np.uint8)
    SKIN_UPPER2 = np.array([180, 180, 255], dtype=np.uint8)

    # Detection thresholds
    SKIN_AREA_THRESHOLD = 0.25  # Skin must occupy 25% of center ROI
    SKIN_AREA_MIN_FOR_VARIANCE = 0.03  # Minimum skin for variance-assisted detection (3%)
    SKIN_AREA_TINY = 0.01  # Tiny skin hint (1%) - required even for very-low-variance
    VARIANCE_THRESHOLD_STRICT = 100  # Very low variance = camera truly blocked (lowered from 150)
    VARIANCE_THRESHOLD_WITH_SKIN = 400  # Higher threshold when skin is also detected
    VARIANCE_DROP_RATIO = 0.25  # Variance must drop to 25% of baseline
    FRAME_SIZE = (320, 240)  # Downsampled size for performance

    # ROI bounds (center 50% of frame)
    ROI_X1_RATIO = 0.25
    ROI_Y1_RATIO = 0.25
    ROI_X2_RATIO = 0.75
    ROI_Y2_RATIO = 0.75

    # Morphological kernel
    KERNEL_SIZE = 5

    # Debounce
    DEBOUNCE_SEC = 2.0

    # Consecutive frame requirement to avoid false positives
    CONSECUTIVE_FRAMES_REQUIRED = 4

    # Debug logging (every Nth frame) - set low for debugging
    DEBUG_INTERVAL = 5

    # Distance threshold - disable low-variance detection when robot is close to obstacle
    # (low variance near obstacle is from wall/floor, not a hand)
    OBSTACLE_DISTANCE_THRESHOLD = 30.0  # cm - disable variance detection when closer than this

    def __init__(self, speak_callback=None):
        """
        Initialize gesture detector.

        Args:
            speak_callback: Optional callback to speak when gesture detected
        """
        self.speak_callback = speak_callback
        self._last_trigger_time = 0.0
        self._kernel = np.ones((self.KERNEL_SIZE, self.KERNEL_SIZE), np.uint8)
        self._frame_count = 0
        self._consecutive_detections = 0  # Track consecutive positive detections
        self._baseline_variance = None  # Track normal scene variance
        self._last_distance = None  # Track obstacle distance to avoid false positives

    def set_obstacle_distance(self, distance: float):
        """
        Update the current obstacle distance for context-aware detection.

        When robot is close to obstacle, low-variance detection is disabled
        because low variance is expected (wall/floor in view, not a hand).

        Args:
            distance: Distance to nearest obstacle in cm (negative = invalid)
        """
        self._last_distance = distance

    def detect_stop_gesture(self, frame: np.ndarray, obstacle_distance: float = None) -> bool:
        """
        Detect if hand/object is blocking the camera.

        Uses TWO detection methods:
        1. Skin-color detection for bare hands (always active)
        2. Low-variance detection for any object blocking camera (disabled when near obstacle)

        Requires detection in multiple consecutive frames to avoid false positives.

        Args:
            frame: Image frame from camera (RGB or BGR)
            obstacle_distance: Optional distance to nearest obstacle in cm.
                              If provided, disables variance detection when close to obstacle.

        Returns:
            bool: True if stop gesture detected (after consecutive frame check)
        """
        if frame is None:
            self._consecutive_detections = 0
            return False

        # Update distance if provided
        if obstacle_distance is not None:
            self._last_distance = obstacle_distance

        self._frame_count += 1

        # Determine if we're close to an obstacle (disables variance-only detection)
        near_obstacle = False
        if self._last_distance is not None:
            # Valid positive distance that's close
            if 0 < self._last_distance < self.OBSTACLE_DISTANCE_THRESHOLD:
                near_obstacle = True
            # Invalid/negative distance often means VERY close (sensor error)
            elif self._last_distance < 0:
                near_obstacle = True

        try:
            # Handle XBGR8888 format (4 channels) - common from Picamera2
            if len(frame.shape) == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]

            # Downscale for performance
            frame_small = cv2.resize(frame, self.FRAME_SIZE)

            # Extract center ROI for analysis
            h, w = frame_small.shape[:2]
            cx1 = int(w * self.ROI_X1_RATIO)
            cy1 = int(h * self.ROI_Y1_RATIO)
            cx2 = int(w * self.ROI_X2_RATIO)
            cy2 = int(h * self.ROI_Y2_RATIO)
            roi_color = frame_small[cy1:cy2, cx1:cx2]

            # METHOD 1: Low-variance detection (works for gloves, any blocking object)
            # When something is very close to camera, the image becomes uniform/blurry
            gray_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            variance = gray_roi.var()

            # Update baseline variance (rolling average of normal scene)
            # Only update when scene looks normal (not blocked)
            if self._baseline_variance is None:
                self._baseline_variance = variance
            else:
                # Only adapt baseline when variance is high enough (normal scene)
                if variance > self.VARIANCE_THRESHOLD_WITH_SKIN:
                    self._baseline_variance = 0.95 * self._baseline_variance + 0.05 * variance

            # METHOD 2: Skin color detection (for bare hands)
            hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            mask1 = cv2.inRange(hsv, self.SKIN_LOWER1, self.SKIN_UPPER1)
            mask2 = cv2.inRange(hsv, self.SKIN_LOWER2, self.SKIN_UPPER2)
            skin_mask = cv2.bitwise_or(mask1, mask2)

            # Clean up mask
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, self._kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, self._kernel)

            # Calculate skin coverage
            skin_pixels = cv2.countNonZero(skin_mask)
            roi_area = skin_mask.shape[0] * skin_mask.shape[1]
            skin_ratio = skin_pixels / roi_area if roi_area > 0 else 0

            # Detection logic - MUST have skin OR extremely low variance WITH tiny skin hint
            # This prevents false positives from walls/floors during navigation
            skin_detected = skin_ratio > self.SKIN_AREA_THRESHOLD
            some_skin_detected = skin_ratio > self.SKIN_AREA_MIN_FOR_VARIANCE
            tiny_skin_hint = skin_ratio > self.SKIN_AREA_TINY  # At least 1% skin

            # Low variance detection requires EITHER:
            # 1. Extremely low variance WITH tiny skin hint (camera truly blocked by hand)
            #    - DISABLED when near obstacle
            #    - REQUIRES at least tiny skin to avoid floor/wall false positives
            # 2. Low variance WITH some skin detected (glove/partial hand)
            # 3. Variance dropped significantly from baseline WITH skin hint
            #    - DISABLED when near obstacle
            #
            # IMPORTANT: When robot is close to obstacle, low variance is EXPECTED
            # (camera sees uniform wall/floor) - this is NOT a hand gesture!
            # Also: Very low variance WITHOUT any skin is likely floor/wall, not hand!
            very_low_variance = (
                variance < self.VARIANCE_THRESHOLD_STRICT and
                tiny_skin_hint and  # MUST have at least 1% skin to rule out floor/wall
                not near_obstacle  # Disable when close to obstacle
            )
            low_variance_with_skin = (
                variance < self.VARIANCE_THRESHOLD_WITH_SKIN and
                some_skin_detected
            )
            variance_dropped = (
                self._baseline_variance is not None and
                self._baseline_variance > 500 and  # Only if baseline is reasonable
                variance < self._baseline_variance * self.VARIANCE_DROP_RATIO and
                some_skin_detected and  # Still require some skin indication
                not near_obstacle  # Disable when close to obstacle
            )

            # Determine detection method
            is_potential_gesture = False
            detection_method = None

            if skin_detected:
                # Skin detection always works (hands have skin color)
                is_potential_gesture = True
                detection_method = "skin-color"
            elif very_low_variance:
                # Very low variance + tiny skin hint (rules out floor/wall)
                is_potential_gesture = True
                detection_method = "very-low-variance+skin-hint"
            elif low_variance_with_skin:
                # Low variance + skin works always (gloves still have some skin visible)
                is_potential_gesture = True
                detection_method = "low-variance+skin"
            elif variance_dropped:
                # Variance drop + skin hint
                is_potential_gesture = True
                detection_method = "variance-drop+skin"

            # Debug logging (every Nth frame)
            if self._frame_count % self.DEBUG_INTERVAL == 0:
                baseline_str = f"{self._baseline_variance:.0f}" if self._baseline_variance else "?"
                dist_str = f"{self._last_distance:.0f}cm" if self._last_distance is not None else "?"
                near_str = "NEAR" if near_obstacle else "far"
                print(f"[GESTURE] var={variance:.0f} base={baseline_str} skin={skin_ratio*100:.0f}% dist={dist_str}({near_str}) consec={self._consecutive_detections}", flush=True)

            if is_potential_gesture:
                self._consecutive_detections += 1
                if self._consecutive_detections >= self.CONSECUTIVE_FRAMES_REQUIRED:
                    print(f"[GESTURE] Confirmed via {detection_method} after {self._consecutive_detections} frames!")
                    return True
            else:
                # Reset consecutive count if this frame doesn't detect
                self._consecutive_detections = 0

            return False

        except Exception as e:
            print(f"[GESTURE] Detection error: {e}")
            self._consecutive_detections = 0
            return False

    def check_and_handle(self, frame: np.ndarray, mode_name: str = "behavior") -> bool:
        """
        Check for stop gesture with debounce handling.

        Args:
            frame: BGR image frame from camera
            mode_name: Name of mode for logging (e.g., "roam", "follow", "stare")

        Returns:
            bool: True if stop gesture detected (and debounce passed)
        """
        if not self.detect_stop_gesture(frame):
            return False

        # Check debounce
        now = time.time()
        if now - self._last_trigger_time < self.DEBOUNCE_SEC:
            return False

        self._last_trigger_time = now
        print(f"[GESTURE] STOP gesture detected - stopping {mode_name} mode")

        # Speak if callback available
        if self.speak_callback:
            try:
                self.speak_callback(f"Fine, fine, stopping {mode_name} mode because you waved your hand at me.")
            except Exception as e:
                print(f"[GESTURE] TTS error: {e}")

        return True

    def reset_debounce(self):
        """Reset debounce timer and detection state (call when mode starts)."""
        self._last_trigger_time = 0.0
        self._consecutive_detections = 0
        self._baseline_variance = None  # Re-learn baseline for new scene
