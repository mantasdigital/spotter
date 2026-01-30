"""
Safety Manager.

Handles collision avoidance and cliff detection using ultrasonic and grayscale sensors.
Extracted from voice_active_car_gpt_modified_v58.py lines 2304-2409.
"""

import time
import logging
from typing import Optional

from hardware.interfaces import IRobotCar
from config.settings import SafetyConfig


logger = logging.getLogger(__name__)


class SafetyManager:
    """
    Manages robot safety through sensor monitoring.

    Provides cliff detection and obstacle avoidance using ultrasonic
    distance sensor and grayscale sensors.

    Attributes:
        robot: Robot car instance
        config: Safety configuration
    """

    def __init__(self, robot: IRobotCar, config: SafetyConfig):
        """
        Initialize safety manager.

        Args:
            robot: Robot car instance
            config: Safety configuration with thresholds

        Example:
            >>> from config.settings import SafetyConfig
            >>> config = SafetyConfig()
            >>> safety = SafetyManager(robot, config)
        """
        self.robot = robot
        self.config = config
        self.logger = logging.getLogger(__name__)

    def is_path_safe(self, direction: str = "forward") -> bool:
        """
        Check if path is safe for given direction.

        Checks both cliff detection (grayscale sensors) and obstacle
        detection (ultrasonic sensor). Automatically backs up if cliff
        or obstacle detected.

        Args:
            direction: Direction to check ("forward" or "backward")
                      Only forward is actively checked, backward always safe.

        Returns:
            True if path is safe, False if blocked or dangerous

        Example:
            >>> if not safety.is_path_safe("forward"):
            ...     print("Path blocked!")
        """
        # Backward is treated as safe (only check forward)
        if direction != "forward":
            return True

        # Read sensors
        try:
            dist = self.robot.get_distance()
        except Exception as e:
            self.logger.warning(f"Failed to read distance sensor: {e}")
            dist = 100.0

        try:
            gs = self.robot.get_grayscale_data()
        except Exception as e:
            self.logger.warning(f"Failed to read grayscale sensors: {e}")
            gs = []

        # Cliff detection first (most dangerous)
        if gs and self.robot.get_cliff_status(gs):
            self.logger.warning(f"[SAFETY] CLIFF DETECTED! Sensors: {gs}")
            self.robot.stop()
            # Back up slowly from the edge
            self.robot.backward(30)
            time.sleep(0.8)
            self.robot.stop()
            return False

        # Obstacle too close
        if 0 < dist < self.config.too_close_distance_cm:
            self.logger.warning(f"[SAFETY] OBSTACLE DETECTED at {dist:.1f}cm")
            self.robot.stop()
            # If really close, back up a bit more
            back_ms = 1.0 if dist > self.config.really_close_distance_cm else 1.5
            self.robot.backward(30)
            time.sleep(back_ms)
            self.robot.stop()
            return False

        return True

    def scan_direction(
        self,
        direction: str = "forward",
        steer_angle: int = 35,
        sample_time: float = 0.3
    ) -> bool:
        """
        Quick look-ahead scan in a given direction.

        Temporarily steers to the specified direction and samples
        sensors to check if that direction is clear.

        Args:
            direction: Direction to scan ("forward", "left", "right")
            steer_angle: Steering angle for left/right scans (degrees)
            sample_time: Sampling duration in seconds

        Returns:
            True if direction looks reasonably clear, False if blocked

        Note:
            Returns steering to original position (0) after scan.

        Example:
            >>> if safety.scan_direction("left"):
            ...     print("Left path is clear")
        """
        current_angle = 0

        # Set steering angle based on direction
        if direction == "left":
            angle = -abs(steer_angle)
        elif direction == "right":
            angle = abs(steer_angle)
        else:
            angle = 0

        self.robot.set_dir_servo_angle(angle)
        time.sleep(0.1)

        start = time.time()
        safe = True

        while time.time() - start < sample_time:
            # Read sensors
            try:
                dist = self.robot.get_distance()
            except Exception:
                dist = 100.0

            try:
                gs = self.robot.get_grayscale_data()
            except Exception:
                gs = []

            # For side scanning, only treat as blocked if really close
            block_threshold = 8 if direction in ("left", "right") else 15

            if 0 < dist < block_threshold:
                self.logger.info(f"[SAFETY] SIDE OBSTACLE ({direction}) at {dist:.1f}cm")
                safe = False
                break

            if gs and self.robot.get_cliff_status(gs):
                self.logger.info(f"[SAFETY] SIDE CLIFF ({direction}) Sensors: {gs}")
                safe = False
                break

            time.sleep(0.05)

        # Return steering to original position
        self.robot.set_dir_servo_angle(current_angle)
        return safe

    def find_clear_direction(self) -> Optional[str]:
        """
        Scan all directions to find a clear path.

        Tries directions in order: forward, left, right.

        Returns:
            "forward", "left", "right", or None if all blocked

        Example:
            >>> clear_dir = safety.find_clear_direction()
            >>> if clear_dir:
            ...     print(f"Can move {clear_dir}")
        """
        # Try forward first
        if self.scan_direction("forward"):
            return "forward"

        # Try left
        if self.scan_direction("left"):
            return "left"

        # Try right
        if self.scan_direction("right"):
            return "right"

        # All directions blocked
        return None

    def emergency_stop(self):
        """
        Immediately stop all robot motion.

        Example:
            >>> safety.emergency_stop()
        """
        self.logger.warning("[SAFETY] EMERGENCY STOP")
        self.robot.stop()
