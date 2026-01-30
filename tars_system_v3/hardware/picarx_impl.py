"""
PicarX Hardware Implementation.

Wraps the Sunfounder picarx.picarx.Picarx class to implement our IRobotCar
interface, enabling dependency injection and testability.
"""

import logging
from typing import List

from hardware.interfaces import IRobotCar


class PicarXRobotCar(IRobotCar):
    """
    Real PicarX robot car implementation.

    Wraps the Sunfounder Picarx class to implement our abstract interface.
    All method calls are delegated to the underlying Picarx instance.

    Attributes:
        px: The underlying Picarx hardware instance
        logger: Logger for tracking commands
    """

    def __init__(self, picarx_instance):
        """
        Initialize PicarX wrapper.

        Args:
            picarx_instance: Instance of picarx.picarx.Picarx to wrap

        Example:
            >>> from picarx import picarx
            >>> px = picarx.Picarx()
            >>> robot = PicarXRobotCar(px)
        """
        self.px = picarx_instance
        self.logger = logging.getLogger(__name__)
        self.logger.info("[HARDWARE] PicarX robot car initialized")

    def forward(self, speed: int):
        """
        Move forward at specified speed.

        Args:
            speed: Speed from 0-100

        Example:
            >>> robot.forward(50)
        """
        self.logger.info(f"[HARDWARE] Forward: speed={speed}")
        self.px.forward(speed)

    def backward(self, speed: int):
        """
        Move backward at specified speed.

        Args:
            speed: Speed from 0-100

        Example:
            >>> robot.backward(30)
        """
        self.logger.info(f"[HARDWARE] Backward: speed={speed}")
        self.px.backward(speed)

    def stop(self):
        """
        Stop all motors immediately.

        Example:
            >>> robot.stop()
        """
        self.logger.info("[HARDWARE] Stop")
        self.px.stop()

    def set_dir_servo_angle(self, angle: int):
        """
        Set steering servo angle.

        Args:
            angle: Angle from -30 to +30 degrees (left to right)

        Example:
            >>> robot.set_dir_servo_angle(15)  # Turn right
        """
        self.logger.info(f"[HARDWARE] Steering: angle={angle}")
        self.px.set_dir_servo_angle(angle)

    def set_cam_pan_angle(self, angle: int):
        """
        Set camera pan servo angle.

        Args:
            angle: Angle from -90 to +90 degrees (left to right)

        Example:
            >>> robot.set_cam_pan_angle(45)  # Look right
        """
        self.logger.info(f"[HARDWARE] Camera pan: angle={angle}")
        self.px.set_cam_pan_angle(angle)

    def set_cam_tilt_angle(self, angle: int):
        """
        Set camera tilt servo angle.

        Args:
            angle: Angle from -35 to +65 degrees (down to up)

        Example:
            >>> robot.set_cam_tilt_angle(20)  # Look up
        """
        self.logger.info(f"[HARDWARE] Camera tilt: angle={angle}")
        self.px.set_cam_tilt_angle(angle)

    def get_distance(self) -> float:
        """
        Read ultrasonic distance sensor.

        Returns:
            Distance in centimeters (0-500 range typically)

        Example:
            >>> distance = robot.get_distance()
            >>> if distance < 20:
            ...     robot.stop()
        """
        distance = self.px.get_distance()
        self.logger.debug(f"[HARDWARE] Distance: {distance:.1f} cm")
        return distance

    def get_grayscale_data(self) -> List[int]:
        """
        Read three grayscale sensors.

        Returns:
            List of three sensor values [left, center, right]
            Lower values (300-700) = good reflection = normal floor
            Higher values (>950) = poor reflection = cliff/void/edge

        Example:
            >>> values = robot.get_grayscale_data()
            >>> print(f"Sensors: {values}")
            [650, 700, 680]  # Normal floor
        """
        values = self.px.get_grayscale_data()
        self.logger.debug(f"[HARDWARE] Grayscale: {values}")
        return values

    def get_cliff_status(self, grayscale_values: List[int]) -> bool:
        """
        Check if cliff detected from grayscale sensor values.

        Uses the same threshold as v58: values > 900 indicate cliff
        (high values = poor reflection = looking into void/edge).

        Normal floor readings are typically 300-700.
        Cliff/edge readings go above 900.

        Args:
            grayscale_values: List of three grayscale readings

        Returns:
            True if cliff detected (danger), False if safe

        Example:
            >>> values = robot.get_grayscale_data()
            >>> if robot.get_cliff_status(values):
            ...     robot.stop()
            ...     print("Cliff detected!")
        """
        # Increased threshold from 900 to 950 to reduce false positives
        # Normal floor: 300-700, cliff/edge: >950
        CLIFF_THRESHOLD = 950

        if not grayscale_values:
            return False

        # Validate grayscale values are reasonable
        try:
            # Ensure all values are integers and within expected range
            valid_values = all(isinstance(v, (int, float)) and 0 <= v <= 4095 for v in grayscale_values)
            if not valid_values:
                self.logger.debug(f"[CLIFF] Invalid grayscale values: {grayscale_values}")
                return False
        except (TypeError, ValueError):
            return False

        # Check if ANY sensor reads cliff (high value = poor reflection = void)
        is_cliff = any(val > CLIFF_THRESHOLD for val in grayscale_values)
        return is_cliff

    def reset(self):
        """
        Reset robot to default state.

        Stops motors and centers all servos (direction, camera pan/tilt).

        Example:
            >>> robot.reset()
        """
        self.logger.info("[HARDWARE] Resetting to default state")
        self.px.reset()

    def close(self):
        """
        Clean shutdown of robot hardware.

        Resets robot and closes ultrasonic sensor resources.

        Example:
            >>> robot.close()
        """
        self.logger.info("[HARDWARE] Closing robot hardware")
        self.px.close()
