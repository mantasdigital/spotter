"""
Action Resolver.

Dynamically resolves action strings (like "forward(50)") into executable functions.
Extracted from voice_active_car_gpt_modified_v58.py lines 3243-3354.
"""

import re
import logging
import threading
from typing import Dict, Callable, Optional, Any

from hardware.interfaces import IRobotCar
from core.state_manager import StateManager
from motion import primitives


logger = logging.getLogger(__name__)


class ActionResolver(dict):
    """
    Dynamic action resolver that maps action strings to callable functions.

    Acts as a dictionary that always contains any key and dynamically generates
    appropriate functions based on action name patterns.

    Attributes:
        robot: Robot car instance for motion
        state: State manager for tracking
        stop_flag: Threading event to signal emergency stop
        safety_manager: Optional safety manager for path checking
        actions_db: Optional database of learned actions
    """

    def __init__(
        self,
        robot: IRobotCar,
        state: StateManager,
        stop_flag: threading.Event,
        safety_manager: Optional[Any] = None,
        actions_db: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize action resolver.

        Args:
            robot: Robot car instance
            state: State manager
            stop_flag: Threading event for emergency stop
            safety_manager: Optional safety manager for path checking
            actions_db: Optional database of learned actions

        Example:
            >>> resolver = ActionResolver(robot, state, stop_event)
            >>> action_fn = resolver["forward(50)"]
            >>> action_fn()
        """
        super().__init__()
        self.robot = robot
        self.state = state
        self.stop_flag = stop_flag
        self.safety_manager = safety_manager
        self.actions_db = actions_db or {}

    def __contains__(self, key):
        """Action resolver always contains any key."""
        return True

    def __getitem__(self, key):
        """
        Resolve action string to callable function.

        Args:
            key: Action string (e.g. "forward(50)", "spin_left()", "wiggle()")

        Returns:
            Callable function that executes the action

        Example:
            >>> action_fn = resolver["forward(50)"]
            >>> action_fn()  # Executes forward motion
        """
        key_str = str(key).lower().strip()

        # Extract base name and check learned actions database
        base_name = key_str.split("(", 1)[0]
        base_name = base_name.replace(" ", "_")

        if self.actions_db and base_name in self.actions_db:
            # Could tune durations/speeds from learned actions
            logger.debug(f"[ACTION] Found learned action: {base_name}")

        # Extract numeric parameters
        nums = re.findall(r"\d+", key_str)
        dist = int(nums[0]) if nums else 30

        # --- SLEEP ACTION ---
        if key_str.startswith("sleep("):
            ms = int(nums[0]) if nums else 0
            return lambda *args: primitives.sleep_ms(ms, self.stop_flag)

        # --- FORWARD/BACKWARD MOVEMENT ---
        if "forward" in key_str or "ahead" in key_str:
            def do_forward(*args):
                safety_check = self.safety_manager.is_path_safe if self.safety_manager else None
                primitives.execute_move(
                    self.robot, self.state, "forward", dist,
                    self.stop_flag, safety_check
                )
            return do_forward

        if any(x in key_str for x in ["backward", "back", "reverse"]):
            def do_backward(*args):
                safety_check = self.safety_manager.is_path_safe if self.safety_manager else None
                primitives.execute_move(
                    self.robot, self.state, "backward", dist,
                    self.stop_flag, safety_check
                )
            return do_backward

        # --- TURNING ACTIONS ---
        if "left" in key_str and "spin" not in key_str:
            return lambda *args: primitives.turn_left(self.robot, self.stop_flag)

        if "right" in key_str and "spin" not in key_str:
            return lambda *args: primitives.turn_right(self.robot, self.stop_flag)

        # --- SPINNING ACTIONS ---
        if "spin_left" in key_str or ("spin" in key_str and "left" in key_str):
            def do_spin_left(*args):
                scan_func = self.safety_manager.scan_direction if self.safety_manager else None
                primitives.spin_in_place(
                    self.robot, self.state, "left", 1.0,
                    self.stop_flag, scan_func
                )
            return do_spin_left

        if "spin_right" in key_str or ("spin" in key_str and "right" in key_str):
            def do_spin_right(*args):
                scan_func = self.safety_manager.scan_direction if self.safety_manager else None
                primitives.spin_in_place(
                    self.robot, self.state, "right", 1.0,
                    self.stop_flag, scan_func
                )
            return do_spin_right

        if "spin" in key_str and "left" not in key_str and "right" not in key_str:
            def do_spin(*args):
                scan_func = self.safety_manager.scan_direction if self.safety_manager else None
                primitives.spin_in_place(
                    self.robot, self.state, "right", 1.0,
                    self.stop_flag, scan_func
                )
            return do_spin

        # --- SPECIAL MANEUVERS ---
        if "turn_around" in key_str or "u_turn" in key_str or "u-turn" in key_str:
            return lambda *args: primitives.turn_around(self.robot, self.state, self.stop_flag)

        if "square" in key_str:
            return lambda *args: primitives.drive_square(self.robot, self.state, dist, self.stop_flag)

        # --- WIGGLE VARIATIONS ---
        if "wiggle_drive" in key_str or "snake" in key_str:
            # Extract direction and speed if specified
            direction = "forward" if "forward" in key_str else "backward" if "backward" in key_str else "forward"
            speed = "crazy" if "crazy" in key_str else "fast" if "fast" in key_str else "slow" if "slow" in key_str else "normal"
            return lambda *args: primitives.wiggle_drive(self.robot, direction, 2.5, 35, speed)

        if "wiggle_vary" in key_str or "wiggle vary" in key_str:
            return lambda *args: primitives.wiggle_vary(self.robot, 3.5)

        if "wiggle_full" in key_str or "full wiggle" in key_str:
            return lambda *args: primitives.wiggle_full(self.robot, 3.0)

        if "wiggle_head" in key_str or "head wiggle" in key_str:
            speed = "fast" if "fast" in key_str else "slow" if "slow" in key_str else "normal"
            return lambda *args: primitives.wiggle_head(self.robot, speed, 4)

        if "wiggle" in key_str or "shake" in key_str:
            # Determine speed from command
            speed = "crazy" if "crazy" in key_str else "fast" if "fast" in key_str else "slow" if "slow" in key_str else "normal"
            count = 5 if "long" in key_str else 3
            return lambda *args: primitives.wiggle(self.robot, speed, count)

        # --- DANCE VARIATIONS ---
        if "dance" in key_str:
            # Extract dance style from command
            style = "random"
            if "snake" in key_str:
                style = "snake"
            elif "excited" in key_str or "celebration" in key_str or "happy" in key_str:
                style = "excited"
            elif "smooth" in key_str or "elegant" in key_str or "slow" in key_str:
                style = "smooth"
            elif "robot" in key_str or "mechanical" in key_str:
                style = "robot"
            elif "disco" in key_str or "party" in key_str:
                style = "disco"
            elif "basic" in key_str or "simple" in key_str:
                style = "basic"
            elif "random" in key_str or "surprise" in key_str:
                style = "random"

            def do_dance(*args, dance_style=style):
                primitives.dance(self.robot, self.state, self.stop_flag, dance_style)
            return do_dance

        # --- HEAD MOVEMENTS ---
        if "head_center" in key_str or "head center" in key_str:
            return lambda *args: primitives.head_center(self.robot)

        if "head_left" in key_str or "head left" in key_str:
            return lambda *args: primitives.head_left(self.robot)

        if "head_right" in key_str or "head right" in key_str:
            return lambda *args: primitives.head_right(self.robot)

        if "head_scan" in key_str or "head scan" in key_str:
            return lambda *args: primitives.head_scan(self.robot)

        # --- STOP ---
        if "stop" in key_str:
            def do_stop(*args):
                self.robot.stop()
                self.stop_flag.set()
                logger.info("[ACTION] Emergency stop triggered")
            return do_stop

        # --- GESTURES (delegate to preset_actions if available) ---
        # These would be handled by importing picarx.preset_actions
        # For now, we log them as requiring preset_actions integration
        gesture_keywords = [
            "wave_hands", "wave hands",
            "shake_head", "shake head",
            "nod", "nod_head", "nod head",
            "resist", "act_cute", "rub_hands", "think",
            "twist_body", "celebrate", "depressed"
        ]

        for gesture in gesture_keywords:
            if gesture.replace("_", " ") in key_str or gesture in key_str:
                def do_gesture(*args, gesture_name=gesture):
                    logger.info(f"[ACTION] Gesture action: {gesture_name}")
                    logger.warning(f"[ACTION] Gesture '{gesture_name}' requires picarx.preset_actions integration")
                    # TODO: Integrate with picarx.preset_actions when available
                return do_gesture

        # --- SYMBOLIC/HELPER ACTIONS (non-motor) ---
        # Ignore symbolic helper-like actions so they don't look like motor errors
        if any(kw in key_str for kw in ("memory.visual", "what_do_you_see", "remember(")):
            return lambda *args, **kwargs: logger.debug(
                f"[SYSTEM] Skipping non-motor helper action: {key_str}"
            )

        # --- FALLBACK ---
        # Default: log unknown motor-ish actions
        def unknown_action(*args, **kwargs):
            logger.warning(f"[SYSTEM] No motor action defined for: {key_str}")

        return unknown_action


def create_preset_actions_wrapper(
    robot: IRobotCar,
    state: StateManager,
    stop_flag: threading.Event,
    safety_manager: Optional[Any] = None
) -> ActionResolver:
    """
    Create an ActionResolver that can be used as picarx.preset_actions.actions_dict.

    This allows integration with existing code that expects preset_actions.

    Args:
        robot: Robot car instance
        state: State manager
        stop_flag: Threading event for emergency stop
        safety_manager: Optional safety manager

    Returns:
        ActionResolver instance

    Example:
        >>> import picarx.preset_actions
        >>> resolver = create_preset_actions_wrapper(robot, state, stop_event, safety)
        >>> picarx.preset_actions.actions_dict = resolver
    """
    return ActionResolver(robot, state, stop_flag, safety_manager)
