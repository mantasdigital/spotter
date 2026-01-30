"""
Motion Primitives.

Low-level movement functions extracted from voice_active_car_gpt_modified_v58.py.
All functions use dependency injection instead of globals for testability.
"""

import time
import logging
import threading
import random
from typing import Optional, Callable

from hardware.interfaces import IRobotCar
from core.state_manager import StateManager
from utils.music_player import play_dance_music, stop_dance_music


logger = logging.getLogger(__name__)


def execute_move(
    robot: IRobotCar,
    state: StateManager,
    direction: str,
    dist_cm: float,
    stop_flag: threading.Event,
    safety_check: Optional[Callable[[str], bool]] = None
):
    """
    Execute basic movement for a distance.

    Args:
        robot: Robot car instance
        state: State manager
        direction: "forward" or "backward"
        dist_cm: Distance to move in centimeters
        stop_flag: Threading event to signal emergency stop
        safety_check: Optional function to check if path is safe

    Example:
        >>> execute_move(robot, state, "forward", 50, stop_event)
    """
    logger.info(f"[MOTION] {direction.capitalize()}ing: {dist_cm}cm")

    # Approximate duration (tuned for 50% speed)
    duration = (dist_cm / 100.0) * 1.8
    start_time = time.time()

    while time.time() - start_time < duration:
        if stop_flag.is_set():
            robot.stop()
            break

        if safety_check and not safety_check(direction):
            break

        if direction == "forward":
            robot.forward(50)
        else:
            robot.backward(50)

        time.sleep(0.05)

    robot.stop()


def spin_in_place(
    robot: IRobotCar,
    state: StateManager,
    direction: str,
    duration: float,
    stop_flag: threading.Event,
    scan_func: Optional[Callable[[str], bool]] = None
):
    """
    Spin robot in place.

    Args:
        robot: Robot car instance
        state: State manager
        direction: "left" or "right"
        duration: Spin duration in seconds
        stop_flag: Threading event to signal emergency stop
        scan_func: Optional function to pre-scan direction

    Note:
        Updates heading in roam state for odometry tracking.

    Example:
        >>> spin_in_place(robot, state, "left", 1.0, stop_event)
    """
    if stop_flag.is_set():
        robot.stop()
        return

    # Optional pre-scan safety check
    if scan_func and not scan_func(direction):
        logger.warning(f"[SAFETY] Spin {direction} aborted by pre-scan.")
        robot.stop()
        return

    # Set steering angle for spin
    angle = -35 if direction == "left" else 35
    robot.set_dir_servo_angle(angle)

    start = time.time()
    while time.time() - start < duration:
        if stop_flag.is_set():
            robot.stop()
            break
        robot.forward(50)

        # Update approximate heading (assume ~90 deg per 1.0s spin)
        turn_sign = -1.0 if direction == "left" else 1.0
        delta = turn_sign * (90.0 * 0.05 / duration)
        state.roam.integrate_motion(distance_cm=0, turn_deg=delta)

        time.sleep(0.05)

    robot.set_dir_servo_angle(0)
    robot.stop()


def side_step(
    robot: IRobotCar,
    state: StateManager,
    direction: str,
    step_time: float,
    speed: int,
    stop_flag: threading.Event,
    force: bool = False,
    safety_check: Optional[Callable[[str], bool]] = None
):
    """
    Small lateral translation by steering and driving forward.

    Used in roam mode to get around obstacles that only block straight path.

    Args:
        robot: Robot car instance
        state: State manager
        direction: "left" or "right"
        step_time: Step duration in seconds
        speed: Movement speed (0-100)
        stop_flag: Threading event to signal emergency stop
        force: If True, skip safety checks during step
        safety_check: Optional function to check if path is safe

    Example:
        >>> side_step(robot, state, "left", 0.7, 45, stop_event)
    """
    if stop_flag.is_set():
        robot.stop()
        return

    angle = -25 if direction == "left" else 25
    robot.set_dir_servo_angle(angle)

    start = time.time()
    while time.time() - start < step_time:
        if stop_flag.is_set():
            break
        if not force and safety_check and not safety_check("forward"):
            break
        robot.forward(speed)
        time.sleep(0.05)

    # Side-step still moves roughly forward
    approx_cm = speed * step_time * 0.3
    state.roam.integrate_motion(distance_cm=approx_cm)

    robot.set_dir_servo_angle(0)
    robot.stop()


def hard_escape_turn(
    robot: IRobotCar,
    state: StateManager,
    stop_flag: threading.Event
):
    """
    Aggressive escape move for when deeply boxed in.

    Backs up more and turns harder/longer than normal spin.

    Args:
        robot: Robot car instance
        state: State manager
        stop_flag: Threading event to signal emergency stop

    Example:
        >>> hard_escape_turn(robot, state, stop_event)
    """
    if stop_flag.is_set():
        robot.stop()
        return

    logger.info("[ROAM] HARD ESCAPE: backing up and turning strongly.")
    robot.backward(40)
    time.sleep(1.2)
    robot.stop()

    # Stronger spin, pick random direction
    angle = 35 if (int(time.time() * 1000) % 2 == 0) else -35
    robot.set_dir_servo_angle(angle)

    start = time.time()
    duration = 1.6
    while time.time() - start < duration:
        if stop_flag.is_set():
            break
        robot.forward(50)

        # Update heading
        turn_sign = 1.0 if angle > 0 else -1.0
        delta = turn_sign * (90.0 * 0.05 / duration)
        state.roam.integrate_motion(distance_cm=0, turn_deg=delta)

        time.sleep(0.05)

    robot.set_dir_servo_angle(0)
    robot.stop()


def safe_forward_burst(
    robot: IRobotCar,
    state: StateManager,
    speed: int,
    duration: float,
    stop_flag: threading.Event,
    scan_dir: str = "forward",
    scan_func: Optional[Callable[[str], bool]] = None,
    safety_check: Optional[Callable[[str], bool]] = None
):
    """
    Move forward with continuous safety checking.

    Used by roam mode for exploration with collision avoidance.

    Args:
        robot: Robot car instance
        state: State manager
        speed: Movement speed (0-100)
        duration: Movement duration in seconds
        stop_flag: Threading event to signal emergency stop
        scan_dir: Direction to pre-scan ("forward", "left", "right")
        scan_func: Optional function to pre-scan direction
        safety_check: Optional function for continuous safety checking

    Example:
        >>> safe_forward_burst(robot, state, 50, 0.8, stop_event)
    """
    if stop_flag.is_set():
        robot.stop()
        return

    # Pre-scan before moving
    if scan_func and not scan_func(scan_dir):
        logger.warning(f"[SAFETY] Pre-scan blocked {scan_dir} movement.")
        robot.stop()
        return

    start = time.time()
    while time.time() - start < duration:
        if stop_flag.is_set():
            robot.stop()
            break
        if safety_check and not safety_check("forward"):
            break
        robot.forward(speed)
        time.sleep(0.05)

    # Approximate distance traveled
    approx_cm = speed * duration * 0.5  # Tune scale factor as needed
    state.roam.integrate_motion(distance_cm=approx_cm)

    # Mark current pose as visited
    state.roam.mark_visited(cell_size_cm=25.0)

    robot.stop()


def wiggle(robot: IRobotCar, speed: str = "normal", count: int = 3):
    """
    Wiggle steering back and forth.

    Simple gesture action for expressiveness with variable speed.

    Args:
        robot: Robot car instance
        speed: "slow", "normal", "fast", or "crazy"
        count: Number of wiggle cycles

    Example:
        >>> wiggle(robot)
        >>> wiggle(robot, speed="fast", count=5)
    """
    # Speed presets (delay between direction changes)
    speed_map = {
        "slow": 0.35,
        "normal": 0.2,
        "fast": 0.12,
        "crazy": 0.06
    }
    delay = speed_map.get(speed, 0.2)

    for _ in range(count):
        robot.set_dir_servo_angle(-35)
        time.sleep(delay)
        robot.set_dir_servo_angle(35)
        time.sleep(delay)
    robot.set_dir_servo_angle(0)
    robot.stop()


def wiggle_drive(
    robot: IRobotCar,
    direction: str = "forward",
    duration: float = 2.0,
    drive_speed: int = 30,
    wiggle_speed: str = "normal"
):
    """
    Wiggle while driving forward or backward.

    Creates a fun snake-like movement pattern.

    Args:
        robot: Robot car instance
        direction: "forward" or "backward"
        duration: How long to wiggle-drive in seconds
        drive_speed: Movement speed (0-100)
        wiggle_speed: "slow", "normal", "fast", or "crazy"

    Example:
        >>> wiggle_drive(robot, "forward", 3.0, 40, "fast")
    """
    speed_map = {
        "slow": 0.4,
        "normal": 0.25,
        "fast": 0.15,
        "crazy": 0.08
    }
    wiggle_delay = speed_map.get(wiggle_speed, 0.25)

    start_time = time.time()
    angle = -30
    angle_step = 60  # Swing from -30 to +30

    while time.time() - start_time < duration:
        robot.set_dir_servo_angle(angle)
        if direction == "forward":
            robot.forward(drive_speed)
        else:
            robot.backward(drive_speed)
        time.sleep(wiggle_delay)
        angle = -angle  # Flip direction

    robot.set_dir_servo_angle(0)
    robot.stop()


def wiggle_vary(robot: IRobotCar, duration: float = 3.0):
    """
    Wiggle with varying speeds - accelerates and decelerates.

    Creates an organic, playful wiggle effect.

    Args:
        robot: Robot car instance
        duration: Total duration in seconds

    Example:
        >>> wiggle_vary(robot, 4.0)
    """
    start_time = time.time()
    angle = -35
    delay = 0.3  # Start slow

    while time.time() - start_time < duration:
        robot.set_dir_servo_angle(angle)
        time.sleep(delay)
        angle = -angle

        # Vary the speed - accelerate then decelerate
        elapsed = time.time() - start_time
        progress = elapsed / duration

        if progress < 0.3:
            # Accelerating phase
            delay = 0.3 - (progress * 0.6)  # 0.3 -> 0.12
        elif progress < 0.7:
            # Fast phase
            delay = 0.08 + (random.random() * 0.04)  # 0.08-0.12 with randomness
        else:
            # Decelerating phase
            delay = 0.12 + ((progress - 0.7) * 0.6)  # 0.12 -> 0.3

        delay = max(0.05, min(0.4, delay))  # Clamp

    robot.set_dir_servo_angle(0)
    robot.stop()


def wiggle_head(robot: IRobotCar, speed: str = "normal", count: int = 3):
    """
    Wiggle the camera/head left and right.

    Args:
        robot: Robot car instance
        speed: "slow", "normal", "fast"
        count: Number of wiggle cycles

    Example:
        >>> wiggle_head(robot, "fast", 5)
    """
    speed_map = {"slow": 0.3, "normal": 0.18, "fast": 0.1}
    delay = speed_map.get(speed, 0.18)

    for _ in range(count):
        robot.set_cam_pan_angle(-25)
        time.sleep(delay)
        robot.set_cam_pan_angle(25)
        time.sleep(delay)
    robot.set_cam_pan_angle(0)


def wiggle_full(robot: IRobotCar, duration: float = 3.0):
    """
    Full body wiggle - steering AND head together.

    Args:
        robot: Robot car instance
        duration: Duration in seconds

    Example:
        >>> wiggle_full(robot, 4.0)
    """
    start_time = time.time()
    angle = -30
    delay = 0.15

    while time.time() - start_time < duration:
        robot.set_dir_servo_angle(angle)
        robot.set_cam_pan_angle(int(angle * 0.7))  # Head follows but less extreme
        time.sleep(delay)
        angle = -angle
        # Slight speed variation
        delay = 0.12 + (random.random() * 0.06)

    robot.set_dir_servo_angle(0)
    robot.set_cam_pan_angle(0)
    robot.stop()


def drive_square(
    robot: IRobotCar,
    state: StateManager,
    side_cm: float,
    stop_flag: threading.Event
):
    """
    Drive in a square pattern.

    Args:
        robot: Robot car instance
        state: State manager
        side_cm: Side length in centimeters
        stop_flag: Threading event to signal emergency stop

    Example:
        >>> drive_square(robot, state, 20, stop_event)
    """
    for _ in range(4):
        execute_move(robot, state, "forward", side_cm, stop_flag)
        spin_in_place(robot, state, "right", 0.8, stop_flag)


def head_center(robot: IRobotCar):
    """
    Center camera/head servos.

    Args:
        robot: Robot car instance

    Example:
        >>> head_center(robot)
    """
    try:
        robot.set_cam_pan_angle(0)
        robot.set_cam_tilt_angle(0)
    except Exception as e:
        logger.error(f"Failed to center head: {e}")


def head_left(robot: IRobotCar):
    """
    Small head tilt to the left.

    Args:
        robot: Robot car instance

    Example:
        >>> head_left(robot)
    """
    try:
        robot.set_cam_pan_angle(-25)
    except Exception as e:
        logger.error(f"Failed to tilt head left: {e}")


def head_right(robot: IRobotCar):
    """
    Small head tilt to the right.

    Args:
        robot: Robot car instance

    Example:
        >>> head_right(robot)
    """
    try:
        robot.set_cam_pan_angle(25)
    except Exception as e:
        logger.error(f"Failed to tilt head right: {e}")


def head_scan(robot: IRobotCar):
    """
    Simple 'look around' scan.

    Pans camera left, center, right, center.

    Args:
        robot: Robot car instance

    Example:
        >>> head_scan(robot)
    """
    try:
        for angle in (-25, 0, 25, 0):
            robot.set_cam_pan_angle(angle)
            time.sleep(0.4)
    except Exception as e:
        logger.error(f"Failed to scan head: {e}")


def sleep_ms(ms: int, stop_flag: threading.Event):
    """
    Sleep for milliseconds with interruptible checking.

    Args:
        ms: Milliseconds to sleep
        stop_flag: Threading event to signal early termination

    Example:
        >>> sleep_ms(1000, stop_event)  # Sleep 1 second
    """
    try:
        ms_val = int(ms)
    except Exception:
        ms_val = 0

    end_time = time.time() + (ms_val / 1000.0)
    while time.time() < end_time:
        if stop_flag.is_set():
            break
        time.sleep(0.05)


def turn_left(
    robot: IRobotCar,
    stop_flag: threading.Event,
    duration: float = 0.8
):
    """
    Simple left turn.

    Steer left and drive forward briefly.

    Args:
        robot: Robot car instance
        stop_flag: Threading event to signal emergency stop
        duration: Turn duration in seconds

    Example:
        >>> turn_left(robot, stop_event)
    """
    if stop_flag.is_set():
        robot.stop()
        return

    robot.set_dir_servo_angle(-35)
    robot.forward(50)
    time.sleep(duration)
    robot.set_dir_servo_angle(0)
    robot.stop()


def turn_right(
    robot: IRobotCar,
    stop_flag: threading.Event,
    duration: float = 0.8
):
    """
    Simple right turn.

    Steer right and drive forward briefly.

    Args:
        robot: Robot car instance
        stop_flag: Threading event to signal emergency stop
        duration: Turn duration in seconds

    Example:
        >>> turn_right(robot, stop_event)
    """
    if stop_flag.is_set():
        robot.stop()
        return

    robot.set_dir_servo_angle(35)
    robot.forward(50)
    time.sleep(duration)
    robot.set_dir_servo_angle(0)
    robot.stop()


def turn_around(
    robot: IRobotCar,
    state: StateManager,
    stop_flag: threading.Event
):
    """
    180-degree turn.

    Args:
        robot: Robot car instance
        state: State manager
        stop_flag: Threading event to signal emergency stop

    Example:
        >>> turn_around(robot, state, stop_event)
    """
    spin_in_place(robot, state, "right", duration=2.0, stop_flag=stop_flag)


# =============================================================================
# DANCE ROUTINES - Various dance moves and combinations
# =============================================================================

def dance_basic(robot: IRobotCar, state: StateManager, stop_flag: threading.Event):
    """
    Basic dance routine - wiggle and spin.

    Duration: ~6 seconds

    Args:
        robot: Robot car instance
        state: State manager
        stop_flag: Threading event to signal emergency stop
    """
    if stop_flag.is_set():
        return

    # Start with a wiggle
    wiggle(robot, speed="normal", count=4)

    if stop_flag.is_set():
        return

    # Spin right
    spin_in_place(robot, state, "right", 0.8, stop_flag)

    if stop_flag.is_set():
        return

    # Fast wiggle
    wiggle(robot, speed="fast", count=5)

    if stop_flag.is_set():
        return

    # Spin left
    spin_in_place(robot, state, "left", 0.8, stop_flag)

    # Final wiggle
    wiggle(robot, speed="normal", count=3)


def dance_snake(robot: IRobotCar, state: StateManager, stop_flag: threading.Event):
    """
    Snake dance - wiggle while driving forward and backward.

    Duration: ~10 seconds

    Args:
        robot: Robot car instance
        state: State manager
        stop_flag: Threading event to signal emergency stop
    """
    if stop_flag.is_set():
        return

    # Snake forward - slow start
    wiggle_drive(robot, "forward", 2.0, 25, "slow")

    if stop_flag.is_set():
        return

    # Speed up
    wiggle_drive(robot, "forward", 2.0, 35, "fast")

    if stop_flag.is_set():
        return

    # Snake backward - crazy speed
    wiggle_drive(robot, "backward", 2.0, 30, "crazy")

    if stop_flag.is_set():
        return

    # Slow down elegantly
    wiggle_drive(robot, "backward", 1.5, 20, "slow")

    if stop_flag.is_set():
        return

    # Finish with varying wiggle in place
    wiggle_vary(robot, 2.5)


def dance_excited(robot: IRobotCar, state: StateManager, stop_flag: threading.Event):
    """
    Excited/celebration dance - fast movements and spins.

    Duration: ~12 seconds

    Args:
        robot: Robot car instance
        state: State manager
        stop_flag: Threading event to signal emergency stop
    """
    if stop_flag.is_set():
        return

    # Crazy wiggle to start
    wiggle(robot, speed="crazy", count=6)

    if stop_flag.is_set():
        return

    # Quick spin right
    spin_in_place(robot, state, "right", 0.5, stop_flag)

    if stop_flag.is_set():
        return

    # Wiggle with head
    wiggle_full(robot, 2.0)

    if stop_flag.is_set():
        return

    # Quick spin left
    spin_in_place(robot, state, "left", 0.5, stop_flag)

    if stop_flag.is_set():
        return

    # Fast forward wiggle
    wiggle_drive(robot, "forward", 1.5, 40, "crazy")

    if stop_flag.is_set():
        return

    # More crazy wiggles
    wiggle(robot, speed="crazy", count=8)

    if stop_flag.is_set():
        return

    # Full spin
    spin_in_place(robot, state, "right", 1.5, stop_flag)

    # Final celebration wiggle
    wiggle_vary(robot, 2.0)


def dance_smooth(robot: IRobotCar, state: StateManager, stop_flag: threading.Event):
    """
    Smooth/elegant dance - slow, flowing movements.

    Duration: ~15 seconds

    Args:
        robot: Robot car instance
        state: State manager
        stop_flag: Threading event to signal emergency stop
    """
    if stop_flag.is_set():
        return

    # Slow graceful wiggle
    wiggle(robot, speed="slow", count=4)

    if stop_flag.is_set():
        return

    # Slow turn with head scan
    robot.set_dir_servo_angle(-25)
    robot.forward(30)
    robot.set_cam_pan_angle(-20)
    time.sleep(1.0)

    if stop_flag.is_set():
        robot.stop()
        return

    robot.set_cam_pan_angle(20)
    time.sleep(1.0)
    robot.stop()
    robot.set_dir_servo_angle(0)
    robot.set_cam_pan_angle(0)

    if stop_flag.is_set():
        return

    # Graceful backward glide with slow wiggle
    wiggle_drive(robot, "backward", 2.5, 25, "slow")

    if stop_flag.is_set():
        return

    # Slow spin
    spin_in_place(robot, state, "left", 1.5, stop_flag)

    if stop_flag.is_set():
        return

    # Forward glide
    wiggle_drive(robot, "forward", 2.0, 30, "slow")

    if stop_flag.is_set():
        return

    # Elegant varying wiggle finish
    wiggle_vary(robot, 3.0)


def dance_robot(robot: IRobotCar, state: StateManager, stop_flag: threading.Event):
    """
    Robot-style dance - mechanical, rhythmic movements.

    Duration: ~14 seconds

    Args:
        robot: Robot car instance
        state: State manager
        stop_flag: Threading event to signal emergency stop
    """
    if stop_flag.is_set():
        return

    # Mechanical head movements
    for _ in range(3):
        if stop_flag.is_set():
            break
        robot.set_cam_pan_angle(-30)
        time.sleep(0.3)
        robot.set_cam_pan_angle(30)
        time.sleep(0.3)
    robot.set_cam_pan_angle(0)

    if stop_flag.is_set():
        return

    # Step left, pause, step right, pause (robotic)
    robot.set_dir_servo_angle(-30)
    robot.forward(40)
    time.sleep(0.4)
    robot.stop()
    time.sleep(0.2)

    if stop_flag.is_set():
        return

    robot.set_dir_servo_angle(30)
    robot.forward(40)
    time.sleep(0.4)
    robot.stop()
    time.sleep(0.2)
    robot.set_dir_servo_angle(0)

    if stop_flag.is_set():
        return

    # Precise wiggle (normal speed, exact counts)
    wiggle(robot, speed="normal", count=4)

    if stop_flag.is_set():
        return

    # 90-degree turn pause 90-degree turn
    spin_in_place(robot, state, "right", 0.6, stop_flag)
    time.sleep(0.3)

    if stop_flag.is_set():
        return

    spin_in_place(robot, state, "right", 0.6, stop_flag)

    if stop_flag.is_set():
        return

    # Forward march with wiggle
    wiggle_drive(robot, "forward", 2.0, 35, "normal")

    if stop_flag.is_set():
        return

    # Final mechanical wiggle
    wiggle(robot, speed="normal", count=5)


def dance_disco(robot: IRobotCar, state: StateManager, stop_flag: threading.Event):
    """
    Disco dance - alternating fast and slow with dramatic pauses.

    Duration: ~16 seconds

    Args:
        robot: Robot car instance
        state: State manager
        stop_flag: Threading event to signal emergency stop
    """
    if stop_flag.is_set():
        return

    # Start with pose (head tilt)
    robot.set_cam_pan_angle(-25)
    robot.set_cam_tilt_angle(15)
    time.sleep(0.5)

    if stop_flag.is_set():
        return

    # Burst into crazy wiggle
    wiggle(robot, speed="crazy", count=6)

    if stop_flag.is_set():
        return

    # Pose again
    robot.set_cam_pan_angle(25)
    robot.set_cam_tilt_angle(-10)
    time.sleep(0.5)
    robot.set_cam_pan_angle(0)
    robot.set_cam_tilt_angle(0)

    if stop_flag.is_set():
        return

    # Spin!
    spin_in_place(robot, state, "left", 1.0, stop_flag)

    if stop_flag.is_set():
        return

    # Wiggle drive combo - fast forward
    wiggle_drive(robot, "forward", 1.5, 45, "fast")

    if stop_flag.is_set():
        return

    # Dramatic pause
    time.sleep(0.3)

    if stop_flag.is_set():
        return

    # Fast backward
    wiggle_drive(robot, "backward", 1.5, 45, "fast")

    if stop_flag.is_set():
        return

    # Another spin
    spin_in_place(robot, state, "right", 1.0, stop_flag)

    if stop_flag.is_set():
        return

    # Full body finale
    wiggle_full(robot, 3.0)

    # End pose
    robot.set_cam_pan_angle(-20)
    time.sleep(0.3)
    robot.set_cam_pan_angle(0)


def dance_random(robot: IRobotCar, state: StateManager, stop_flag: threading.Event):
    """
    Random dance - picks random moves for variety each time.

    Duration: ~12-18 seconds (varies)

    Args:
        robot: Robot car instance
        state: State manager
        stop_flag: Threading event to signal emergency stop
    """
    moves = [
        lambda: wiggle(robot, speed=random.choice(["slow", "normal", "fast", "crazy"]),
                      count=random.randint(3, 7)),
        lambda: wiggle_drive(robot, random.choice(["forward", "backward"]),
                            random.uniform(1.0, 2.5),
                            random.randint(25, 45),
                            random.choice(["slow", "normal", "fast"])),
        lambda: wiggle_vary(robot, random.uniform(2.0, 4.0)),
        lambda: wiggle_full(robot, random.uniform(1.5, 3.0)),
        lambda: wiggle_head(robot, random.choice(["slow", "normal", "fast"]),
                           random.randint(3, 6)),
        lambda: spin_in_place(robot, state, random.choice(["left", "right"]),
                             random.uniform(0.5, 1.2), stop_flag),
    ]

    # Do 5-8 random moves
    num_moves = random.randint(5, 8)
    for i in range(num_moves):
        if stop_flag.is_set():
            break
        move = random.choice(moves)
        try:
            move()
        except Exception as e:
            logger.warning(f"Dance move failed: {e}")
        time.sleep(0.2)  # Small pause between moves

    robot.set_dir_servo_angle(0)
    robot.set_cam_pan_angle(0)
    robot.stop()


def dance(robot: IRobotCar, state: StateManager, stop_flag: threading.Event,
          style: str = "random"):
    """
    Main dance function - selects and executes a dance style with music.

    Automatically plays a random dance track during the dance and stops
    the music when the dance completes or is interrupted.

    Args:
        robot: Robot car instance
        state: State manager
        stop_flag: Threading event to signal emergency stop
        style: Dance style - "basic", "snake", "excited", "smooth",
               "robot", "disco", "random", or "surprise"

    Example:
        >>> dance(robot, state, stop_event, "excited")
        >>> dance(robot, state, stop_event)  # Random style
    """
    style = style.lower().strip()

    dance_map = {
        "basic": dance_basic,
        "snake": dance_snake,
        "excited": dance_excited,
        "celebration": dance_excited,
        "smooth": dance_smooth,
        "elegant": dance_smooth,
        "robot": dance_robot,
        "mechanical": dance_robot,
        "disco": dance_disco,
        "party": dance_disco,
        "random": dance_random,
        "surprise": dance_random,
    }

    # If style not found or "surprise", pick random
    if style == "surprise" or style not in dance_map:
        style = random.choice(["basic", "snake", "excited", "smooth", "robot", "disco"])

    # Start playing dance music
    track_name = play_dance_music()
    if track_name:
        logger.info(f"[DANCE] Music: {track_name}")

    try:
        logger.info(f"[DANCE] Starting {style} dance!")
        dance_func = dance_map.get(style, dance_random)
        dance_func(robot, state, stop_flag)
        logger.info(f"[DANCE] {style.capitalize()} dance complete!")
    finally:
        # Always stop the music when dance ends (success or interrupt)
        stop_dance_music()
        logger.info("[DANCE] Music stopped")
