"""
Behavior subsystem for TARS.

Provides autonomous behaviors including roaming, face tracking, and following.
"""

from behaviors.roam import RoamBehavior
from behaviors.stare_behavior import StareBehavior, FollowBehavior
from behaviors.auto_roam_watchdog import AutoRoamWatchdog

__all__ = [
    'RoamBehavior',
    'StareBehavior',
    'FollowBehavior',
    'AutoRoamWatchdog',
]
