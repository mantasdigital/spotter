"""
Motion System.

Handles all robot movement including primitives, safety checking, action resolution,
and execution.
"""

from motion import primitives
from motion.safety_manager import SafetyManager
from motion.action_resolver import ActionResolver, create_preset_actions_wrapper
from motion.action_executor import ActionExecutor

__all__ = [
    "primitives",
    "SafetyManager",
    "ActionResolver",
    "create_preset_actions_wrapper",
    "ActionExecutor",
]
