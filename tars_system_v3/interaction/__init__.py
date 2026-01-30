"""
Interaction subsystem for TARS.

Provides command processing and voice interaction capabilities.
"""

from interaction.command_processor import CommandProcessor, SimpleCommandRouter, MockCommandProcessor
from interaction.command_handlers import MotionCommandHandler, BehaviorCommandHandler

__all__ = [
    'CommandProcessor',
    'SimpleCommandRouter',
    'MockCommandProcessor',
    'MotionCommandHandler',
    'BehaviorCommandHandler',
]
