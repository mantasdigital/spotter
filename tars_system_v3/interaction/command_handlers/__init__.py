"""
Command handlers for processing voice commands.
"""

from interaction.command_handlers.motion_commands import MotionCommandHandler
from interaction.command_handlers.behavior_commands import BehaviorCommandHandler
from interaction.command_handlers.system_commands import SystemCommandHandler
from interaction.command_handlers.macro_commands import MacroCommandHandler
from interaction.command_handlers.web_search_commands import WebSearchCommandHandler
from interaction.command_handlers.visual_commands import VisualCommandHandler

__all__ = [
    'MotionCommandHandler',
    'BehaviorCommandHandler',
    'SystemCommandHandler',
    'MacroCommandHandler',
    'WebSearchCommandHandler',
    'VisualCommandHandler',
]
