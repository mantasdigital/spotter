"""
Speech subsystem for TARS.

Provides STT, TTS, and language management capabilities.
"""

from speech.providers import VoskSTT, MockSTT, OpenAITTS, PiperTTS, MockTTS
from speech.language_manager import LanguageManager, MockLanguageManager

__all__ = [
    'VoskSTT',
    'MockSTT',
    'OpenAITTS',
    'PiperTTS',
    'MockTTS',
    'LanguageManager',
    'MockLanguageManager',
]
