"""
Speech subsystem for TARS.

Provides STT, TTS, and language management capabilities.
"""

from speech.providers import VoskSTT, MockSTT, OpenAITTS, PiperTTS, MockTTS
from speech.language_manager import LanguageManager, MockLanguageManager
from speech.lt_fuzzy import fuzzy_match_lt_command, normalize_lt, is_lithuanian_text

__all__ = [
    'VoskSTT',
    'MockSTT',
    'OpenAITTS',
    'PiperTTS',
    'MockTTS',
    'LanguageManager',
    'MockLanguageManager',
    'fuzzy_match_lt_command',
    'normalize_lt',
    'is_lithuanian_text',
]
