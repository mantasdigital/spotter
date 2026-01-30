"""
Language Detection and Management.

Handles language detection from speech and manages language switching
for STT and TTS systems.
"""

import re
from typing import Optional, Set, Dict, Any

from hardware.interfaces import ISTT, ITTS


class LanguageManager:
    """
    Language detection and switching manager.

    Manages language state and coordinates language switching across
    STT and TTS systems.
    """

    # Language detection keywords (from v58 + Vosk misrecognitions)
    LANGUAGE_KEYWORDS = {
        "en": [
            "switch to english",
            "speak english",
            "english mode",
            "change to english",
            "talk english",
            "talk in english",
            "kalbek angliskai",
        ],
        "lt": [
            # Explicit patterns
            "speak lithuanian", "talk lithuanian", "talk in lithuanian",
            "switch to lithuanian", "switch to a lithuanian",
            "switched to a lithuanian", "switch to lithuanians",
            "switch to lithuania", "switched to lithuanian",
            "switched to lithuanians", "switch the lithuanian",
            "switch to lithuanian and", "switched to lithuanian and",
            # Vosk misrecognitions (from v58)
            "switch to live waning", "switch to live way",
            "switch to live way inin", "switch to look way in",
            "switched to live way in", "switch to live way the and",
            "switch to live for anyone", "switch to live way neon",
            # Additional Vosk misrecognitions
            "switch to live for any", "switch to live for any and",
            "switch to live", "switch to lith", "switch to lit",
            "switch to leave", "switch to life", "switch to liv",
            "switched to live", "switch to living", "switch to live in",
            "switch to live away", "switch to live weigh",
            "switch to live alien", "switch to live anyone",
            "switch live", "switch to the lithuanian",
            # Lithuanian language patterns
            "pakeisk i lietuviu", "lietuviu kalba",
            "kalbek lietuviskai",
        ]
    }

    # Common greetings for language detection
    GREETINGS = {
        "en": ["hello", "hi", "hey", "greetings"],
        "lt": ["labas", "sveikas", "sveiki", "labą dieną"]
    }

    def __init__(
        self,
        stt: ISTT,
        tts: ITTS,
        supported_languages: Optional[Set[str]] = None,
        default_language: str = "en"
    ):
        """
        Initialize language manager.

        Args:
            stt: Speech-to-text provider
            tts: Text-to-speech provider
            supported_languages: Set of supported language codes (defaults to en, lt)
            default_language: Default language to use
        """
        self.stt = stt
        self.tts = tts
        self.supported_languages = supported_languages or {"en", "lt"}
        self.current_language = default_language

        # Set initial language
        self.switch_language(default_language)

    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect language from text.

        Checks for language-specific keywords and patterns to identify
        the language of the input text.

        Args:
            text: Input text to analyze

        Returns:
            Language code if detected, None otherwise
        """
        if not text:
            return None

        text_lower = text.lower().strip()

        # Check for explicit language switch commands
        for lang, keywords in self.LANGUAGE_KEYWORDS.items():
            if lang not in self.supported_languages:
                continue

            for keyword in keywords:
                if keyword in text_lower:
                    return lang

        # Check for language-specific greetings
        for lang, greetings in self.GREETINGS.items():
            if lang not in self.supported_languages:
                continue

            for greeting in greetings:
                if text_lower.startswith(greeting):
                    return lang

        # Check for Lithuanian characters
        if re.search(r'[ąčęėįšųūž]', text_lower):
            if "lt" in self.supported_languages:
                return "lt"

        # Default: no detection
        return None

    def check_and_switch_language(self, text: str) -> bool:
        """
        Check text for language switch command and switch if detected.

        Args:
            text: Text to check for language switch commands

        Returns:
            bool: True if language was switched, False otherwise
        """
        detected_lang = self.detect_language(text)

        if detected_lang and detected_lang != self.current_language:
            self.switch_language(detected_lang)
            return True

        return False

    def switch_language(self, language: str):
        """
        Switch to specified language.

        Updates both STT and TTS to use the new language.

        Args:
            language: Language code to switch to

        Raises:
            ValueError: If language is not supported
        """
        if language not in self.supported_languages:
            raise ValueError(
                f"Language '{language}' not supported. "
                f"Supported: {', '.join(sorted(self.supported_languages))}"
            )

        self.current_language = language

        # Update STT language
        self.stt.set_language(language)

        # TTS typically auto-detects language from text, but we track it
        print(f"Language switched to: {language}")

    def get_current_language(self) -> str:
        """
        Get current language.

        Returns:
            Current language code
        """
        return self.current_language

    def get_language_name(self, lang_code: Optional[str] = None) -> str:
        """
        Get human-readable language name.

        Args:
            lang_code: Language code (uses current language if None)

        Returns:
            Language name in English
        """
        lang = lang_code or self.current_language

        names = {
            "en": "English",
            "lt": "Lithuanian",
            "es": "Spanish",
            "fr": "French",
            "de": "German"
        }

        return names.get(lang, lang.upper())

    def get_confirmation_message(self, language: str) -> str:
        """
        Get language switch confirmation message.

        Args:
            language: Language code that was switched to

        Returns:
            Confirmation message in the target language
        """
        messages = {
            "en": "Switched to English",
            "lt": "Gerai, dabar kalbÄ—siu lietuviÅ¡kai."  # From v58
        }

        return messages.get(language, f"Switched to {language}")

    def is_language_command(self, text: str) -> bool:
        """
        Check if text is a language switch command.

        Args:
            text: Text to check

        Returns:
            bool: True if text is a language switch command
        """
        text_lower = text.lower().strip()

        for keywords in self.LANGUAGE_KEYWORDS.values():
            for keyword in keywords:
                if keyword in text_lower:
                    return True

        return False


class MockLanguageManager:
    """
    Mock language manager for testing.

    Simulates language detection and switching.
    """

    def __init__(self, supported_languages: Optional[Set[str]] = None):
        """Initialize mock language manager."""
        self.supported_languages = supported_languages or {"en", "lt"}
        self.current_language = "en"

    def detect_language(self, text: str) -> Optional[str]:
        """Mock language detection."""
        if "lithuanian" in text.lower() or "lietuvių" in text.lower():
            return "lt"
        elif "english" in text.lower():
            return "en"
        return None

    def check_and_switch_language(self, text: str) -> bool:
        """Mock language switching."""
        detected = self.detect_language(text)
        if detected and detected != self.current_language:
            self.current_language = detected
            return True
        return False

    def switch_language(self, language: str):
        """Mock switch language."""
        if language in self.supported_languages:
            self.current_language = language

    def get_current_language(self) -> str:
        """Get current language."""
        return self.current_language

    def get_language_name(self, lang_code: Optional[str] = None) -> str:
        """Get language name."""
        lang = lang_code or self.current_language
        return {"en": "English", "lt": "Lithuanian"}.get(lang, lang)

    def get_confirmation_message(self, language: str) -> str:
        """Get confirmation message."""
        return f"Switched to {self.get_language_name(language)}"

    def is_language_command(self, text: str) -> bool:
        """Check if language command."""
        return self.detect_language(text) is not None
