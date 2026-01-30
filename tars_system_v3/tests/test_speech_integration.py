"""
Integration tests for speech subsystem.

Tests STT, TTS, and language management.
"""

import unittest
import time

from speech import MockSTT, MockTTS, MockLanguageManager


class TestSTT(unittest.TestCase):
    """Test speech-to-text functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.stt = MockSTT()

    def test_transcribe(self):
        """Test audio transcription."""
        # Transcribe mock audio
        text = self.stt.transcribe(b"fake_audio_data")

        # Should return text
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)

    def test_transcribe_wav(self):
        """Test WAV file transcription."""
        # Mock doesn't require real file
        text = self.stt.transcribe_wav("fake_path.wav")

        # Should return text
        self.assertIsInstance(text, str)

    def test_set_language(self):
        """Test language setting."""
        # Should not raise error
        self.stt.set_language("en")
        self.assertEqual(self.stt.current_language, "en")

        self.stt.set_language("lt")
        self.assertEqual(self.stt.current_language, "lt")


class TestTTS(unittest.TestCase):
    """Test text-to-speech functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.tts = MockTTS()

    def test_speak(self):
        """Test speech synthesis."""
        # Speak should complete successfully
        result = self.tts.speak("Hello, world!")

        self.assertTrue(result)

    def test_speak_async(self):
        """Test asynchronous speech."""
        # Start async speech
        result = self.tts.speak_async("Hello, world!")

        self.assertTrue(result)

        # Should be speaking
        time.sleep(0.05)
        # May or may not be speaking depending on timing

        # Wait for completion
        time.sleep(0.5)
        self.assertFalse(self.tts.is_speaking())

    def test_is_speaking(self):
        """Test speaking status check."""
        # Initially not speaking
        self.assertFalse(self.tts.is_speaking())

        # Start speaking
        self.tts.speak_async("Test message")

        # Should be speaking now (or very soon)
        time.sleep(0.05)
        # Status depends on timing

    def test_stop_speaking(self):
        """Test stopping speech."""
        # Start speaking
        self.tts.speak_async("Long message that takes time to say")

        # Stop it
        self.tts.stop_speaking()

        # Should not be speaking
        self.assertFalse(self.tts.is_speaking())

    def test_empty_text(self):
        """Test speaking empty text."""
        result = self.tts.speak("")

        # Should handle gracefully
        self.assertFalse(result)


class TestLanguageManager(unittest.TestCase):
    """Test language management functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.lang_mgr = MockLanguageManager()

    def test_detect_language(self):
        """Test language detection."""
        # English command
        lang = self.lang_mgr.detect_language("switch to english")
        self.assertEqual(lang, "en")

        # Lithuanian command
        lang = self.lang_mgr.detect_language("pakeisk į lietuvių")
        self.assertEqual(lang, "lt")

        # Unknown command
        lang = self.lang_mgr.detect_language("random text")
        self.assertIsNone(lang)

    def test_switch_language(self):
        """Test language switching."""
        # Switch to Lithuanian
        self.lang_mgr.switch_language("lt")
        self.assertEqual(self.lang_mgr.get_current_language(), "lt")

        # Switch to English
        self.lang_mgr.switch_language("en")
        self.assertEqual(self.lang_mgr.get_current_language(), "en")

    def test_check_and_switch_language(self):
        """Test automatic language switching from command."""
        # Start in English
        self.assertEqual(self.lang_mgr.get_current_language(), "en")

        # Command to switch to Lithuanian
        switched = self.lang_mgr.check_and_switch_language("switch to lithuanian")

        # Should have switched
        self.assertTrue(switched)
        self.assertEqual(self.lang_mgr.get_current_language(), "lt")

    def test_get_language_name(self):
        """Test getting language name."""
        name = self.lang_mgr.get_language_name("en")
        self.assertEqual(name, "English")

        name = self.lang_mgr.get_language_name("lt")
        self.assertEqual(name, "Lithuanian")

    def test_get_confirmation_message(self):
        """Test confirmation message generation."""
        msg = self.lang_mgr.get_confirmation_message("en")
        self.assertIsInstance(msg, str)
        self.assertGreater(len(msg), 0)

    def test_is_language_command(self):
        """Test language command detection."""
        # Language commands
        self.assertTrue(self.lang_mgr.is_language_command("switch to english"))
        self.assertTrue(self.lang_mgr.is_language_command("lietuvių kalba"))

        # Non-language commands
        self.assertFalse(self.lang_mgr.is_language_command("move forward"))


if __name__ == '__main__':
    unittest.main()
