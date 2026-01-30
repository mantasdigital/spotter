"""
Music Player for Dance Routines.

Plays random background music during dance sessions.
Uses locally generated audio patterns for reliability (no external dependencies).
"""

import subprocess
import threading
import random
import logging
import os
import signal
import wave
import struct
import math
import tempfile
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


# Dance beat patterns - frequency sequences for different "songs"
# Each pattern: (name, bpm, [(freq_hz, duration_beats), ...])
DANCE_PATTERNS = [
    {
        "name": "Disco Fever",
        "bpm": 120,
        "notes": [
            (440, 0.5), (0, 0.5), (440, 0.5), (0, 0.5),  # A4 beat
            (523, 0.5), (0, 0.5), (523, 0.5), (0, 0.5),  # C5 beat
            (659, 0.5), (0, 0.5), (659, 0.5), (0, 0.5),  # E5 beat
            (523, 0.5), (0, 0.5), (440, 0.5), (0, 0.5),  # back down
        ] * 4  # Repeat 4 times
    },
    {
        "name": "Robot Groove",
        "bpm": 100,
        "notes": [
            (330, 0.25), (0, 0.25), (330, 0.25), (0, 0.25),  # E4 staccato
            (392, 0.5), (0, 0.5),  # G4
            (330, 0.25), (0, 0.25), (330, 0.25), (0, 0.25),
            (494, 0.5), (0, 0.5),  # B4
            (440, 1.0), (0, 0.5),  # A4 hold
            (392, 0.5), (330, 0.5), (0, 0.5),  # G4 E4
        ] * 3
    },
    {
        "name": "Happy Dance",
        "bpm": 140,
        "notes": [
            (523, 0.25), (587, 0.25), (659, 0.25), (698, 0.25),  # C D E F run up
            (784, 0.5), (0, 0.25), (784, 0.25),  # G G
            (698, 0.25), (659, 0.25), (587, 0.25), (523, 0.25),  # F E D C run down
            (440, 0.5), (0, 0.5),  # A
            (523, 0.5), (659, 0.5), (784, 0.5), (0, 0.5),  # C E G chord arpegg
        ] * 3
    },
    {
        "name": "Funky Beat",
        "bpm": 110,
        "notes": [
            (147, 0.5), (0, 0.25), (147, 0.25),  # D3 bass
            (175, 0.5), (0, 0.25), (196, 0.25),  # F3 G3
            (220, 0.25), (0, 0.25), (220, 0.25), (0, 0.25),  # A3 staccato
            (175, 0.5), (147, 0.5),  # F3 D3
            (0, 0.5),  # rest
            (294, 0.25), (330, 0.25), (349, 0.25), (330, 0.25),  # D4 E4 F4 E4
        ] * 4
    },
    {
        "name": "Party Time",
        "bpm": 128,
        "notes": [
            (262, 0.5), (0, 0.5), (330, 0.5), (0, 0.5),  # C4 E4
            (392, 0.5), (0, 0.5), (330, 0.5), (0, 0.5),  # G4 E4
            (349, 0.5), (0, 0.5), (294, 0.5), (0, 0.5),  # F4 D4
            (262, 1.0), (0, 0.5),  # C4 hold
            (392, 0.25), (440, 0.25), (494, 0.25), (523, 0.25),  # run up
            (494, 0.5), (440, 0.5), (392, 0.5), (0, 0.5),  # back down
        ] * 3
    },
    {
        "name": "Electronic Pulse",
        "bpm": 135,
        "notes": [
            (220, 0.125), (0, 0.125), (220, 0.125), (0, 0.125),  # A3 fast pulse
            (220, 0.125), (0, 0.125), (220, 0.125), (0, 0.125),
            (330, 0.25), (0, 0.25), (440, 0.25), (0, 0.25),  # E4 A4
            (220, 0.125), (0, 0.125), (220, 0.125), (0, 0.125),
            (220, 0.125), (0, 0.125), (220, 0.125), (0, 0.125),
            (494, 0.25), (0, 0.25), (392, 0.25), (0, 0.25),  # B4 G4
        ] * 4
    },
    {
        "name": "Swing Dance",
        "bpm": 150,
        "notes": [
            (349, 0.75), (392, 0.25),  # F4 swing to G4
            (440, 0.5), (0, 0.5),  # A4
            (392, 0.75), (349, 0.25),  # G4 swing to F4
            (330, 0.5), (0, 0.5),  # E4
            (294, 0.75), (330, 0.25),  # D4 swing to E4
            (349, 0.5), (392, 0.5),  # F4 G4
            (440, 1.0), (0, 0.5),  # A4 hold
        ] * 3
    },
    {
        "name": "Techno Beat",
        "bpm": 138,
        "notes": [
            (110, 0.25), (0, 0.25), (110, 0.25), (0, 0.25),  # A2 bass kick
            (110, 0.25), (0, 0.25), (165, 0.25), (0, 0.25),  # A2 E3
            (220, 0.25), (0, 0.25), (220, 0.25), (0, 0.25),  # A3
            (330, 0.5), (0, 0.5),  # E4 accent
            (110, 0.25), (0, 0.25), (110, 0.25), (0, 0.25),
            (440, 0.25), (494, 0.25), (440, 0.25), (0, 0.25),  # high notes
        ] * 4
    },
]


def generate_tone(frequency: float, duration: float, sample_rate: int = 22050,
                  volume: float = 0.5) -> bytes:
    """
    Generate a sine wave tone.

    Args:
        frequency: Frequency in Hz (0 for silence)
        duration: Duration in seconds
        sample_rate: Audio sample rate
        volume: Volume 0.0 to 1.0

    Returns:
        Raw audio bytes (16-bit signed)
    """
    num_samples = int(duration * sample_rate)

    if frequency <= 0:
        # Silence
        return b'\x00\x00' * num_samples

    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        # Sine wave with slight attack/decay envelope for smoother sound
        envelope = 1.0
        attack_samples = int(0.01 * sample_rate)
        decay_samples = int(0.01 * sample_rate)

        if i < attack_samples:
            envelope = i / attack_samples
        elif i > num_samples - decay_samples:
            envelope = (num_samples - i) / decay_samples

        value = math.sin(2 * math.pi * frequency * t) * volume * envelope
        # Convert to 16-bit signed integer
        sample = int(value * 32767)
        sample = max(-32768, min(32767, sample))
        samples.append(struct.pack('<h', sample))

    return b''.join(samples)


def generate_dance_audio(pattern: dict, sample_rate: int = 22050) -> bytes:
    """
    Generate audio data for a dance pattern.

    Args:
        pattern: Dance pattern dictionary with name, bpm, notes
        sample_rate: Audio sample rate

    Returns:
        Raw audio bytes
    """
    bpm = pattern["bpm"]
    beat_duration = 60.0 / bpm  # Duration of one beat in seconds

    audio_data = []
    for freq, beats in pattern["notes"]:
        duration = beats * beat_duration
        tone = generate_tone(freq, duration, sample_rate, volume=0.6)
        audio_data.append(tone)

    return b''.join(audio_data)


def create_wav_file(audio_data: bytes, filepath: str, sample_rate: int = 22050):
    """
    Write audio data to a WAV file.

    Args:
        audio_data: Raw 16-bit signed audio bytes
        filepath: Output file path
        sample_rate: Audio sample rate
    """
    with wave.open(filepath, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)


class DanceMusicPlayer:
    """
    Manages background music playback for dance routines.

    Generates and plays procedural dance music using system audio tools.
    Thread-safe for use during robot dance execution.
    """

    def __init__(self):
        """Initialize music player."""
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._current_track: Optional[str] = None
        self._temp_file: Optional[str] = None
        self._cache_dir = "/tmp/tars_dance_music"

        # Ensure cache directory exists
        os.makedirs(self._cache_dir, exist_ok=True)

    def play_random_track(self) -> Optional[str]:
        """
        Start playing a random dance track.

        Returns:
            Track name if started successfully, None on failure
        """
        pattern = random.choice(DANCE_PATTERNS)
        return self._play_pattern(pattern)

    def _play_pattern(self, pattern: dict) -> Optional[str]:
        """
        Generate and play a dance pattern.

        Args:
            pattern: Dance pattern dictionary

        Returns:
            Track name if started successfully, None on failure
        """
        name = pattern["name"]

        with self._lock:
            # Stop any currently playing track
            self._stop_internal()

            try:
                # Generate audio file (cache it for reuse)
                cache_file = os.path.join(self._cache_dir, f"{name.replace(' ', '_')}.wav")

                if not os.path.exists(cache_file):
                    logger.info(f"[MUSIC] Generating: {name}")
                    audio_data = generate_dance_audio(pattern)
                    create_wav_file(audio_data, cache_file)

                # Play using aplay (most reliable on Pi)
                self._process = subprocess.Popen(
                    ["aplay", "-q", cache_file],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    preexec_fn=os.setsid
                )
                self._current_track = name
                self._temp_file = cache_file
                logger.info(f"[MUSIC] Now playing: {name}")
                print(f"[MUSIC] Now playing: {name}")
                return name

            except FileNotFoundError:
                logger.warning("[MUSIC] aplay not found, trying paplay")
                return self._play_with_paplay(pattern)
            except Exception as e:
                logger.error(f"[MUSIC] Failed to play: {e}")
                print(f"[MUSIC] Error: {e}")
                return None

    def _play_with_paplay(self, pattern: dict) -> Optional[str]:
        """Fallback to paplay (PulseAudio)."""
        name = pattern["name"]

        try:
            cache_file = os.path.join(self._cache_dir, f"{name.replace(' ', '_')}.wav")

            if not os.path.exists(cache_file):
                audio_data = generate_dance_audio(pattern)
                create_wav_file(audio_data, cache_file)

            self._process = subprocess.Popen(
                ["paplay", cache_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
            self._current_track = name
            self._temp_file = cache_file
            logger.info(f"[MUSIC] Playing with paplay: {name}")
            return name

        except Exception as e:
            logger.error(f"[MUSIC] paplay also failed: {e}")
            return None

    def play_track(self, url: str, name: str = "Unknown") -> Optional[str]:
        """
        Play a specific track (for backwards compatibility).

        Now ignores URL and plays a random generated pattern instead.
        """
        return self.play_random_track()

    def stop(self):
        """Stop currently playing music."""
        with self._lock:
            self._stop_internal()

    def _stop_internal(self):
        """Internal stop without lock."""
        if self._process is not None:
            try:
                # Kill the entire process group
                os.killpg(os.getpgid(self._process.pid), signal.SIGTERM)
                self._process.wait(timeout=1.0)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
            except Exception as e:
                logger.debug(f"[MUSIC] Stop cleanup: {e}")
            finally:
                self._process = None
                if self._current_track:
                    logger.info(f"[MUSIC] Stopped: {self._current_track}")
                    print(f"[MUSIC] Stopped: {self._current_track}")
                self._current_track = None

    def is_playing(self) -> bool:
        """Check if music is currently playing."""
        with self._lock:
            if self._process is None:
                return False
            return self._process.poll() is None

    def get_current_track(self) -> Optional[str]:
        """Get name of currently playing track."""
        with self._lock:
            return self._current_track if self.is_playing() else None

    def __del__(self):
        """Cleanup on destruction."""
        self.stop()


# Global singleton instance
_player_instance: Optional[DanceMusicPlayer] = None


def get_music_player() -> DanceMusicPlayer:
    """Get the global music player instance."""
    global _player_instance
    if _player_instance is None:
        _player_instance = DanceMusicPlayer()
    return _player_instance


def play_dance_music() -> Optional[str]:
    """Start playing random dance music."""
    return get_music_player().play_random_track()


def stop_dance_music():
    """Stop dance music playback."""
    get_music_player().stop()


def is_music_playing() -> bool:
    """Check if dance music is playing."""
    return get_music_player().is_playing()
