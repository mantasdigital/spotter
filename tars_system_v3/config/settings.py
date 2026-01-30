"""
TARS System Configuration Settings.

This module centralizes all configuration constants that were previously scattered
as global variables throughout the v58 implementation.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Set


@dataclass
class PathsConfig:
    """File and directory paths for TARS system."""

    home_dir: str = field(default_factory=lambda: os.path.expanduser("~"))

    # Memory and state files
    character_file: str = field(
        default_factory=lambda: os.path.expanduser("~/.tars_character.json")
    )
    memory_file: str = field(
        default_factory=lambda: os.path.expanduser("~/tars_memory.json")
    )
    visual_memory_file: str = field(
        default_factory=lambda: os.path.expanduser("~/tars_visual_memory.json")
    )
    macro_file: str = field(
        default_factory=lambda: os.path.expanduser("~/tars_macros.json")
    )
    actions_file: str = field(
        default_factory=lambda: os.path.expanduser("~/tars_actions.json")
    )

    # Image storage
    image_dir: str = field(
        default_factory=lambda: os.path.expanduser("~/tars_images")
    )

    # Model paths
    piper_model_en: str = "/home/{user}/picar-x/picarx/models/en_US-lessac-medium.onnx"
    haar_cascade_paths: List[str] = field(default_factory=lambda: [
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
        "./haarcascade_frontalface_default.xml",
    ])


@dataclass
class LanguageConfig:
    """Language-specific settings for TARS."""

    default_language: str = "en"
    supported_languages: Set[str] = field(default_factory=lambda: {"en", "lt"})

    # English STT/TTS
    tts_model_en: str = "en_US-ryan-low"
    stt_language_en: str = "en-us"

    # Lithuanian STT/TTS
    tts_model_lt: str = "tts_models/lt/cv/vits"
    stt_language_lt: str = "lt-lt"
    use_external_lt_tts: bool = True


@dataclass
class WakeWordConfig:
    """Wake word configuration."""

    wake_words: List[str] = field(default_factory=lambda: [
        "tars", "rob", "alien master", "alien", "robert", "avatar", "bob", "gnome",
        "gideon", "robot", "robote", "robotė", "robotę"
    ])


@dataclass
class SafetyConfig:
    """Safety thresholds and limits."""

    cliff_threshold: int = 900
    too_close_distance_cm: float = 18.0
    really_close_distance_cm: float = 10.0

    # Direction servo limits
    dir_min: int = -30
    dir_max: int = 30

    # Camera servo limits
    cam_pan_min: int = -90
    cam_pan_max: int = 90
    cam_tilt_min: int = -35
    cam_tilt_max: int = 65


@dataclass
class RoamConfig:
    """Roaming behavior configuration."""

    # Inactivity and runtime limits
    inactivity_timeout_sec: float = 60.0
    max_runtime_sec: float = 600.0  # 10 minutes

    # Observation and speech intervals
    observation_interval_sec: float = 8.0
    speech_min_interval_sec: float = 6.0

    # Spatial tracking
    visit_cell_size_cm: float = 25.0
    visit_heading_bucket_deg: float = 45.0

    # Movement parameters
    forward_speed: int = 45
    backup_speed: int = 30
    turn_speed: int = 30

    # Obstacle handling
    max_forward_blocks_before_escape: int = 8
    blocked_heading_memory_sec: float = 30.0


@dataclass
class LLMConfig:
    """LLM configuration."""

    model: str = "gpt-4o"
    model_mini: str = "gpt-4o-mini"
    timeout_sec: int = 15
    max_tokens: int = 1000
    temperature: float = 0.7


@dataclass
class VisionConfig:
    """Vision and scene analysis configuration."""

    # Scene similarity thresholds
    similarity_threshold_new_scene: float = 0.7
    max_visual_memories: int = 100

    # Face detection
    face_detection_scale_factor: float = 1.1
    face_detection_min_neighbors: int = 5
    face_detection_min_size: tuple = (30, 30)

    # Stare mode
    stare_pan_gain: float = 0.15
    stare_tilt_gain: float = 0.10
    stare_max_pan_step: float = 10.0
    stare_max_tilt_step: float = 8.0


@dataclass
class CuriosityConfig:
    """Tags that indicate interesting scenes during roam."""

    curious_tags: Set[str] = field(default_factory=lambda: {
        "person", "face", "dog", "cat", "vehicle", "bicycle",
        "unusual", "bright", "colorful"
    })

    object_type_tags: Set[str] = field(default_factory=lambda: {
        "person", "car", "truck", "bicycle", "dog", "cat",
        "chair", "table", "bottle", "cup", "book", "plant"
    })

    label_patterns: List[str] = field(default_factory=lambda: [
        r"this is (\w+)",
        r"that's (\w+)",
        r"it's (\w+)",
        r"you are (\w+)",
        r"your name is (\w+)"
    ])


@dataclass
class TarsConfig:
    """
    Main TARS configuration container.

    Aggregates all sub-configurations into a single object that can be
    easily passed around and tested.
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    wake_words: WakeWordConfig = field(default_factory=WakeWordConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    roam: RoamConfig = field(default_factory=RoamConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    curiosity: CuriosityConfig = field(default_factory=CuriosityConfig)

    @classmethod
    def from_env(cls) -> 'TarsConfig':
        """
        Load configuration from environment variables.

        Environment variables can override default values:
        - TARS_LANGUAGE: default language ("en" or "lt")
        - TARS_LLM_MODEL: LLM model to use
        - TARS_IMAGE_DIR: directory for storing images
        - TARS_MEMORY_FILE: path to memory file

        Returns:
            TarsConfig: Configuration object with environment overrides applied

        Example:
            >>> os.environ['TARS_LANGUAGE'] = 'lt'
            >>> config = TarsConfig.from_env()
            >>> assert config.language.default_language == 'lt'
        """
        config = cls()

        # Language overrides
        if 'TARS_LANGUAGE' in os.environ:
            lang = os.environ['TARS_LANGUAGE']
            if lang in config.language.supported_languages:
                config.language.default_language = lang

        # LLM overrides
        if 'TARS_LLM_MODEL' in os.environ:
            config.llm.model = os.environ['TARS_LLM_MODEL']

        # Path overrides
        if 'TARS_IMAGE_DIR' in os.environ:
            config.paths.image_dir = os.environ['TARS_IMAGE_DIR']

        if 'TARS_MEMORY_FILE' in os.environ:
            config.paths.memory_file = os.environ['TARS_MEMORY_FILE']

        return config

    def ensure_directories(self):
        """
        Ensure all required directories exist.

        Creates image directory and any other required paths.
        """
        os.makedirs(self.paths.image_dir, exist_ok=True)


# Default configuration instance
default_config = TarsConfig()
