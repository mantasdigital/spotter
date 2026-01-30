#!/usr/bin/env python3
"""
TARS Main Entry Point - V2 Development.

Integrates all subsystems with real hardware to create the complete TARS robot.
"""

__version__ = "2.0.0-dev"
TARS_VERSION = "V2 Development"

import os
import sys
import time
import threading
import signal
from pathlib import Path

# Add parent directory to path for picarx and voice_active_car imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add example directory to path for voice_active_car
example_dir = str(Path(__file__).parent.parent / "example")
if example_dir not in sys.path:
    sys.path.insert(0, example_dir)

# Pre-load cascades in background (before slow hardware init)
# Import triggers background thread that loads OpenCV cascades
from behaviors.stare_behavior import CascadeLoader as _CascadePreloader
_cascade_preloader = _CascadePreloader()  # Starts loading immediately
print("[PRELOAD] Started background cascade loading")

# Hardware
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    print("Warning: Picamera2 not available")
    PICAMERA2_AVAILABLE = False

try:
    import picarx.picarx as picarx_module
    PICARX_AVAILABLE = True
except ImportError:
    print("Warning: picarx not available")
    PICARX_AVAILABLE = False

from hardware.picarx_impl import PicarXRobotCar
from hardware.camera_impl import Picamera2Camera
from hardware.mock_hardware import MockRobotCar, MockCamera

# Core systems
from core.state_manager import StateManager
from config.settings import SafetyConfig

# Motion
from motion.safety_manager import SafetyManager
from motion.action_resolver import create_preset_actions_wrapper
from motion.action_executor import ActionExecutor

# Vision
from vision import HaarFaceDetector, SceneAnalyzer, VisualMemory, GestureDetector

# Speech
from speech import VoskSTT, OpenAITTS, LanguageManager

# LLM
from llm.agents import ConversationAgent

# Behaviors
from behaviors import RoamBehavior, StareBehavior, FollowBehavior, AutoRoamWatchdog

# Interaction
from interaction import CommandProcessor
from interaction.command_handlers import (
    MotionCommandHandler, BehaviorCommandHandler, SystemCommandHandler,
    MacroCommandHandler, WebSearchCommandHandler, VisualCommandHandler
)

# Memory
from memory.character_state import CharacterState
from memory.conversation_memory import ConversationMemory
from memory.macro_store import MacroStore


class TARSRobot:
    """
    Main TARS robot controller.

    Integrates all subsystems and provides lifecycle management.
    """

    def __init__(self):
        """Initialize TARS robot."""
        print("=" * 60)
        print(f"TARS Robot System {TARS_VERSION} - Initializing...")
        print("=" * 60)

        # Initialize state
        self.state = StateManager()
        self.stop_flag = threading.Event()

        # Initialize hardware
        print("\n[1/8] Initializing hardware...")
        self._init_hardware()

        # Initialize memory systems
        print("[2/8] Initializing memory systems...")
        self._init_memory()

        # Initialize vision
        print("[3/8] Initializing vision...")
        self._init_vision()

        # Initialize speech
        print("[4/8] Initializing speech...")
        self._init_speech()

        # Initialize motion
        print("[5/8] Initializing motion...")
        self._init_motion()

        # Initialize LLM
        print("[6/8] Initializing LLM...")
        self._init_llm()

        # Initialize behaviors
        print("[7/8] Initializing behaviors...")
        self._init_behaviors()

        # Initialize command processing
        print("[8/8] Initializing command processor...")
        self._init_command_processor()

        print("\n" + "=" * 60)
        print(f"TARS Robot System {TARS_VERSION} - Ready!")
        print("=" * 60)

    def _init_hardware(self):
        """Initialize hardware components."""
        # Both PiCar-X and Camera will be initialized by VoiceActiveCar
        # to avoid GPIO pin conflicts (VoiceActiveCar creates its own Picarx instance)
        # Use mock hardware for now, will swap in real hardware from VoiceActiveCar later
        self.car = MockRobotCar()
        self.camera = MockCamera()
        self.state.hardware.car_instance = self.car
        self.state.hardware.camera_instance = self.camera
        print("   ✓ Hardware will be initialized by voice system")

    def _init_memory(self):
        """Initialize memory systems."""
        # Character state
        char_file = Path.home() / ".tars_character.json"
        self.character = CharacterState(str(char_file))

        # Conversation memory
        conv_file = Path.home() / ".tars_conversation.json"
        visual_conv_file = Path.home() / ".tars_conversation_visual.json"
        self.conversation = ConversationMemory(str(conv_file), str(visual_conv_file))

        # Macro store
        macro_file = Path.home() / ".tars_macros.json"
        self.macros = MacroStore(str(macro_file))

        # Visual memory
        visual_dir = Path.home() / ".tars_visual"
        self.visual_memory = VisualMemory(str(visual_dir))

        print("   ✓ Memory systems initialized")

    def _init_vision(self):
        """Initialize vision components."""
        # Face detector
        self.face_detector = HaarFaceDetector()

        # Scene analyzer (requires LLM - will set up in _init_llm)
        self.scene_analyzer = None

        print("   ✓ Face detector initialized")

    def _init_speech(self):
        """Initialize speech components."""
        # STT - Vosk
        # Note: The main STT is handled by VoiceActiveCar/VoiceAssistant via sunfounder_voice_assistant
        # This VoskSTT is used by TARSRobot for language management and direct WAV transcription
        try:
            self.stt = VoskSTT(
                model_path_en="/opt/vosk_models/vosk-model-small-en-us-0.15",
                model_path_lt="/opt/vosk_models/vosk-model-lt-0.1"  # Optional
            )
            print("   ✓ Vosk STT initialized")
        except Exception as e:
            print(f"   ✗ Vosk STT failed: {e}")
            print("   Using mock STT")
            from speech import MockSTT
            self.stt = MockSTT()

        # TTS - OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                self.tts = OpenAITTS(api_key=api_key, voice="alloy")
                print("   ✓ OpenAI TTS initialized")
            except Exception as e:
                print(f"   ✗ OpenAI TTS failed: {e}")
                from speech import MockTTS
                self.tts = MockTTS()
        else:
            print("   ✗ OPENAI_API_KEY not set, using mock TTS")
            from speech import MockTTS
            self.tts = MockTTS()

        # Language manager
        self.language = LanguageManager(
            stt=self.stt,
            tts=self.tts,
            supported_languages={"en", "lt"}
        )

        print("   ✓ Language manager initialized")

    def _init_motion(self):
        """Initialize motion system."""
        # Safety manager
        safety_config = SafetyConfig()
        self.safety = SafetyManager(self.car, safety_config)

        # Action resolver
        self.action_resolver = create_preset_actions_wrapper(
            robot=self.car,
            state=self.state,
            stop_flag=self.stop_flag,
            safety_manager=self.safety
        )

        # Action executor
        def action_callback(action_name: str):
            self.state.interaction.track_action(action_name)

        self.executor = ActionExecutor(
            action_resolver=self.action_resolver,
            macro_store=self.macros,
            action_callback=action_callback
        )

        # Set the executor on MacroStore so it can run macros
        self.macros.executor = self.executor.execute

        print("   ✓ Motion system initialized")

    def _init_llm(self):
        """Initialize LLM and dependent systems."""
        # LLM provider
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("   ✗ OPENAI_API_KEY not set")
            print("   ✗ LLM features will not be available")
            self.llm = None
            return

        try:
            from picarx.llm import OpenAI as LLM
            from llm.provider_wrapper import OpenAIProviderWrapper

            # Create raw LLM instance
            raw_llm = LLM(api_key, model="gpt-4o")

            # Wrap it to add ILLMProvider interface compatibility
            self.llm = OpenAIProviderWrapper(raw_llm)

            # Create scene analyzer now that we have LLM
            self.scene_analyzer = SceneAnalyzer(self.llm, model_name="gpt-4o")

            # Create conversation agent
            self.agent = ConversationAgent(
                llm_provider=self.llm,
                character_state=self.character,
                conversation_memory=self.conversation,
                visual_memory=self.visual_memory,
                executor=self.executor
            )

            print("   ✓ LLM initialized (gpt-4o)")

        except Exception as e:
            print(f"   ✗ LLM initialization failed: {e}")
            self.llm = None
            self.scene_analyzer = None
            self.agent = None

    def _init_behaviors(self):
        """Initialize autonomous behaviors."""
        # Behaviors will be initialized later when camera is available
        # (after VoiceActiveCar starts)
        self.roam = None
        self.stare = None
        self.follow = None
        self.auto_roam_watchdog = None

        print("   ✓ Autonomous behaviors will be initialized with camera")

    def _init_command_processor(self):
        """Initialize command processor."""
        self.command_processor = CommandProcessor(
            state=self.state,
            executor=self.executor
        )

        # Add command handlers in priority order
        # 1. System commands (highest priority)
        system_handler = SystemCommandHandler(
            state=self.state,
            language_manager=self.language,
            conversation_memory=self.conversation
        )
        self.command_processor.add_handler(system_handler)

        # 2. Macro commands
        macro_handler = MacroCommandHandler(
            macro_store=self.macros
        )
        self.command_processor.add_handler(macro_handler)

        # 3. Visual labeling and recall
        visual_handler = VisualCommandHandler(
            visual_memory=self.visual_memory
        )
        self.command_processor.add_handler(visual_handler)

        # 4. Web search and browse
        web_handler = WebSearchCommandHandler(
            llm_provider=self.llm,
            language=self.language.get_current_language()
        )
        self.command_processor.add_handler(web_handler)

        # 5. Motion commands
        motion_handler = MotionCommandHandler(
            executor=self.executor,
            state=self.state
        )
        self.command_processor.add_handler(motion_handler)

        # 6. Behavior commands
        behavior_handler = BehaviorCommandHandler(
            state=self.state,
            roam_behavior=self.roam,
            stare_behavior=self.stare,
            follow_behavior=self.follow,
            language_manager=self.language
        )
        self.command_processor.add_handler(behavior_handler)

        print("   ✓ Command processor initialized with 6 handlers")

    def process_voice_command(self, audio_path: str):
        """
        Process a voice command from audio file.

        Args:
            audio_path: Path to WAV audio file
        """
        # Transcribe audio
        text = self.stt.transcribe_wav(audio_path, language=self.language.get_current_language())

        if not text:
            print("[TARS] (no speech detected)")
            return

        print(f"[USER] {text}")

        # Check for language switch
        if self.language.check_and_switch_language(text):
            lang_name = self.language.get_language_name()
            msg = self.language.get_confirmation_message(self.language.get_current_language())
            print(f"[TARS] {msg}")
            self.tts.speak(msg, language=self.language.get_current_language())
            return

        # Mark that we're processing a command - roam should skip observations
        self.state.interaction.start_command_processing()

        try:
            # Process command
            result = self.command_processor.process_command(text)

            if result["success"]:
                # Speak response
                if result.get("message"):
                    print(f"[TARS] {result['message']}")
                    self.tts.speak(result["message"], language=self.language.get_current_language())
            else:
                # Command not recognized - use LLM for conversation
                if self.agent:
                    current_lang = self.language.get_current_language()
                    response = self.agent.chat(text, language=current_lang)
                    print(f"[TARS] {response}")
                    self.tts.speak(response, language=current_lang)
                else:
                    msg = "I don't understand that command"
                    print(f"[TARS] {msg}")
                    self.tts.speak(msg, language=self.language.get_current_language())
        finally:
            # Clear command processing flag
            self.state.interaction.end_command_processing()

    def run_voice_loop(self):
        """
        Run main voice interaction loop.

        Uses VoiceAssistant's run() method for the main loop,
        with TARS integrated via TARSVoiceActiveCar.
        """
        print("\n[TARS] Starting voice interaction system...")
        print("[TARS] Press Ctrl+C to exit\n")

        try:
            # Import TARS voice car
            from tars_voice_car import TARSVoiceActiveCar

            # Check LLM availability
            if not self.llm:
                print("[TARS] WARNING: No LLM available, creating fallback")
                from picarx.llm import OpenAI as LLM
                from llm.provider_wrapper import OpenAIProviderWrapper
                api_key = os.getenv("OPENAI_API_KEY", "dummy")
                raw_llm = LLM(api_key, model="gpt-4o")
                self.llm = OpenAIProviderWrapper(raw_llm)

            # Define wake words
            wake_words = [
                "tars", "bobby", "rob", "robert",
                "alien master", "avatar", "gnome"
            ]

            print("[TARS] Initializing voice system (this may take a moment)...")
            print("       - Setting up hardware (PiCar-X, Picamera2)")
            print("       - Loading STT/TTS models")
            print("       - Configuring wake word detection")

            # Create TARS voice car with wake words passed to parent
            # Parent needs wake words for STT detection to work
            vac = TARSVoiceActiveCar(
                tars_robot=self,
                llm=self.llm,
                name="TARS",
                too_close=10,
                with_image=True,
                keyboard_enable=True,
                wake_enable=True,
                wake_word=wake_words,  # Pass to parent for STT wake detection
                answer_on_wake="",  # Empty to reduce delay
                welcome="TARS Reconnaissance Unit online. Wake words: " + ", ".join(wake_words),
                tars_wake_words=wake_words,  # Also store in TARS for inline detection

                instructions="""You are TARS, a sarcastic reconnaissance robot.

Your personality:
- Humor setting at 75%
- Sarcastic and witty but helpful
- Military-style responses
- Self-aware that you're a robot

You can:
- Move (forward, backward, turn, spin, dance, wiggle)
- Look around (pan/tilt camera)
- Detect faces and track people
- Roam autonomously
- Analyze scenes with your camera

Keep responses concise and witty."""
            )

            self.state.hardware.vac_instance = vac

            # Run the VoiceAssistant main loop
            # TARSVoiceActiveCar overrides think() to use TARS command processing
            print("[TARS] Voice loop started. Listening for wake word...")
            vac.run()

        except KeyboardInterrupt:
            print("\n[TARS] Interrupt received")
        except Exception as e:
            print(f"\n[TARS] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()

    def _init_behaviors_with_camera(self):
        """Initialize behaviors now that camera is available."""
        # Create speak callback for roam observations and gesture detection
        def speak_callback(text: str):
            """Speak using TTS."""
            try:
                current_lang = self.language.get_current_language()
                self.tts.speak(text, language=current_lang)
            except Exception as e:
                print(f"[TTS] Error: {e}")

        # Create gesture detector for stop gesture (hand-in-face)
        gesture_detector = GestureDetector(speak_callback=speak_callback)

        # Language getter for roam behavior
        def language_getter():
            return self.language.get_current_language()

        # Roaming behavior with v58-style obstacle avoidance and visual memory
        self.roam = RoamBehavior(
            car=self.car,
            camera=self.camera,
            state=self.state,
            action_executor=self.executor,
            scene_analyzer=self.scene_analyzer,
            visual_memory=self.visual_memory,  # Enable memory saving during roam
            observation_interval_sec=6.0,  # Same as v58
            speak_callback=speak_callback,
            gesture_detector=gesture_detector,
            language_getter=language_getter
        )

        # Stare behavior
        self.stare = StareBehavior(
            car=self.car,
            camera=self.camera,
            face_detector=self.face_detector,
            state=self.state,
            gesture_detector=gesture_detector
        )

        # Follow behavior
        self.follow = FollowBehavior(
            car=self.car,
            camera=self.camera,
            face_detector=self.face_detector,
            state=self.state,
            gesture_detector=gesture_detector
        )

        # Update command processor with behaviors
        behavior_handler = BehaviorCommandHandler(
            state=self.state,
            roam_behavior=self.roam,
            stare_behavior=self.stare,
            follow_behavior=self.follow,
            language_manager=self.language
        )
        # Replace the old handler
        self.command_processor.handlers = [
            h for h in self.command_processor.handlers
            if not isinstance(h, BehaviorCommandHandler)
        ]
        self.command_processor.add_handler(behavior_handler)

        # Initialize auto-roam watchdog (v58-style)
        # Starts roaming automatically after 60 seconds of inactivity
        self.auto_roam_watchdog = AutoRoamWatchdog(
            state=self.state,
            start_roam_callback=self.roam.start,
            inactivity_threshold_sec=60.0,  # 1 minute like v58
            speak_callback=speak_callback
        )
        # Start the watchdog (it will auto-trigger roam after inactivity)
        self.auto_roam_watchdog.start()

        print("[TARS] Behaviors initialized with camera")
        print("[TARS] Auto-roam watchdog started (60s inactivity threshold)")

    def shutdown(self):
        """Shutdown TARS robot cleanly."""
        print("\n[TARS] Shutting down systems...")

        # Set stop flag
        self.stop_flag.set()

        # Stop auto-roam watchdog first
        if hasattr(self, 'auto_roam_watchdog') and self.auto_roam_watchdog is not None:
            self.auto_roam_watchdog.stop()

        # Stop behaviors
        if hasattr(self, 'roam') and self.roam is not None:
            self.roam.stop()
        if hasattr(self, 'stare') and self.stare is not None:
            self.stare.stop()
        if hasattr(self, 'follow') and self.follow is not None:
            self.follow.stop()

        # Stop hardware - but only if we're using mock hardware
        # Real hardware is owned by VoiceActiveCar and will be cleaned up by it
        if hasattr(self, 'car') and isinstance(self.car, MockRobotCar):
            self.car.stop()
            self.car.reset()
        elif hasattr(self, 'car'):
            # Real car - just stop it, don't reset (VoiceActiveCar manages it)
            self.car.stop()

        # Camera is managed by VoiceActiveCar, don't stop it here
        # (VoiceActiveCar will clean it up)

        # Save memory
        if hasattr(self, 'character'):
            self.character.save()
        if hasattr(self, 'conversation'):
            self.conversation.save()
        if hasattr(self, 'macros'):
            self.macros.save_macros()  # MacroStore uses save_macros()
        if hasattr(self, 'visual_memory'):
            self.visual_memory.save()

        print("[TARS] Shutdown complete")


def main():
    """Main entry point."""
    # Initialize robot
    tars = TARSRobot()

    # Set up signal handler for clean shutdown
    def signal_handler(sig, frame):
        print("\n[TARS] Interrupt received")
        tars.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run voice loop
    tars.run_voice_loop()


if __name__ == "__main__":
    main()
