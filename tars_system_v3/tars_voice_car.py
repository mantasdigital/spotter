#!/usr/bin/env python3
"""
TARS Voice Car Integration.

Extends VoiceActiveCar to integrate TARS subsystems while using
VoiceAssistant's main loop for voice detection and interaction.
"""

import os
import sys
import wave
import threading
import signal
import queue
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add example directory for voice_active_car
example_dir = str(Path(__file__).parent.parent / "example")
if example_dir not in sys.path:
    sys.path.insert(0, example_dir)

from voice_active_car import VoiceActiveCar
from hardware.picarx_impl import PicarXRobotCar
from hardware.camera_impl import Picamera2Camera
import time

# Image capture cache for faster repeated queries
_IMAGE_CACHE = {
    "data": None,
    "timestamp": 0,
    "ttl": 0.5  # Cache valid for 500ms
}


def _coordinated_speak(tars, text: str, language: str = "en", source: str = "command") -> bool:
    """
    Speak text with global TTS coordination to prevent voice overlap.

    Waits for any current speech (e.g., roam observation) to finish,
    then acquires TTS lock before speaking.

    CRITICAL: Command speech has priority. We wait for roam to finish
    but roam should detect command_processing flag and abort.

    Args:
        tars: TARSRobot instance (has tts and state)
        text: Text to speak
        language: Language code
        source: Who is speaking ("command", "response", etc.)

    Returns:
        bool: True if speech was successful
    """
    if not text or not text.strip():
        return False

    # Mark command processing FIRST so roam knows to stop/skip speaking
    # This ensures roam's _safe_speak will see the flag and abort
    already_processing = tars.state.interaction.is_command_processing()
    if not already_processing:
        tars.state.interaction.start_command_processing()

    try:
        # Wait for any current roam speech to finish (max 10 seconds)
        # Roam should detect command_processing and abort quickly
        if tars.state.tts.is_currently_speaking():
            current_source = tars.state.tts.get_speaking_source()
            print(f"[TTS-COORD] Waiting for {current_source} to finish...")
            # Wait with longer timeout - roam should detect command flag and abort
            if not tars.state.tts.wait_for_speech_end(timeout=10.0):
                print(f"[TTS-COORD] Timeout waiting for {current_source}")
                # Force clear ONLY if roam is the source (command has priority)
                if current_source == "roam":
                    print(f"[TTS-COORD] Force clearing roam lock for command priority")
                    tars.state.tts.stop_speaking()
                else:
                    # Another command is speaking - wait more or skip
                    print(f"[TTS-COORD] Another command speaking, waiting more...")
                    tars.state.tts.wait_for_speech_end(timeout=5.0)

        # Acquire global TTS lock
        if not tars.state.tts.start_speaking(source):
            # If we couldn't acquire, another speech started - wait briefly
            time.sleep(0.5)
            if not tars.state.tts.start_speaking(source):
                print(f"[TTS-COORD] Couldn't acquire TTS lock after retry")
                # Force acquire for command priority
                tars.state.tts.stop_speaking()
                tars.state.tts.start_speaking(source)

        tars.tts.speak(text, language=language)
        return True
    except Exception as e:
        print(f"[TTS-COORD] TTS error: {e}")
        return False
    finally:
        tars.state.tts.stop_speaking()
        # Only clear command_processing if we set it
        if not already_processing:
            tars.state.interaction.end_command_processing()


def _capture_image_cached(picam2) -> str:
    """
    Capture image with caching to avoid redundant processing.

    Returns cached image if captured within TTL, otherwise captures fresh.

    Args:
        picam2: Picamera2 instance

    Returns:
        str: Base64-encoded JPEG image, or None if capture fails
    """
    import base64
    import io
    from PIL import Image

    now = time.time()

    # Return cached if still valid
    if _IMAGE_CACHE["data"] and (now - _IMAGE_CACHE["timestamp"]) < _IMAGE_CACHE["ttl"]:
        return _IMAGE_CACHE["data"]

    try:
        frame = picam2.capture_array()
        if len(frame.shape) == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        img = Image.fromarray(frame)

        # Use lower quality for faster encoding (70 instead of 85)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=70)
        image_data = base64.b64encode(buffered.getvalue()).decode()

        # Update cache
        _IMAGE_CACHE["data"] = image_data
        _IMAGE_CACHE["timestamp"] = now

        return image_data

    except Exception as e:
        print(f"[VISION] Capture failed: {e}")
        return None

# Global Lithuanian ASR instance (loaded lazily in background)
LT_ASR = None
LT_ASR_LOADING = False

# Audio recording state for Lithuanian ASR
_AUDIO_RECORDING = False
_AUDIO_CHUNKS = []
_AUDIO_SAMPLE_RATE = 44100  # Default, will be updated from STT


def _setup_signal_handlers():
    """Set up signal handlers for graceful crash handling."""
    def handle_sigabrt(signum, frame):
        print("\n[TARS] SIGABRT caught - system crash prevented")
        print("[TARS] This may indicate a native library issue (libcamera, OpenCV, etc.)")
        # Don't exit - try to continue
        pass

    def handle_sigsegv(signum, frame):
        print("\n[TARS] SIGSEGV caught - attempting graceful shutdown")
        sys.exit(1)

    # Only register handlers if not already set
    try:
        signal.signal(signal.SIGABRT, handle_sigabrt)
    except Exception:
        pass


# Set up signal handlers early
_setup_signal_handlers()

# Path for saving audio for Lithuanian transcription
LT_COMMAND_WAV_PATH = "/tmp/vac_last_command.wav"


def _init_lt_asr_async():
    """
    Start loading the Lithuanian ASR model in a background thread.
    Model is heavy so we load it asynchronously to not block startup.
    """
    global LT_ASR, LT_ASR_LOADING

    if LT_ASR is not None or LT_ASR_LOADING:
        return

    LT_ASR_LOADING = True

    def _loader():
        global LT_ASR, LT_ASR_LOADING
        try:
            from lt_asr import LithuanianASR
            LT_ASR = LithuanianASR(device="cpu")
            print("[LT-ASR] Lithuanian model loaded.")
        except Exception as e:
            print(f"[LT-ASR] Failed to load Lithuanian ASR: {e}")
        finally:
            LT_ASR_LOADING = False

    threading.Thread(target=_loader, daemon=True).start()
    print("[LT-ASR] Loading Lithuanian ASR model in background...")


def _wrap_vosk_stt_to_save_command_wav(vac):
    """
    Wrap vac.stt.listen(stream=True) so that for each final user utterance,
    all PCM chunks from that utterance are saved to /tmp/vac_last_command.wav.

    This version patches the STT's _listen_streaming method to capture ALL audio
    chunks, not just those that result in yields (fixes missing silent chunks).
    """
    global _AUDIO_SAMPLE_RATE

    try:
        stt = getattr(vac, "stt", None)
        if stt is None:
            print("[LT-ASR] No stt attribute on VoiceActiveCar; cannot hook audio.")
            return

        # Get sample rate from STT
        _AUDIO_SAMPLE_RATE = getattr(stt, "_samplerate", 44100)

        original_listen_streaming = getattr(stt, "_listen_streaming", None)
        if not callable(original_listen_streaming):
            print("[LT-ASR] stt._listen_streaming not callable; cannot hook audio.")
            return

        def patched_listen_streaming(q, device=None, samplerate=None, callback=None):
            """
            Patched version that accumulates ALL audio chunks, including those
            that don't produce partial results (fixes the missing chunk bug).
            """
            global _AUDIO_RECORDING, _AUDIO_CHUNKS, _AUDIO_SAMPLE_RATE

            # Start recording
            _AUDIO_RECORDING = True
            _AUDIO_CHUNKS = []
            sr = samplerate or _AUDIO_SAMPLE_RATE

            try:
                import sounddevice as sd

                with sd.RawInputStream(
                    samplerate=samplerate,
                    blocksize=1024,
                    device=device,
                    dtype="int16",
                    channels=1,
                    callback=callback):

                    while True:
                        if stt.stop_listening_event.is_set():
                            return None

                        try:
                            data = q.get(timeout=0.5)
                        except queue.Empty:
                            continue

                        # CAPTURE ALL AUDIO for Lithuanian ASR
                        if isinstance(data, (bytes, bytearray)):
                            _AUDIO_CHUNKS.append(bytes(data))

                        # Set last_audio for compatibility
                        stt.last_audio = (data, stt._samplerate)
                        stt.last_sample_rate = stt._samplerate

                        result = {
                            "done": False,
                            "partial": "",
                            "final": ""
                        }

                        import json
                        if stt.recognizer.AcceptWaveform(data):
                            text = stt.recognizer.Result()
                            text = json.loads(text)["text"]
                            if text == "":
                                continue
                            result["done"] = True
                            result["final"] = text.strip()

                            # Save accumulated audio to WAV when utterance ends
                            if _AUDIO_CHUNKS:
                                try:
                                    with wave.open(LT_COMMAND_WAV_PATH, "wb") as wf:
                                        wf.setnchannels(1)
                                        wf.setsampwidth(2)  # 16-bit PCM
                                        wf.setframerate(sr)
                                        wf.writeframes(b"".join(_AUDIO_CHUNKS))
                                    total_bytes = sum(len(c) for c in _AUDIO_CHUNKS)
                                    print(f"[LT-ASR] Saved {total_bytes} bytes audio (sr={sr})")
                                except Exception as e:
                                    print(f"[LT-ASR] Failed to save WAV: {e}")

                            yield result
                            break
                        else:
                            partial = stt.recognizer.PartialResult()
                            partial = json.loads(partial)["partial"]
                            if partial == "" or partial.isspace():
                                continue
                            result["partial"] = partial.strip()
                            yield result
            finally:
                _AUDIO_RECORDING = False

        # Replace the internal streaming method
        stt._listen_streaming = patched_listen_streaming
        print("[LT-ASR] Patched STT to capture all audio for Lithuanian transcription.")

    except Exception as e:
        print(f"[LT-ASR] Could not patch Vosk STT: {e}")
        import traceback
        traceback.print_exc()


class LogInterceptor:
    """
    Intercepts stdout to detect inline wake + command patterns.

    Example: "tars move forward" should be processed immediately
    without waiting for two separate inputs.
    """

    def __init__(self, original_stdout, tars_vac):
        """
        Initialize interceptor.

        Args:
            original_stdout: Original stdout stream
            tars_vac: TARSVoiceActiveCar instance to call back
        """
        self.original_stdout = original_stdout
        self.tars_vac = tars_vac
        self.is_waiting_for_command = False
        self.buffer = ""  # Buffer for incomplete lines
        self.processing = False  # Prevent re-entrant processing

    def write(self, message):
        """Write to original stdout and check for wake patterns."""
        # Always write to original stdout first
        self.original_stdout.write(message)

        # CRITICAL: Prevent infinite loop - don't process our own debug output
        if self.processing:
            return

        # Buffer text until we get a complete line
        self.buffer += message

        # CRITICAL FIX: STT "heard:" output doesn't always end with \n
        # Process "heard:" lines immediately when we see them (like v58)
        # Ignore partial results that end with \r (carriage return)
        if "heard:" in self.buffer.lower():
            # Check if this is a final result (no \r at the end, or followed by other content)
            # Partial results look like: "heard: partial text\r"
            # Final results look like: ">>> heard: final text" (no \r, may or may not have \n)

            # If we have a \r at the very end, it's a partial result - wait for more
            if self.buffer.endswith('\r'):
                return

            # Set processing flag to prevent recursive calls
            self.processing = True

            try:
                # Clean and process the buffer
                clean_line = self.buffer.replace("\x1b[K", "").replace("\r", "")
                # Remove >>> prompt if present
                clean_line = clean_line.replace(">>>", "").strip()
                lower = clean_line.lower()

                if "heard:" in lower and clean_line.strip():
                    # Process the command
                    self._process_heard_text(clean_line)
            finally:
                # Always clear processing flag
                self.processing = False

            # Clear buffer after processing
            self.buffer = ""
            return

        # Also process complete lines ending with newline (for other output)
        if '\n' in self.buffer:
            # Process all complete lines in buffer
            lines = self.buffer.split('\n')
            # Keep the last incomplete part in buffer
            self.buffer = lines[-1]

            # Set processing flag
            self.processing = True

            try:
                # Process complete lines
                for line in lines[:-1]:
                    clean_line = line.replace("\x1b[K", "").replace("\r", "")
                    # Remove >>> prompt if present
                    clean_line = clean_line.replace(">>>", "").strip()
                    lower = clean_line.lower()

                    # Look for "heard:" output from STT (shouldn't hit this now, but keep as fallback)
                    if "heard:" in lower and clean_line.strip():
                        self._process_heard_text(clean_line)
            finally:
                # Always clear processing flag
                self.processing = False

    def flush(self):
        """Flush original stdout."""
        self.original_stdout.flush()

    def _process_heard_text(self, text):
        """
        Process text that was heard by STT.

        Handles inline wake + command patterns like "tars move forward".
        Processes immediately like v58 instead of waiting for parent's flow.

        Args:
            text: Text containing "heard:" output
        """
        # Extract what was heard
        raw_heard = text.lower().split("heard:")[-1].strip()

        if not raw_heard:
            return

        # FILTER: Skip internal sensor messages (ultrasonic, etc.)
        if raw_heard.startswith('<<<') or '<<<' in raw_heard or 'ultrasonic' in raw_heard:
            return

        # Check if any wake word is present
        has_wake = any(w in raw_heard for w in self.tars_vac.wake_words)

        # CRITICAL: If no wake word, silently ignore (don't waste memory logging)
        # This is how v58 worked - only log/process actual commands
        if not has_wake and not self.is_waiting_for_command:
            return

        # CRITICAL: If waiting for command (two-part wake), DON'T process here
        # Let the parent's normal flow handle it to get the full command
        # Only process INLINE wake+command (both in one phrase)
        if self.is_waiting_for_command:
            # Reset flag so parent can handle it
            self.is_waiting_for_command = False
            return

        # Extract command by removing wake word
        cmd_only = raw_heard
        if has_wake:
            for w in self.tars_vac.wake_words:
                if w in cmd_only:
                    cmd_only = cmd_only.replace(w, "", 1).strip()
                    break

        # Wake word only (no command after it) - let parent handle the command
        if has_wake and not cmd_only:
            self.is_waiting_for_command = True
            print(f"[DEBUG] Wake word only, waiting for command")
            return

        # Have INLINE command (wake+command in one phrase) - PROCESS IT IMMEDIATELY (like v58)
        if cmd_only:
            print(f"[INLINE] Detected command: {cmd_only}")

            # Process directly using TARS command processor
            self._execute_command(cmd_only)

    def _execute_command(self, command_text):
        """
        Execute command immediately (like v58 does in interceptor).

        Supports multi-command sequences separated by "then", "than", "and then".
        Example: "go forward 20 then go back 20" executes both commands.

        Args:
            command_text: Command to execute
        """
        print(f"[DEBUG] Executing command: {repr(command_text)}")

        # FILTER: Skip internal sensor messages (ultrasonic, etc.)
        if command_text.startswith('<<<') or '<<<' in command_text or 'Ultrasonic' in command_text or 'ultrasonic' in command_text.lower():
            print(f"[DEBUG] Skipping sensor message in _execute_command")
            return

        # Clear accumulated actions ONCE at start of new command for macro recording
        # BUT skip clearing for save/run macro commands (they need the previous actions)
        import re
        cmd_lower = command_text.lower()
        is_macro_command = any(kw in cmd_lower for kw in [
            'save macro', 'same macro', 'run macro', 'one macro', 'won macro',
            'macro ', 'repeat macro', 'stop macro', 'delete macro', 'forget macro'
        ])
        if not is_macro_command:
            if hasattr(self.tars_vac, 'tars') and hasattr(self.tars_vac.tars, 'executor'):
                if hasattr(self.tars_vac.tars.executor, 'clear_accumulated_actions'):
                    self.tars_vac.tars.executor.clear_accumulated_actions()

        # MULTI-COMMAND SUPPORT: Split on "then", "than", "them" (STT misrecognitions), "and then"
        # Split on common delimiters while preserving "and" in phrases like "back and forth"
        # "them" is a common STT misrecognition of "then"
        sub_commands = re.split(r'\b(?:and\s+then|then|than|them)\b', command_text, flags=re.IGNORECASE)
        sub_commands = [cmd.strip() for cmd in sub_commands if cmd.strip()]

        if len(sub_commands) > 1:
            print(f"[MULTI-CMD] Split into {len(sub_commands)} commands: {sub_commands}")
            for i, sub_cmd in enumerate(sub_commands):
                print(f"[MULTI-CMD] Executing ({i+1}/{len(sub_commands)}): {sub_cmd}")
                self._execute_single_command(sub_cmd, clear_actions=False)
                # Small delay between commands
                import time
                time.sleep(0.3)
            return

        # Single command - execute directly (already cleared above)
        self._execute_single_command(command_text, clear_actions=False)

    def _execute_single_command(self, command_text, clear_actions=True):
        """
        Execute a single command (no multi-command splitting).

        Args:
            command_text: Single command to execute
            clear_actions: If True, clear accumulated actions before processing.
                          Set to False when processing sub-commands in a multi-command sequence.
        """
        # LITHUANIAN ASR OVERRIDE (from v58)
        # If in Lithuanian mode and LT_ASR is available, use it to transcribe the saved audio
        current_lang = self.tars_vac.tars.language.get_current_language()
        if current_lang == "lt" and LT_ASR is not None:
            try:
                if os.path.exists(LT_COMMAND_WAV_PATH) and os.path.getsize(LT_COMMAND_WAV_PATH) > 0:
                    lt_text = LT_ASR.transcribe_wav(LT_COMMAND_WAV_PATH)
                    print(f"[LT-ASR] Raw LT text: {repr(lt_text)}")
                    if lt_text and lt_text.strip():
                        print(f"[LT-ASR] Override Vosk text with LT: {lt_text.strip()}")
                        command_text = lt_text.strip()
            except Exception as e:
                msg = str(e)
                if "Kernel size can't be greater than actual input size" in msg:
                    print(f"[LT-ASR] Short audio, skipping LT override: {e}")
                else:
                    print(f"[LT-ASR] Failed to transcribe LT audio: {e}")

        # Get camera image if available (using cached capture)
        image_data = None
        if hasattr(self.tars_vac, 'picam2'):
            image_data = _capture_image_cached(self.tars_vac.picam2)

        # Mark that we're processing a command - roam should skip observations
        self.tars_vac.tars.state.interaction.start_command_processing()

        try:
            # Process with command processor
            result = self.tars_vac.tars.command_processor.process_command(command_text, image_data=image_data, clear_actions=clear_actions)
            print(f"[DEBUG] Result: {result}")

            if result["success"]:
                response = result.get("message", "Done.")
            else:
                # Try LLM agent
                if self.tars_vac.tars.agent:
                    try:
                        current_lang = self.tars_vac.tars.language.get_current_language()
                        # Pass image_data for vision queries like "what do you see"
                        response = self.tars_vac.tars.agent.chat(command_text, language=current_lang, image_data=image_data)
                    except Exception as e:
                        print(f"[ERROR] Agent failed: {e}")
                        response = "I don't understand."
                else:
                    response = "I don't understand."

            # Speak response using coordinated TTS (prevents overlap with roam)
            current_lang = self.tars_vac.tars.language.get_current_language()
            print(f"[TARS] {response}")
            _coordinated_speak(self.tars_vac.tars, response, language=current_lang, source="command")
        finally:
            # Clear command processing flag
            self.tars_vac.tars.state.interaction.end_command_processing()


class TARSVoiceActiveCar(VoiceActiveCar):
    """
    TARS-integrated voice-activated car.

    Extends VoiceActiveCar to add TARS functionality:
    - Command processing
    - LLM conversation
    - Behaviors (roam, stare, follow)
    - Memory systems
    """

    def __init__(self, tars_robot, *args, **kwargs):
        """
        Initialize TARS voice car.

        Args:
            tars_robot: TARSRobot instance with all subsystems initialized
            *args, **kwargs: Passed to VoiceActiveCar parent
        """
        self.tars = tars_robot

        # Track intercepted commands from stdout
        self.intercepted_command = None
        self.processing_inline_command = False
        self.last_processed_command = None  # Track commands already processed by on_heard()

        # Get wake words from tars_wake_words parameter
        self.wake_words = kwargs.pop('tars_wake_words', ['tars'])
        if isinstance(self.wake_words, str):
            self.wake_words = [self.wake_words]
        # Normalize wake words to lowercase for matching
        self.wake_words = [w.lower() for w in self.wake_words]

        # Set up stdout interceptor for inline commands
        self.log_interceptor = LogInterceptor(sys.stdout, self)
        sys.stdout = self.log_interceptor

        # Initialize VoiceActiveCar (which creates car and camera)
        # This is the slow part - loads wake word models, STT/TTS
        print("[TARS] Calling VoiceActiveCar.__init__ (loading wake word models)...")
        super().__init__(*args, **kwargs)
        print("[TARS] VoiceActiveCar initialization complete")

        # Wrap STT to save audio for Lithuanian transcription (from v58)
        _wrap_vosk_stt_to_save_command_wav(self)

        # Start loading Lithuanian ASR in background (for when user switches to LT)
        _init_lt_asr_async()

        # DISABLE the built-in ultrasonic trigger which causes crashes
        # VoiceActiveCar's is_too_close() calls action_flow.add_action('backward')
        # which conflicts with our operations and causes SIGABRT crashes
        # We handle obstacles ourselves through SafetyManager
        self.triggers = [t for t in self.triggers if t != self.is_too_close]

        # Add custom trigger for inline wake detection
        # This allows "bob move forward" to work as one phrase
        self.continuous_listening_active = False
        self.add_trigger(self.trigger_inline_wake)

        # Now swap TARS components to use VoiceActiveCar's hardware
        self._integrate_tars_with_hardware()

        print("[TARS] Inline wake command detection enabled")
        print("[DEBUG] TARSVoiceActiveCar initialization complete")

    def on_start(self) -> None:
        """Called when voice assistant starts."""
        print("[DEBUG] on_start() called")
        print(f"[DEBUG] wake_enable: {self.wake_enable}")
        print(f"[DEBUG] keyboard_enable: {self.keyboard_enable}")
        print(f"[DEBUG] Number of triggers: {len(self.triggers)}")
        super().on_start()

    def on_finish_a_round(self) -> None:
        """Called after finishing a round of processing."""
        print("[DEBUG] on_finish_a_round() called")

        # Clear command processing flag at end of each round
        # This ensures flag is cleared even if command processing didn't complete normally
        self.tars.state.interaction.end_command_processing()

        super().on_finish_a_round()

    def main(self) -> None:
        """
        Override main loop to ensure our methods are called.

        The parent caches method references during __init__, so our overrides
        don't work. We override main() to call our methods directly.
        """
        print("[DEBUG] main() override called")
        self.running = True
        super().on_start()  # Call parent's on_start
        self.tts.say(self.welcome)

        # Main loop
        while self.running:
          try:  # Wrap entire loop iteration in try/except to prevent crashes
            triggered = False
            message = ''
            disable_image = False

            # Start listening wake words if wake enabled
            if self.wake_enable:
                self.stt.start_listening_wake_words()

            # Start keyboard input
            if self.keyboard_enable:
                self.keyboard_input.start()

            # Wait for triggers
            while self.running:
                for trigger in self.triggers:
                    triggered, disable_image, message = trigger()
                    if triggered:
                        break
                if triggered:
                    break
                import time
                time.sleep(0.01)

            # Stop listening
            if self.wake_enable:
                self.stt.stop_listening()

            if self.keyboard_enable:
                self.keyboard_input.stop()

            print(f"[DEBUG] Trigger fired with message: {repr(message)}")

            # Process message if we have text
            if message:
                # FILTER: Skip internal sensor messages (ultrasonic, etc.)
                # These are VoiceActiveCar's built-in triggers, not user commands
                msg_stripped = message.strip()
                if msg_stripped.startswith('<<<') or '<<<' in msg_stripped or 'Ultrasonic sense' in message:
                    print(f"[DEBUG] Skipping sensor message: {message}")
                    continue

                # CRITICAL: Skip if this command was already processed by on_heard()
                # This prevents duplicate responses for two-part wake commands
                if message.lower().strip() == self.last_processed_command:
                    print(f"[DEBUG] Skipping - already processed by on_heard()")
                    self.last_processed_command = None  # Reset for next command
                    continue
                # INLINE WAKE DETECTION: Check if message contains wake word
                # Only strip wake word if at START of phrase (not in middle like "that the gnome")
                message_lower = message.lower().strip()
                has_wake = any(wake_word in message_lower for wake_word in self.wake_words)

                if has_wake:
                    # Check if wake word is at start (possibly after filler words)
                    cmd_only = None
                    filler_prefixes = ['hey ', 'ok ', 'okay ', 'hi ', 'hello ']
                    check_text = message_lower
                    for filler in filler_prefixes:
                        if check_text.startswith(filler):
                            check_text = check_text[len(filler):]
                            break

                    for wake_word in self.wake_words:
                        if check_text.startswith(wake_word):
                            # Wake word at start - extract command
                            cmd_only = check_text[len(wake_word):].strip()
                            break

                    if cmd_only:
                        print(f"[INLINE] Detected: '{message}' → Command: '{cmd_only}'")
                        message = cmd_only  # Use cleaned command
                    elif not cmd_only and check_text == message_lower:
                        # Wake word NOT at start - don't strip, process full phrase
                        print(f"[USER] {message} (wake word in middle, keeping full phrase)")
                    else:
                        # Wake word only, skip processing
                        print(f"[USER] Wake word only")
                        continue
                else:
                    print(f"[USER] {message}")

                # Process command (with multi-command support)
                print(f"[DEBUG] Processing command: {repr(message)}")

                # Clear accumulated actions ONCE at start for macro recording
                # BUT skip clearing for save/run macro commands (they need the previous actions)
                import re
                msg_lower = message.lower()
                is_macro_command = any(kw in msg_lower for kw in [
                    'save macro', 'same macro', 'run macro', 'one macro', 'won macro',
                    'macro ', 'repeat macro', 'stop macro', 'delete macro', 'forget macro'
                ])
                if not is_macro_command:
                    if hasattr(self, 'tars') and hasattr(self.tars, 'executor'):
                        if hasattr(self.tars.executor, 'clear_accumulated_actions'):
                            self.tars.executor.clear_accumulated_actions()

                # MULTI-COMMAND SUPPORT: Split on "then", "than", "them" (STT misrecognitions), "and then"
                sub_commands = re.split(r'\b(?:and\s+then|then|than|them)\b', message, flags=re.IGNORECASE)
                sub_commands = [cmd.strip() for cmd in sub_commands if cmd.strip()]

                if len(sub_commands) > 1:
                    print(f"[MULTI-CMD] Split into {len(sub_commands)} commands: {sub_commands}")

                for cmd_idx, sub_message in enumerate(sub_commands):
                    if len(sub_commands) > 1:
                        print(f"[MULTI-CMD] Executing ({cmd_idx+1}/{len(sub_commands)}): {sub_message}")

                    # Get camera image (using cached capture)
                    image_data = None
                    if hasattr(self, 'picam2'):
                        image_data = _capture_image_cached(self.picam2)

                    # Process with command processor (don't clear - already cleared above)
                    result = self.tars.command_processor.process_command(sub_message, image_data=image_data, clear_actions=False)
                    print(f"[DEBUG] Result: {result}")

                    if result["success"]:
                        response = result.get("message", "Done.")
                    else:
                        # Try LLM agent
                        if self.tars.agent:
                            try:
                                current_lang = self.tars.language.get_current_language()
                                # Pass image_data for vision queries like "what do you see"
                                response = self.tars.agent.chat(sub_message, language=current_lang, image_data=image_data)
                            except Exception as e:
                                print(f"[ERROR] Agent failed: {e}")
                                response = "I don't understand."
                        else:
                            response = "I don't understand."

                    # Speak response using coordinated TTS (prevents overlap with roam)
                    current_lang = self.tars.language.get_current_language()
                    print(f"[TARS] {response}")
                    _coordinated_speak(self.tars, response, language=current_lang, source="command")

                    # Small delay between multi-commands
                    if len(sub_commands) > 1 and cmd_idx < len(sub_commands) - 1:
                        import time
                        time.sleep(0.3)

            # Call on_finish_a_round
            super().on_finish_a_round()

            # Wait before next round
            import time
            time.sleep(1)

          except KeyboardInterrupt:
            print("\n[TARS] Interrupted by user")
            self.running = False
            break
          except Exception as loop_error:
            # Log error but don't crash - continue to next iteration
            print(f"[ERROR] Loop iteration error: {loop_error}")
            import traceback
            traceback.print_exc()
            time.sleep(1)  # Brief pause before retry

    def _integrate_tars_with_hardware(self):
        """Swap TARS components to use VoiceActiveCar's real hardware."""
        # Wrap car
        real_car = PicarXRobotCar(self.car)
        self.tars.car = real_car
        self.tars.state.hardware.car_instance = real_car
        self.tars.safety.robot = real_car

        # Recreate action resolver with real car
        from motion.action_resolver import create_preset_actions_wrapper
        self.tars.action_resolver = create_preset_actions_wrapper(
            robot=real_car,
            state=self.tars.state,
            stop_flag=self.tars.stop_flag,
            safety_manager=self.tars.safety
        )
        self.tars.executor.action_resolver = self.tars.action_resolver

        # Wrap camera
        if hasattr(self, 'picam2'):
            real_camera = Picamera2Camera(self.picam2)
            real_camera._active = True  # Already started by VoiceActiveCar
            self.tars.camera = real_camera
            self.tars.state.hardware.camera_instance = real_camera

            # Initialize behaviors with real hardware
            self.tars._init_behaviors_with_camera()

        print("[TARS] Hardware integration complete")

    def listen(self) -> str:
        """
        Override listen to detect inline wake commands.

        Returns:
            str: Transcribed text
        """
        print("[DEBUG] listen() called")

        # Call parent's listen to get the transcribed text
        result = super().listen()

        print(f"[DEBUG] listen() got result: {repr(result)}")

        if result:
            result_lower = result.lower().strip()

            # Check if the result contains a wake word (inline command)
            has_wake = any(wake_word in result_lower for wake_word in self.wake_words)

            print(f"[DEBUG] has_wake: {has_wake}, wake_words: {self.wake_words}")

            if has_wake:
                # This is an inline command - extract just the command part
                # BUT only strip wake word if it's at/near the START of the phrase
                # Don't strip "gnome" from "that the gnome name is doris"
                cmd_only = result_lower
                wake_at_start = False
                for wake_word in self.wake_words:
                    # Check if wake word is at start (possibly after filler words like "hey", "ok")
                    stripped = cmd_only.lstrip()
                    filler_prefixes = ['hey ', 'ok ', 'okay ', 'hi ', 'hello ']
                    check_text = stripped
                    for filler in filler_prefixes:
                        if check_text.startswith(filler):
                            check_text = check_text[len(filler):]
                            break

                    if check_text.startswith(wake_word):
                        # Wake word at start - this is a real inline command
                        cmd_only = check_text[len(wake_word):].strip()
                        wake_at_start = True
                        break

                if wake_at_start and cmd_only:
                    # Inline command detected with wake word at start
                    print(f"[INLINE] Detected: {result} → Command: {cmd_only}")
                    self.processing_inline_command = True
                    return cmd_only  # Return just the command without wake word
                # else: wake word in middle/end - don't treat as inline command

        return result

    def trigger_inline_wake(self) -> tuple[bool, bool, str]:
        """
        Custom trigger for inline wake + command detection.

        Checks if the interceptor detected an inline command and returns it.

        Returns:
            tuple: (triggered, disable_image, message)
        """
        triggered = False
        disable_image = False
        message = ''

        # Check if interceptor captured an inline command
        if self.intercepted_command:
            message = self.intercepted_command
            self.intercepted_command = None  # Clear it
            self.processing_inline_command = True
            triggered = True
            print(f"[INLINE] Processing: {message}")

        return triggered, disable_image, message

    def init_camera(self) -> None:
        """Initialize camera with progress message."""
        print("[TARS] Initializing camera (Picamera2)...")
        super().init_camera()
        print("[TARS] Camera initialized")

    def on_heard(self, text: str) -> None:
        """
        Called when speech is heard after wake word detection.

        This is called by the parent when text is heard via keyboard or after wake.
        We process it here since think() isn't being called reliably.

        Args:
            text: The transcribed text that was heard
        """
        print(f"[DEBUG] on_heard() called with text: {repr(text)}")

        if not text:
            return

        # CRITICAL FIX: Save original Vosk text BEFORE LT-ASR override for duplicate detection
        # The trigger in main() uses original Vosk text, not LT-ASR overridden text
        # So we must compare against the original to prevent double processing
        original_text_for_dedup = text.lower().strip()

        # LITHUANIAN ASR OVERRIDE (from v58)
        # If in Lithuanian mode and LT_ASR is available, use it to transcribe the saved audio
        current_lang = self.tars.language.get_current_language()
        if current_lang == "lt" and LT_ASR is not None:
            try:
                if os.path.exists(LT_COMMAND_WAV_PATH) and os.path.getsize(LT_COMMAND_WAV_PATH) > 0:
                    lt_text = LT_ASR.transcribe_wav(LT_COMMAND_WAV_PATH)
                    print(f"[LT-ASR] Raw LT text: {repr(lt_text)}")
                    if lt_text and lt_text.strip():
                        print(f"[LT-ASR] Override Vosk text with LT: {lt_text.strip()}")
                        text = lt_text.strip()
                else:
                    print(f"[LT-ASR] No usable WAV at {LT_COMMAND_WAV_PATH} (file missing or empty).")
            except Exception as e:
                msg = str(e)
                if "Kernel size can't be greater than actual input size" in msg:
                    # LT-ASR chunk too short; keep Vosk text but don't crash
                    print(f"[LT-ASR] Short audio, skipping LT override: {e}")
                else:
                    print(f"[LT-ASR] Failed to transcribe LT audio: {e}")

        # INLINE WAKE DETECTION: Check if text contains wake word
        # Only strip wake word if at START of phrase (not in middle like "that the gnome")
        text_lower = text.lower().strip()
        has_wake = any(wake_word in text_lower for wake_word in self.wake_words)

        if has_wake:
            # Check if wake word is at start (possibly after filler words)
            cmd_only = None
            filler_prefixes = ['hey ', 'ok ', 'okay ', 'hi ', 'hello ']
            check_text = text_lower
            for filler in filler_prefixes:
                if check_text.startswith(filler):
                    check_text = check_text[len(filler):]
                    break

            for wake_word in self.wake_words:
                if check_text.startswith(wake_word):
                    # Wake word at start - extract command
                    cmd_only = check_text[len(wake_word):].strip()
                    break

            if cmd_only:
                print(f"[INLINE] Detected: '{text}' → Command: '{cmd_only}'")
                text = cmd_only  # Use cleaned text
            else:
                # Wake word NOT at start or wake word only - keep full phrase
                print(f"[USER] {text}")
                super().on_heard(text)
                return
        else:
            # Normal command after wake
            print(f"[USER] {text}")

        # Mark this command as processed to prevent duplicate processing in main()
        # CRITICAL: Use original_text_for_dedup (pre-LT-ASR override) because
        # main() loop receives the original Vosk text, not the LT-ASR text
        self.last_processed_command = original_text_for_dedup

        # Process the command directly here
        print(f"[DEBUG] Processing command in on_heard: {repr(text)}")

        # Check for language switch
        if self.tars.language.check_and_switch_language(text):
            msg = self.tars.language.get_confirmation_message(
                self.tars.language.get_current_language()
            )
            print(f"[TARS] {msg}")
            _coordinated_speak(self.tars, msg, language=self.tars.language.get_current_language(), source="command")
            super().on_heard(text)
            return

        # Clear accumulated actions ONCE at start for macro recording
        # BUT skip clearing for save/run macro commands (they need the previous actions)
        import re
        text_lower = text.lower()
        is_macro_command = any(kw in text_lower for kw in [
            'save macro', 'same macro', 'run macro', 'one macro', 'won macro',
            'macro ', 'repeat macro', 'stop macro', 'delete macro', 'forget macro'
        ])
        if not is_macro_command:
            if hasattr(self, 'tars') and hasattr(self.tars, 'executor'):
                if hasattr(self.tars.executor, 'clear_accumulated_actions'):
                    self.tars.executor.clear_accumulated_actions()

        # MULTI-COMMAND SUPPORT: Split on "then", "than", "them" (STT misrecognitions), "and then"
        sub_commands = re.split(r'\b(?:and\s+then|then|than|them)\b', text, flags=re.IGNORECASE)
        sub_commands = [cmd.strip() for cmd in sub_commands if cmd.strip()]

        if len(sub_commands) > 1:
            print(f"[MULTI-CMD] Split into {len(sub_commands)} commands: {sub_commands}")

        for cmd_idx, sub_text in enumerate(sub_commands):
            if len(sub_commands) > 1:
                print(f"[MULTI-CMD] Executing ({cmd_idx+1}/{len(sub_commands)}): {sub_text}")

            # Get camera image if available (using cached capture)
            image_data = None
            if hasattr(self, 'picam2'):
                image_data = _capture_image_cached(self.picam2)

            # Process command (don't clear - already cleared above)
            result = self.tars.command_processor.process_command(sub_text, image_data=image_data, clear_actions=False)
            print(f"[DEBUG] Command result: {result}")

            if result["success"]:
                response = result.get("message", "Command executed.")
            else:
                # Try LLM agent if command not recognized
                if self.tars.agent:
                    try:
                        current_lang = self.tars.language.get_current_language()
                        # Pass image_data for vision queries like "what do you see"
                        response = self.tars.agent.chat(sub_text, language=current_lang, image_data=image_data)
                    except Exception as e:
                        print(f"[ERROR] Agent failed: {e}")
                        response = "I don't understand that command."
                else:
                    response = "I don't understand that command."

            # Speak response using coordinated TTS (prevents overlap with roam)
            current_lang = self.tars.language.get_current_language()
            print(f"[TARS] {response}")
            _coordinated_speak(self.tars, response, language=current_lang, source="command")

            # Small delay between multi-commands
            if len(sub_commands) > 1 and cmd_idx < len(sub_commands) - 1:
                import time
                time.sleep(0.3)

        # Pass to parent's on_heard (does nothing by default)
        super().on_heard(text)

    def is_too_close(self) -> tuple:
        """
        Override VoiceActiveCar's ultrasonic trigger.

        The parent's is_too_close() causes crashes by calling action_flow.add_action()
        which conflicts with our operations. We handle obstacles through SafetyManager instead.
        """
        # Return (triggered=False, disable_image=False, message='') - never trigger
        return (False, False, '')

    def on_wake(self) -> None:
        """Called when wake word is detected."""
        print("[DEBUG] on_wake() called")

        # CRITICAL: Set command processing flag IMMEDIATELY when wake detected
        # This blocks roam observations from speaking while we listen for command
        self.tars.state.interaction.start_command_processing()

        # Only announce if it's not an inline command
        if not self.processing_inline_command:
            print("Waked, listening...")
        super().on_wake()

    def before_think(self, text: str) -> None:
        """Called before processing user input."""
        print(f"[DEBUG] before_think() called with text: {repr(text)}")
        # Don't print here - already printed in on_heard
        super().before_think(text)

    def think(self, text: str, disable_image: bool = False) -> str:
        """
        Process user input and generate response.

        This is the main integration point - we use TARS command
        processing instead of VoiceActiveCar's default behavior.

        Args:
            text: User's input text
            disable_image: Whether to disable image processing

        Returns:
            str: Response text to speak
        """
        print(f"[DEBUG] think() called with text: {repr(text)}, disable_image: {disable_image}")

        # INLINE WAKE DETECTION: Check if text contains wake word
        # Only strip wake word if at START of phrase (not in middle like "that the gnome")
        if text:
            text_lower = text.lower().strip()
            has_wake = any(wake_word in text_lower for wake_word in self.wake_words)

            if has_wake:
                # Check if wake word is at start (possibly after filler words)
                cmd_only = None
                filler_prefixes = ['hey ', 'ok ', 'okay ', 'hi ', 'hello ']
                check_text = text_lower
                for filler in filler_prefixes:
                    if check_text.startswith(filler):
                        check_text = check_text[len(filler):]
                        break

                for wake_word in self.wake_words:
                    if check_text.startswith(wake_word):
                        # Wake word at start - extract command
                        cmd_only = check_text[len(wake_word):].strip()
                        break

                if cmd_only:
                    print(f"[INLINE] Detected wake word in input. Cleaning: '{text}' → '{cmd_only}'")
                    text = cmd_only  # Use cleaned text for processing
                # else: wake word NOT at start - keep full phrase

        # Reset inline command flag at start of think
        if self.processing_inline_command:
            self.processing_inline_command = False
        # Check for language switch
        if self.tars.language.check_and_switch_language(text):
            msg = self.tars.language.get_confirmation_message(
                self.tars.language.get_current_language()
            )
            print(f"[DEBUG] think() returning language switch message: {repr(msg)}")
            return msg

        # Get current camera frame if available for vision queries (using cached capture)
        image_data = None
        if not disable_image and hasattr(self, 'picam2'):
            image_data = _capture_image_cached(self.picam2)

        # Try command processing with vision
        print(f"[DEBUG] Processing command: {repr(text)}")
        result = self.tars.command_processor.process_command(text, image_data=image_data)
        print(f"[DEBUG] Command result: {result}")

        if result["success"]:
            # Command was recognized and executed
            response = result.get("message", "Command executed.")
            print(f"[DEBUG] think() returning success: {repr(response)}")
            return response
        else:
            # Fallback
            response = "I don't understand that command."
            print(f"[DEBUG] think() returning fallback: {repr(response)}")
            return response

    def parse_response(self, text: str) -> str:
        """
        Parse LLM response to extract just the speech text.

        Handles ACTIONS: format from responses.

        Args:
            text: Raw response from think()

        Returns:
            str: Text to speak (without action commands)
        """
        # Remove ACTIONS: lines (in case LLM included them)
        lines = text.split('\n')
        speech_lines = []
        for line in lines:
            if not line.strip().upper().startswith('ACTIONS:'):
                speech_lines.append(line)

        return '\n'.join(speech_lines).strip()

    def before_say(self, text: str) -> None:
        """Called before speaking response."""
        print(f"[DEBUG] before_say() called with text: {repr(text)}")
        print(f"[TARS] {text}")
        super().before_say(text)

    def on_stop(self) -> None:
        """Called when stopping - clean up TARS systems."""
        # Restore original stdout
        if hasattr(self, 'log_interceptor'):
            sys.stdout = self.log_interceptor.original_stdout

        super().on_stop()
        self.tars.shutdown()
