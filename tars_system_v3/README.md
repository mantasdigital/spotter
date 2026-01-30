# TARS Robot System V2 (Development)

**Version 2.0 Development Branch** - Separate development copy for testing new features.

Based on the stable tars_system (v1) modular architecture.

## Features

All features from v58 are implemented in a modular architecture:

✅ **Vision**
- Face detection (OpenCV Haar cascades)
- Scene analysis (GPT-4o vision)
- Visual memory with tagging

✅ **Speech**
- Speech-to-Text (Vosk offline)
- Text-to-Speech (OpenAI TTS)
- Multi-language support (English, Lithuanian)

✅ **Motion**
- Safety-checked movement primitives
- Action resolver with preset actions
- Macro recording and playback

✅ **Behaviors**
- Autonomous roaming with obstacle avoidance
- Face tracking (stare mode)
- Person following

✅ **Interaction**
- Voice command processing
- Pattern-based command handlers
- LLM fallback for conversation

✅ **Memory**
- Character state (personality)
- Conversation history
- Visual memories
- Learned action macros

## Quick Start

### 1. Install Dependencies

```bash
# Install Python packages
pip install -r requirements.txt

# Download Vosk models (if using offline STT)
mkdir -p models
cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
```

### 2. Set Environment Variables

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Run TARS

```bash
# From tars_system directory
python main.py

# Or make it executable
chmod +x main.py
./main.py
```

## Usage

Once running, TARS will listen for voice commands:

### Motion Commands
- "move forward" / "go ahead"
- "move backward" / "go back"
- "turn left" / "turn right"
- "stop"

### Behavior Commands
- "start roaming" / "explore"
- "stop roaming"
- "look at me" / "track my face"
- "follow me"
- "stop following"

### Language Commands
- "switch to english"
- "pakeisk į lietuvių"

## Architecture

```
tars_system/
├── main.py                 # Main entry point
├── hardware/              # Hardware abstractions
│   ├── interfaces.py      # Abstract interfaces
│   ├── picarx_impl.py    # Real PiCar-X implementation
│   ├── camera_impl.py    # Real camera implementation
│   └── mock_hardware.py  # Mock implementations for testing
├── core/                  # Core systems
│   └── state_manager.py  # Centralized state management
├── config/               # Configuration
│   └── settings.py       # Safety and system settings
├── motion/               # Motion control
│   ├── primitives.py     # Low-level movement functions
│   ├── safety_manager.py # Safety checking
│   ├── action_resolver.py # Action dictionary
│   └── action_executor.py # Action execution
├── vision/               # Vision processing
│   ├── face_detector.py  # Face/hand detection
│   ├── scene_analyzer.py # LLM-based scene analysis
│   └── visual_memory.py  # Visual memory storage
├── speech/               # Speech I/O
│   ├── providers/        # STT/TTS implementations
│   └── language_manager.py # Language detection/switching
├── llm/                  # LLM integration
│   ├── agents.py         # Conversation agent
│   └── prompts.py        # System prompts
├── behaviors/            # Autonomous behaviors
│   ├── roam/            # Roaming behavior
│   └── stare_behavior.py # Face tracking & following
├── interaction/          # Command processing
│   ├── command_processor.py # Main processor
│   └── command_handlers/ # Command handlers
├── memory/              # Memory systems
│   ├── character_state.py    # Personality state
│   ├── conversation_memory.py # Dialogue history
│   ├── visual_memory.py      # Visual memories
│   └── macro_store.py        # Action macros
└── tests/               # Integration tests
    ├── test_vision_integration.py
    ├── test_speech_integration.py
    └── test_command_integration.py
```

## Testing

Run all integration tests:

```bash
python -m unittest discover -s tests -v
```

Run specific test suite:

```bash
python -m unittest tests.test_vision_integration -v
python -m unittest tests.test_speech_integration -v
python -m unittest tests.test_command_integration -v
```

## Differences from v58

### Improvements
- **Modular Architecture**: Everything is properly separated into modules
- **Testable**: All components have interfaces and mock implementations
- **Type Hints**: Full type annotations throughout
- **Documentation**: Comprehensive docstrings
- **Clean Dependencies**: Proper dependency injection
- **48 Integration Tests**: Full test coverage

### What's the Same
- All functionality from v58 is preserved
- Uses same hardware (PiCar-X, Picamera2)
- Same LLM integration (OpenAI GPT-4o)
- Same voice processing pipeline
- Compatible with existing configs and memory files

## Troubleshooting

### Camera not working
```bash
# Check if camera is detected
vcgencmd get_camera

# Test camera
libcamera-hello
```

### Vosk models not found
Download models to `models/` directory. See Quick Start section.

### OPENAI_API_KEY not set
```bash
export OPENAI_API_KEY="sk-..."
# Or add to ~/.bashrc for persistence
```

### Import errors
```bash
# Make sure you're in the tars_system_v2 directory
cd /home/mdigital/picar-x/tars_system_v2
python main.py
```

## Contributing

The modular architecture makes it easy to:
- Add new command handlers
- Implement new behaviors
- Swap out providers (STT, TTS, LLM)
- Add new hardware interfaces

See individual module docstrings for API details.
