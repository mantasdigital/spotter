# 🤖 TARS — An AI-Powered Autonomous Robot

**Built with Claude AI, Raspberry Pi 5, and a lot of late nights.**

> *"What's my sarcasm setting?"*
> *"100%."*

TARS is a fully autonomous, AI-powered robot that sees, speaks, listens, dances, explores, and makes sarcastic comments about everything — all in Lithuanian. Built as a passion project over 3 weeks of evenings and weekends, starting on Kūčios (Lithuanian Christmas Eve) 2025.

---

## What Can TARS Do?

**🧠 Intelligence & Conversation**
- Full voice interaction — listens, understands, and responds in Lithuanian
- Sarcastic personality with genuine humor (adjustable... in theory)
- Google search integration — can look things up and share what it finds
- Great memory — remembers faces, names, objects, and conversations
- If idle too long, starts exploring on its own out of "boredom"

**👁️ Vision & Recognition**
- Real-time camera-based object recognition
- Distinguishes between specific object models (different shoes, different microphones, etc.)
- Face recognition — remembers who you are
- Explains what it sees and comments on it (often sarcastically)

**🚶 Movement & Navigation**
- Autonomous perimeter patrol with obstacle avoidance
- Infrared cliff detection — won't drive off edges
- Follows humans around the room
- Learns and remembers new pathways over time

**💃 Dance**
- 10+ pre-programmed dance routines
- Can learn new dance moves
- Yes, it actually dances. No, it's not graceful. Yes, it's entertaining.

**🇱🇹 Lithuanian Language**
- Full Lithuanian speech recognition
- Lithuanian speech output
- One of very few hobby robots with native Lithuanian language support

---

## Hardware

| Component | Details |
|-----------|---------|
| **Brain** | Raspberry Pi 5 |
| **Vision** | Camera module |
| **Hearing** | Microphone |
| **Voice** | Speaker |
| **Movement** | DC motors + servo motors |
| **Senses** | Infrared sensors (obstacle avoidance & cliff detection) |

---

## Software Stack

- **Language:** Python
- **AI Assistant:** Built with [Claude](https://claude.ai) by Anthropic
- **Architecture:** `tars_system_v3` — modular system handling vision, movement, speech, memory, and personality

---

## The Story

This project started on **Kūčios (December 24, 2025)** — Lithuanian Christmas Eve. While most people were eating kūčiukai, I opened my laptop and started building a robot.

Over the next **3 weeks**, working evenings after my day job and weekends, TARS went from a pile of parts to a fully autonomous robot that patrols the house, recognizes family members, dances on command, and makes sarcastic comments about the furniture.

The entire software was built with **Claude AI** as my coding partner. From motor control to computer vision to natural language processing — Claude helped architect, debug, and iterate on every component.

The highlight? **Making grandparents happy.** Watching them interact with TARS, laughing at its jokes and responses, made every late night worth it.

---

## Setup

### Default (Cloud AI — OpenAI)

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=sk-your-key-here

# Run from inside picar-x directory
cd ~/picar-x/tars_system_v3
chmod +x start_tars.sh
./start_tars.sh
```

### Local AI (Ollama — no cloud needed)

Run TARS with a local AI model instead of OpenAI. No API key required, fully offline.

**Step 1: Install Ollama (~150MB)**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Step 2: Pull a model (pick one)**

| Model | Size | Speed on Pi | Quality | Command |
|-------|------|-------------|---------|---------|
| gemma2:2b | 1.6GB | Fast (~5 tok/s) | Good for commands | `ollama pull gemma2:2b` |
| phi3:mini | 2.3GB | Medium (~3 tok/s) | Better conversation | `ollama pull phi3:mini` |
| mistral | 4GB | Slow (~2 tok/s) | Best quality | `ollama pull mistral` |

**Step 3: Run TARS in local mode**
```bash
export TARS_LLM_BACKEND=local
export TARS_LOCAL_MODEL=gemma2:2b   # optional, this is the default

cd ~/picar-x/tars_system_v3
python3 main.py
```

**Notes:**
- If Ollama is not running or the model is not pulled, TARS automatically falls back to OpenAI
- Vision queries ("what do you see") need OpenAI — local models can't process images
- All voice commands (movement, roaming, Lithuanian) work the same regardless of backend
- You can switch back anytime: `export TARS_LLM_BACKEND=openai` or just unset the variable

---

## Project Structure

```
spotter/
└── tars_system_v3/          # Main robot system (v3)
    ├── main.py              # Entry point
    ├── tars_voice_car.py    # Voice loop + wake word detection
    ├── config/              # Settings
    ├── speech/              # STT (Vosk), TTS (OpenAI/Piper), Lithuanian fuzzy matching
    ├── llm/                 # AI agents (OpenAI or local Ollama)
    ├── vision/              # Camera, face detection, scene analysis
    ├── motion/              # Movement, safety, action execution
    ├── behaviors/           # Roaming, following, face tracking
    ├── interaction/         # Command processing + handlers
    └── memory/              # Conversation, character, macros, visual memory
```

---

## Built With

- 🧠 [Claude AI](https://claude.ai) — AI coding partner for the entire build
- 🍓 [Raspberry Pi 5](https://www.raspberrypi.com/) — The brain
- 🐍 Python — The language
- ❤️ Curiosity — The fuel

---

## Media

*Photos and videos coming soon — stay tuned.*

---

## Author

**Mantas** — Founder and Digital engineer at [Mantas Digital](https://mantasdigital.com)

This was my very first robot. Built in 2025.

---

## License

This project is open source. Feel free to learn from it, fork it, and build your own sarcastic robot.

---

*"I have a cue light I can use when I'm joking, if you want."*
*— TARS*
