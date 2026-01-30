# TARS Robot - SSH Startup Guide

Complete guide for starting TARS when connected via SSH.

## 🔌 Connect to Your Robot

```bash
# From your computer, connect to the Raspberry Pi
ssh pi@<robot-ip-address>
# or
ssh pi@raspberrypi.local
```

## 🚀 Start TARS (Quick Method)

```bash
# 1. Navigate to TARS directory
cd /home/mdigital/picar-x/tars_system

# 2. Set your OpenAI API key (if not already in ~/.bashrc)
export OPENAI_API_KEY="sk-your-key-here"

# 3. Run the startup script
./start_tars.sh
```

That's it! TARS will initialize and start listening for voice commands.

## 🎯 Alternative: Direct Python Execution

```bash
cd /home/mdigital/picar-x/tars_system
export OPENAI_API_KEY="sk-your-key-here"
python3 main.py
```

## ⚙️ Make API Key Permanent (Recommended)

To avoid setting the API key every time:

```bash
# Edit your bash profile
nano ~/.bashrc

# Add this line at the end:
export OPENAI_API_KEY="sk-your-actual-key-here"

# Save (Ctrl+X, Y, Enter)

# Reload the profile
source ~/.bashrc
```

Now the API key will be set automatically on every login!

## 🎤 All Available Voice Commands

### Basic Motion
```
"move forward"
"move backward" / "go back"
"turn left"
"turn right"
"stop"
```

### Fun Actions
```
"dance"                    # Does wiggle + square dance
"wiggle" / "shake"        # Shakes the robot
"turn around" / "u-turn"  # 180 degree turn
"spin" / "spin left" / "spin right"
"drive in a square"
```

### Head Movements
```
"look left" / "head left"
"look right" / "head right"
"look center" / "head center"
"scan around" / "head scan"
```

### Autonomous Behaviors
```
"start roaming" / "explore"    # Autonomous exploration
"stop roaming"
"look at me" / "track my face" # Face tracking mode
"stop staring"
"follow me"                     # Person following mode
"stop following"
```

### Language Switching
```
"switch to english"
"pakeisk į lietuvių"           # Switch to Lithuanian
```

### Conversations
```
"what do you see?"
"tell me about yourself"
"what's your name?"
Any question or conversation...
```

## 🛑 Stop TARS

Press **Ctrl+C** to safely stop TARS. It will:
- Stop all motors
- Stop all behaviors
- Save all memory
- Clean shutdown

## 📊 Running in Background (tmux method)

If you want TARS to keep running after you disconnect from SSH:

```bash
# Install tmux (one time)
sudo apt-get install tmux

# Start a tmux session
tmux new -s tars

# Inside tmux, start TARS
cd /home/mdigital/picar-x/tars_system
./start_tars.sh

# Detach from tmux (TARS keeps running)
# Press: Ctrl+B, then D

# Reconnect later
tmux attach -t tars

# Kill the session when done
tmux kill-session -t tars
```

## 📊 Running in Background (nohup method)

Alternative background method:

```bash
cd /home/mdigital/picar-x/tars_system
export OPENAI_API_KEY="sk-your-key-here"
nohup python3 main.py > tars.log 2>&1 &

# View logs
tail -f tars.log

# Stop TARS
pkill -f "python3 main.py"
```

## 🔧 Troubleshooting

### Camera Issues
```bash
# Check camera
vcgencmd get_camera

# Test camera
libcamera-hello

# If camera not working, enable it
sudo raspi-config
# Go to: Interface Options -> Camera -> Enable
```

### Audio Issues
```bash
# List audio devices
arecord -l
aplay -l

# Test microphone
arecord -d 5 test.wav
aplay test.wav

# Adjust microphone volume
alsamixer
# Press F4 for capture, adjust with arrow keys
```

### Python Import Errors
```bash
# Make sure you're in the right directory
cd /home/mdigital/picar-x/tars_system
pwd  # Should show: /home/mdigital/picar-x/tars_system

# Check Python path
python3 -c "import sys; print('\n'.join(sys.path))"
```

### Vosk Models Missing
```bash
# Download English model
cd /home/mdigital/picar-x/tars_system
mkdir -p models
cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip

# Optional: Lithuanian model
wget https://alphacephei.com/vosk/models/vosk-model-lt-0.1.zip
unzip vosk-model-lt-0.1.zip
```

### Check System Status
```bash
# Check what's running
ps aux | grep python

# Check memory
free -h

# Check CPU temperature
vcgencmd measure_temp

# Check disk space
df -h
```

## 🧪 Test Mode (Without Robot Hardware)

If you want to test without the robot hardware:

```bash
cd /home/mdigital/picar-x/tars_system

# Run tests
python3 -m unittest discover -s tests -v

# This will use mock hardware
```

## 📝 Logs and Memory Files

TARS saves data to your home directory:

```bash
~/.tars_character.json     # Personality/character state
~/.tars_conversation.json  # Conversation history
~/.tars_macros.json        # Learned action macros
~/.tars_visual/            # Visual memories (images + index)
```

View memory:
```bash
cat ~/.tars_character.json | python3 -m json.tool
cat ~/.tars_conversation.json | python3 -m json.tool
```

## 🎬 Quick Start Checklist

- [ ] SSH into robot
- [ ] Navigate to: `cd /home/mdigital/picar-x/tars_system`
- [ ] Set API key: `export OPENAI_API_KEY="sk-..."`
- [ ] Run: `./start_tars.sh`
- [ ] Wait for "Voice loop started. Listening..."
- [ ] Say "Hello TARS!"
- [ ] Have fun! 🤖

## 💡 Pro Tips

1. **Use tmux** for persistent sessions
2. **Add API key to ~/.bashrc** for convenience
3. **Monitor logs** with `tail -f tars.log` if running in background
4. **Test camera and mic** before starting TARS
5. **Keep robot on flat surface** during startup
6. **Say commands clearly** and wait for response

---

**Need help?** Check the main README.md or run the integration tests to verify everything works.
