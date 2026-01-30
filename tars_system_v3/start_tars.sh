#!/bin/bash
# TARS Robot Startup Script - V2 Development Version

echo "========================================="
echo "TARS Robot System V2 (Development)"
echo "========================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found. Please run this script from the tars_system_v2 directory."
    exit 1
fi

# Check for OPENAI_API_KEY
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Warning: OPENAI_API_KEY environment variable not set."
    echo "LLM features will not be available."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for Vosk models (in system-wide location used by sunfounder_voice_assistant)
if [ ! -d "/opt/vosk_models/vosk-model-small-en-us-0.15" ]; then
    echo "Warning: Vosk English model not found at /opt/vosk_models/vosk-model-small-en-us-0.15"
    echo "STT will fall back to mock implementation or auto-download on first use."
    echo ""
fi

# Check if camera is available
echo "Checking camera..."
if command -v vcgencmd &> /dev/null; then
    CAM_STATUS=$(vcgencmd get_camera)
    echo "Camera status: $CAM_STATUS"
else
    echo "Warning: Cannot check camera status (vcgencmd not found)"
fi

echo ""
echo "Starting TARS..."
echo "Press Ctrl+C to stop"
echo "========================================="
echo ""

# Run TARS
python3 main.py
