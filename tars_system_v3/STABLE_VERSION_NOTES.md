# TARS System - Stable Version Notes
## Version: 2026-01-06 v3 (Production Ready)

This is the most stable version of the TARS system with:
- Roam/Explore mode with robust obstacle avoidance (49/50+ success rate)
- Follow mode with roam-style obstacle detection
- Gesture detection without false positives
- Cliff detection with immediate response
- Soft object detection (trash bags, fabric)
- Exploration variety (curiosity wanders)
- Voice commands (English and Lithuanian)

---

## What's New in v3 (from v2)

### 1. Follow Mode - Roam-Style Obstacle Detection
- Added invalid distance handling (negative readings = too close)
- Added soft object detection (variance + jump patterns)
- Matching thresholds with roam mode

### 2. Increased Obstacle Thresholds
- DIST_CAUTION: 45 → 48 cm
- DIST_BLOCKED: 35 → 38 cm
- DIST_CRITICAL: 18 → 20 cm

### 3. Exploration Variety
- Robot no longer goes straight forever on clear paths
- Random "curiosity wander" turns after 8 consecutive forwards
- 40% chance to turn 30-60 degrees to explore different areas

---

## Critical Thresholds and Configurations

### 1. Obstacle Avoidance (`behaviors/roam/roam_behavior.py`)

```python
# Distance thresholds (in cm)
DIST_CLEAR = 55.0       # Path clear - full speed forward
DIST_CAUTION = 48.0     # Slow down to 60% speed
DIST_BLOCKED = 38.0     # Stop and turn
DIST_CRITICAL = 20.0    # Emergency backup

# Soft object detection
SOFT_OBJECT_JUMP_CM = 50.0    # Sudden distance increase threshold
SOFT_OBJECT_CONSECUTIVE = 3   # Readings before treating as obstacle

# Exploration variety
WANDER_TRIGGER_COUNT = 8      # Forwards before wander check
WANDER_CHANCE = 0.4           # 40% chance to turn
WANDER_ANGLES = [30, -30, 45, -45, 60, -60]

# Escape escalation (block count thresholds)
SIDE_ESCAPE_BLOCKS = 1      # Simple turn (60 deg)
STUCK_BLOCKS = 3            # Aggressive turn (90+ deg)
VERY_STUCK_BLOCKS = 5       # Backup + turn (120+ deg)
HARD_ESCAPE_BLOCKS = 8      # Long backup + large turn (150+ deg)
ULTRA_STUCK_BLOCKS = 12     # Full reverse + 180 deg + reset

# Timing
CLEAR_COUNT_TO_RESET = 10   # Consecutive clear readings to reset block count
```

### 2. Follow Mode Obstacle Detection (`behaviors/stare_behavior.py`)

```python
# Distance thresholds (matching roam mode)
OBSTACLE_DIST_CRITICAL = 20.0  # Stop immediately (cm)
OBSTACLE_DIST_CAUTION = 38.0   # Slow down zone (cm)
OBSTACLE_DIST_CLEAR = 55.0     # Safe to move forward (cm)

# Soft object detection
SOFT_OBJECT_JUMP_CM = 50.0     # Sudden distance increase
SOFT_OBJECT_VARIANCE = 400     # High variance threshold
SOFT_OBJECT_CONSECUTIVE = 3    # Readings before blocking
```

### 3. Gesture Detection (`vision/gesture_detector.py`)

```python
# Skin color detection thresholds
SKIN_AREA_THRESHOLD = 0.25      # 25% skin = definite hand
SKIN_AREA_MIN_FOR_VARIANCE = 0.03  # 3% skin for variance-assisted
SKIN_AREA_TINY = 0.01           # 1% skin REQUIRED for very-low-variance

# Variance thresholds
VARIANCE_THRESHOLD_STRICT = 100     # Very low variance (blocked camera)
VARIANCE_THRESHOLD_WITH_SKIN = 400  # Low variance + skin detected

# Distance context
OBSTACLE_DISTANCE_THRESHOLD = 30.0  # Disable variance detection when closer

# Debounce and consecutive frames
DEBOUNCE_SEC = 2.0
CONSECUTIVE_FRAMES_REQUIRED = 4
```

### 4. Cliff Detection (`hardware/picarx_impl.py`)

```python
CLIFF_THRESHOLD = 950  # Grayscale values > 950 = cliff/void

# Normal floor readings: 300-700
# Cliff/edge readings: > 950
```

---

## Key Learnings and Fixes (All Versions)

### v2 Fixes Applied:
1. **Gesture False Positives**: Require 1% skin for very-low-variance mode
2. **Soft Objects**: Detect via distance jumps + high variance patterns
3. **Invalid Distance**: Track consecutive invalid readings → emergency backup
4. **Block Count**: Require 10 consecutive clear readings to reset
5. **Escape Gesture**: Disable gesture during escape maneuvers

### v3 Enhancements:
1. **Follow Mode**: Full roam-style obstacle detection
2. **Higher Thresholds**: More conservative distances for safety
3. **Exploration Variety**: Random curiosity wanders

---

## Backup Versions Reference

| Version | Date | Key Features |
|---------|------|--------------|
| tars_system_stable_20260105 | 2025-01-05 | Basic working version |
| tars_system_stable_20260106_macros_working | 2025-01-06 | Macros and commands working |
| tars_system_stable_20260106_v2_perfected | 2025-01-06 | Gesture + obstacle fixes |
| **tars_system_stable_20260106_v3** | 2025-01-06 | **CURRENT BEST** - Follow detection + wander |

---

## Testing Results

### Roam Mode: 49/50 obstacles handled correctly
- Soft objects (trash bags) detected via erratic readings
- Curiosity wanders add exploration variety
- Gesture detection no false positives

### Follow Mode: Enhanced with roam-style detection
- Same soft object detection as roam
- Invalid distance handling
- Stuck detection with escape maneuvers

---

## Voice Commands

### Roam/Explore (English)
- "start roaming", "roam mode", "explore", "exploring mode"
- "wander", "patrol", "watch the perimeter", "scout around"

### Follow Mode
- "follow me", "come with me"
- "stop following", "stay there"

### Stare/Face Tracking
- "stare at me", "look at me", "watch me"

---

## Quick Start

```bash
cd /home/mdigital/picar-x/tars_system
./start_tars.sh

# Or directly:
python3 main.py
```

---

## Debug Logging

- `[ROAM] dist=XXcm` - Distance readings
- `[ROAM] BLOCKED at XXcm` - Obstacle detected
- `[ROAM] Soft object suspect` - Soft object pattern
- `[ROAM] Curiosity wander!` - Random exploration turn
- `[FOLLOW] Distance: XXcm` - Follow mode distance
- `[FOLLOW] Soft object suspect` - Follow mode soft detection
- `[GESTURE] var=XX skin=XX%` - Gesture detection state
- `[ROAM] CLIFF DETECTED!` - Cliff triggered

---

*Last Updated: 2026-01-06*
*Status: STABLE v3 - Production Ready*
