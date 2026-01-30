# Follow Mode Improvement Plan
## Goals:
1. Track back of head when face is lost
2. Continue tracking after obstacle avoidance in last-seen direction

---

## Research Findings

### Current Behavior (Problems)
1. **When face lost** (line 630-635 in stare_behavior.py):
   - Motors stop immediately
   - Camera pan/tilt preserved but no search performed
   - No memory of last-seen direction or position
   - Forward tracking resets

2. **After stuck escape** (line 725-749):
   - Robot backs up and turns to side
   - No attempt to re-acquire target
   - Doesn't look in last-known direction

### Available Detection Cascades
```
/usr/share/opencv4/haarcascades/
├── haarcascade_frontalface_default.xml  ← Current (front face only)
├── haarcascade_profileface.xml          ← Side face (left/right profile)
├── haarcascade_upperbody.xml            ← Upper body (works from behind!)
├── haarcascade_fullbody.xml             ← Full body
└── haarcascade_lowerbody.xml            ← Lower body
```

### Key Insight
- `haarcascade_upperbody.xml` detects torso/shoulders from ANY angle including behind
- `haarcascade_profileface.xml` detects side views of face
- Combined with frontal face = better coverage

---

## Implementation Plan

### Phase 1: Add Multi-Cascade Detection

**File**: `behaviors/stare_behavior.py` (FollowBehavior class)

**Changes**:
1. Load additional cascades in `_init_cascade()`:
   ```python
   self._face_cascade = None       # Frontal face (primary)
   self._profile_cascade = None    # Side face profile
   self._upperbody_cascade = None  # Upper body (back of person)
   ```

2. Create detection priority system:
   - Priority 1: Frontal face (best for distance estimation)
   - Priority 2: Profile face (side view)
   - Priority 3: Upper body (back of person, larger target)

3. Track detection type for logging:
   ```python
   self._last_detection_type = None  # "face", "profile", "body"
   ```

### Phase 2: Add Last-Seen Memory

**New state variables**:
```python
# Last-seen tracking
self._last_seen_x = None          # Last known X position in frame
self._last_seen_y = None          # Last known Y position in frame
self._last_seen_width = None      # Last known target width
self._last_seen_time = 0.0        # When last seen
self._last_seen_pan = 0.0         # Camera pan when last seen
self._last_seen_direction = 0     # -1=left, 0=center, 1=right (relative to robot)
self._frames_since_seen = 0       # Counter for search timeout
```

### Phase 3: Implement Search Behavior

**When target lost**:
1. Don't stop immediately - wait a few frames
2. If lost for 5-10 frames:
   - Pan camera toward last-seen direction
   - Try profile and body detection
3. If lost for 20+ frames:
   - Start slow rotation toward last-seen direction
   - Keep scanning with all detectors
4. If lost for 60+ frames (3 seconds):
   - Stop and wait (current behavior)

**Search pattern**:
```python
SEARCH_PATIENCE_FRAMES = 5      # Frames before starting search
SEARCH_PAN_SPEED = 2.0          # Degrees per frame to pan
SEARCH_MAX_PAN = 45             # Max pan before giving up direction
SEARCH_TIMEOUT_FRAMES = 60      # Give up after this many frames
```

### Phase 4: Post-Obstacle Recovery

**After `_execute_stuck_escape()`**:
1. Remember the escape direction
2. After escape, pan camera to last-seen direction
3. Try all detection cascades
4. If found, resume following
5. If not found, continue search pattern

**New method**: `_search_for_target()`
```python
def _search_for_target(self):
    """
    Search for target using all available detectors.
    Returns (found, detection_type, bbox) or (False, None, None)
    """
    # Try frontal face first
    # Then profile face
    # Then upper body
    # Return best match
```

---

## Detailed Code Changes

### 1. New Constants
```python
# Multi-cascade detection
PROFILE_SCALE_FACTOR = 1.1
PROFILE_MIN_NEIGHBORS = 3
PROFILE_MIN_SIZE = (50, 50)

BODY_SCALE_FACTOR = 1.05
BODY_MIN_NEIGHBORS = 3
BODY_MIN_SIZE = (60, 100)  # Bodies are taller than wide

# Search behavior
SEARCH_PATIENCE_FRAMES = 5
SEARCH_PAN_STEP = 3.0          # Degrees per search step
SEARCH_MAX_PAN_OFFSET = 35     # Max additional pan from last position
SEARCH_TIMEOUT_FRAMES = 60
SEARCH_SLOW_TURN_SPEED = 15    # Speed for searching turn
```

### 2. Modified `_init_cascade()`
```python
def _init_cascade(self):
    # Load frontal face (primary)
    self._face_cascade = self._load_cascade([
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
    ])

    # Load profile face (secondary)
    self._profile_cascade = self._load_cascade([
        "/usr/share/opencv4/haarcascades/haarcascade_profileface.xml",
    ])

    # Load upper body (tertiary - for back of person)
    self._upperbody_cascade = self._load_cascade([
        "/usr/share/opencv4/haarcascades/haarcascade_upperbody.xml",
    ])
```

### 3. New Detection Method
```python
def _detect_person(self, grey, frame_w, frame_h):
    """
    Detect person using cascade priority: face > profile > body.

    Returns:
        tuple: (detected, detection_type, x, y, w, h) or (False, None, ...)
    """
    # Try frontal face
    if self._face_cascade:
        faces = self._face_cascade.detectMultiScale(...)
        if faces:
            return (True, "face", *best_match(faces))

    # Try profile (both orientations)
    if self._profile_cascade:
        profiles = self._profile_cascade.detectMultiScale(...)
        # Also try flipped image for other profile
        if profiles:
            return (True, "profile", *best_match(profiles))

    # Try upper body (works from behind)
    if self._upperbody_cascade:
        bodies = self._upperbody_cascade.detectMultiScale(...)
        if bodies:
            return (True, "body", *best_match(bodies))

    return (False, None, 0, 0, 0, 0)
```

### 4. Modified Main Loop Logic
```python
# In _follow_step_vilib():

detected, det_type, x, y, w, h = self._detect_person(grey, frame_w, frame_h)

if detected:
    # Update last-seen memory
    self._update_last_seen(x, y, w, h, det_type)
    self._frames_since_seen = 0

    # Adjust target size based on detection type
    if det_type == "body":
        # Body is larger, adjust target width expectation
        effective_w = w * 0.6  # Face would be ~60% of body width
    else:
        effective_w = w

    # Continue with tracking using effective_w...

else:
    # Target lost
    self._frames_since_seen += 1

    if self._frames_since_seen < SEARCH_PATIENCE_FRAMES:
        # Brief loss - keep moving in same direction
        pass
    elif self._frames_since_seen < SEARCH_TIMEOUT_FRAMES:
        # Search mode - pan toward last-seen direction
        self._execute_search_step()
    else:
        # Give up - stop
        self.car.stop()
```

### 5. Post-Obstacle Recovery
```python
def _execute_stuck_escape(self):
    # ... existing escape code ...

    # NEW: After escape, try to re-acquire target
    self._post_escape_recovery()

def _post_escape_recovery(self):
    """
    After obstacle escape, search for target in last-known direction.
    """
    # Pan camera toward last-seen direction
    if self._last_seen_direction != 0:
        recovery_pan = self._last_seen_pan + (15 * self._last_seen_direction)
        recovery_pan = _clamp_number(recovery_pan, self.PAN_MIN, self.PAN_MAX)
        self.car.set_cam_pan_angle(recovery_pan)
        time.sleep(0.3)  # Allow camera to settle

    # Reset search state to start fresh search
    self._frames_since_seen = self.SEARCH_PATIENCE_FRAMES  # Skip patience, start searching
```

---

## Testing Plan

1. **Test frontal face tracking** - Should work as before
2. **Test profile tracking** - Walk past robot sideways
3. **Test body tracking** - Walk away from robot (back visible)
4. **Test face loss recovery** - Turn away briefly, robot should find you
5. **Test obstacle recovery** - Have obstacle between robot and person
6. **Test search timeout** - Leave room, robot should stop after timeout

---

## Risk Mitigation

1. **False positives from body detection**:
   - Use higher minNeighbors for body cascade
   - Prioritize face/profile over body
   - Require body detection to be in expected region

2. **Performance impact**:
   - Run cascades sequentially, stop on first match
   - Body detection only when face/profile not found
   - Consider downscaling for body detection

3. **Oscillation during search**:
   - Use exponential smoothing for search direction
   - Add deadband to prevent small oscillations

---

## Files to Modify

1. `behaviors/stare_behavior.py` - FollowBehavior class (main changes)
2. `behaviors/__init__.py` - No changes needed
3. Optionally: Create separate `person_detector.py` for reusable detection

---

## Estimated Changes

- ~100 lines new code for multi-cascade detection
- ~50 lines for last-seen memory tracking
- ~80 lines for search behavior
- ~30 lines for post-obstacle recovery
- Total: ~260 lines of new/modified code

---

*Plan created: 2026-01-06*
*Status: Ready for implementation*
