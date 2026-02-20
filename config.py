"""
config.py — BFMC Autonomous Car System
=======================================
Single source of truth for ALL constants, thresholds, model paths, and
tunable parameters. Every other module imports from here.
No magic numbers anywhere else.
"""

import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# PROJECT ROOT  (directory this file lives in)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()

# ===========================================================================
# SECTION 1 — CAMERA
# ===========================================================================
CAM_W: int = 640        # raw capture width  (pixels)
CAM_H: int = 480        # raw capture height (pixels)
BEV_W: int = 640        # bird's-eye-view output width
BEV_H: int = 480        # bird's-eye-view output height
TARGET_FPS: int = 30
FRAME_PERIOD: float = 1.0 / TARGET_FPS

# ===========================================================================
# SECTION 2 — BEV CALIBRATION
# SRC: 4 trapezoid corners in the RAW frame  [TL, TR, BL, BR]
# DST: corresponding corners in BEV output    [TL, TR, BL, BR]
# CALIBRATE: open Camera window, adjust until lane lines are parallel.
# ===========================================================================
SRC_PTS = np.float32([[200, 260], [440, 260], [40,  450], [600, 450]])
DST_PTS = np.float32([[150,   0], [490,   0], [150, 480], [490, 480]])

# CLAHE enhancement
CLAHE_CLIP_LIMIT: float = 3.0
CLAHE_TILE_GRID: tuple[int, int] = (8, 8)

# Adaptive threshold (applied on L-channel of HLS in BEV space)
ADAPT_BLOCK_SIZE: int = 31    # must be odd; larger → suits wider BEV lines
ADAPT_C: int = -8             # negative = pick bright lines

# S-channel supplemental threshold (yellow / coloured lines)
S_THRESH_LOW: int = 80

# Morphological close kernel
MORPH_KERNEL_SIZE: tuple[int, int] = (5, 5)

# ===========================================================================
# SECTION 3 — PHYSICAL CONSTANTS
# ===========================================================================
WHEELBASE_M: float  = 0.23    # front-to-rear axle distance (m)
LANE_WIDTH_M: float = 0.35    # one-lane physical width (m)

# ===========================================================================
# SECTION 4 — LANE POSITIONING OFFSETS  (pixels in BEV space)
# ===========================================================================
RIGHT_LANE_OFFSET_PX: int  =  70   # default right-lane bias from image centre
DUAL_OFFSET_PX: int        =   0   # extra when BOTH lines visible (0 = centred)
SINGLE_DIV_OFFSET_PX: int  =  40   # GHOST_R: shift right when only divider visible
SINGLE_EDGE_OFFSET_PX: int = -40   # GHOST_L: shift left when only edge visible

# ===========================================================================
# SECTION 5 — LANE TRACKING  (HybridLaneTracker)
# ===========================================================================
TRACKER_NWINDOWS: int          = 9
TRACKER_SW_MARGIN: int         = 60     # sliding-window half-width (px)
TRACKER_MINPIX: int            = 50     # min px to recenter a window
TRACKER_POLY_MARGIN_BASE: int  = 60     # normal polynomial search band (px)
TRACKER_POLY_MARGIN_CURV: int  = 120    # wider band when curvature is high
TRACKER_MIN_PIX_OK: int        = 200    # min px to accept a line
TRACKER_EMA_ALPHA: float       = 0.50   # polynomial EMA weight (higher = more responsive)
TRACKER_STALE_FIT_FRAMES: int  = 5      # frames to hold stale polynomial before dropping
TRACKER_HIST_SPLIT: float      = 0.50   # FIX 22: symmetric lane search midpoint
TRACKER_MIN_LANE_WIDTH_PX: int = 80     # FIX 27: lower bound for _width_sane
TRACKER_MAX_LANE_WIDTH_PX: int = 560    # FIX 27: upper bound for _width_sane
TRACKER_DEFAULT_LANE_WIDTH_PX: int = 280 # FIX 41: default for trackbar/init

# ── JunctionDetector ───────────────────────────────────────────────────────
# BFMC loop runs at ~20-30fps, so we use real-time seconds for timers.
JCT_ENTRY_SECONDS: float      = 0.17    # ~5 frames at 30fps
JCT_EXIT_SECONDS: float       = 0.27    # ~8 frames at 30fps
JCT_CROSS_ENERGY_RATIO: float = 1.4
JCT_WIDTH_RATIO_HIGH: float   = 1.6
JCT_MIN_BOT_ENERGY: int       = 500
JCT_MAX_SECONDS: float        = 5.0     # hard exit timeout

# ── RoundaboutNavigator ────────────────────────────────────────────────────
RBT_ENTRY_WIDTH_RATIO: float  = 0.60
RBT_ENTRY_SECONDS: float      = 0.17    # debounce before entering
RBT_EXIT_WIDTH_RATIO: float   = 0.82
RBT_MIN_CIRCLE_SECONDS: float = 0.83    # ~25 frames
RBT_MAX_CIRCLE_SECONDS: float = 4.0     # ~120 frames
RBT_SPEED_SCALE: float        = 0.50
RBT_LOOKAHEAD_SCALE: float    = 0.55

# ===========================================================================
# SECTION 6 — STEERING CONTROL
# ===========================================================================
# EMA weights
STEER_EMA_SLOW: float = 0.25   # straights — very smooth
STEER_EMA_FAST: float = 0.50   # sharp turns — responsive (>8° error threshold)
STEER_EMA_SWITCH_DEG: float = 8.0   # abs(error) above this → use FAST

GUARD_EMA: float       = 0.55   # EMA weight for divider-guard correction
STEER_DEADBAND: float  = 0.5    # degrees — suppress tiny changes to servo

# Hard limits
MAX_STEER: float      = 25.0   # degrees — matches STM32 serial_handler hardware clamp (±25)
MAX_STEER_RATE: float = 5.0    # degrees per frame — rate limiter

# DividerGuard
GUARD_DIVIDER_SAFE_PX: int  = 55     # min gap: car centre → centre divider (px)
GUARD_EDGE_SAFE_PX: int     = 50     # min gap: car centre → outer edge (px)
GUARD_GAIN: float           = 0.09   # proportional correction gain
GUARD_MAX_CORR: float       = 8.0    # max correction magnitude (degrees)
GUARD_DEADBAND_PX: int      = 5      # ignore gaps smaller than this

# ===========================================================================
# SECTION 7 — SPEED POLICY
# ===========================================================================
HIGH_CURV_THRESH: float = 0.003
MED_CURV_THRESH: float  = 0.0015
HIGH_CURV_SCALE: float  = 0.60
MED_CURV_SCALE: float   = 0.80
DUAL_SPEED_SCALE: float = 1.15   # speed boost when both lines visible on straight

# Safety Caps / Floors (FIX 13, 15)
# Speed values sent to STM32 are throttle units 0–200, not km/h. 
# Calibrate against measured wheel velocity if encoder feedback is available.
HIGHWAY_MAX_SPEED: int      = 130   # prevents compound-multiplier overshoot
MOTOR_STALL_THRESHOLD: int  = 15    # minimum throttle to prevent silent motor stall

JUNCTION_SPEED_SCALE: float = 0.55
HIGH_STEER_SCALE: float     = 0.60   # abs(steer) > HIGH_STEER_DEG
MED_STEER_SCALE: float      = 0.80   # abs(steer) > MED_STEER_DEG
HIGH_STEER_DEG: float       = 18.0
MED_STEER_DEG: float        = 10.0
DUAL_MAX_STEER_DEG: float   = 10.0   # threshold for DUAL speed boost

# Look-ahead scaling per nav state
LA_ROUNDABOUT_SCALE: float = 0.55
LA_JUNCTION_SCALE: float   = 0.75
LA_HIGH_CURV_SCALE: float  = 0.60
LA_MED_CURV_SCALE: float   = 0.80
LA_MIN_PX: int             = 60      # absolute minimum look-ahead pixels

# ===========================================================================
# SECTION 8 — LOST-LANE POLICY
# ===========================================================================
LOST_GRACE_SECONDS: float = 0.27    # ~8 frames at 30fps
LOST_STOP: bool           = False   # True = full stop after grace; False = creep
LOST_CREEP_SPEED: float   = 25.0   # speed while creeping (LOST_STOP=False)

# ===========================================================================
# SECTION 9 — SERIAL CONFIG
# ===========================================================================
SERIAL_PORT: str  = "/dev/ttyACM0"   # override with --config or env var
SERIAL_BAUD: int  = 115200

# ===========================================================================
# SECTION 10 — AI MODEL PATHS
# All .pt files live in  bosch/models/
# Set to empty string "" to disable a detector gracefully.
# ===========================================================================
MODELS_DIR = PROJECT_ROOT / "models"

# FIX 1 — Updated model filenames to match actual directory content.
# Using 'traffic_light_small.pt' as default for better reliability than nano.
MODEL_TRAFFIC_LIGHT: str  = str(MODELS_DIR / "traffic_light_small.pt")  # ~22 MB
# MODEL_TRAFFIC_LIGHT: str = str(MODELS_DIR / "traffic_light_nano.pt")   # ~6 MB
# MODEL_TRAFFIC_LIGHT: str = str(MODELS_DIR / "traffic_light.pt")        # ~50 MB

# ── Road sign / highway sign detector ─────────────────────────────────────
# ENSEMBLE: both models run simultaneously — highest confidence result is used.
#   road_sign.pt    (YOLOv8n, v1 weights — good recall)
#   road_sign_v2.pt (YOLOv8n, v2 weights — better precision)
MODEL_ROAD_SIGN: str    = str(MODELS_DIR / "road_sign.pt")
MODEL_ROAD_SIGN_V2: str = str(MODELS_DIR / "road_sign_v2.pt")

# ── Obstacle detector ─────────────────────────────────────────────────────
MODEL_OBSTACLE: str       = ""   # empty = disabled

# ── Lane divider (optional AI supplement — CV is primary) ─────────────────
MODEL_LANE_DIVIDER: str   = str(MODELS_DIR / "lane_divider.pt")

# FIX 6 — BFMC miniature signs require full resolution — do not reduce below 416×416
AI_INFER_W: int = 640
AI_INFER_H: int = 480

# AI detector target frame-rate
AI_FPS: int = 8   # detector thread run rate (Hz)

# FIX 38 — Watchdog on background threads (seconds)
WATCHDOG_TIMEOUT_S: float = 2.0

# Confidence thresholds (0–1)
CONF_TRAFFIC_LIGHT: float = 0.30   # raised from 0.20 — reduces false RED stops (BFMC Competition Fix)
CONF_ROAD_SIGN: float     = 0.25   # catch small sign models
CONF_LANE_DIVIDER: float  = 0.40
CONF_OBSTACLE: float      = 0.40

# Sign class names
SIGN_CLASSES: dict[str, int] = {
    "HIGHWAY_ENTRY":   0,
    "ZEBRA_CROSSING":  1,
    "STOP_SIGN":       2,
    "HIGHWAY_EXIT":    3,
    "PARKING":         4,
    "ONE_WAY":         5,
}

# ===========================================================================
# SECTION 11 — BEHAVIOR ENGINE THRESHOLDS
# ===========================================================================
# Traffic light
TL_DEBOUNCE_SECONDS: float      = 0.17    # confirm state (5 frames at 30fps)
TL_GREEN_CONFIRM_SECONDS: float = 0.27    # confirmation to resume (8 frames)

# Zebra crossing
ZEBRA_APPROACH_SPEED_SCALE: float = 0.40
ZEBRA_CLEAR_SECONDS: float        = 1.0   # resume after 1s of clear sign/obstacle

# Obstacle / detour
DETOUR_OFFSET_PX: int     = 80
DETOUR_HOLD_SECONDS: float = 1.5   # ~45 frames
DETOUR_RAMP_SECONDS: float = 0.67  # ~20 frames

# Highway mode
HIGHWAY_SPEED_FACTOR: float = 1.40
HIGHWAY_HOLD_SECONDS: float = 8.0    # Increased from 5.0s — longer tracks need more time

# Stop sign
STOP_SIGN_HOLD_SECONDS: float    = 3.0
STOP_SIGN_DEBOUNCE_SECONDS: float = 0.17  # Same pattern as TL debounce — prevents 1-frame false stops

# Lane divider advisory (AI supplemental — CV is primary)
DIVIDER_ADVISORY_OFFSET_PX: int = 20  # small lateral nudge when divider detected near centre

# Robustness: grace period for dropped detection frames
DETECTION_GRACE_SECONDS: float = 0.10  # allow ~3 missed frames before resetting stop timers

# ===========================================================================
# SECTION 12 — DASHBOARD
# ===========================================================================
# BEV car-centre x — single source of truth for DividerGuard, LookAhead, etc.
CAR_X_BEV: int = BEV_W // 2

DASH_W: int   = 1280   # total dashboard window width
DASH_H: int   = 720    # total dashboard window height

# Panel proportions (fractions of DASH_W)
DASH_LEFT_FRAC: float   = 0.33
DASH_CENTER_FRAC: float = 0.34
DASH_RIGHT_FRAC: float  = 0.33

DASH_FPS: int = 15    # dashboard refresh rate

# Colors (BGR)
COLOR_GREEN:  tuple[int, int, int] = (0, 220, 80)
COLOR_RED:    tuple[int, int, int] = (0, 0, 220)
COLOR_YELLOW: tuple[int, int, int] = (0, 200, 220)
COLOR_CYAN:   tuple[int, int, int] = (200, 220, 0)
COLOR_ORANGE: tuple[int, int, int] = (0, 140, 255)
COLOR_GREY:   tuple[int, int, int] = (120, 120, 120)
COLOR_WHITE:  tuple[int, int, int] = (240, 240, 240)
COLOR_BG:     tuple[int, int, int] = (20, 20, 28)     # dark background
COLOR_PANEL:  tuple[int, int, int] = (32, 34, 44)     # panel background

# Anchor state colors
ANCHOR_COLORS: dict[str, tuple[int, int, int]] = {
    "DUAL":     COLOR_GREEN,
    "GHOST_L":  COLOR_YELLOW,
    "GHOST_R":  COLOR_ORANGE,
    "LOST":     COLOR_RED,
    "JCT_EDGE": COLOR_CYAN,
    "JCT_DIV":  COLOR_CYAN,
    "JCT_LOST": COLOR_RED,
    "RBT_INNER": COLOR_CYAN,
    "RBT_OUTER": COLOR_CYAN,
    "RBT_LOST":  COLOR_RED,
}

# Sign display duration (seconds)
DASH_SIGN_DISPLAY_SEC: float = 3.0

# Speed gauge display (dashboard uses these via getattr)
DASH_MAX_SPEED: int       = 200       # throttle units (0-200)
DASH_SPEED_LABEL: str     = "thr"     # label shown under the speed number

# ===========================================================================
# SECTION 13 — HONK / HORN
# Serial command string sent to STM32 for horn
# ===========================================================================
HONK_SERIAL_CMD: str = "HONK"


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# STARTUP VALIDATION (FIX 1)
# ---------------------------------------------------------------------------
def validate_model_paths() -> bool:
    """Check that all configured model paths exist on disk. Logs errors if not."""
    import logging
    l = logging.getLogger("config")
    all_ok = True
    for name, path_str in [
        ("Traffic Light", MODEL_TRAFFIC_LIGHT),
        ("Road Sign V1", MODEL_ROAD_SIGN),
        ("Road Sign V2", MODEL_ROAD_SIGN_V2),
        ("Lane Divider", MODEL_LANE_DIVIDER),
    ]:
        if not path_str: continue
        p = Path(path_str)
        if not p.exists():
            l.error("FIX 1 CRITICAL: %s model MISSING! Expected at: %s", name, p.absolute())
            all_ok = False
    if all_ok:
        l.info("AI Model Paths: ALL VALIDATED")
    return all_ok


if __name__ == "__main__":
    print("config.py — sanity check")
    validate_model_paths()
    print(f"  CAM: {CAM_W}x{CAM_H} @ {TARGET_FPS}fps")
    print(f"  BEV: {BEV_W}x{BEV_H}")
    print(f"  AI models: traffic={MODEL_TRAFFIC_LIGHT}")
    print("  All constants loaded OK.")
