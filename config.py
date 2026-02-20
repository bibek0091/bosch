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

# JunctionDetector
JCT_ENTRY_FRAMES: int        = 5
JCT_EXIT_FRAMES: int         = 8
JCT_CROSS_ENERGY_RATIO: float = 1.4
JCT_WIDTH_RATIO_HIGH: float  = 1.6
JCT_MIN_BOT_ENERGY: int      = 500
JCT_MAX_FRAMES: int          = 150     # hard exit timeout

# RoundaboutNavigator
RBT_ENTRY_WIDTH_RATIO: float = 0.60
RBT_ENTRY_FRAMES: int        = 5       # consecutive frames required to enter roundabout
RBT_EXIT_WIDTH_RATIO: float  = 0.82
RBT_MIN_CIRCLE_FRAMES: int   = 25
RBT_MAX_CIRCLE_FRAMES: int   = 120
RBT_SPEED_SCALE: float       = 0.50
RBT_LOOKAHEAD_SCALE: float   = 0.55

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
MAX_STEER: float      = 30.0   # degrees (serial_handler clamps ±25, keep ≤25 on hardware)
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
LOST_GRACE_FRAMES: int   = 8      # frames before stopping
LOST_STOP: bool          = False   # True = full stop after grace; False = creep
LOST_CREEP_SPEED: float  = 25.0   # speed while creeping (LOST_STOP=False)

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

# ── Traffic light detector ─────────────────────────────────────────────────
# Three weight variants — swap by changing the active line.
MODEL_TRAFFIC_LIGHT: str  = str(MODELS_DIR / "traffic_light.pt")        # med  ~50 MB  (default)
# MODEL_TRAFFIC_LIGHT: str = str(MODELS_DIR / "traffic_light_small.pt")  # small ~22 MB
# MODEL_TRAFFIC_LIGHT: str = str(MODELS_DIR / "traffic_light_nano.pt")   # nano  ~6 MB (fastest for Pi)

# ── Road sign / highway sign detector ─────────────────────────────────────
# Two variants: best.pt (v1) and last.pt (v2) — both detect highway signs.
# Use v1 by default; swap to v2 to compare.
MODEL_ROAD_SIGN: str      = str(MODELS_DIR / "road_sign.pt")     # from best.pt  ~6 MB  (v1, default)
# MODEL_ROAD_SIGN: str     = str(MODELS_DIR / "road_sign_v2.pt")  # from last.pt  ~6 MB  (v2 alternative)

# ── Obstacle detector ─────────────────────────────────────────────────────
# No dedicated obstacle .pt model supplied yet — detector disabled.
# Drop an obstacle.pt into models/ to enable it automatically.
MODEL_OBSTACLE: str       = ""   # empty = disabled

# ── Lane divider (optional AI supplement — CV is primary) ─────────────────
MODEL_LANE_DIVIDER: str   = str(MODELS_DIR / "lane_divider.pt")  # disabled if file missing

# Inference resize (smaller = faster; bboxes are projected back to CAM_W x CAM_H)
AI_INFER_W: int = 320
AI_INFER_H: int = 240

# AI detector target frame-rate (separate from control loop fps).
# YOLO on Pi 5 achieves ~8-12 fps — set lower than that to avoid CPU starvation.
# Increase if Pi handles it; decrease if main loop is starved.
AI_FPS: int = 8   # detector thread run rate (Hz)

# Confidence thresholds (0–1)
# BFMC small-scale models at 1-2m distance produce lower scores than real-world training.
# Lower thresholds to catch dim/partial views of miniature traffic lights and signs.
# If too many false positives appear, raise CONF_TRAFFIC_LIGHT to 0.30 first.
CONF_TRAFFIC_LIGHT: float = 0.20   # lowered from 0.40 — small TL models at close range
CONF_ROAD_SIGN: float     = 0.25   # lowered from 0.50 — small sign models
CONF_LANE_DIVIDER: float  = 0.40
CONF_OBSTACLE: float      = 0.40

# Sign class names (must match model's class indices)
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
# Traffic light debounce
TL_DEBOUNCE_FRAMES: int     = 5    # N consecutive frames to confirm state
TL_GREEN_CONFIRM_FRAMES: int = 8   # M consecutive GREEN frames to resume after RED

# Zebra crossing
ZEBRA_APPROACH_SPEED_SCALE: float = 0.40  # slow down when sign seen
ZEBRA_CLEAR_FRAMES: int           = 30    # frames with no obstacle → resume

# Obstacle / detour
DETOUR_OFFSET_PX: int    = 80    # lane offset when detouring around obstacle
DETOUR_HOLD_FRAMES: int  = 45    # min frames to hold offset after obstacle clears
DETOUR_RAMP_FRAMES: int  = 20    # frames to smoothly ramp offset back to 0

# Highway mode
HIGHWAY_SPEED_FACTOR: float = 1.40   # multiply base_speed in HIGHWAY_MODE
HIGHWAY_HOLD_FRAMES: int    = 150    # maintain highway mode for N frames

# Stop sign
STOP_SIGN_HOLD_FRAMES: int = 90     # full stop duration (frames at 30fps ≈ 3s)

# ===========================================================================
# SECTION 12 — DASHBOARD
# ===========================================================================
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

# ===========================================================================
# SECTION 13 — HONK / HORN
# Serial command string sent to STM32 for horn
# ===========================================================================
HONK_SERIAL_CMD: str = "HONK"


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("config.py — sanity check")
    print(f"  CAM: {CAM_W}x{CAM_H} @ {TARGET_FPS}fps")
    print(f"  BEV: {BEV_W}x{BEV_H}")
    print(f"  SRC_PTS: {SRC_PTS.tolist()}")
    print(f"  Models dir: {PROJECT_ROOT / 'models'}")
    print(f"  AI models: traffic={MODEL_TRAFFIC_LIGHT}")
    print("  All constants loaded OK.")
