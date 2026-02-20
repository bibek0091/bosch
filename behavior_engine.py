"""
behavior_engine.py — BFMC Autonomous Car System
================================================
Priority-ordered state machine that translates AI vision signals into
driving behavior commands.

Priority order (highest first):
  1. Traffic Light   (RED/DARK → FULL_STOP)
  2. Zebra Crossing  (slow → stop → detour)
  3. Highway Entry   (speed up)
  4. Stop Sign       (timed full stop)

Outputs a BehaviorCommand dataclass consumed by SteeringController.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import config

log = logging.getLogger(__name__)


# ===========================================================================
# SHARED ENUMS & DATACLASSES
# ===========================================================================

class BehaviorMode(Enum):
    NORMAL    = auto()
    FULL_STOP = auto()
    SLOW      = auto()
    HIGHWAY   = auto()
    DETOUR    = auto()
    HONK      = auto()


@dataclass
class BehaviorCommand:
    mode:                BehaviorMode
    speed_multiplier:    float        = 1.0
    lane_offset_override: Optional[int] = None
    hold_frames:         int          = 0


# ---------------------------------------------------------------------------
# Vision state enums / dataclasses (also imported by vision_ai)
# ---------------------------------------------------------------------------

class TrafficLightState(Enum):
    NONE   = auto()   # no traffic light detected
    GREEN  = auto()
    YELLOW = auto()
    RED    = auto()
    DARK   = auto()   # light visible but all dark → treat as RED


class ObstacleSide(Enum):
    NONE   = auto()
    LEFT   = auto()
    RIGHT  = auto()
    CENTER = auto()


@dataclass
class SignDetection:
    sign_type:  str
    confidence: float
    bbox:       tuple[int, int, int, int]   # x1, y1, x2, y2


@dataclass
class LaneDividerDetection:
    x_position: float   # estimated BEV x of divider
    divider_type: str   # "solid" | "dashed" | "unknown"
    confidence: float


@dataclass
class ObstacleDetection:
    present:        bool
    bbox:           Optional[tuple[int, int, int, int]]
    estimated_side: ObstacleSide


@dataclass
class VisionState:
    """
    Shared state written by vision_ai threads, read by BehaviorEngine.
    Protected by threading.Lock (caller must acquire before read/write).
    """
    import threading as _threading

    traffic_light:  TrafficLightState      = field(default=TrafficLightState.NONE)
    sign:           Optional[SignDetection] = field(default=None)
    lane_divider:   Optional[LaneDividerDetection] = field(default=None)
    obstacle:       Optional[ObstacleDetection]    = field(default=None)
    lock:           _threading.Lock                = field(default_factory=_threading.Lock)


# ===========================================================================
# BEHAVIOR ENGINE
# ===========================================================================
class BehaviorEngine:
    """
    Reads VisionState and current nav_state; outputs BehaviorCommand.

    Usage::

        engine = BehaviorEngine()
        cmd = engine.update(vision_state, nav_state="NORMAL")
    """

    # Internal state enum for the engine itself
    class _EngineState(Enum):
        NORMAL         = auto()
        RED_STOP       = auto()
        YELLOW_SLOW    = auto()
        ZEBRA_APPROACH = auto()
        ZEBRA_STOP     = auto()
        ZEBRA_DETOUR   = auto()
        HIGHWAY        = auto()
        STOP_SIGN_HOLD = auto()

    def __init__(self) -> None:
        self._state = self._EngineState.NORMAL

        # Debounce / hold counters
        self._red_debounce:    int = 0    # consecutive RED/DARK frames
        self._green_confirm:   int = 0    # consecutive GREEN frames (while stopped)
        self._yellow_debounce: int = 0

        self._stop_sign_frames:   int = 0
        self._highway_frames:     int = 0
        self._zebra_clear_frames: int = 0

        # Last sign timestamp (for display timeout)
        self._last_sign_time: float = 0.0
        self._last_sign: Optional[SignDetection] = None

        # Honk state
        self._honk_active: bool = False

        log.info("BehaviorEngine initialised")

    def update(
        self,
        vision_state: VisionState,
        nav_state:    str = "NORMAL",
    ) -> BehaviorCommand:
        """
        One-shot update; returns the current BehaviorCommand.
        Thread-safe: acquires vision_state.lock internally.
        """
        with vision_state.lock:
            tl       = vision_state.traffic_light
            sign     = vision_state.sign
            obstacle = vision_state.obstacle

        return self._tick(tl, sign, obstacle, nav_state)

    # ------------------------------------------------------------------
    # PRIVATE — STATE MACHINE
    # ------------------------------------------------------------------

    def _tick(
        self,
        tl:       TrafficLightState,
        sign:     Optional[SignDetection],
        obstacle: Optional[ObstacleDetection],
        nav_state: str,
    ) -> BehaviorCommand:

        # ----------------------------------------------------------------
        # PRIORITY 1 — TRAFFIC LIGHT
        # ----------------------------------------------------------------
        cmd = self._handle_traffic_light(tl, nav_state)
        if cmd is not None:
            return cmd

        # ----------------------------------------------------------------
        # PRIORITY 2 — ZEBRA CROSSING
        # ----------------------------------------------------------------
        cmd = self._handle_zebra(sign, obstacle)
        if cmd is not None:
            return cmd

        # ----------------------------------------------------------------
        # PRIORITY 3 — HIGHWAY ENTRY
        # ----------------------------------------------------------------
        cmd = self._handle_highway(sign, nav_state)
        if cmd is not None:
            return cmd

        # ----------------------------------------------------------------
        # PRIORITY 4 — STOP SIGN
        # ----------------------------------------------------------------
        cmd = self._handle_stop_sign(sign)
        if cmd is not None:
            return cmd

        # ----------------------------------------------------------------
        # No override — normal driving
        # ----------------------------------------------------------------
        self._state = self._EngineState.NORMAL
        return BehaviorCommand(mode=BehaviorMode.NORMAL, speed_multiplier=1.0)

    # ------------------------------------------------------------------

    def _handle_traffic_light(
        self, tl: TrafficLightState, nav_state: str
    ) -> Optional[BehaviorCommand]:

        is_stop_color = tl in (TrafficLightState.RED, TrafficLightState.DARK)
        is_green      = tl == TrafficLightState.GREEN
        is_yellow     = tl == TrafficLightState.YELLOW

        # --- Already stopped for RED ---
        if self._state == self._EngineState.RED_STOP:
            if is_green:
                self._green_confirm += 1
                if self._green_confirm >= config.TL_GREEN_CONFIRM_FRAMES:
                    log.info("BehaviorEngine: GREEN confirmed — resuming")
                    self._state        = self._EngineState.NORMAL
                    self._green_confirm = 0
                    self._red_debounce  = 0
                    return None   # release to lower priorities
            else:
                self._green_confirm = 0   # reset if not consistently green
            return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

        # --- Debounce RED/DARK ---
        if is_stop_color:
            self._red_debounce += 1
            if self._red_debounce >= config.TL_DEBOUNCE_FRAMES:
                self._state         = self._EngineState.RED_STOP
                self._green_confirm = 0
                log.info("BehaviorEngine: RED/DARK confirmed — FULL_STOP")
                return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)
            # Not yet confirmed — fall through
        else:
            self._red_debounce = 0

        # --- YELLOW slow ---
        if is_yellow:
            self._yellow_debounce += 1
            if self._yellow_debounce >= config.TL_DEBOUNCE_FRAMES:
                return BehaviorCommand(
                    mode=BehaviorMode.SLOW,
                    speed_multiplier=0.50,
                )
        else:
            self._yellow_debounce = 0

        return None

    # ------------------------------------------------------------------

    def _handle_zebra(
        self,
        sign:     Optional[SignDetection],
        obstacle: Optional[ObstacleDetection],
    ) -> Optional[BehaviorCommand]:

        zebra_visible = (
            sign is not None
            and sign.sign_type == "ZEBRA_CROSSING"
        )

        if self._state not in (
            self._EngineState.ZEBRA_APPROACH,
            self._EngineState.ZEBRA_STOP,
            self._EngineState.ZEBRA_DETOUR,
        ):
            if not zebra_visible:
                return None
            # Newly detected zebra
            self._state              = self._EngineState.ZEBRA_APPROACH
            self._zebra_clear_frames = 0
            log.info("BehaviorEngine: ZEBRA_CROSSING detected — approaching slowly")

        # --- In ZEBRA_APPROACH ---
        if self._state == self._EngineState.ZEBRA_APPROACH:
            has_obstacle = obstacle is not None and obstacle.present

            if not has_obstacle:
                # FIX: count frames where zebra sign is gone (car has passed)
                # not just frames where there's no obstacle (wrong condition)
                if not zebra_visible:
                    self._zebra_clear_frames += 1
                else:
                    self._zebra_clear_frames = 0   # still approaching — reset
                if self._zebra_clear_frames > config.ZEBRA_CLEAR_FRAMES:
                    log.info("BehaviorEngine: zebra cleared — resuming normal speed")
                    self._state = self._EngineState.NORMAL
                    return None
                return BehaviorCommand(
                    mode=BehaviorMode.SLOW,
                    speed_multiplier=config.ZEBRA_APPROACH_SPEED_SCALE,
                )

            # Obstacle detected — stop and honk
            self._state              = self._EngineState.ZEBRA_STOP
            self._zebra_clear_frames = 0
            self._honk_active        = True
            log.info("BehaviorEngine: obstacle on zebra — FULL_STOP + HONK")

        # --- In ZEBRA_STOP ---
        if self._state == self._EngineState.ZEBRA_STOP:
            has_obstacle = obstacle is not None and obstacle.present

            if not has_obstacle:
                self._zebra_clear_frames += 1
                if self._zebra_clear_frames > config.ZEBRA_CLEAR_FRAMES:
                    self._state       = self._EngineState.NORMAL
                    self._honk_active = False
                    log.info("BehaviorEngine: obstacle cleared — resuming")
                    return None
            else:
                self._zebra_clear_frames = 0
                side = obstacle.estimated_side if obstacle else ObstacleSide.CENTER

                # Determine detour offset
                if side == ObstacleSide.LEFT:
                    offset = config.DETOUR_OFFSET_PX   # go right
                    self._state = self._EngineState.ZEBRA_DETOUR
                    return BehaviorCommand(
                        mode=BehaviorMode.DETOUR,
                        speed_multiplier=config.ZEBRA_APPROACH_SPEED_SCALE,
                        lane_offset_override=offset,
                    )
                elif side == ObstacleSide.RIGHT:
                    offset = -config.DETOUR_OFFSET_PX  # go left
                    self._state = self._EngineState.ZEBRA_DETOUR
                    return BehaviorCommand(
                        mode=BehaviorMode.DETOUR,
                        speed_multiplier=config.ZEBRA_APPROACH_SPEED_SCALE,
                        lane_offset_override=offset,
                    )
                else:
                    # CENTER — remain stopped and honk
                    return BehaviorCommand(
                        mode=BehaviorMode.FULL_STOP,
                        speed_multiplier=0.0,
                    )

            return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

        # --- In ZEBRA_DETOUR ---
        if self._state == self._EngineState.ZEBRA_DETOUR:
            has_obstacle = obstacle is not None and obstacle.present
            if not has_obstacle:
                self._state = self._EngineState.NORMAL
                self._honk_active = False
                return None

        return None

    # ------------------------------------------------------------------

    def _handle_highway(
        self,
        sign:      Optional[SignDetection],
        nav_state: str,
    ) -> Optional[BehaviorCommand]:

        is_highway_sign = (
            sign is not None and sign.sign_type == "HIGHWAY_ENTRY"
        )

        if self._state == self._EngineState.HIGHWAY:
            self._highway_frames += 1
            exit_sign = sign is not None and sign.sign_type == "HIGHWAY_EXIT"
            if self._highway_frames > config.HIGHWAY_HOLD_FRAMES or exit_sign:
                self._state = self._EngineState.NORMAL
                log.info("BehaviorEngine: HIGHWAY_MODE ended")
                return None
            return BehaviorCommand(
                mode=BehaviorMode.HIGHWAY,
                speed_multiplier=config.HIGHWAY_SPEED_FACTOR,
            )

        # Only activate highway mode during NORMAL driving
        if is_highway_sign and nav_state == "NORMAL":
            self._state          = self._EngineState.HIGHWAY
            self._highway_frames = 0
            log.info("BehaviorEngine: HIGHWAY_ENTRY detected — boosting speed")
            return BehaviorCommand(
                mode=BehaviorMode.HIGHWAY,
                speed_multiplier=config.HIGHWAY_SPEED_FACTOR,
            )

        return None

    # ------------------------------------------------------------------

    def _handle_stop_sign(
        self,
        sign: Optional[SignDetection],
    ) -> Optional[BehaviorCommand]:

        is_stop = sign is not None and sign.sign_type == "STOP_SIGN"

        if self._state == self._EngineState.STOP_SIGN_HOLD:
            self._stop_sign_frames += 1
            if self._stop_sign_frames >= config.STOP_SIGN_HOLD_FRAMES:
                self._state = self._EngineState.NORMAL
                log.info("BehaviorEngine: STOP_SIGN hold complete — resuming")
                return None
            return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

        if is_stop:
            self._state            = self._EngineState.STOP_SIGN_HOLD
            self._stop_sign_frames = 0
            log.info("BehaviorEngine: STOP_SIGN detected — holding")
            return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

        return None

    @property
    def honk_active(self) -> bool:
        """True when the engine wants the horn active."""
        return self._honk_active


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    engine = BehaviorEngine()
    vs     = VisionState()

    # 1. Normal — no vision signals
    cmd = engine.update(vs, "NORMAL")
    assert cmd.mode == BehaviorMode.NORMAL, f"Expected NORMAL, got {cmd.mode}"

    # 2. RED light — needs TL_DEBOUNCE_FRAMES consecutive RED frames
    n = config.TL_DEBOUNCE_FRAMES
    for _ in range(n):
        with vs.lock:
            vs.traffic_light = TrafficLightState.RED
        cmd = engine.update(vs, "NORMAL")
    assert cmd.mode == BehaviorMode.FULL_STOP, f"Expected FULL_STOP, got {cmd.mode}"

    # 3. GREEN after RED — needs TL_GREEN_CONFIRM_FRAMES consecutive GREEN frames
    for _ in range(config.TL_GREEN_CONFIRM_FRAMES):
        with vs.lock:
            vs.traffic_light = TrafficLightState.GREEN
        cmd = engine.update(vs, "NORMAL")
    # After confirmation, next normal tick should be NORMAL
    with vs.lock:
        vs.traffic_light = TrafficLightState.NONE
    cmd = engine.update(vs, "NORMAL")
    assert cmd.mode == BehaviorMode.NORMAL, f"Expected NORMAL after GREEN, got {cmd.mode}"

    print("behavior_engine smoke-test PASSED")
