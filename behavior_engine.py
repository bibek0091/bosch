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
import threading
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
    traffic_light:  TrafficLightState      = field(default=TrafficLightState.NONE)
    sign:           Optional[SignDetection] = field(default=None)
    lane_divider:   Optional[LaneDividerDetection] = field(default=None)
    obstacle:       Optional[ObstacleDetection]    = field(default=None)
    lock:           threading.Lock                 = field(default_factory=threading.Lock)


# ===========================================================================
# BEHAVIOR ENGINE
# ===========================================================================
class BehaviorEngine:
    """
    Reads VisionState and current nav_state; outputs BehaviorCommand.
    """

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

        # Timestamps for real-time transitions (Fix 37)
        self._red_start:     float = 0.0
        self._green_confirm_start: float = 0.0
        self._yellow_start:  float = 0.0

        self._stop_sign_start:   float = 0.0
        self._highway_start_time: float = 0.0
        self._zebra_clear_start: float = 0.0
        self._zebra_obstacle_start: float = 0.0  # Fix 10 debounce

        # Last sign tracker
        self._last_sign_time: float = 0.0
        self._last_sign: Optional[SignDetection] = None

        # Honk state
        self._honk_active: bool = False

        log.info("BehaviorEngine initialised")

    @property
    def state_name(self) -> str:
        """Fix 18: public property for state inspection."""
        return self._state.name

    @property
    def honk_active(self) -> bool:
        """True when the engine wants the horn active."""
        return self._honk_active

    def update(
        self,
        vision_state: VisionState,
        nav_state:    str = "NORMAL",
    ) -> BehaviorCommand:
        with vision_state.lock:
            tl       = vision_state.traffic_light
            sign     = vision_state.sign
            obstacle = vision_state.obstacle

        return self._tick(tl, sign, obstacle, nav_state)

    def _tick(
        self,
        tl:       TrafficLightState,
        sign:     Optional[SignDetection],
        obstacle: Optional[ObstacleDetection],
        nav_state: str,
    ) -> BehaviorCommand:

        # Fix 9: default reset to False, logic sets it True if needed
        self._honk_active = False

        # PRIORITY 1 — TRAFFIC LIGHT
        cmd = self._handle_traffic_light(tl, nav_state)
        if cmd is not None:
            return cmd

        # PRIORITY 2 — ZEBRA CROSSING
        cmd = self._handle_zebra(sign, obstacle)
        if cmd is not None:
            return cmd

        # PRIORITY 3 — HIGHWAY ENTRY
        cmd = self._handle_highway(sign, nav_state)
        if cmd is not None:
            return cmd

        # PRIORITY 4 — STOP SIGN
        cmd = self._handle_stop_sign(sign)
        if cmd is not None:
            return cmd

        # NORMAL DRIVING
        self._state = self._EngineState.NORMAL
        return BehaviorCommand(mode=BehaviorMode.NORMAL, speed_multiplier=1.0)

    def _handle_traffic_light(
        self, tl: TrafficLightState, nav_state: str
    ) -> Optional[BehaviorCommand]:

        is_stop_color = tl in (TrafficLightState.RED, TrafficLightState.DARK)
        is_green      = tl == TrafficLightState.GREEN
        is_yellow     = tl == TrafficLightState.YELLOW
        now = time.monotonic()

        # Stopped for RED
        if self._state == self._EngineState.RED_STOP:
            if is_green:
                if self._green_confirm_start == 0:
                    self._green_confirm_start = now
                elif now - self._green_confirm_start >= config.TL_GREEN_CONFIRM_SECONDS:
                    log.info("BehaviorEngine: GREEN confirmed — resuming")
                    self._state = self._EngineState.NORMAL
                    self._green_confirm_start = 0
                    self._red_start = 0
                    return None
            else:
                self._green_confirm_start = 0
            
            # Fix 9: Kill honk if stopped by red light
            if self._honk_active:
                log.info("BehaviorEngine: Traffic light preempted zebra stop — clearing honk")
                self._honk_active = False

            return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

        # Confirm RED/DARK (Fix 37 real-time)
        if is_stop_color:
            if self._red_start == 0:
                self._red_start = now
            elif now - self._red_start >= config.TL_DEBOUNCE_SECONDS:
                self._state = self._EngineState.RED_STOP
                self._green_confirm_start = 0
                log.info("BehaviorEngine: RED/DARK confirmed — FULL_STOP")
                # Fix 9: also reset honk here if transition occurs
                self._honk_active = False
                return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)
        else:
            self._red_start = 0

        # YELLOW slow
        if is_yellow:
            if self._yellow_start == 0:
                self._yellow_start = now
            elif now - self._yellow_start >= config.TL_DEBOUNCE_SECONDS:
                return BehaviorCommand(mode=BehaviorMode.SLOW, speed_multiplier=0.50)
        else:
            self._yellow_start = 0

        return None

    def _handle_zebra(
        self,
        sign:     Optional[SignDetection],
        obstacle: Optional[ObstacleDetection],
    ) -> Optional[BehaviorCommand]:

        zebra_visible = (sign is not None and sign.sign_type == "ZEBRA_CROSSING")
        now = time.monotonic()

        if self._state not in (self._EngineState.ZEBRA_APPROACH, 
                               self._EngineState.ZEBRA_STOP, 
                               self._EngineState.ZEBRA_DETOUR):
            if not zebra_visible:
                return None
            self._state = self._EngineState.ZEBRA_APPROACH
            self._zebra_clear_start = 0
            log.info("BehaviorEngine: ZEBRA_CROSSING detected — approaching slowly")

        # Approach
        if self._state == self._EngineState.ZEBRA_APPROACH:
            has_obstacle = obstacle is not None and obstacle.present
            if not has_obstacle:
                if not zebra_visible:
                    if self._zebra_clear_start == 0: self._zebra_clear_start = now
                else:
                    self._zebra_clear_start = 0
                
                if (self._zebra_clear_start > 0 and 
                    now - self._zebra_clear_start > config.ZEBRA_CLEAR_SECONDS):
                    log.info("BehaviorEngine: zebra cleared — resuming")
                    self._state = self._EngineState.NORMAL
                    return None
                    
                return BehaviorCommand(mode=BehaviorMode.SLOW, 
                                      speed_multiplier=config.ZEBRA_APPROACH_SPEED_SCALE)

            # Obstacle detected — enter stop state
            self._state = self._EngineState.ZEBRA_STOP
            self._zebra_clear_start = 0
            self._zebra_obstacle_start = now  # Fix 10: start debounce timer
            self._honk_active = True
            log.info("BehaviorEngine: obstacle on zebra — entering ZEBRA_STOP")
            # Fix 10: return immediately to avoid fall-through detour
            return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

        # Stop
        if self._state == self._EngineState.ZEBRA_STOP:
            self._honk_active = True
            has_obstacle = obstacle is not None and obstacle.present

            if not has_obstacle:
                self._zebra_obstacle_start = 0
                if self._zebra_clear_start == 0: self._zebra_clear_start = now
                if now - self._zebra_clear_start > config.ZEBRA_CLEAR_SECONDS:
                    self._state = self._EngineState.NORMAL
                    self._honk_active = False
                    log.info("BehaviorEngine: obstacle cleared — resuming")
                    return None
            else:
                self._zebra_clear_start = 0
                # Fix 10: Debounce obstacle detection before detour
                if now - self._zebra_obstacle_start < config.TL_DEBOUNCE_SECONDS:
                    return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

                side = obstacle.estimated_side if obstacle else ObstacleSide.CENTER
                if side in (ObstacleSide.LEFT, ObstacleSide.RIGHT):
                    offset = config.DETOUR_OFFSET_PX if side == ObstacleSide.LEFT else -config.DETOUR_OFFSET_PX
                    self._state = self._EngineState.ZEBRA_DETOUR
                    log.info(f"BehaviorEngine: detouring to the {'RIGHT' if side==ObstacleSide.LEFT else 'LEFT'}")
                    return BehaviorCommand(mode=BehaviorMode.DETOUR,
                                          speed_multiplier=config.ZEBRA_APPROACH_SPEED_SCALE,
                                          lane_offset_override=offset)
                else:
                    return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

            return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

        # Detour
        if self._state == self._EngineState.ZEBRA_DETOUR:
            if obstacle is None or not obstacle.present:
                self._state = self._EngineState.NORMAL
                self._honk_active = False
                return None
            # Maintain detour offset (handled by logic above)
            offset = config.DETOUR_OFFSET_PX if obstacle.estimated_side == ObstacleSide.LEFT else -config.DETOUR_OFFSET_PX
            return BehaviorCommand(mode=BehaviorMode.DETOUR, 
                                  speed_multiplier=config.ZEBRA_APPROACH_SPEED_SCALE,
                                  lane_offset_override=offset)

        return None

    def _handle_highway(self, sign: Optional[SignDetection], nav_state: str) -> Optional[BehaviorCommand]:
        is_entry = (sign is not None and sign.sign_type == "HIGHWAY_ENTRY")
        now = time.monotonic()

        if self._state == self._EngineState.HIGHWAY:
            exit_sign = sign is not None and sign.sign_type == "HIGHWAY_EXIT"
            if now - self._highway_start_time > config.HIGHWAY_HOLD_SECONDS or exit_sign:
                self._state = self._EngineState.NORMAL
                log.info("BehaviorEngine: HIGHWAY_MODE ended")
                return None
            return BehaviorCommand(mode=BehaviorMode.HIGHWAY, speed_multiplier=config.HIGHWAY_SPEED_FACTOR)

        if is_entry and nav_state == "NORMAL":
            self._state = self._EngineState.HIGHWAY
            self._highway_start_time = now
            log.info("BehaviorEngine: HIGHWAY detected — boosting speed")
            return BehaviorCommand(mode=BehaviorMode.HIGHWAY, speed_multiplier=config.HIGHWAY_SPEED_FACTOR)

        return None

    def _handle_stop_sign(self, sign: Optional[SignDetection]) -> Optional[BehaviorCommand]:
        is_stop = sign is not None and sign.sign_type == "STOP_SIGN"
        now = time.monotonic()

        if self._state == self._EngineState.STOP_SIGN_HOLD:
            if now - self._stop_sign_start >= config.STOP_SIGN_HOLD_SECONDS:
                self._state = self._EngineState.NORMAL
                log.info("BehaviorEngine: STOP_SIGN hold complete — resuming")
                return None
            return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

        if is_stop:
            self._state = self._EngineState.STOP_SIGN_HOLD
            self._stop_sign_start = now
            log.info("BehaviorEngine: STOP_SIGN detected — holding")
            return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

        return None


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    engine = BehaviorEngine()
    vs     = VisionState()

    # 1. Normal
    cmd = engine.update(vs, "NORMAL")
    assert cmd.mode == BehaviorMode.NORMAL

    # 2. RED light real-time
    with vs.lock: vs.traffic_light = TrafficLightState.RED
    engine.update(vs, "NORMAL")
    time.sleep(config.TL_DEBOUNCE_SECONDS + 0.1)
    cmd = engine.update(vs, "NORMAL")
    assert cmd.mode == BehaviorMode.FULL_STOP

    # 3. GREEN confirm real-time
    with vs.lock: vs.traffic_light = TrafficLightState.GREEN
    engine.update(vs, "NORMAL")
    time.sleep(config.TL_GREEN_CONFIRM_SECONDS + 0.1)
    cmd = engine.update(vs, "NORMAL")
    assert engine._state == engine._EngineState.NORMAL

    print("behavior_engine smoke-test PASSED")
