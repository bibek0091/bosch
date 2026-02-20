"""
behavior_engine.py — BFMC Autonomous Car System
================================================
Priority-ordered state machine that translates AI vision signals into
driving behavior commands.

Priority order (highest first):
  1. Traffic Light   (RED/DARK → FULL_STOP, YELLOW → SLOW, GREEN confirm)
  2. Zebra Crossing  (slow → stop → detour → honk)
  3. Highway Entry   (speed up, exit on sign/TL/stop_sign)
  4. Stop Sign       (debounced timed full stop)
  5. Lane Divider    (advisory lateral offset — AI-supplemental)

Outputs a BehaviorCommand dataclass consumed by SteeringController.

BFMC Competition Fixes:
  - Stop sign debounce (STOP_SIGN_DEBOUNCE_SECONDS): prevents 1-frame false stops
  - Highway safety exits on RED light or STOP_SIGN while in HIGHWAY mode
  - Lane divider advisory offset wired in from VisionState
  - All timers use time.monotonic() (real-time, frame-rate independent)
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

        # ── Traffic light timers ─────────────────────────────────────────
        self._red_start:            float = 0.0
        self._green_confirm_start:  float = 0.0
        self._yellow_start:         float = 0.0

        # ── Stop sign timers ─────────────────────────────────────────────
        self._stop_sign_start:      float = 0.0
        self._stop_sign_debounce:   float = 0.0   # Competition Fix: debounce timer

        # ── Highway timers ───────────────────────────────────────────────
        self._highway_start_time:   float = 0.0

        # ── Zebra timers ─────────────────────────────────────────────────
        self._zebra_clear_start:    float = 0.0
        self._zebra_obstacle_start: float = 0.0

        # ── Last sign tracker ────────────────────────────────────────────
        self._last_sign_time: float = 0.0
        self._last_sign: Optional[SignDetection] = None

        # ── Honk state ───────────────────────────────────────────────────
        self._honk_active: bool = False

        log.info("BehaviorEngine initialised (Competition Edition)")

    @property
    def state_name(self) -> str:
        """Public property for state inspection."""
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
            divider  = vision_state.lane_divider   # Competition Fix: wired in

        return self._tick(tl, sign, obstacle, divider, nav_state)

    def _tick(
        self,
        tl:       TrafficLightState,
        sign:     Optional[SignDetection],
        obstacle: Optional[ObstacleDetection],
        divider:  Optional[LaneDividerDetection],
        nav_state: str,
    ) -> BehaviorCommand:

        # Reset honk every frame; handlers set it True as needed
        self._honk_active = False

        # PRIORITY 1 — TRAFFIC LIGHT (highest: RED/DARK = stop no matter what)
        cmd = self._handle_traffic_light(tl, nav_state)
        if cmd is not None:
            return cmd

        # PRIORITY 2 — ZEBRA CROSSING
        cmd = self._handle_zebra(sign, obstacle)
        if cmd is not None:
            return cmd

        # PRIORITY 3 — HIGHWAY ENTRY/MODE
        cmd = self._handle_highway(sign, tl, nav_state)
        if cmd is not None:
            return cmd

        # PRIORITY 4 — STOP SIGN (debounced)
        cmd = self._handle_stop_sign(sign)
        if cmd is not None:
            return cmd

        # PRIORITY 5 — LANE DIVIDER ADVISORY (lowest: only a nudge, not a stop)
        cmd = self._handle_divider_advisory(divider)
        if cmd is not None:
            return cmd

        # NORMAL DRIVING
        self._state = self._EngineState.NORMAL
        return BehaviorCommand(mode=BehaviorMode.NORMAL, speed_multiplier=1.0)

    # -----------------------------------------------------------------------
    # HANDLER 1 — TRAFFIC LIGHT
    # -----------------------------------------------------------------------
    def _handle_traffic_light(
        self, tl: TrafficLightState, nav_state: str
    ) -> Optional[BehaviorCommand]:

        is_stop_color = tl in (TrafficLightState.RED, TrafficLightState.DARK)
        is_green      = tl == TrafficLightState.GREEN
        is_yellow     = tl == TrafficLightState.YELLOW
        now = time.monotonic()

        # ── Already stopped for RED ──────────────────────────────────────
        if self._state == self._EngineState.RED_STOP:
            if is_green:
                if self._green_confirm_start == 0:
                    self._green_confirm_start = now
                elif now - self._green_confirm_start >= config.TL_GREEN_CONFIRM_SECONDS:
                    log.info("BehaviorEngine: GREEN confirmed — resuming")
                    self._state               = self._EngineState.NORMAL
                    self._green_confirm_start = 0.0
                    self._red_start           = 0.0
                    return None
            else:
                self._green_confirm_start = 0.0

            # Kill honk if a red light interrupts a zebra stop
            self._honk_active = False
            return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

        # ── Confirm RED/DARK (debounced) ─────────────────────────────────
        if is_stop_color:
            if self._red_start == 0:
                self._red_start = now
            elif now - self._red_start >= config.TL_DEBOUNCE_SECONDS:
                self._state               = self._EngineState.RED_STOP
                self._green_confirm_start = 0.0
                log.info("BehaviorEngine: RED/DARK confirmed — FULL_STOP")
                self._honk_active = False
                return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)
        else:
            self._red_start = 0.0

        # ── YELLOW — slow down ───────────────────────────────────────────
        if is_yellow:
            if self._yellow_start == 0:
                self._yellow_start = now
            elif now - self._yellow_start >= config.TL_DEBOUNCE_SECONDS:
                return BehaviorCommand(mode=BehaviorMode.SLOW, speed_multiplier=0.50)
        else:
            self._yellow_start = 0.0

        return None

    # -----------------------------------------------------------------------
    # HANDLER 2 — ZEBRA CROSSING
    # -----------------------------------------------------------------------
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
            self._zebra_clear_start = 0.0
            log.info("BehaviorEngine: ZEBRA_CROSSING detected — approaching slowly")

        # Approach
        if self._state == self._EngineState.ZEBRA_APPROACH:
            has_obstacle = obstacle is not None and obstacle.present
            if not has_obstacle:
                if not zebra_visible:
                    if self._zebra_clear_start == 0:
                        self._zebra_clear_start = now
                else:
                    self._zebra_clear_start = 0.0

                if (self._zebra_clear_start > 0 and
                        now - self._zebra_clear_start > config.ZEBRA_CLEAR_SECONDS):
                    log.info("BehaviorEngine: zebra cleared — resuming")
                    self._state = self._EngineState.NORMAL
                    return None

                return BehaviorCommand(mode=BehaviorMode.SLOW,
                                       speed_multiplier=config.ZEBRA_APPROACH_SPEED_SCALE)

            # Obstacle detected → stop
            self._state = self._EngineState.ZEBRA_STOP
            self._zebra_clear_start    = 0.0
            self._zebra_obstacle_start = now
            self._honk_active = True
            log.info("BehaviorEngine: obstacle on zebra — ZEBRA_STOP (side=%s)",
                     obstacle.estimated_side.name)
            return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

        # Stop
        if self._state == self._EngineState.ZEBRA_STOP:
            self._honk_active = True
            has_obstacle = obstacle is not None and obstacle.present

            if not has_obstacle:
                self._zebra_obstacle_start = 0.0
                if self._zebra_clear_start == 0:
                    self._zebra_clear_start = now
                if now - self._zebra_clear_start > config.ZEBRA_CLEAR_SECONDS:
                    self._state       = self._EngineState.NORMAL
                    self._honk_active = False
                    log.info("BehaviorEngine: zebra obstacle cleared — resuming")
                    return None
            else:
                self._zebra_clear_start = 0.0
                # Debounce before detour
                if now - self._zebra_obstacle_start < config.TL_DEBOUNCE_SECONDS:
                    return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

                side = obstacle.estimated_side if obstacle else ObstacleSide.CENTER
                if side in (ObstacleSide.LEFT, ObstacleSide.RIGHT):
                    offset = (config.DETOUR_OFFSET_PX if side == ObstacleSide.LEFT
                              else -config.DETOUR_OFFSET_PX)
                    self._state = self._EngineState.ZEBRA_DETOUR
                    log.info("BehaviorEngine: detouring to the %s",
                             "RIGHT" if side == ObstacleSide.LEFT else "LEFT")
                    return BehaviorCommand(mode=BehaviorMode.DETOUR,
                                          speed_multiplier=config.ZEBRA_APPROACH_SPEED_SCALE,
                                          lane_offset_override=offset)
                else:
                    return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

            return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

        # Detour
        if self._state == self._EngineState.ZEBRA_DETOUR:
            if obstacle is None or not obstacle.present:
                self._state       = self._EngineState.NORMAL
                self._honk_active = False
                return None
            offset = (config.DETOUR_OFFSET_PX if obstacle.estimated_side == ObstacleSide.LEFT
                      else -config.DETOUR_OFFSET_PX)
            return BehaviorCommand(mode=BehaviorMode.DETOUR,
                                   speed_multiplier=config.ZEBRA_APPROACH_SPEED_SCALE,
                                   lane_offset_override=offset)

        return None

    # -----------------------------------------------------------------------
    # HANDLER 3 — HIGHWAY
    # -----------------------------------------------------------------------
    def _handle_highway(
        self,
        sign:      Optional[SignDetection],
        tl:        TrafficLightState,
        nav_state: str,
    ) -> Optional[BehaviorCommand]:
        is_entry = (sign is not None and sign.sign_type == "HIGHWAY_ENTRY")
        is_exit  = (sign is not None and sign.sign_type == "HIGHWAY_EXIT")
        now = time.monotonic()

        if self._state == self._EngineState.HIGHWAY:
            timeout_exit = now - self._highway_start_time > config.HIGHWAY_HOLD_SECONDS
            sign_exit    = is_exit
            # Competition Fix: also exit highway mode on RED light or STOP sign
            safety_exit  = (tl in (TrafficLightState.RED, TrafficLightState.DARK) or
                            (sign is not None and sign.sign_type == "STOP_SIGN"))

            if timeout_exit or sign_exit or safety_exit:
                reason = ("timeout" if timeout_exit else
                          "STOP_SIGN/RED" if safety_exit else "EXIT sign")
                self._state = self._EngineState.NORMAL
                log.info("BehaviorEngine: HIGHWAY_MODE ended (%s)", reason)
                return None
            return BehaviorCommand(mode=BehaviorMode.HIGHWAY,
                                   speed_multiplier=config.HIGHWAY_SPEED_FACTOR)

        if is_entry and nav_state == "NORMAL":
            self._state              = self._EngineState.HIGHWAY
            self._highway_start_time = now
            log.info("BehaviorEngine: HIGHWAY detected — boosting speed x%.2f",
                     config.HIGHWAY_SPEED_FACTOR)
            return BehaviorCommand(mode=BehaviorMode.HIGHWAY,
                                   speed_multiplier=config.HIGHWAY_SPEED_FACTOR)

        return None

    # -----------------------------------------------------------------------
    # HANDLER 4 — STOP SIGN (debounced)
    # -----------------------------------------------------------------------
    def _handle_stop_sign(self, sign: Optional[SignDetection]) -> Optional[BehaviorCommand]:
        is_stop = sign is not None and sign.sign_type == "STOP_SIGN"
        now = time.monotonic()

        if self._state == self._EngineState.STOP_SIGN_HOLD:
            if now - self._stop_sign_start >= config.STOP_SIGN_HOLD_SECONDS:
                self._state           = self._EngineState.NORMAL
                self._stop_sign_debounce = 0.0
                log.info("BehaviorEngine: STOP_SIGN hold complete — resuming")
                return None
            return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

        if is_stop:
            # Competition Fix: debounce the stop sign — require a sustained detection
            if self._stop_sign_debounce == 0.0:
                self._stop_sign_debounce = now
                log.debug("BehaviorEngine: STOP_SIGN debounce started (conf=%.2f)",
                          sign.confidence)
                return None  # Don't commit to stop yet
            elif now - self._stop_sign_debounce >= config.STOP_SIGN_DEBOUNCE_SECONDS:
                self._state              = self._EngineState.STOP_SIGN_HOLD
                self._stop_sign_start    = now
                self._stop_sign_debounce = 0.0
                log.info("BehaviorEngine: STOP_SIGN confirmed — holding %.1fs",
                         config.STOP_SIGN_HOLD_SECONDS)
                return BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)
            # Within debounce window — still tracking
            return None
        else:
            # Sign disappeared — reset debounce
            self._stop_sign_debounce = 0.0

        return None

    # -----------------------------------------------------------------------
    # HANDLER 5 — LANE DIVIDER ADVISORY (Competition Fix: wired in)
    # -----------------------------------------------------------------------
    def _handle_divider_advisory(
        self, divider: Optional[LaneDividerDetection]
    ) -> Optional[BehaviorCommand]:
        """
        If the AI lane-divider detector sees the centre divider unusually close,
        apply a small advisory offset to nudge the car right.
        This is supplemental to CV lane tracking — it acts like a soft correction,
        not a full-stop.
        """
        if divider is None or divider.confidence < config.CONF_LANE_DIVIDER:
            return None

        # Only nudge if divider is left of BEV centre (i.e., car is drifting left)
        bev_mid = float(config.CAR_X_BEV)
        if divider.x_position < bev_mid - 20:
            offset = config.DIVIDER_ADVISORY_OFFSET_PX
            log.debug("BehaviorEngine: divider advisory +%dpx (divider x=%.0f)",
                      offset, divider.x_position)
            return BehaviorCommand(
                mode=BehaviorMode.NORMAL,
                speed_multiplier=1.0,
                lane_offset_override=offset,
            )
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
    assert cmd.mode == BehaviorMode.NORMAL, f"Expected NORMAL, got {cmd.mode}"
    print("  [PASS] Normal driving")

    # 2. RED light (real-time debounce)
    with vs.lock: vs.traffic_light = TrafficLightState.RED
    engine.update(vs, "NORMAL")
    time.sleep(config.TL_DEBOUNCE_SECONDS + 0.05)
    cmd = engine.update(vs, "NORMAL")
    assert cmd.mode == BehaviorMode.FULL_STOP, f"Expected FULL_STOP, got {cmd.mode}"
    print("  [PASS] RED light -> FULL_STOP")

    # 3. GREEN confirm
    with vs.lock: vs.traffic_light = TrafficLightState.GREEN
    engine.update(vs, "NORMAL")
    time.sleep(config.TL_GREEN_CONFIRM_SECONDS + 0.05)
    cmd = engine.update(vs, "NORMAL")
    assert engine._state == engine._EngineState.NORMAL, f"Expected NORMAL, got {engine._state}"
    print("  [PASS] GREEN confirm -> resume")

    # 4. Stop sign debounce (should NOT stop on first frame)
    with vs.lock:
        vs.traffic_light = TrafficLightState.NONE
        vs.sign = SignDetection("STOP_SIGN", 0.90, (0, 0, 50, 50))
    cmd = engine.update(vs, "NORMAL")
    assert cmd.mode == BehaviorMode.NORMAL, f"Debounce failed — stopped on frame 1"
    print("  [PASS] Stop sign debounce (no stop on frame 1)")

    # 5. Stop sign confirmed after debounce
    time.sleep(config.STOP_SIGN_DEBOUNCE_SECONDS + 0.05)
    cmd = engine.update(vs, "NORMAL")
    assert cmd.mode == BehaviorMode.FULL_STOP, f"Expected FULL_STOP after debounce"
    print("  [PASS] Stop sign confirmed -> FULL_STOP after debounce")

    # 6. Highway safety exit on RED
    with vs.lock:
        vs.sign = SignDetection("HIGHWAY_ENTRY", 0.85, (0, 0, 50, 50))
        vs.traffic_light = TrafficLightState.NONE
    engine._state = engine._EngineState.NORMAL
    cmd = engine.update(vs, "NORMAL")
    assert cmd.mode == BehaviorMode.HIGHWAY, f"Expected HIGHWAY, got {cmd.mode}"
    with vs.lock:
        vs.traffic_light = TrafficLightState.RED
        vs.sign = None
    # First call starts the RED debounce timer (highway exits immediately on safety_exit check)
    engine.update(vs, "NORMAL")
    time.sleep(config.TL_DEBOUNCE_SECONDS + 0.05)
    cmd = engine.update(vs, "NORMAL")
    # RED should now preempt and produce FULL_STOP
    assert cmd.mode == BehaviorMode.FULL_STOP, f"Expected RED FULL_STOP, got {cmd.mode}"
    print("  [PASS] RED light preempts highway mode -> FULL_STOP")


    print("\nbehavior_engine smoke-test PASSED (Competition Edition)")
