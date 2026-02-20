"""
steering_controller.py — BFMC Autonomous Car System
====================================================
All steering and speed logic extracted from BFMC_Pilot.run().

Classes
-------
PurePursuitController  — converts (target_x, look_ahead, lane_width) → steer_angle°
SteeringRateLimiter    — caps per-frame steering change (prevents servo overshoot)
SteeringEMA            — dual-speed EMA for smooth / responsive tracking
DividerGuard           — hard safety layer: pushes car away from divider / edge
SpeedPolicy            — rules-based speed selection from nav state + behavior cmd
"""

from __future__ import annotations

import math
import logging
from typing import Optional

import numpy as np

import config
from behavior_engine import BehaviorCommand, BehaviorMode   # noqa: TC002 (TYPE_CHECKING fine here)

log = logging.getLogger(__name__)


# ===========================================================================
# DIVIDER GUARD
# ===========================================================================
class DividerGuard:
    """
    Hard safety layer — runs every frame.

    If the car is within DIVIDER_SAFE_PX of the centre divider or within
    EDGE_SAFE_PX of the outer edge, a proportional steering correction is
    applied to push the car back into the safe zone.

    The divider (safety-critical) takes priority when both fire together.
    The correction is EMA-smoothed by the caller to prevent spikes.
    """

    DIVIDER_SAFE_PX = config.GUARD_DIVIDER_SAFE_PX
    EDGE_SAFE_PX    = config.GUARD_EDGE_SAFE_PX
    GAIN            = config.GUARD_GAIN
    MAX_CORR        = config.GUARD_MAX_CORR
    DEADBAND_PX     = config.GUARD_DEADBAND_PX

    def apply(
        self,
        steer_angle: float,
        left_fit:  Optional[np.ndarray],
        right_fit: Optional[np.ndarray],
        y_eval: int = 440,
        car_x:  int = 320,
    ) -> tuple[float, float, bool]:
        """
        Parameters
        ----------
        steer_angle : current steering angle (degrees)
        left_fit    : smoothed left polynomial (None = ignore)
        right_fit   : smoothed right polynomial (None = ignore)
        y_eval      : BEV row at which to evaluate the polynomial
        car_x       : assumed car centre x (pixels) in BEV

        Returns
        -------
        (corrected_steer, speed_scale, triggered)
        """
        correction  = 0.0
        speed_scale = 1.0
        triggered   = False

        div_corr = 0.0
        if left_fit is not None:
            div_x = float(np.polyval(left_fit, y_eval))
            gap   = car_x - div_x
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err      = float(self.DIVIDER_SAFE_PX - gap)
                div_corr = min(self.GAIN * err, self.MAX_CORR)
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 120.0))
                triggered   = True

        edge_corr = 0.0
        if right_fit is not None:
            edge_x = float(np.polyval(right_fit, y_eval))
            gap    = edge_x - car_x
            if gap < self.EDGE_SAFE_PX - self.DEADBAND_PX:
                err       = float(self.EDGE_SAFE_PX - gap)
                edge_corr = min(self.GAIN * err, self.MAX_CORR)
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 120.0))
                triggered   = True

        # Divider priority: if both fire, net correction favours divider side
        if div_corr > 0 and edge_corr > 0:
            correction = max(div_corr - edge_corr, self.DEADBAND_PX * self.GAIN)
        else:
            correction = div_corr - edge_corr

        return steer_angle + correction, speed_scale, triggered


# ===========================================================================
# PURE PURSUIT CONTROLLER
# ===========================================================================
class PurePursuitController:
    """
    Converts a BEV target pixel position to a steering angle using the
    Pure Pursuit geometric steering law.
    """

    def compute_steer(
        self,
        target_x:     float,
        look_ahead_px: int,
        lane_width_px: int,
    ) -> float:
        """
        Parameters
        ----------
        target_x      : x-pixel in BEV the car should head toward
        look_ahead_px : look-ahead distance (pixels)
        lane_width_px : measured lane width in BEV pixels (for scaling)

        Returns
        -------
        steer_angle : degrees (positive = right)
        """
        lane_width_px = max(lane_width_px, 50)                 # guard ÷0
        ppm   = lane_width_px / config.LANE_WIDTH_M            # pixels per metre
        dx    = target_x - float(config.BEV_W // 2)
        dy    = max(float(look_ahead_px), 1.0)
        ld    = math.sqrt(dx * dx + dy * dy)
        alpha = math.atan2(dx, dy)
        wb_px = config.WHEELBASE_M * ppm
        steer = math.atan2(2.0 * wb_px * math.sin(alpha), ld)
        return math.degrees(steer)


# ===========================================================================
# STEERING RATE LIMITER
# ===========================================================================
class SteeringRateLimiter:
    """
    Caps the per-frame change in steering angle to MAX_STEER_RATE degrees.
    Prevents servo overshoot on sharp transitions.
    """

    MAX_RATE = config.MAX_STEER_RATE

    def __init__(self) -> None:
        self._prev: float = 0.0

    def apply(self, raw: float) -> float:
        """Apply rate limiting and return the clamped steering angle."""
        delta       = raw - self._prev
        delta       = max(-self.MAX_RATE, min(self.MAX_RATE, delta))
        limited     = self._prev + delta
        self._prev  = limited
        return limited

    def reset(self) -> None:
        self._prev = 0.0


# ===========================================================================
# STEERING EMA
# ===========================================================================
class SteeringEMA:
    """
    Dual-speed EMA: uses a faster weight when the correction error is large
    (responsive on sharp turns) and a slower weight on straights.
    """

    def __init__(self) -> None:
        self._smooth: float = 0.0

    def update(self, raw: float) -> float:
        err   = abs(raw - self._smooth)
        alpha = (config.STEER_EMA_FAST if err > config.STEER_EMA_SWITCH_DEG
                 else config.STEER_EMA_SLOW)
        self._smooth = alpha * raw + (1.0 - alpha) * self._smooth
        return self._smooth

    def reset(self) -> None:
        self._smooth = 0.0


# ===========================================================================
# SPEED POLICY
# ===========================================================================
class SpeedPolicy:
    """
    Rules-based speed selector.

    Priority (highest first):
      1. behavior_command overrides  (FULL_STOP → 0, SLOW → multiplier, HIGHWAY → multiplier)
      2. Lost-lane grace period
      3. nav_state  (roundabout / junction)
      4. Curvature
      5. Anchor mode + steer angle
    """

    def compute_speed(
        self,
        base_speed:      float,
        nav_state:       str,
        anchor:          str,
        steer_angle:     float,
        curvature:       float,
        lost_frames:     int,
        behavior_cmd:    Optional["BehaviorCommand"] = None,
        guard_on:        bool  = False,
        guard_spd:       float = 1.0,
    ) -> float:
        """
        Returns the final speed value (0–200).
        """
        # --- Behavior overrides ---
        if behavior_cmd is not None:
            if behavior_cmd.mode == BehaviorMode.FULL_STOP:
                return 0.0
            effective_base = base_speed * behavior_cmd.speed_multiplier
        else:
            effective_base = base_speed

        # --- Lost-lane hard stop ---
        if config.LOST_STOP and lost_frames > config.LOST_GRACE_FRAMES:
            return 0.0

        # --- Base = 0 ---
        if effective_base == 0:
            return 0.0

        # --- Nav state ---
        if nav_state == "ROUNDABOUT":
            speed = effective_base * config.RBT_SPEED_SCALE
        elif nav_state == "JUNCTION":
            speed = effective_base * config.JUNCTION_SPEED_SCALE
        # --- Curvature ---
        elif curvature > config.HIGH_CURV_THRESH:
            speed = effective_base * config.HIGH_CURV_SCALE
        elif curvature > config.MED_CURV_THRESH:
            speed = effective_base * config.MED_CURV_SCALE
        # --- Anchor + steer ---
        elif anchor == "DUAL" and abs(steer_angle) < config.DUAL_MAX_STEER_DEG:
            speed = effective_base * config.DUAL_SPEED_SCALE
        elif abs(steer_angle) > config.HIGH_STEER_DEG:
            speed = effective_base * config.HIGH_STEER_SCALE
        elif abs(steer_angle) > config.MED_STEER_DEG:
            speed = effective_base * config.MED_STEER_SCALE
        else:
            speed = effective_base

        # --- Lost-lane creep / slow-down ---
        if 0 < lost_frames <= config.LOST_GRACE_FRAMES:
            frac  = min(lost_frames / max(config.LOST_GRACE_FRAMES, 1), 1.0)
            speed = max(config.LOST_CREEP_SPEED, speed * (1.0 - frac * 0.70))

        # --- Guard speed penalty ---
        if guard_on:
            speed *= guard_spd

        return float(max(0.0, min(200.0, speed)))


# ===========================================================================
# CONVENIENCE BUNDLE
# ===========================================================================
class SteeringController:
    """
    Thin façade that bundles all four steering sub-systems into one object,
    mirroring the per-frame logic from BFMC_Pilot.run().

    Usage in main.py::

        ctrl = SteeringController()
        steer = ctrl.compute_steer(target_x, eff_la, lane_width_px)
        steer = ctrl.apply_guard(steer, sl, sr, y_eval)
        speed = ctrl.compute_speed(base_speed, nav_state, anchor,
                                   steer, curvature, lost_frames, behavior_cmd)
    """

    def __init__(self) -> None:
        self._pp      = PurePursuitController()
        self._ema     = SteeringEMA()
        self._limiter = SteeringRateLimiter()
        self._guard   = DividerGuard()
        self._policy  = SpeedPolicy()

        self._smooth_guard: float = 0.0   # EMA of guard correction

    def compute_steer(
        self,
        target_x:     float,
        look_ahead_px: int,
        lane_width_px: int,
    ) -> float:
        """Pure pursuit → EMA smooth → rate-limited steering angle."""
        raw     = self._pp.compute_steer(target_x, look_ahead_px, lane_width_px)
        smooth  = self._ema.update(raw)
        limited = self._limiter.apply(smooth)
        return float(max(-config.MAX_STEER, min(config.MAX_STEER, limited)))

    def apply_guard(
        self,
        steer_angle: float,
        left_fit:    Optional[np.ndarray],
        right_fit:   Optional[np.ndarray],
        y_eval:      int,
        lost:        bool,
    ) -> tuple[float, float, bool]:
        """
        Apply DividerGuard with EMA smoothing.

        Returns (guarded_steer, guard_speed_scale, guard_triggered).
        """
        raw_guarded, guard_spd, guard_on = self._guard.apply(
            steer_angle, left_fit, right_fit, y_eval=y_eval
        )

        if lost:
            self._smooth_guard = 0.0
            guard_on = False
        else:
            g_delta            = raw_guarded - steer_angle
            self._smooth_guard = (config.GUARD_EMA * g_delta
                                  + (1.0 - config.GUARD_EMA) * self._smooth_guard)

        steer_angle += self._smooth_guard
        steer_angle  = float(max(-config.MAX_STEER, min(config.MAX_STEER, steer_angle)))
        return steer_angle, guard_spd, guard_on

    def compute_speed(
        self,
        base_speed:   float,
        nav_state:    str,
        anchor:       str,
        steer_angle:  float,
        curvature:    float,
        lost_frames:  int,
        behavior_cmd: Optional["BehaviorCommand"] = None,
        guard_on:     bool  = False,
        guard_spd:    float = 1.0,
    ) -> float:
        return self._policy.compute_speed(
            base_speed, nav_state, anchor, steer_angle,
            curvature, lost_frames, behavior_cmd, guard_on, guard_spd,
        )

    def reset(self) -> None:
        """Reset all stateful components (e.g., after a full-stop event)."""
        self._ema.reset()
        self._limiter.reset()
        self._smooth_guard = 0.0


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    ctrl = SteeringController()

    # Target at image centre → steer should be ~0
    s = ctrl.compute_steer(320.0, 150, 280)
    assert abs(s) < 1.0, f"Expected ~0 steer, got {s}"

    # Target right of centre → steer should be positive
    s_right = ctrl.compute_steer(400.0, 150, 280)
    assert s_right > 0, f"Expected positive steer, got {s_right}"

    # Speed at 0 base → 0
    spd = ctrl.compute_speed(0, "NORMAL", "DUAL", 0.0, 0.0, 0)
    assert spd == 0.0

    # Speed FULL_STOP behavior
    from behavior_engine import BehaviorCommand, BehaviorMode
    cmd  = BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0,
                           lane_offset_override=None, hold_frames=0)
    spd  = ctrl.compute_speed(50, "NORMAL", "DUAL", 0.0, 0.0, 0, cmd)
    assert spd == 0.0

    print("steering_controller smoke-test PASSED")
