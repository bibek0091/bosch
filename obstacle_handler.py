"""
obstacle_handler.py — BFMC Autonomous Car System
=================================================
Computes a lateral lane offset to steer around an obstacle detected.
Implements hysteresis so detouring is stable and smooth.

Logic:
  - When an obstacle is detected: compute offset based on obstacle side
  - Hold offset for DETOUR_HOLD_SECONDS after obstacle disappears (prevent oscillation)
  - Ramp offset smoothly back to zero over DETOUR_RAMP_SECONDS (smooth exit)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import config
from behavior_engine import ObstacleDetection, ObstacleSide, BehaviorCommand, BehaviorMode

log = logging.getLogger(__name__)


class ObstacleHandler:
    """
    Stateful detour offset calculator.
    """

    def __init__(self) -> None:
        self._state      = "CLEAR"
        self._side       = 0   # -1=left, 1=right
        self._start_time = 0.0 # timestamp of current state start
        self._ramp_start_offset = 0.0
        self.heartbeat   = time.monotonic()

        log.info("ObstacleHandler initialised")

    def compute_offset(
        self,
        obstacle:  Optional[ObstacleDetection],
        lane_width_px: int = 280,
    ) -> tuple[float, Optional[BehaviorCommand]]:
        """
        Returns (lateral_offset_px, optional_behavior_command).
        Fix 21: Returns BehaviorMode.FULL_STOP if obstacle is centered.
        """
        now = time.monotonic()
        self.heartbeat = now

        has_obstacle = obstacle is not None and obstacle.present
        cmd: Optional[BehaviorCommand] = None

        # State Transitions
        if self._state == "CLEAR":
            if has_obstacle:
                self._state = "DETOURING"
                self._side  = 1 if obstacle.estimated_side == ObstacleSide.LEFT else -1
                log.info("Obstacle DETOUR started (side=%s)", obstacle.estimated_side)

        elif self._state == "DETOURING":
            if not has_obstacle:
                self._state      = "HOLDING"
                self._start_time = now
                log.debug("Obstacle cleared, entering HOLD")
            else:
                # Update side if obstacle moves
                self._side = 1 if obstacle.estimated_side == ObstacleSide.LEFT else -1

        elif self._state == "HOLDING":
            if has_obstacle:
                self._state = "DETOURING"
                self._side  = 1 if obstacle.estimated_side == ObstacleSide.LEFT else -1
            elif now - self._start_time >= config.DETOUR_HOLD_SECONDS:
                self._state             = "RAMPING"
                self._start_time        = now
                self._ramp_start_offset = float(self._side * config.DETOUR_OFFSET_PX)
                log.debug("HOLD timeout, entering RAMP")

        elif self._state == "RAMPING":
            if has_obstacle:
                self._state = "DETOURING"
                self._side  = 1 if obstacle.estimated_side == ObstacleSide.LEFT else -1
            elif now - self._start_time >= config.DETOUR_RAMP_SECONDS:
                self._state = "CLEAR"
                self._side  = 0
                log.debug("Detour RAMP complete")

        # Offset Calculation
        offset = 0.0
        if self._state == "DETOURING":
            offset = float(self._side * config.DETOUR_OFFSET_PX)
        elif self._state == "HOLDING":
            offset = float(self._side * config.DETOUR_OFFSET_PX)
        elif self._state == "RAMPING":
            dt   = now - self._start_time
            frac = dt / max(config.DETOUR_RAMP_SECONDS, 0.01)
            frac = min(max(frac, 0.0), 1.0)
            offset = self._ramp_start_offset * (1.0 - frac)

        # Fix 19, 21: Priority stopping if centered
        if has_obstacle and obstacle.estimated_side == ObstacleSide.CENTER:
            cmd = BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)
            log.warning("PRIORITY STOP: Obstacle centred")

        return offset, cmd

    def reset(self) -> None:
        """Immediately zero the offset."""
        self._state      = "CLEAR"
        self._side       = 0
        self._start_time = 0.0
        log.info("ObstacleHandler RESET")

    @property
    def state(self) -> str:
        """Current handler state: CLEAR | DETOURING | HOLDING | RAMPING"""
        return self._state


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    handler = ObstacleHandler()

    # No obstacle → offset = 0
    res, cmd = handler.compute_offset(None, 280)
    assert res == 0 and cmd is None

    # Obstacle on LEFT → positive offset
    obs_left = ObstacleDetection(present=True, bbox=(0,0,10,10), estimated_side=ObstacleSide.LEFT)
    res, cmd = handler.compute_offset(obs_left, 280)
    assert res == config.DETOUR_OFFSET_PX
    assert handler.state == "DETOURING"

    print("obstacle_handler smoke-test PASSED")
