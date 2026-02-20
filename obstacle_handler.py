"""
obstacle_handler.py — BFMC Autonomous Car System
=================================================
Computes a lateral lane offset to steer around an obstacle detected on a
zebra crossing. Implements hysteresis so detouring is stable and smooth.

Logic:
  - When an obstacle is detected: compute offset based on obstacle side
  - Hold offset for DETOUR_HOLD_FRAMES after obstacle disappears (prevent oscillation)
  - Ramp offset smoothly back to zero over DETOUR_RAMP_FRAMES (smooth exit)
  - If obstacle reappears during ramp-down: re-engage immediately
"""

from __future__ import annotations

import logging
from typing import Optional

import config
from behavior_engine import ObstacleDetection, ObstacleSide

log = logging.getLogger(__name__)


class ObstacleHandler:
    """
    Stateful detour offset calculator.

    Usage::

        handler = ObstacleHandler()
        offset = handler.compute_offset(obstacle_detection, lane_width_px=280)
        # offset is added to the normal lane target_x inside main.py
    """

    def __init__(self) -> None:
        self._current_offset:  int = 0       # active offset (px, signed)
        self._target_offset:   int = 0       # desired offset
        self._hold_frames_left: int = 0      # frames remaining in hold phase
        self._ramp_frames_left: int = 0      # frames remaining in ramp phase
        self._state: str = "CLEAR"           # CLEAR | DETOURING | HOLDING | RAMPING

        log.info("ObstacleHandler initialised")

    def compute_offset(
        self,
        obstacle:  Optional[ObstacleDetection],
        lane_width_px: int = 280,
    ) -> tuple[int, Optional[BehaviorCommand]]:
        """
        Returns (lateral_offset_px, optional_behavior_command).
        Fix 21: Returns BehaviorMode.FULL_STOP if obstacle is CENTER.
        """
        has_obstacle = obstacle is not None and obstacle.present
        cmd: Optional[BehaviorCommand] = None

        # ----------------------------------------------------------------
        # ACTIVE OBSTACLE → engage or maintain detour
        # ----------------------------------------------------------------
        if has_obstacle and obstacle is not None:
            side = obstacle.estimated_side

            if side == ObstacleSide.LEFT:
                self._target_offset = config.DETOUR_OFFSET_PX
                self._state = "DETOURING"
            elif side == ObstacleSide.RIGHT:
                self._target_offset = -config.DETOUR_OFFSET_PX
                self._state = "DETOURING"
            else:
                # CENTER — Fix 21: Hard stop
                self._target_offset = 0
                self._state = "DETOURING"
                cmd = BehaviorCommand(mode=BehaviorMode.FULL_STOP, speed_multiplier=0.0)

            self._hold_frames_left = config.DETOUR_HOLD_FRAMES
            self._ramp_frames_left = 0
            self._current_offset   = self._target_offset
            return self._current_offset, cmd

        # ----------------------------------------------------------------
        # NO OBSTACLE — work through hold → ramp → clear
        # ----------------------------------------------------------------
        if self._state == "DETOURING":
            self._state            = "HOLDING"
            self._hold_frames_left = config.DETOUR_HOLD_FRAMES
            self._ramp_frames_left = config.DETOUR_RAMP_FRAMES
            log.debug("ObstacleHandler: entering HOLD phase (%d frames)",
                      self._hold_frames_left)

        if self._state == "HOLDING":
            if self._hold_frames_left > 0:
                self._hold_frames_left -= 1
            else:
                self._state            = "RAMPING"
                self._ramp_frames_left = config.DETOUR_RAMP_FRAMES
                log.debug("ObstacleHandler: entering RAMP phase (%d frames)",
                          self._ramp_frames_left)

        if self._state == "RAMPING":
            if self._ramp_frames_left > 0:
                n_frames = max(config.DETOUR_RAMP_FRAMES, 1)
                step = self._current_offset / n_frames
                self._current_offset -= int(step)
                self._ramp_frames_left -= 1
            else:
                self._current_offset = 0
                self._target_offset  = 0
                self._state          = "CLEAR"
                log.info("ObstacleHandler: offset ramped to zero")

        return self._current_offset, None

    def reset(self) -> None:
        """Immediately zero the offset (e.g., at startup or after a full stop)."""
        self._current_offset   = 0
        self._target_offset    = 0
        self._hold_frames_left = 0
        self._ramp_frames_left = 0
        self._state            = "CLEAR"

    @property
    def state(self) -> str:
        """Current handler state: CLEAR | DETOURING | HOLDING | RAMPING"""
        return self._state

    @property
    def current_offset(self) -> int:
        """Latest computed offset (px, signed)."""
        return self._current_offset


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    handler = ObstacleHandler()

    # No obstacle → offset = 0
    result = handler.compute_offset(None, 280)
    assert result == 0, f"Expected 0, got {result}"

    # Obstacle on LEFT → positive offset (go right)
    obs_left = ObstacleDetection(present=True, bbox=(10, 10, 100, 200),
                                 estimated_side=ObstacleSide.LEFT)
    result = handler.compute_offset(obs_left, 280)
    assert result == config.DETOUR_OFFSET_PX, f"Expected {config.DETOUR_OFFSET_PX}, got {result}"
    assert handler.state == "DETOURING"

    # Obstacle clears → hold phase
    result = handler.compute_offset(None, 280)
    assert handler.state == "HOLDING"
    assert result == config.DETOUR_OFFSET_PX   # still held

    # Drain hold phase
    for _ in range(config.DETOUR_HOLD_FRAMES):
        handler.compute_offset(None, 280)
    assert handler.state == "RAMPING"

    # Drain ramp phase
    for _ in range(config.DETOUR_RAMP_FRAMES + 2):
        handler.compute_offset(None, 280)
    assert handler.state == "CLEAR"
    assert handler.current_offset == 0

    print("obstacle_handler smoke-test PASSED")
