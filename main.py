"""
main.py — BFMC Autonomous Car System
======================================
Thin orchestrator. Wires all 9 modules together and runs the main control loop.

Startup order:
  1. Parse CLI / apply config overrides
  2. Init CameraManager
  3. Init ImageProcessor
  4. Init HybridLaneTracker + JunctionDetector + RoundaboutNavigator
  5. Init SteeringController
  6. Init VisionAI (spawns 4 detector threads)
  7. Init BehaviorEngine
  8. Init ObstacleHandler
  9. Init Dashboard (optional, spawns render thread)
  10. Connect STM32 serial
  11. Main loop at TARGET_FPS

CLI flags:
  --sim       : No camera, no serial (blank frames, logged actuation)
  --nodash    : Skip dashboard window
  --debug     : DEBUG-level logging
  --config    : Override a config key (e.g. --config CAM_W=800)

Exit: press 'q' in any CV window or Ctrl-C → clean shutdown.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
import config
from camera_manager    import CameraManager
from image_processing  import ImageProcessor
from lane_detection    import HybridLaneTracker, JunctionDetector, RoundaboutNavigator
from steering_controller import SteeringController
from vision_ai         import VisionAI
from behavior_engine   import BehaviorEngine, VisionState, BehaviorMode
from obstacle_handler  import ObstacleHandler
from dashboard         import Dashboard, DashboardState

# ---------------------------------------------------------------------------
# STM32 serial — graceful fallback
# ---------------------------------------------------------------------------
_SERIAL_AVAILABLE = False
try:
    from serial_handler import STM32_SerialHandler  # type: ignore
    _SERIAL_AVAILABLE = True
except ImportError:
    class STM32_SerialHandler:   # type: ignore[no-redef]
        def connect(self)         -> bool : return False
        def set_speed(self, s)    -> None : pass
        def set_steering(self, a) -> None : pass
        def disconnect(self)      -> None : pass

log = logging.getLogger("bfmc")


# ===========================================================================
# MAIN PILOT
# ===========================================================================
class BFMCPilot:
    """
    Main pilot class. Owns all sub-systems and the control loop.
    """

    # -----------------------------------------------------------------------
    def __init__(self, sim_mode: bool, nodash: bool) -> None:
        self._sim_mode = sim_mode

        # ── Sub-systems ──────────────────────────────────────────────────
        self.camera    = CameraManager(sim_mode=sim_mode)
        self.proc      = ImageProcessor()
        self.tracker   = HybridLaneTracker()
        self.junction  = JunctionDetector()
        self.roundabout = RoundaboutNavigator()
        self.ctrl      = SteeringController()
        self.vision_st = VisionState()
        self.vision    = VisionAI()
        self.behavior  = BehaviorEngine()
        self.obs_hdlr  = ObstacleHandler()

        # Optional dashboard
        self.dash_state: Optional[DashboardState] = None
        self.dash: Optional[Dashboard]            = None
        self._nodash = nodash
        if not nodash:
            self.dash_state = DashboardState()
            self.dash       = Dashboard(self.dash_state)

        # Serial
        self.handler   = STM32_SerialHandler()
        self.connected = False

        # ── State ────────────────────────────────────────────────────────
        self.base_speed:    float = 50.0   # set via trackbar in non-sim mode
        self.look_ahead_px: int   = 150
        self.lane_width_px: int   = 280

        self.smooth_steer:  float = 0.0
        self.prev_steer:    float = 0.0
        self.last_target:   Optional[float] = None
        self.lost_frames:   int   = 0

        # EMA FPS tracker
        self._fps: float = 0.0

    # -----------------------------------------------------------------------
    def start(self) -> None:
        """Start all sub-systems."""
        # Camera
        self.camera.start()

        # Serial (non-blocking; retry after CONNECT_RETRY_S if needed)
        if not self._sim_mode:
            self.connected = self.handler.connect()
            if self.connected:
                log.info("Serial: connected")
            else:
                log.warning("Serial: NOT connected — trying again after 2s")
                time.sleep(2.0)
                self.connected = self.handler.connect()
                if not self.connected:
                    log.warning("Serial: still not connected — running blind")

        # VisionAI
        self.vision.push_frame(np.zeros((config.CAM_H, config.CAM_W, 3), np.uint8))
        self.vision.start(self.vision_st)

        # Dashboard
        if self.dash:
            self.dash.start()

        # OpenCV windows (must be created in main thread)
        cv2.namedWindow("BEV Debug", cv2.WINDOW_NORMAL)
        cv2.createTrackbar("Base Speed", "BEV Debug",
                           int(self.base_speed), 200, self._on_speed)
        cv2.createTrackbar("Look Ahead",  "BEV Debug",
                           self.look_ahead_px, 400, self._on_la)
        cv2.createTrackbar("Lane Width",  "BEV Debug",
                           self.lane_width_px, 500, self._on_lw)

        # Dashboard window (main thread owns ALL cv2.imshow calls)
        if self.dash is not None:
            blank = np.full((config.DASH_H, config.DASH_W, 3), (13, 14, 23), np.uint8)
            cv2.namedWindow("BFMC Dashboard", cv2.WINDOW_NORMAL)
            cv2.imshow("BFMC Dashboard", blank)   # must imshow BEFORE resize on Linux Qt
            cv2.resizeWindow("BFMC Dashboard", config.DASH_W, config.DASH_H)
            cv2.moveWindow("BFMC Dashboard", 0, 0)   # force on-screen top-left

        # Raw camera + AI detection window (separate from BEV debug)
        cv2.namedWindow("Camera + AI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera + AI", config.CAM_W, config.CAM_H)

        log.info("BFMCPilot: all sub-systems started (sim=%s)", self._sim_mode)

    def stop(self) -> None:
        """Clean shutdown — zero actuators, stop all threads."""
        log.info("BFMCPilot: stopping …")

        # Zero actuators
        if self.connected:
            try:
                self.handler.set_speed(0)
                self.handler.set_steering(0)
                self.handler.disconnect()
            except Exception:
                pass

        # Stop threads
        self.vision.stop()
        self.camera.stop()
        if self.dash:
            self.dash.stop()

        cv2.destroyAllWindows()
        log.info("BFMCPilot: stopped.")

    # -----------------------------------------------------------------------
    def run(self) -> None:
        """Main control loop."""
        self.start()

        try:
            while True:
                t_frame = time.time()

                # ── 1. Capture frame ────────────────────────────────────
                frame = self.camera.get_frame()
                self.vision.push_frame(frame)

                # ── 2. Image processing → BEV ───────────────────────────
                warped_binary, warped_colour = self.proc.process(frame)

                # ── 3. Lane detection ────────────────────────────────────
                sl, sr, dbg, detect_mode = self.tracker.update(warped_binary)
                left_conf  = self.tracker.left_conf
                right_conf = self.tracker.right_conf
                left_fit   = self.tracker.left_fit
                right_fit  = self.tracker.right_fit

                # ── 4. Junction / Roundabout state machines ──────────────
                jct_state = self.junction.update(
                    warped_binary, left_conf, right_conf,
                    left_fit, right_fit, self.lane_width_px,
                )
                rbt_state = self.roundabout.update(
                    left_fit, right_fit, self.lane_width_px,
                )
                nav_state = (
                    "ROUNDABOUT" if rbt_state == "ROUNDABOUT"
                    else ("JUNCTION" if jct_state == "JUNCTION" else "NORMAL")
                )

                # ── 5. Behavior engine ───────────────────────────────────
                behavior_cmd = self.behavior.update(self.vision_st, nav_state)

                # ── 6. Obstacle handler → extra lane offset ──────────────
                obs_offset = self.obs_hdlr.compute_offset(
                    self.vision_st.obstacle,   # read without lock (atomic read on CPython)
                    self.lane_width_px,
                )

                # Total offset: behavior override takes priority over obstacle
                if (behavior_cmd.lane_offset_override is not None
                        and behavior_cmd.mode == BehaviorMode.DETOUR):
                    total_offset = behavior_cmd.lane_offset_override
                else:
                    total_offset = obs_offset

                # ── 7. Look-ahead (adaptive) ─────────────────────────────
                curvature = self.tracker.get_curvature(self.tracker.h // 2)
                eff_la    = self._compute_lookahead(nav_state, curvature)

                # ── 8. Target x ──────────────────────────────────────────
                y_eval = max(0, self.tracker.h - eff_la)    # FIX: use eff_la not hardcoded 40
                raw_target_x, anchor = self.tracker.get_target_x(
                    y_eval, self.lane_width_px, total_offset, nav_state
                )

                # ── 9. Lost-lane handling (grace period) ─────────────────
                lost = raw_target_x is None
                if lost:
                    self.lost_frames += 1
                    target_x = self.last_target if self.last_target is not None else float(config.BEV_W // 2)
                else:
                    self.lost_frames = 0
                    self.last_target = raw_target_x
                    target_x         = raw_target_x

                # ── 10. Steering computation ─────────────────────────────
                steer_angle = self.ctrl.compute_steer(target_x, eff_la, self.lane_width_px)

                # Guard (divider safety)
                steer_angle, guard_spd, guard_on = self.ctrl.apply_guard(
                    steer_angle, sl, sr, y_eval, lost
                )

                # ── 11. Speed policy ─────────────────────────────────────
                speed = self.ctrl.compute_speed(
                    base_speed   = self.base_speed,
                    nav_state    = nav_state,
                    anchor       = anchor,
                    steer_angle  = steer_angle,
                    curvature    = curvature,
                    lost_frames  = self.lost_frames,
                    behavior_cmd = behavior_cmd,
                    guard_on     = guard_on,
                    guard_spd    = guard_spd,
                )

                # ── 12. Actuation ────────────────────────────────────────
                self._actuate(speed, steer_angle)

                # ── 13. Honk ────────────────────────────────────────────
                if self.behavior.honk_active and self.connected:
                    try:
                        self.handler.set_speed(0)   # stop briefly before honk
                    except Exception:
                        pass

                # ── 14. EMA FPS ──────────────────────────────────────────
                elapsed    = time.time() - t_frame
                inst_fps   = 1.0 / max(elapsed, 1e-6)
                self._fps  = 0.9 * self._fps + 0.1 * inst_fps

                # ── 15. Raw camera window with AI detection overlays ──────
                self._show_raw(frame)

                # ── 16. BEV debug window ─────────────────────────────────
                self._show_debug(dbg, steer_angle, speed, anchor, nav_state,
                                 detect_mode, guard_on, curvature)

                # ── 17. Dashboard: imshow from MAIN THREAD ───────────────
                self._update_dashboard(
                    frame, warped_colour, dbg, steer_angle, speed, anchor,
                    nav_state, curvature, guard_on, detect_mode,
                )
                if self.dash is not None:
                    cv2.imshow("BFMC Dashboard", self.dash.get_canvas())

                # ── 17. Frame-rate cap ────────────────────────────────────
                elapsed = time.time() - t_frame
                wait_ms = max(1, int((config.FRAME_PERIOD - elapsed) * 1000))
                key = cv2.waitKey(wait_ms) & 0xFF
                if key == ord("q"):
                    break

        except KeyboardInterrupt:
            log.info("Keyboard interrupt — stopping …")
        finally:
            self.stop()

    # -----------------------------------------------------------------------
    # PRIVATE HELPERS
    # -----------------------------------------------------------------------

    def _actuate(self, speed: float, steer: float) -> None:
        s_clamped = float(max(-config.MAX_STEER, min(config.MAX_STEER, steer)))
        if self._sim_mode:
            log.debug("SIM actuation: speed=%.1f  steer=%.2f°", speed, s_clamped)
            return
        if self.connected:
            try:
                self.handler.set_speed(int(speed))
                self.handler.set_steering(s_clamped)
            except Exception as exc:
                log.warning("Actuation error: %s", exc)

    def _compute_lookahead(self, nav_state: str, curvature: float) -> int:
        la = self.look_ahead_px
        if nav_state == "ROUNDABOUT":
            la = int(la * config.LA_ROUNDABOUT_SCALE)
        elif nav_state == "JUNCTION":
            la = int(la * config.LA_JUNCTION_SCALE)
        elif curvature > config.HIGH_CURV_THRESH:
            la = int(la * config.LA_HIGH_CURV_SCALE)
        elif curvature > config.MED_CURV_THRESH:
            la = int(la * config.LA_MED_CURV_SCALE)
        return max(la, config.LA_MIN_PX)

    def _show_raw(self, frame: np.ndarray) -> None:
        """Show raw camera frame with AI bounding-box overlays."""
        viz    = frame.copy()
        overlays = self.vision.get_overlays()   # [(label, conf, bbox), ...]
        for label, conf, bbox in overlays:
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            # Colour code by detection type
            if label.startswith("TL:"):
                tl_name = label.split(":")[-1]
                col = (60, 220, 60) if "GREEN" in tl_name else \
                      (0, 200, 220) if "YELLOW" in tl_name else \
                      (50, 50, 220) if tl_name in ("RED", "DARK") else \
                      (200, 200, 200)
                cv2.rectangle(viz, (x1, y1), (x2, y2), col, 3)
                cv2.putText(viz, label, (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
                # Glow halo for traffic light
                cv2.rectangle(viz, (x1-2, y1-2), (x2+2, y2+2), col, 1)
            else:
                col = (255, 160, 0)
                cv2.rectangle(viz, (x1, y1), (x2, y2), col, 2)
                cv2.putText(viz, f"{label} {conf:.0%}",
                            (x1, max(y1 - 6, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1, cv2.LINE_AA)

        # AI FPS + TL state in corner
        tl_s, _ = self.vision.last_tl_result()
        tl_str  = f"TL:{tl_s.name}" if hasattr(tl_s, 'name') else str(tl_s)
        tl_col  = (60, 220, 60) if tl_str.endswith("GREEN") else \
                  (0, 200, 220) if tl_str.endswith("YELLOW") else \
                  (50, 50, 220) if "RED" in tl_str or "DARK" in tl_str else \
                  (170, 170, 170)
        cv2.putText(viz, tl_str,
                    (6, 22), cv2.FONT_HERSHEY_DUPLEX, 0.70, tl_col, 2, cv2.LINE_AA)
        cv2.putText(viz, f"AI {self.vision.get_fps():.1f} fps",
                    (6, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.imshow("Camera + AI", viz)

    def _show_debug(
        self,
        dbg:         np.ndarray,
        steer:       float,
        speed:       float,
        anchor:      str,
        nav_state:   str,
        detect_mode: str,
        guard_on:    bool,
        curvature:   float,
    ) -> None:
        """Overlay telemetry on the BEV debug image and show it."""
        viz = dbg.copy()
        col_anc = config.ANCHOR_COLORS.get(anchor, config.COLOR_WHITE)

        def put(text: str, y: int, color=config.COLOR_WHITE, scale: float = 0.5) -> None:
            cv2.putText(viz, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, color, 1, cv2.LINE_AA)

        put(f"FPS {self._fps:.1f}", 20)
        put(f"STEER {steer:+.1f}°", 40, config.COLOR_CYAN)
        put(f"SPEED {speed:.0f}", 60, config.COLOR_GREEN)
        put(f"NAV   {nav_state}", 80)
        put(f"ANCHOR {anchor}", 100, col_anc)
        put(f"CURV  {curvature:.4f}", 120)
        put(detect_mode, 140, config.COLOR_YELLOW)
        if guard_on:
            put("GUARD!", 160, config.COLOR_RED)
        if self.lost_frames > 0:
            put(f"LOST {self.lost_frames}", 180, config.COLOR_RED)

        cv2.imshow("BEV Debug", viz)

    def _update_dashboard(
        self,
        frame:       np.ndarray,
        bev_bgr:     np.ndarray,
        bev_dbg:     np.ndarray,
        steer:       float,
        speed:       float,
        anchor:      str,
        nav_state:   str,
        curvature:   float,
        guard_on:    bool,
        detect_mode: str,
    ) -> None:
        if self.dash_state is None:
            return

        with self.vision_st.lock:    # FIX: always read obstacle under lock
            tl   = self.vision_st.traffic_light
            sign = self.vision_st.sign
            obs  = self.vision_st.obstacle

        obs_str = self.obs_hdlr.state

        # FIX: use vision.get_overlays() — includes BOTH TL bbox and sign bbox
        overlays = self.vision.get_overlays()

        with self.dash_state.lock:
            self.dash_state.raw_frame        = frame
            self.dash_state.bev_bgr          = bev_bgr
            self.dash_state.bev_dbg          = bev_dbg
            self.dash_state.steer_angle      = steer
            self.dash_state.speed            = speed
            self.dash_state.base_speed       = self.base_speed
            self.dash_state.nav_state        = nav_state
            self.dash_state.anchor           = anchor
            self.dash_state.curvature        = curvature
            self.dash_state.lost_frames      = self.lost_frames
            self.dash_state.guard_active     = guard_on
            self.dash_state.detect_mode      = detect_mode
            self.dash_state.traffic_light    = tl
            self.dash_state.sign_label       = sign.sign_type if sign else None
            self.dash_state.sign_conf        = sign.confidence if sign else None
            self.dash_state.obstacle_state   = obs_str
            self.dash_state.behavior_mode    = self.behavior._state.name   # type: ignore[attr-defined]
            self.dash_state.main_fps         = self._fps
            self.dash_state.vision_fps       = self.vision.get_fps()
            self.dash_state.serial_connected = self.connected
            self.dash_state.frame_ts         = time.time()
            self.dash_state.ai_overlays      = overlays

    # Trackbar callbacks
    def _on_speed(self, v: int) -> None: self.base_speed    = float(v)
    def _on_la(self,    v: int) -> None: self.look_ahead_px = max(v, config.LA_MIN_PX)
    def _on_lw(self,    v: int) -> None: self.lane_width_px = max(v, 50)


# ===========================================================================
# ENTRY POINT
# ===========================================================================
def _apply_config_overrides(overrides: list[str]) -> None:
    """Apply --config KEY=VALUE overrides to the live config module."""
    for kv in overrides:
        if "=" not in kv:
            log.warning("Ignoring bad --config entry: %r (need KEY=VALUE)", kv)
            continue
        k, v = kv.split("=", 1)
        k = k.strip()
        if not hasattr(config, k):
            log.warning("--config: unknown key %r — ignored", k)
            continue
        orig = getattr(config, k)
        try:
            if isinstance(orig, bool):
                setattr(config, k, v.lower() in ("1", "true", "yes"))
            elif isinstance(orig, int):
                setattr(config, k, int(v))
            elif isinstance(orig, float):
                setattr(config, k, float(v))
            else:
                setattr(config, k, v)
            log.info("config override: %s = %s", k, getattr(config, k))
        except ValueError as exc:
            log.warning("--config: bad value for %r: %s", k, exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BFMC Autonomous Car — Modular Pilot"
    )
    parser.add_argument("--sim",    action="store_true",
                        help="Simulation mode: no camera, no serial")
    parser.add_argument("--nodash", action="store_true",
                        help="Disable dashboard window")
    parser.add_argument("--debug",  action="store_true",
                        help="Enable DEBUG-level logging")
    parser.add_argument("--config", nargs="*", default=[],
                        metavar="KEY=VALUE",
                        help="Override config values at runtime")
    args = parser.parse_args()

    logging.basicConfig(
        level     = logging.DEBUG if args.debug else logging.INFO,
        format    = "%(asctime)s  %(name)-22s  %(levelname)-7s  %(message)s",
        datefmt   = "%H:%M:%S",
        stream    = sys.stdout,
    )

    if args.config:
        _apply_config_overrides(args.config)

    log.info("=" * 60)
    log.info("BFMC Autonomous Car  —  Modular Pilot")
    log.info("  sim_mode : %s", args.sim)
    log.info("  nodash   : %s", args.nodash)
    log.info("  debug    : %s", args.debug)
    log.info("=" * 60)

    pilot = BFMCPilot(sim_mode=args.sim, nodash=args.nodash)
    pilot.run()


if __name__ == "__main__":
    main()
