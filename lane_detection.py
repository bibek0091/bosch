"""
lane_detection.py — BFMC Autonomous Car System
===============================================
Direct lift of the three state-machine classes from bfmc_pilot_v2.py
with zero logic changes. All constants pulled from config.

Classes exported:
  HybridLaneTracker   — sliding-window + polynomial lane tracker
  JunctionDetector    — detects intersection approach / exit
  RoundaboutNavigator — detects roundabout entry / exit
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)


# ===========================================================================
# HYBRID LANE TRACKER
# ===========================================================================
class HybridLaneTracker:
    """
    Hybrid sliding-window + polynomial lane tracker.

    Public interface
    ----------------
    update(warped_binary) -> (sl, sr, dbg, mode_label)
    get_target_x(y_eval, lane_width_px, extra_offset_px, nav_state) -> (x, anchor)
    get_curvature(y_eval) -> float
    reset()
    """

    # All tuning constants pulled from config
    NWINDOWS         = config.TRACKER_NWINDOWS
    SW_MARGIN        = config.TRACKER_SW_MARGIN
    MINPIX           = config.TRACKER_MINPIX
    POLY_MARGIN_BASE = config.TRACKER_POLY_MARGIN_BASE
    POLY_MARGIN_CURV = config.TRACKER_POLY_MARGIN_CURV
    MIN_PIX_OK       = config.TRACKER_MIN_PIX_OK
    EMA_ALPHA        = config.TRACKER_EMA_ALPHA
    STALE_FIT_FRAMES = config.TRACKER_STALE_FIT_FRAMES

    def __init__(self, img_shape: tuple[int, int] = (config.BEV_H, config.BEV_W)) -> None:
        self.h, self.w = img_shape
        self.mode        = "SEARCH"
        self.left_fit:  Optional[np.ndarray] = None
        self.right_fit: Optional[np.ndarray] = None
        self.sl:        Optional[np.ndarray] = None   # EMA-smoothed left poly
        self.sr:        Optional[np.ndarray] = None   # EMA-smoothed right poly
        self.left_conf:  int = 0
        self.right_conf: int = 0
        self.left_stale:  int = 0
        self.right_stale: int = 0

    # ------------------------------------------------------------------
    def update(
        self, warped_binary: np.ndarray
    ) -> tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, str]:
        """
        Run one tracking step on the binary BEV image.

        Returns
        -------
        sl : smoothed left polynomial coefficients (or None)
        sr : smoothed right polynomial coefficients (or None)
        dbg : colour debug image (BGR, BEV_H × BEV_W)
        mode_label : "POLY" or "SLIDE"
        """
        nz  = warped_binary.nonzero()
        nzy = np.array(nz[0])
        nzx = np.array(nz[1])

        if self.mode == "TRACKING" and (self.sl is not None or self.sr is not None):
            curv = self.get_curvature(self.h // 2)
            li, ri, dbg = self._poly_search(warped_binary, nzx, nzy, curvature=curv)
            mode_label  = "POLY"
        else:
            li, ri, dbg = self._sliding_window(warped_binary, nzx, nzy)
            mode_label  = "SLIDE"

        self.left_conf  = len(li)
        self.right_conf = len(ri)
        has_l = self.left_conf  >= self.MIN_PIX_OK
        has_r = self.right_conf >= self.MIN_PIX_OK

        if has_l:
            fl = np.polyfit(nzy[li], nzx[li], 2)
            self.left_fit    = fl
            self.sl          = self._ema(self.sl, fl)
            self.left_stale  = 0
        else:
            self.left_stale += 1
            if self.left_stale > self.STALE_FIT_FRAMES:
                self.left_fit = None
                self.sl       = None

        if has_r:
            fr = np.polyfit(nzy[ri], nzx[ri], 2)
            self.right_fit   = fr
            self.sr          = self._ema(self.sr, fr)
            self.right_stale = 0
        else:
            self.right_stale += 1
            if self.right_stale > self.STALE_FIT_FRAMES:
                self.right_fit = None
                self.sr        = None

        # Sanity: if both found, check lane width is plausible
        if has_l and has_r:
            if not self._width_sane(self.left_fit, self.right_fit):
                if self.left_conf < self.right_conf:
                    self.left_fit    = None
                    self.sl          = None
                    self.left_stale  = self.STALE_FIT_FRAMES
                    has_l            = False
                else:
                    self.right_fit   = None
                    self.sr          = None
                    self.right_stale = self.STALE_FIT_FRAMES
                    has_r            = False

        self.mode = (
            "TRACKING"
            if (has_l or has_r or self.sl is not None or self.sr is not None)
            else "SEARCH"
        )
        return self.sl, self.sr, dbg, mode_label

    def get_target_x(
        self,
        y_eval: int,
        lane_width_px: int,
        extra_offset_px: int = 0,
        nav_state: str = "NORMAL",
    ) -> tuple[Optional[float], str]:
        """
        Compute the BEV x-pixel the car should steer toward.
        Fix 28: Guards for None polynomial evaluation.
        """
        sl = self.sl
        sr = self.sr
        hw = lane_width_px / 2.0

        def ev(fit: Optional[np.ndarray]) -> Optional[float]:
            if fit is None: return None
            return float(np.polyval(fit, y_eval))

        xl = ev(sl)
        xr = ev(sr)

        if nav_state == "ROUNDABOUT":
            if xl is not None:
                return xl + hw + extra_offset_px, "RBT_INNER"
            if xr is not None:
                return xr - hw + extra_offset_px, "RBT_OUTER"
            return None, "RBT_LOST"

        if nav_state == "JUNCTION":
            if xr is not None:
                return xr - hw + extra_offset_px, "JCT_EDGE"
            if xl is not None:
                return xl + hw + extra_offset_px, "JCT_DIV"
            return None, "JCT_LOST"

        # NORMAL right-lane driving (Fix 24: ensure xl and xr are valid)
        if xl is not None and xr is not None:
            return (xl + xr) / 2.0 + config.DUAL_OFFSET_PX, "DUAL"

        if xr is not None and xl is None:
            ghost_xl = xr - lane_width_px
            return (ghost_xl + xr) / 2.0 + config.SINGLE_EDGE_OFFSET_PX, "GHOST_L"

        if xl is not None and xr is None:
            ghost_xr = xl + lane_width_px
            return (xl + ghost_xr) / 2.0 + config.SINGLE_DIV_OFFSET_PX, "GHOST_R"

        return None, "LOST"

    def get_curvature(self, y_eval: int) -> float:
        """
        vature = |2a| / (1 + (2ay+b)^2)^1.5
        """
        fit = self.sr if self.sr is not None else self.sl
        if fit is None:
            return 0.0
        # Fix 23 corollary: ensure fit has 3 coeffs for degree-2
        if len(fit) < 3: return 0.0
        a, b  = fit[0], fit[1]
        num   = abs(2.0 * a)
        denom = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5
        return num / max(denom, 1e-6)

    def reset(self) -> None:
        """Reset tracker to initial SEARCH state."""
        self.mode        = "SEARCH"
        self.left_fit    = None
        self.right_fit   = None
        self.sl          = None
        self.sr          = None
        self.left_stale  = 0
        self.right_stale = 0

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _sliding_window(
        self,
        warped: np.ndarray,
        nzx: np.ndarray,
        nzy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dbg  = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        
        # Fix 25: use bottom half for histogram
        y_start = self.h // 2
        hist = np.sum(warped[y_start:, :], axis=0)

        # Fix 22: Histogram split via TRACKER_HIST_SPLIT
        split_frac = getattr(config, "TRACKER_HIST_SPLIT", 0.5)
        mid = int(self.w * split_frac)
        margin = self.SW_MARGIN

        # Peak detection with margin
        lb = int(np.argmax(hist[margin : mid - margin])) + margin
        rb = int(np.argmax(hist[mid + margin : self.w - margin])) + mid + margin

        # Peak collision fallback
        if abs(rb - lb) < 100:
            smoothed = np.convolve(hist.astype(float), np.ones(20) / 20, mode="same")
            p1  = int(np.argmax(smoothed))
            tmp = smoothed.copy()
            tmp[max(0, p1 - 50):min(self.w, p1 + 50)] = 0
            p2  = int(np.argmax(tmp))
            lb, rb = (min(p1, p2), max(p1, p2))

        wh = self.h // self.NWINDOWS
        lx, rx = lb, rb
        li, ri = [], []

        for win in range(self.NWINDOWS):
            y_lo = self.h - (win + 1) * wh
            y_hi = self.h - win * wh
            xl0, xl1 = max(0, lx - margin), min(self.w, lx + margin)
            xr0, xr1 = max(0, rx - margin), min(self.w, rx + margin)

            cv2.rectangle(dbg, (xl0, y_lo), (xl1, y_hi), (0, 255, 0), 2)
            cv2.rectangle(dbg, (xr0, y_lo), (xr1, y_hi), (0, 255, 0), 2)

            gl = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xl0) & (nzx < xl1)).nonzero()[0]
            gr = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xr0) & (nzx < xr1)).nonzero()[0]

            li.append(gl)
            ri.append(gr)

            if len(gl) > self.MINPIX: lx = int(np.mean(nzx[gl]))
            if len(gr) > self.MINPIX: rx = int(np.mean(nzx[gr]))

        li = np.concatenate(li) if li else np.array([], dtype=int)
        ri = np.concatenate(ri) if ri else np.array([], dtype=int)

        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80, 80, 255]
        return li, ri, dbg

    def _poly_search(
        self,
        warped: np.ndarray,
        nzx: np.ndarray,
        nzy: np.ndarray,
        curvature: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fix 28: polynomial existence guards."""
        dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        m   = self.POLY_MARGIN_CURV if curvature > 0.0015 else self.POLY_MARGIN_BASE

        def band(fit: Optional[np.ndarray]) -> np.ndarray:
            if fit is None: return np.array([], dtype=int)
            cx = np.polyval(fit, nzy)
            return ((nzx > cx - m) & (nzx < cx + m)).nonzero()[0]

        li = band(self.sl)
        ri = band(self.sr)

        if len(li) < self.MIN_PIX_OK and len(ri) < self.MIN_PIX_OK:
            self.mode = "SEARCH"
            return self._sliding_window(warped, nzx, nzy)

        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80, 80, 255]
        return li, ri, dbg

    def _width_sane(self, lf: np.ndarray, rf: np.ndarray) -> bool:
        """Fix 27: use config bounds."""
        y = config.BEV_H - 50
        w = np.polyval(rf, y) - np.polyval(lf, y)
        min_w = getattr(config, "TRACKER_MIN_LANE_WIDTH_PX", 120)
        max_w = getattr(config, "TRACKER_MAX_LANE_WIDTH_PX", 500)
        return min_w < w < max_w

    def _ema(self, prev: Optional[np.ndarray], new: np.ndarray) -> np.ndarray:
        if prev is None:
            return new.copy()
        return self.EMA_ALPHA * new + (1.0 - self.EMA_ALPHA) * prev


# ===========================================================================
# JUNCTION DETECTOR
# ===========================================================================
class JunctionDetector:
    """
    Detects junction (intersection) approach via three independent cues:
      1. Both lines lost  (both_lost)
      2. Top-half energy > bottom-half energy  (cross_energy)
      3. Measured lane width much wider than calibrated  (wide_lane)

    Public interface
    ----------------
    update(warped_binary, left_conf, right_conf,
           left_fit, right_fit, lane_width_px) -> state_str
    """

    ENTRY_FRAMES       = config.JCT_ENTRY_FRAMES
    EXIT_FRAMES        = config.JCT_EXIT_FRAMES
    CROSS_ENERGY_RATIO = config.JCT_CROSS_ENERGY_RATIO
    WIDTH_RATIO_HIGH   = config.JCT_WIDTH_RATIO_HIGH
    MIN_BOT_ENERGY     = config.JCT_MIN_BOT_ENERGY
    MAX_JCT_FRAMES     = config.JCT_MAX_FRAMES

    def __init__(self) -> None:
        self.state         = "NORMAL"
        self.entry_count   = 0
        self.exit_count    = 0
        self.frames_in_jct = 0

    def update(
        self,
        warped_binary: np.ndarray,
        left_conf:    int,
        right_conf:   int,
        left_fit:     Optional[np.ndarray],
        right_fit:    Optional[np.ndarray],
        lane_width_px: int,
    ) -> str:
        h, w = warped_binary.shape

        both_lost = (left_conf < 200) and (right_conf < 200)

        hist_top = float(np.sum(warped_binary[:h // 2, :]))
        hist_bot = float(np.sum(warped_binary[h // 2:, :]))

        cross_energy = False
        if hist_bot > self.MIN_BOT_ENERGY:
            cross_energy = (hist_top / hist_bot) > self.CROSS_ENERGY_RATIO

        wide_lane = False
        if left_fit is not None and right_fit is not None:
            lx = np.polyval(left_fit,  h - 50)
            rx = np.polyval(right_fit, h - 50)
            if (rx - lx) > lane_width_px * self.WIDTH_RATIO_HIGH:
                wide_lane = True

        evidence = both_lost or cross_energy or wide_lane

        if self.state == "NORMAL":
            self.entry_count = self.entry_count + 1 if evidence else 0
            if self.entry_count >= self.ENTRY_FRAMES:
                self.state         = "JUNCTION"
                self.exit_count    = 0
                self.frames_in_jct = 0
                log.info("Junction ENTERED")

        elif self.state == "JUNCTION":
            self.frames_in_jct += 1
            self.exit_count = self.exit_count + 1 if not evidence else 0
            normal_exit  = (self.exit_count >= self.EXIT_FRAMES and self.frames_in_jct > 15)
            timeout_exit = self.frames_in_jct > self.MAX_JCT_FRAMES
            if normal_exit or timeout_exit:
                reason = "timeout" if timeout_exit else "clear"
                self.state       = "NORMAL"
                self.entry_count = 0
                log.info("Junction EXITED (%s)", reason)

        return self.state


# ===========================================================================
# ROUNDABOUT NAVIGATOR
# ===========================================================================
class RoundaboutNavigator:
    """
    Detects roundabout entry (lane narrows) and exit (lane widens).

    Public interface
    ----------------
    update(left_fit, right_fit, lane_width_px, img_h) -> state_str
    """

    ENTRY_WIDTH_RATIO = config.RBT_ENTRY_WIDTH_RATIO
    ENTRY_FRAMES      = config.RBT_ENTRY_FRAMES       # debounce before entering
    EXIT_WIDTH_RATIO  = config.RBT_EXIT_WIDTH_RATIO
    MIN_CIRCLE_FRAMES = config.RBT_MIN_CIRCLE_FRAMES
    MAX_CIRCLE_FRAMES = config.RBT_MAX_CIRCLE_FRAMES
    SPEED_SCALE       = config.RBT_SPEED_SCALE
    LOOKAHEAD_SCALE   = config.RBT_LOOKAHEAD_SCALE

    def __init__(self) -> None:
        self.state       = "NORMAL"
        self.frames      = 0
        self.entry_count = 0   # consecutive frames with narrow ratio

    def update(
        self,
        left_fit:     Optional[np.ndarray],
        right_fit:    Optional[np.ndarray],
        lane_width_px: int,
        img_h: int = config.BEV_H,
    ) -> str:
        y = img_h - 50

        if left_fit is not None and right_fit is not None:
            lx    = np.polyval(left_fit,  y)
            rx    = np.polyval(right_fit, y)
            ratio = (rx - lx) / max(float(lane_width_px), 1.0)

            if self.state == "NORMAL":
                if ratio < self.ENTRY_WIDTH_RATIO:
                    self.entry_count += 1
                    if self.entry_count >= self.ENTRY_FRAMES:
                        self.state       = "ROUNDABOUT"
                        self.frames      = 0
                        self.entry_count = 0
                        log.info("Roundabout ENTRY detected")
                else:
                    self.entry_count = 0   # reset on non-narrow frame

            elif self.state == "ROUNDABOUT":
                self.frames += 1
                normal_exit  = (self.frames > self.MIN_CIRCLE_FRAMES and
                                ratio > self.EXIT_WIDTH_RATIO)
                timeout_exit = self.frames > self.MAX_CIRCLE_FRAMES
                if normal_exit or timeout_exit:
                    reason = "timeout" if timeout_exit else "width ratio"
                    self.state  = "NORMAL"
                    self.frames = 0
                    log.info("Roundabout EXIT (%s)", reason)

        elif self.state == "ROUNDABOUT":
            # One line lost inside roundabout — count frames, timeout exit
            self.frames += 1
            if self.frames > self.MAX_CIRCLE_FRAMES:
                self.state  = "NORMAL"
                self.frames = 0
                log.info("Roundabout EXIT (timeout, one line lost)")

        return self.state


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    tracker = HybridLaneTracker()
    jct     = JunctionDetector()
    rbt     = RoundaboutNavigator()

    # Blank binary frame (lane lost)
    blank = np.zeros((config.BEV_H, config.BEV_W), dtype=np.uint8)
    sl, sr, dbg, mode = tracker.update(blank)
    assert sl is None and sr is None and mode == "SLIDE"

    jct_state = jct.update(blank, 0, 0, None, None, 280)
    assert jct_state == "NORMAL"   # no evidence on blank frame

    rbt_state = rbt.update(None, None, 280)
    assert rbt_state == "NORMAL"

    print("lane_detection smoke-test PASSED")
