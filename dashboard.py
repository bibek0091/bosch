"""
dashboard.py — BFMC Autonomous Car System
==========================================
Tesla-style real-time HUD.

Layout  (1280 × 720)
────────────────────────────────────────────────────────────
│  CAMERA + AI    │   SPEED GAUGE + STEER   │   BEV + STATUS │
│   (420 × 580)   │      (440 × 580)        │   (420 × 580)  │
└────────────────────────────── TELEMETRY BAR ───────────────┘

Concurrency model
─────────────────
  • _render_loop runs in a daemon thread and writes to self._canvas.
  • Main thread calls get_canvas() → cv2.imshow().   ← OpenCV-safe
  • cv2.imshow / cv2.waitKey are NEVER called from this module.
"""

from __future__ import annotations

import math
import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared state  (written by main loop, read by dashboard thread)
# ---------------------------------------------------------------------------
@dataclass
class DashboardState:
    # Raw images
    raw_frame:   Optional[np.ndarray] = None
    bev_bgr:     Optional[np.ndarray] = None
    bev_dbg:     Optional[np.ndarray] = None

    # Vehicle telemetry
    steer_angle:  float = 0.0
    speed:        float = 0.0
    base_speed:   float = 50.0
    curvature:    float = 0.0
    lost_frames:  int   = 0
    guard_active: bool  = False

    # Navigation
    nav_state:   str = "NORMAL"
    anchor:      str = "LOST"
    detect_mode: str = "SLIDE"

    # AI / vision
    traffic_light:  Optional[object] = None   # TrafficLightState
    sign_label:     Optional[str]    = None
    sign_conf:      Optional[float]  = None
    obstacle_state: str              = "CLEAR"
    behavior_mode:  str              = "NORMAL"

    # System
    main_fps:         float = 0.0
    vision_fps:       float = 0.0
    serial_connected: bool  = False
    frame_ts:         float = 0.0
    ai_overlays:      list  = field(default_factory=list)

    lock: threading.Lock = field(default_factory=threading.Lock)


# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
C_BG        = (13,  14,  23)      # near-black background
C_PANEL     = (22,  25,  42)      # panel fill
C_BORDER    = (40,  44,  72)      # subtle border
C_BLUE      = (255, 180,  0)      # electric blue (BGR!)
C_CYAN      = (220, 200,  0)      # accent cyan
C_GREEN     = (80,  220,  80)     # status green
C_RED       = (60,   60, 220)     # alert red
C_YELLOW    = (0,   200, 220)     # warning yellow
C_ORANGE    = (0,   140, 255)     # info orange
C_WHITE     = (240, 240, 245)     # text white
C_DIMWHITE  = (140, 145, 160)     # dim text
C_GREY      = (55,  58,  78)      # grid lines

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD   = cv2.FONT_HERSHEY_DUPLEX


# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------
class Dashboard:
    W = config.DASH_W   # 1280
    H = config.DASH_H   # 720

    # Panel widths
    PW_L  = 420
    PW_C  = 440
    PW_R  = 420
    PH    = 580        # panel height (excl. telemetry bar)
    BAR_H = 140        # bottom telemetry bar height

    # Panel x-offsets
    X_L = 0
    X_C = PW_L
    X_R = PW_L + PW_C

    def __init__(self, state: DashboardState) -> None:
        self._state   = state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._period  = 1.0 / config.DASH_FPS

        # Canvas shared between render thread and main thread
        self._canvas: np.ndarray = np.full((self.H, self.W, 3), C_BG, dtype=np.uint8)
        self._canvas_lock = threading.Lock()

        # Sign display hold timer
        self._sign_cache: str   = ""
        self._sign_conf_cache: float = 0.0
        self._sign_until: float = 0.0

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._render_loop, daemon=True, name="dashboard")
        self._thread.start()
        log.info("Dashboard: render thread started at %d fps", config.DASH_FPS)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        log.info("Dashboard: stopped")

    def get_canvas(self) -> np.ndarray:
        """
        Called by the MAIN THREAD to get the latest canvas for cv2.imshow.
        Thread-safe: returns a copy.
        """
        with self._canvas_lock:
            return self._canvas.copy()

    # ------------------------------------------------------------------
    # RENDER LOOP  (daemon thread — zero cv2.imshow / cv2.waitKey calls)
    # ------------------------------------------------------------------
    def _render_loop(self) -> None:
        while self._running:
            t0 = time.monotonic()
            try:
                canvas = self._render()
                with self._canvas_lock:
                    self._canvas = canvas
            except Exception as exc:
                log.debug("Dashboard render error: %s", exc)

            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, self._period - elapsed))

    # ------------------------------------------------------------------
    # TOP-LEVEL RENDER
    # ------------------------------------------------------------------
    def _render(self) -> np.ndarray:
        with self._state.lock:
            s = self._snapshot()

        canvas = np.full((self.H, self.W, 3), C_BG, dtype=np.uint8)

        self._draw_panel_backgrounds(canvas)
        self._draw_left_panel(canvas, s)
        self._draw_center_panel(canvas, s)
        self._draw_right_panel(canvas, s)
        self._draw_telemetry_bar(canvas, s)
        self._draw_header(canvas, s)

        return canvas

    def _snapshot(self) -> dict:
        """Copy relevant state fields while holding the lock."""
        s = self._state
        return dict(
            raw_frame       = _clone(s.raw_frame),
            bev_bgr         = _clone(s.bev_bgr),
            bev_dbg         = _clone(s.bev_dbg),
            steer           = s.steer_angle,
            speed           = s.speed,
            base_speed      = s.base_speed,
            curvature       = s.curvature,
            lost_frames     = s.lost_frames,
            guard           = s.guard_active,
            nav_state       = s.nav_state,
            anchor          = s.anchor,
            detect_mode     = s.detect_mode,
            traffic_light   = s.traffic_light,
            sign_label      = s.sign_label,
            sign_conf       = s.sign_conf,
            obstacle_state  = s.obstacle_state,
            behavior_mode   = s.behavior_mode,
            main_fps        = s.main_fps,
            vision_fps      = s.vision_fps,
            serial          = s.serial_connected,
            ai_overlays     = list(s.ai_overlays),
            frame_ts        = s.frame_ts,
        )

    # ------------------------------------------------------------------
    # PANEL BACKGROUNDS
    # ------------------------------------------------------------------
    def _draw_panel_backgrounds(self, c: np.ndarray) -> None:
        pad = 8
        for x in [self.X_L, self.X_C, self.X_R]:
            _filled_rect(c, x + pad, 52, x + (self.PW_L if x == 0 else
                         (self.PW_C if x == self.X_C else self.PW_R)) - pad,
                         52 + self.PH, C_PANEL, radius=12)

        # Vertical separators
        for x in [self.X_C, self.X_R]:
            cv2.line(c, (x, 52), (x, 52 + self.PH), C_BORDER, 1)

        # Top glow line
        _glow_line(c, 0, 50, self.W, 50, C_BLUE, thickness=2, blur=3)

    # ------------------------------------------------------------------
    # HEADER BAR
    # ------------------------------------------------------------------
    def _draw_header(self, c: np.ndarray, s: dict) -> None:
        _filled_rect(c, 0, 0, self.W, 50, (18, 20, 35), radius=0)
        cv2.line(c, (0, 50), (self.W, 50), C_BORDER, 1)

        _text(c, "BFMC  AUTONOMOUS  PILOT", (self.W // 2, 30),
              FONT_BOLD, 0.65, C_WHITE, 1, center=True)

        # Clock
        ts = time.strftime("%H:%M:%S")
        _text(c, ts, (self.W - 90, 30), FONT, 0.45, C_DIMWHITE, 1)

        # Serial dot
        dot_col = C_GREEN if s["serial"] else C_RED
        cv2.circle(c, (20, 25), 6, dot_col, -1)
        _text(c, "SERIAL", (32, 30), FONT, 0.38, C_DIMWHITE, 1)

    # ------------------------------------------------------------------
    # LEFT PANEL — Camera feed + AI overlays
    # ------------------------------------------------------------------
    def _draw_left_panel(self, c: np.ndarray, s: dict) -> None:
        x0, y0 = self.X_L + 10, 60
        pw = self.PW_L - 20
        ph = self.PH  - 48

        _section_label(c, "CAMERA", x0, y0 - 4)

        frame = s["raw_frame"]
        if frame is not None:
            img = cv2.resize(frame, (pw, ph))
            # AI detection overlays
            for label, conf, bbox in s["ai_overlays"]:
                if bbox:
                    x1, y1, x2, y2 = [int(v * pw / config.CAM_W) if i % 2 == 0
                                       else int(v * ph / config.CAM_H)
                                       for i, v in enumerate(bbox)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), C_BLUE, 2)
                    _text(img, f"{label} {conf:.0%}", (x1 + 4, y1 + 16),
                          FONT, 0.42, C_BLUE, 1)
            # Traffic light overlay
            tl = s["traffic_light"]
            if tl is not None:
                tl_name = tl.name if hasattr(tl, "name") else str(tl)
                tl_col = C_RED if "RED" in tl_name else (C_GREEN if "GREEN" in tl_name else C_YELLOW)
                _filled_rect(img, pw - 140, 6, pw - 6, 30, (0, 0, 0, 160), radius=4)
                _text(img, f"TL: {tl_name}", (pw - 135, 23), FONT, 0.42, tl_col, 1)

            c[y0:y0 + ph, x0:x0 + pw] = img
            _border_rect(c, x0, y0, x0 + pw, y0 + ph, C_BORDER, 1, radius=6)
        else:
            _filled_rect(c, x0, y0, x0 + pw, y0 + ph, C_GREY, radius=6)
            _text(c, "NO SIGNAL", (x0 + pw // 2, y0 + ph // 2),
                  FONT_BOLD, 0.65, C_DIMWHITE, 1, center=True)

        # Sign badge
        now = time.monotonic()
        if s["sign_label"]:
            self._sign_cache     = s["sign_label"]
            self._sign_conf_cache = s["sign_conf"] or 0.0
            self._sign_until     = now + config.DASH_SIGN_DISPLAY_SEC
        if now < self._sign_until:
            sy = y0 + ph + 10
            _filled_rect(c, x0, sy, x0 + pw, sy + 30, C_BLUE, radius=6)
            _text(c, f"SIGN: {self._sign_cache}  {self._sign_conf_cache:.0%}",
                  (x0 + pw // 2, sy + 20), FONT_BOLD, 0.48, C_BG, 1, center=True)

    # ------------------------------------------------------------------
    # CENTRE PANEL — Speed arc gauge + steering wheel
    # ------------------------------------------------------------------
    def _draw_center_panel(self, c: np.ndarray, s: dict) -> None:
        cx = self.X_C + self.PW_C // 2
        speed = s["speed"]
        steer = s["steer"]
        behavior = s["behavior_mode"]
        nav_state = s["nav_state"]

        # ── Section label ──────────────────────────────────────────────
        _section_label(c, "VEHICLE STATUS", self.X_C + 12, 56)

        # ── Speed arc gauge (top half of panel) ───────────────────────
        gauge_cy = 210
        gauge_r  = 120

        # Background track
        _draw_arc(c, cx, gauge_cy, gauge_r, -220, 40, C_GREY, 12)
        # Value arc (0→200 speed range)
        max_spd = 200.0
        pct     = min(speed / max_spd, 1.0)
        end_deg = -220 + int(260 * pct)
        spd_col = C_RED if speed > 150 else (C_YELLOW if speed > 100 else C_BLUE)
        if pct > 0:
            _draw_arc(c, cx, gauge_cy, gauge_r, -220, end_deg, spd_col, 12)

        # Glow dot at tip
        if pct > 0:
            tip_rad = math.radians(end_deg)
            tx = int(cx + gauge_r * math.cos(tip_rad))
            ty = int(gauge_cy + gauge_r * math.sin(tip_rad))
            cv2.circle(c, (tx, ty), 10, spd_col, -1)
            cv2.circle(c, (tx, ty),  6, C_WHITE,  -1)

        # Speed number
        _text(c, f"{int(speed)}", (cx, gauge_cy + 12),
              FONT_BOLD, 2.2, C_WHITE, 2, center=True)
        _text(c, "km/h", (cx, gauge_cy + 48),
              FONT, 0.55, C_DIMWHITE, 1, center=True)

        # Speed tick marks
        for deg in range(-220, 41, 26):
            r1, r2 = gauge_r - 22, gauge_r - 8
            a = math.radians(deg)
            p1 = (int(cx + r1 * math.cos(a)), int(gauge_cy + r1 * math.sin(a)))
            p2 = (int(cx + r2 * math.cos(a)), int(gauge_cy + r2 * math.sin(a)))
            cv2.line(c, p1, p2, C_GREY, 1)

        # Speed labels (every 50 km/h)
        for v in range(0, 201, 50):
            deg = -220 + int(260 * v / max_spd)
            a   = math.radians(deg)
            lx  = int(cx + (gauge_r - 38) * math.cos(a))
            ly  = int(gauge_cy + (gauge_r - 38) * math.sin(a))
            _text(c, str(v), (lx, ly + 5), FONT, 0.32, C_DIMWHITE, 1, center=True)

        # ── Steering wheel (bottom half of panel) ─────────────────────
        steer_cy = 420
        steer_r  = 70
        steer_clamp = max(-30.0, min(30.0, steer))
        # Outer ring
        cv2.circle(c, (cx, steer_cy), steer_r, C_BORDER, 3)
        # Angle arc on the wheel rim
        steer_pct = steer_clamp / 30.0         # -1..+1
        arc_start = -90
        arc_end   = int(-90 + steer_pct * 120)
        arc_col   = C_RED if abs(steer_clamp) > 20 else (C_YELLOW if abs(steer_clamp) > 10 else C_BLUE)
        if abs(arc_end - arc_start) > 2:
            _draw_arc(c, cx, steer_cy, steer_r - 6, min(arc_start, arc_end),
                      max(arc_start, arc_end), arc_col, 5)
        # Spokes (rotated by steer angle)
        for spoke_base in [0, 120, 240]:
            angle_deg = spoke_base + steer_clamp * 3
            a = math.radians(angle_deg)
            sx1 = int(cx + 12 * math.cos(a))
            sy1 = int(steer_cy + 12 * math.sin(a))
            sx2 = int(cx + (steer_r - 10) * math.cos(a))
            sy2 = int(steer_cy + (steer_r - 10) * math.sin(a))
            cv2.line(c, (sx1, sy1), (sx2, sy2), C_WHITE, 3)
        # Hub
        cv2.circle(c, (cx, steer_cy), 12, C_PANEL, -1)
        cv2.circle(c, (cx, steer_cy),  8, C_BLUE,   -1)

        # Steering angle number
        _text(c, f"{steer_clamp:+.1f}°", (cx, steer_cy + steer_r + 24),
              FONT_BOLD, 0.65, arc_col, 1, center=True)
        _text(c, "STEERING", (cx, steer_cy + steer_r + 44),
              FONT, 0.38, C_DIMWHITE, 1, center=True)

        # ── Mode badges ────────────────────────────────────────────────
        badge_y = 520
        badges = [
            (nav_state, _nav_color(nav_state)),
            (behavior, _behavior_color(behavior)),
        ]
        total_w = 0
        for txt, _ in badges:
            total_w += cv2.getTextSize(txt, FONT, 0.45, 1)[0][0] + 24
        bx = cx - total_w // 2
        for txt, col in badges:
            tw = cv2.getTextSize(txt, FONT, 0.45, 1)[0][0]
            _filled_rect(c, bx, badge_y, bx + tw + 20, badge_y + 26, col, radius=6)
            _text(c, txt, (bx + 10, badge_y + 18), FONT, 0.42, C_BG, 1)
            bx += tw + 28

        # Guard alert
        if s["guard"]:
            _text(c, "⚠ GUARD ACTIVE", (cx, 560),
                  FONT_BOLD, 0.52, C_RED, 1, center=True)

    # ------------------------------------------------------------------
    # RIGHT PANEL — BEV + status cards
    # ------------------------------------------------------------------
    def _draw_right_panel(self, c: np.ndarray, s: dict) -> None:
        x0 = self.X_R + 10
        y0 = 60
        pw = self.PW_R - 20

        _section_label(c, "BIRD'S EYE VIEW", x0, y0 - 4)

        # BEV image (top portion)
        bev_h = 280
        bev = s["bev_dbg"] if s["bev_dbg"] is not None else s["bev_bgr"]
        if bev is not None:
            bev_img = cv2.resize(bev, (pw, bev_h))
            c[y0:y0 + bev_h, x0:x0 + pw] = bev_img
            _border_rect(c, x0, y0, x0 + pw, y0 + bev_h, C_BORDER, 1, radius=6)

            # Anchor label on BEV
            anchor = s["anchor"]
            acol   = config.ANCHOR_COLORS.get(anchor, C_WHITE)
            _text(c, anchor, (x0 + pw // 2, y0 + bev_h - 10),
                  FONT_BOLD, 0.48, acol, 1, center=True)
        else:
            _filled_rect(c, x0, y0, x0 + pw, y0 + bev_h, C_GREY, radius=6)

        # ── Status cards ──────────────────────────────────────────────
        cy = y0 + bev_h + 14
        cards = [
            ("LOST FRAMES", str(s["lost_frames"]),
             C_RED if s["lost_frames"] > 3 else C_GREEN),
            ("MODE",        s["detect_mode"], C_CYAN),
            ("CURVATURE",   f"{s['curvature']:.4f}", C_BLUE),
            ("ANCHOR",      s["anchor"],
             config.ANCHOR_COLORS.get(s["anchor"], C_DIMWHITE)),
            ("OBSTACLE",    s["obstacle_state"],
             C_RED if s["obstacle_state"] != "CLEAR" else C_GREEN),
        ]
        card_h = 42
        for label, val, col in cards:
            _card(c, x0, cy, pw, card_h, label, val, col)
            cy += card_h + 4

    # ------------------------------------------------------------------
    # BOTTOM TELEMETRY BAR
    # ------------------------------------------------------------------
    def _draw_telemetry_bar(self, c: np.ndarray, s: dict) -> None:
        by = self.PH + 52 + 8     # top of bar
        bh = self.BAR_H - 8
        _filled_rect(c, 0, by, self.W, by + bh, C_PANEL, radius=0)
        cv2.line(c, (0, by), (self.W, by), C_BORDER, 1)

        # ── Traffic light indicator (left) ────────────────────────────
        tl  = s["traffic_light"]
        tlx = 30
        # Three circles: RED YELLOW GREEN (stacked horizontally)
        for i, (name, col, y_off) in enumerate([
            ("RED",    C_RED,    0),
            ("YELLOW", C_YELLOW, 0),
            ("GREEN",  C_GREEN,  0),
        ]):
            cx2 = tlx + i * 36
            cy2 = by + bh // 2
            active = (tl is not None and name in (tl.name if hasattr(tl, "name") else str(tl)).upper())
            fill = col if active else C_GREY
            cv2.circle(c, (cx2, cy2), 14, fill, -1)
            cv2.circle(c, (cx2, cy2), 14, C_BORDER, 1)
        _text(c, "TRAFFIC LIGHT", (tlx + 120, by + bh // 2 + 5),
              FONT, 0.38, C_DIMWHITE, 1, center=True)

        # ── FPS meters ────────────────────────────────────────────────
        fx = 280
        _metric_block(c, fx, by + 10, "CTRL FPS", f"{s['main_fps']:.1f}",
                      C_BLUE if s["main_fps"] > 20 else C_YELLOW)
        _metric_block(c, fx + 160, by + 10, "AI FPS", f"{s['vision_fps']:.1f}",
                      C_BLUE if s["vision_fps"] > 5 else C_YELLOW)

        # ── Separator ─────────────────────────────────────────────────
        cv2.line(c, (fx + 300, by + 10), (fx + 300, by + bh - 10), C_BORDER, 1)

        # ── Heading + curvature ───────────────────────────────────────
        mx = fx + 320
        _metric_block(c, mx, by + 10, "STEER°", f"{s['steer']:+.1f}", C_WHITE)
        _metric_block(c, mx + 160, by + 10, "CURVATURE", f"{s['curvature']:.4f}", C_DIMWHITE)

        # ── Right: signal bars (WiFi/Serial style) ────────────────────
        rx = self.W - 120
        sig_col = C_GREEN if s["serial"] else C_RED
        for i, h in enumerate([8, 14, 20, 26]):
            bx2 = rx + i * 20
            by2 = by + bh - 16 - h
            _filled_rect(c, bx2, by2, bx2 + 14, by + bh - 14, sig_col if i < 4 else C_GREY, radius=2)
        _text(c, "SERIAL" if s["serial"] else "NO CONN",
              (rx + 40, by + 18), FONT, 0.38,
              C_GREEN if s["serial"] else C_RED, 1, center=True)


# ===========================================================================
# DRAWING HELPERS
# ===========================================================================

def _clone(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    return arr.copy() if arr is not None else None


def _text(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font,
    scale: float,
    color: tuple,
    thickness: int = 1,
    center: bool = False,
) -> None:
    if center:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        pos = (pos[0] - tw // 2, pos[1])
    cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


def _filled_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
    radius: int = 0,
    alpha: float = 1.0,
) -> None:
    """Filled rectangle with optional rounded corners."""
    if radius > 0:
        overlay = img.copy()
        _round_rect(overlay, x1, y1, x2, y2, color, radius, -1)
        if alpha < 1.0:
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        else:
            _round_rect(img, x1, y1, x2, y2, color, radius, -1)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)


def _border_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
    thickness: int = 1,
    radius: int = 0,
) -> None:
    if radius > 0:
        _round_rect(img, x1, y1, x2, y2, color, radius, thickness)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def _round_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple, radius: int, thickness: int
) -> None:
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
    for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r),
                   (x1 + r, y2 - r), (x2 - r, y2 - r)]:
        cv2.circle(img, (cx, cy), r, color, thickness)


def _draw_arc(
    img: np.ndarray,
    cx: int, cy: int, r: int,
    start_deg: int, end_deg: int,
    color: tuple, thickness: int,
) -> None:
    """Draw arc using polyline approximation for anti-aliased look."""
    if start_deg >= end_deg:
        return
    pts = []
    for deg in range(start_deg, end_deg + 1, 2):
        a = math.radians(deg)
        pts.append((int(cx + r * math.cos(a)), int(cy + r * math.sin(a))))
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts, np.int32)], False, color, thickness, cv2.LINE_AA)


def _glow_line(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple, thickness: int = 2, blur: int = 3,
) -> None:
    overlay = np.zeros_like(img)
    cv2.line(overlay, (x1, y1), (x2, y2), color, thickness)
    if blur > 0:
        overlay = cv2.GaussianBlur(overlay, (blur * 2 + 1, blur * 2 + 1), 0)
    cv2.add(img, overlay, img)


def _section_label(img: np.ndarray, text: str, x: int, y: int) -> None:
    _text(img, text, (x, y + 14), FONT, 0.38, C_DIMWHITE, 1)
    cv2.line(img, (x, y + 18), (x + 200, y + 18), C_BORDER, 1)


def _card(
    img: np.ndarray,
    x: int, y: int, w: int, h: int,
    label: str, value: str, val_color: tuple,
) -> None:
    _filled_rect(img, x, y, x + w, y + h, C_BG, radius=6)
    _border_rect(img, x, y, x + w, y + h, C_BORDER, 1, radius=6)
    _text(img, label, (x + 8, y + 14), FONT, 0.35, C_DIMWHITE, 1)
    _text(img, value, (x + w - 8, y + h - 8), FONT_BOLD, 0.50, val_color, 1,
          center=False)
    # Right-align value
    (tw, _), _ = cv2.getTextSize(value, FONT_BOLD, 0.50, 1)
    _text(img, value, (x + w - tw - 8, y + h - 8), FONT_BOLD, 0.50, val_color, 1)


def _metric_block(
    img: np.ndarray, x: int, y: int,
    label: str, value: str, color: tuple,
) -> None:
    _text(img, label, (x, y + 12), FONT, 0.34, C_DIMWHITE, 1)
    _text(img, value, (x, y + 36), FONT_BOLD, 0.70, color, 1)


# ---------------------------------------------------------------------------
# COLOR HELPERS
# ---------------------------------------------------------------------------

def _nav_color(state: str) -> tuple:
    return {
        "ROUNDABOUT": C_ORANGE,
        "JUNCTION":   C_YELLOW,
        "NORMAL":     C_GREEN,
    }.get(state, C_DIMWHITE)


def _behavior_color(mode: str) -> tuple:
    return {
        "FULL_STOP":  C_RED,
        "SLOW_STOP":  C_ORANGE,
        "YIELD":      C_YELLOW,
        "HIGHWAY":    C_BLUE,
        "DETOUR":     C_CYAN,
        "NORMAL":     C_GREEN,
    }.get(mode, C_DIMWHITE)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    state = DashboardState()
    state.speed       = 72.5
    state.steer_angle = -12.3
    state.nav_state   = "NORMAL"
    state.behavior_mode = "HIGHWAY"
    state.main_fps    = 28.4
    state.serial_connected = True

    dash = Dashboard(state)
    canvas = dash._render()
    assert canvas.shape == (720, 1280, 3)
    print("dashboard smoke-test PASSED  (shape:", canvas.shape, ")")
