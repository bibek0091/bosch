"""
dashboard.py — BFMC Autonomous Car System
==========================================
Tesla-style real-time HUD.

Layout (1280 × 720)
────────────────────────────────────────────────────────────
│  CAMERA + DETECTIONS  │  GAUGES  │  BEV + STATUS CARDS  │
│      (420 × 560)      │ (440×560)│     (420 × 560)      │
└───────────── AI DETECTION BAR (full width, 120px) ───────┘

Concurrency model
-----------------
  Render thread writes to self._canvas under _canvas_lock.
  Main thread calls get_canvas() → cv2.imshow().   ← CV-safe
  cv2.imshow / cv2.waitKey are NEVER called from this module.
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
# Shared state
# ---------------------------------------------------------------------------
@dataclass
class DashboardState:
    raw_frame:   Optional[np.ndarray] = None
    bev_bgr:     Optional[np.ndarray] = None
    bev_dbg:     Optional[np.ndarray] = None

    steer_angle:  float = 0.0
    speed:        float = 0.0
    base_speed:   float = 50.0
    curvature:    float = 0.0
    lost_frames:  int   = 0
    guard_active: bool  = False

    nav_state:   str = "NORMAL"
    anchor:      str = "LOST"
    detect_mode: str = "SLIDE"

    traffic_light:  Optional[object] = None
    sign_label:     Optional[str]    = None
    sign_conf:      Optional[float]  = None
    obstacle_state: str              = "CLEAR"
    behavior_mode:  str              = "NORMAL"

    main_fps:         float = 0.0
    vision_fps:       float = 0.0
    serial_connected: bool  = False
    frame_ts:         float = 0.0
    ai_overlays:      list  = field(default_factory=list)

    lock: threading.Lock = field(default_factory=threading.Lock)


# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
C_BG       = (13,  14,  23)
C_PANEL    = (22,  25,  42)
C_BORDER   = (40,  44,  72)
C_BLUE     = (255, 160,  0)     # electric blue (BGR)
C_CYAN     = (220, 200,  0)
C_GREEN    = (60,  220,  60)
C_RED      = (50,   50, 220)
C_YELLOW   = (0,   200, 220)
C_ORANGE   = (0,   140, 255)
C_WHITE    = (240, 240, 245)
C_DIMWHITE = (130, 135, 155)
C_GREY     = (50,  53,  73)

# Traffic light bulb colours
TL_RED_ON    = (60,  60, 220)
TL_RED_OFF   = (20,  20,  60)
TL_YEL_ON   = (0,  200, 220)
TL_YEL_OFF  = (10,  60,  60)
TL_GRN_ON   = (60, 220,  60)
TL_GRN_OFF  = (10,  50,  10)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def _tl_name(tl) -> str:
    """Safely get string name from TrafficLightState enum or None."""
    if tl is None:
        return "NONE"
    return tl.name if hasattr(tl, "name") else str(tl)


# ===========================================================================
# DASHBOARD
# ===========================================================================
class Dashboard:
    W   = config.DASH_W    # 1280
    H   = config.DASH_H    # 720

    PW_L = 420
    PW_C = 440
    PW_R = 420
    PH   = 580            # panel area height (header bar = 48, AI bar = rest)
    BAR_H = 92            # bottom AI detection bar height

    X_L  = 0
    X_C  = PW_L
    X_R  = PW_L + PW_C

    PANEL_TOP = 50        # top of panels (below header)

    def __init__(self, state: DashboardState) -> None:
        self._state   = state
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._period  = 1.0 / config.DASH_FPS

        self._canvas      = np.full((self.H, self.W, 3), C_BG, dtype=np.uint8)
        self._canvas_lock = threading.Lock()

        self._sign_cache:      str   = ""
        self._sign_conf_cache: float = 0.0
        self._sign_until:      float = 0.0

    def start(self) -> None:
        self._running = True
        self._thread  = threading.Thread(target=self._render_loop,
                                         daemon=True, name="dashboard")
        self._thread.start()
        log.info("Dashboard: render thread started at %d fps", config.DASH_FPS)

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_canvas(self) -> np.ndarray:
        """Called by MAIN THREAD for cv2.imshow — thread-safe."""
        with self._canvas_lock:
            return self._canvas.copy()

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
            time.sleep(max(0.0, self._period - (time.monotonic() - t0)))

    def _render(self) -> np.ndarray:
        with self._state.lock:
            s = self._snapshot()

        c = np.full((self.H, self.W, 3), C_BG, dtype=np.uint8)
        self._draw_header(c, s)
        self._draw_panels_bg(c)
        self._draw_left_panel(c, s)
        self._draw_center_panel(c, s)
        self._draw_right_panel(c, s)
        self._draw_ai_bar(c, s)
        return c

    def _snapshot(self) -> dict:
        s = self._state
        return dict(
            raw_frame      = _clone(s.raw_frame),
            bev_bgr        = _clone(s.bev_bgr),
            bev_dbg        = _clone(s.bev_dbg),
            steer          = s.steer_angle,
            speed          = s.speed,
            base_speed     = s.base_speed,
            curvature      = s.curvature,
            lost_frames    = s.lost_frames,
            guard          = s.guard_active,
            nav_state      = s.nav_state,
            anchor         = s.anchor,
            detect_mode    = s.detect_mode,
            traffic_light  = s.traffic_light,
            sign_label     = s.sign_label,
            sign_conf      = s.sign_conf,
            obstacle_state = s.obstacle_state,
            behavior_mode  = s.behavior_mode,
            main_fps       = s.main_fps,
            vision_fps     = s.vision_fps,
            serial         = s.serial_connected,
            ai_overlays    = list(s.ai_overlays),
            frame_ts       = s.frame_ts,
        )

    # ------------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------------
    def _draw_header(self, c: np.ndarray, s: dict) -> None:
        cv2.rectangle(c, (0, 0), (self.W, self.PANEL_TOP), (15, 17, 30), -1)
        cv2.line(c, (0, self.PANEL_TOP), (self.W, self.PANEL_TOP), C_BORDER, 1)
        _glow_line(c, 0, self.PANEL_TOP, self.W, self.PANEL_TOP, C_BLUE, 2, 3)

        _text(c, "BFMC  AUTONOMOUS  PILOT",
              (self.W // 2, 32), FONT_BOLD, 0.65, C_WHITE, 1, center=True)
        _text(c, time.strftime("%H:%M:%S"),
              (self.W - 90, 32), FONT, 0.42, C_DIMWHITE, 1)

        dot = C_GREEN if s["serial"] else C_RED
        cv2.circle(c, (20, 25), 6, dot, -1)
        _text(c, "SERIAL", (32, 30), FONT, 0.36, C_DIMWHITE, 1)

    # ------------------------------------------------------------------
    # PANEL BACKGROUNDS
    # ------------------------------------------------------------------
    def _draw_panels_bg(self, c: np.ndarray) -> None:
        pt   = self.PANEL_TOP + 8
        pb   = self.PANEL_TOP + self.PH
        pad  = 6
        for x, pw in [(self.X_L, self.PW_L),
                      (self.X_C, self.PW_C),
                      (self.X_R, self.PW_R)]:
            _rr(c, x + pad, pt, x + pw - pad, pb, C_PANEL, 12)
        cv2.line(c, (self.X_C, pt), (self.X_C, pb), C_BORDER, 1)
        cv2.line(c, (self.X_R, pt), (self.X_R, pb), C_BORDER, 1)

    # ------------------------------------------------------------------
    # LEFT PANEL — camera + AI detection overlays
    # ------------------------------------------------------------------
    def _draw_left_panel(self, c: np.ndarray, s: dict) -> None:
        x0  = self.X_L + 12
        y0  = self.PANEL_TOP + 18
        pw  = self.PW_L - 24
        ph  = self.PH  - 36

        _section_label(c, "CAMERA", x0, y0 - 6)

        frame = s["raw_frame"]
        if frame is not None:
            img = cv2.resize(frame, (pw, ph))

            # AI bounding box overlays on camera view
            for label, conf, bbox in s["ai_overlays"]:
                if bbox:
                    x1 = int(bbox[0] * pw / config.CAM_W)
                    y1 = int(bbox[1] * ph / config.CAM_H)
                    x2 = int(bbox[2] * pw / config.CAM_W)
                    y2 = int(bbox[3] * ph / config.CAM_H)
                    cv2.rectangle(img, (x1, y1), (x2, y2), C_BLUE, 2)
                    _text(img, f"{label} {conf:.0%}",
                          (x1 + 4, y1 + 16), FONT, 0.40, C_BLUE, 1)

            c[y0:y0 + ph, x0:x0 + pw] = img
            _border_rect(c, x0, y0, x0 + pw, y0 + ph, C_BORDER, 1, 6)
        else:
            _rr(c, x0, y0, x0 + pw, y0 + ph, C_GREY, 6)
            _text(c, "NO SIGNAL",
                  (x0 + pw // 2, y0 + ph // 2),
                  FONT_BOLD, 0.65, C_DIMWHITE, 1, center=True)

        # Sign hold badge (below camera)
        now = time.monotonic()
        if s["sign_label"]:
            self._sign_cache      = s["sign_label"]
            self._sign_conf_cache = s["sign_conf"] or 0.0
            self._sign_until      = now + config.DASH_SIGN_DISPLAY_SEC
        if now < self._sign_until:
            sy = y0 + ph + 10
            _rr(c, x0, sy, x0 + pw, sy + 28, C_BLUE, 6)
            _text(c, f"SIGN: {self._sign_cache}  {self._sign_conf_cache:.0%}",
                  (x0 + pw // 2, sy + 20),
                  FONT_BOLD, 0.46, C_BG, 1, center=True)

    # ------------------------------------------------------------------
    # CENTRE PANEL — speed arc gauge + steering wheel
    # ------------------------------------------------------------------
    def _draw_center_panel(self, c: np.ndarray, s: dict) -> None:
        cx      = self.X_C + self.PW_C // 2
        speed   = s["speed"]
        steer   = s["steer"]
        nav     = s["nav_state"]
        behav   = s["behavior_mode"]

        _section_label(c, "VEHICLE STATUS", self.X_C + 14, self.PANEL_TOP + 12)

        # ── Speed arc gauge ──────────────────────────────────────────
        g_cy = self.PANEL_TOP + 180
        g_r  = 115
        _draw_arc(c, cx, g_cy, g_r, -220, 40, C_GREY, 12)
        max_sp = 200.0
        pct    = min(speed / max_sp, 1.0)
        if pct > 0:
            s_col   = C_RED if speed > 150 else (C_YELLOW if speed > 100 else C_BLUE)
            end_deg = -220 + int(260 * pct)
            _draw_arc(c, cx, g_cy, g_r, -220, end_deg, s_col, 12)
            tip_a = math.radians(end_deg)
            tx = int(cx + g_r * math.cos(tip_a))
            ty = int(g_cy + g_r * math.sin(tip_a))
            cv2.circle(c, (tx, ty), 10, s_col, -1)
            cv2.circle(c, (tx, ty),  6, C_WHITE, -1)
        else:
            s_col = C_DIMWHITE

        # Speed number
        _text(c, str(int(speed)), (cx, g_cy + 10), FONT_BOLD, 2.4, C_WHITE, 2, center=True)
        _text(c, "km/h",          (cx, g_cy + 48), FONT,      0.52, C_DIMWHITE, 1, center=True)

        # Tick marks every 26°
        for deg in range(-220, 41, 26):
            a  = math.radians(deg)
            p1 = (int(cx + (g_r-22)*math.cos(a)), int(g_cy+(g_r-22)*math.sin(a)))
            p2 = (int(cx + (g_r-8) *math.cos(a)), int(g_cy+(g_r-8) *math.sin(a)))
            cv2.line(c, p1, p2, C_GREY, 1)
        for v in range(0, 201, 50):
            deg = -220 + int(260 * v / max_sp)
            a   = math.radians(deg)
            lx  = int(cx + (g_r-38)*math.cos(a))
            ly  = int(g_cy+(g_r-38)*math.sin(a))
            _text(c, str(v), (lx, ly+5), FONT, 0.30, C_DIMWHITE, 1, center=True)

        # ── Steering wheel ───────────────────────────────────────────
        w_cy  = self.PANEL_TOP + 390
        w_r   = 65
        sc    = max(-30.0, min(30.0, steer))
        cv2.circle(c, (cx, w_cy), w_r, C_BORDER, 3)
        pct_s = sc / 30.0
        a_end = int(-90 + pct_s * 120)
        a_col = C_RED if abs(sc) > 20 else (C_YELLOW if abs(sc) > 10 else C_BLUE)
        if abs(a_end - (-90)) > 2:
            _draw_arc(c, cx, w_cy, w_r - 5, min(-90, a_end), max(-90, a_end), a_col, 5)
        for base in [0, 120, 240]:
            a = math.radians(base + sc * 3)
            cv2.line(c,
                     (int(cx+12*math.cos(a)), int(w_cy+12*math.sin(a))),
                     (int(cx+(w_r-10)*math.cos(a)), int(w_cy+(w_r-10)*math.sin(a))),
                     C_WHITE, 3)
        cv2.circle(c, (cx, w_cy), 12, C_PANEL, -1)
        cv2.circle(c, (cx, w_cy),  8, C_BLUE,  -1)
        _text(c, f"{sc:+.1f}°", (cx, w_cy + w_r + 22), FONT_BOLD, 0.62, a_col, 1, center=True)
        _text(c, "STEERING",    (cx, w_cy + w_r + 42), FONT,      0.36, C_DIMWHITE, 1, center=True)

        # ── Mode badges ──────────────────────────────────────────────
        by = self.PANEL_TOP + 492
        for txt, col in [(nav, _nav_col(nav)), (behav, _behav_col(behav))]:
            tw = cv2.getTextSize(txt, FONT, 0.42, 1)[0][0]
            _rr(c, cx - tw//2 - 10, by, cx + tw//2 + 10, by + 24, col, 6)
            _text(c, txt, (cx, by + 17), FONT, 0.42, C_BG, 1, center=True)
            by += 32

        if s["guard"]:
            _text(c, "  GUARD ACTIVE  ",
                  (cx, self.PANEL_TOP + 560),
                  FONT_BOLD, 0.52, C_RED, 1, center=True)

    # ------------------------------------------------------------------
    # RIGHT PANEL — BEV + status cards
    # ------------------------------------------------------------------
    def _draw_right_panel(self, c: np.ndarray, s: dict) -> None:
        x0  = self.X_R + 10
        y0  = self.PANEL_TOP + 18
        pw  = self.PW_R - 20

        _section_label(c, "BIRD'S EYE VIEW", x0, y0 - 6)

        bev_h = 260
        bev   = s["bev_dbg"] if s["bev_dbg"] is not None else s["bev_bgr"]
        if bev is not None:
            bev_img = cv2.resize(bev, (pw, bev_h))
            c[y0:y0 + bev_h, x0:x0 + pw] = bev_img
            _border_rect(c, x0, y0, x0 + pw, y0 + bev_h, C_BORDER, 1, 6)
            anc = s["anchor"]
            _text(c, anc,
                  (x0 + pw // 2, y0 + bev_h - 10),
                  FONT_BOLD, 0.46,
                  config.ANCHOR_COLORS.get(anc, C_WHITE), 1, center=True)
        else:
            _rr(c, x0, y0, x0 + pw, y0 + bev_h, C_GREY, 6)

        cy = y0 + bev_h + 12
        cards = [
            ("LOST FRAMES", str(s["lost_frames"]),
             C_RED if s["lost_frames"] > 3 else C_GREEN),
            ("DETECT MODE", s["detect_mode"],   C_CYAN),
            ("CURVATURE",  f"{s['curvature']:.4f}", C_BLUE),
            ("ANCHOR",      s["anchor"],
             config.ANCHOR_COLORS.get(s["anchor"], C_DIMWHITE)),
            ("OBSTACLE",    s["obstacle_state"],
             C_RED if s["obstacle_state"] != "CLEAR" else C_GREEN),
        ]
        for label, val, col in cards:
            _card(c, x0, cy, pw, 40, label, val, col)
            cy += 44

    # ------------------------------------------------------------------
    # BOTTOM AI DETECTION BAR  (full width)
    # ------------------------------------------------------------------
    def _draw_ai_bar(self, c: np.ndarray, s: dict) -> None:
        by  = self.PANEL_TOP + self.PH + 4
        bh  = self.H - by
        cv2.rectangle(c, (0, by), (self.W, self.H), (15, 17, 30), -1)
        cv2.line(c, (0, by), (self.W, by), C_BORDER, 1)
        _glow_line(c, 0, by, self.W, by, C_BLUE, 1, 2)

        tl_name = _tl_name(s["traffic_light"])
        cy_mid  = by + bh // 2

        # ── Large traffic light widget (left) ────────────────────────
        tlx   = 30
        bul_r = 16
        bul_gap = 46
        for i, (name, on_c, off_c) in enumerate([
            ("RED",    TL_RED_ON,  TL_RED_OFF),
            ("YELLOW", TL_YEL_ON,  TL_YEL_OFF),
            ("GREEN",  TL_GRN_ON,  TL_GRN_OFF),
        ]):
            bx    = tlx + i * bul_gap
            lit   = name in tl_name.upper() and tl_name != "NONE"
            color = on_c if lit else off_c
            # housing
            cv2.circle(c, (bx, cy_mid), bul_r + 4, C_GREY, -1)
            # bulb
            cv2.circle(c, (bx, cy_mid), bul_r, color, -1)
            if lit:
                # glow halo
                overlay = np.zeros_like(c)
                cv2.circle(overlay, (bx, cy_mid), bul_r + 8, color, -1)
                overlay = cv2.GaussianBlur(overlay, (15, 15), 0)
                cv2.add(c, overlay, c)
            _text(c, name[0], (bx, cy_mid + 5),
                  FONT, 0.30, C_WHITE if lit else C_GREY, 1, center=True)

        # Traffic light label
        tl_col = (C_GREEN if "GREEN" in tl_name
                  else C_RED if tl_name in ("RED","DARK")
                  else C_YELLOW if "YELLOW" in tl_name
                  else C_DIMWHITE)
        _text(c, f"TL: {tl_name}", (tlx + bul_gap * 3 + 10, cy_mid + 6),
              FONT_BOLD, 0.55, tl_col, 1)

        # ── Vertical divider ─────────────────────────────────────────
        sep1 = 310
        cv2.line(c, (sep1, by + 8), (sep1, self.H - 8), C_BORDER, 1)

        # ── Sign detection ───────────────────────────────────────────
        sx = sep1 + 18
        _text(c, "SIGN", (sx, by + 20), FONT, 0.38, C_DIMWHITE, 1)
        now = time.monotonic()
        if s["sign_label"]:
            self._sign_cache      = s["sign_label"]
            self._sign_conf_cache = s["sign_conf"] or 0.0
            self._sign_until      = now + config.DASH_SIGN_DISPLAY_SEC
        if now < self._sign_until:
            _text(c, self._sign_cache,
                  (sx, cy_mid + 4), FONT_BOLD, 0.60, C_ORANGE, 1)
            _text(c, f"{self._sign_conf_cache:.0%}",
                  (sx + 180, cy_mid + 4), FONT, 0.48, C_DIMWHITE, 1)
        else:
            _text(c, "---", (sx, cy_mid + 4), FONT_BOLD, 0.60, C_GREY, 1)

        # ── Vertical divider ─────────────────────────────────────────
        sep2 = 560
        cv2.line(c, (sep2, by + 8), (sep2, self.H - 8), C_BORDER, 1)

        # ── FPS meters ───────────────────────────────────────────────
        fx = sep2 + 20
        _metric(c, fx,        by + 10, "CTRL FPS", f"{s['main_fps']:.1f}",
                C_BLUE if s["main_fps"] > 20 else C_YELLOW)
        _metric(c, fx + 160,  by + 10, "AI FPS",   f"{s['vision_fps']:.1f}",
                C_BLUE if s["vision_fps"] > 4  else C_YELLOW)

        # ── Vertical divider ─────────────────────────────────────────
        sep3 = 830
        cv2.line(c, (sep3, by + 8), (sep3, self.H - 8), C_BORDER, 1)

        # ── Obstacle + behavior ──────────────────────────────────────
        ox   = sep3 + 20
        obs  = s["obstacle_state"]
        obs_col = C_RED if obs != "CLEAR" else C_GREEN
        _metric(c, ox,      by + 10, "OBSTACLE", obs,         obs_col)
        _metric(c, ox + 160, by + 10, "BEHAV",   s["behavior_mode"], _behav_col(s["behavior_mode"]))

        # ── Serial signal bars (far right) ────────────────────────
        rx    = self.W - 110
        sc_c  = C_GREEN if s["serial"] else C_RED
        for i, h in enumerate([8, 14, 20, 26]):
            bx2  = rx + i * 20
            top  = self.H - 14 - h
            _rr(c, bx2, top, bx2 + 14, self.H - 14, sc_c, 2)
        _text(c, "OK" if s["serial"] else "NO",
              (rx + 44, by + 20), FONT_BOLD, 0.42,
              C_GREEN if s["serial"] else C_RED, 1, center=True)


# ===========================================================================
# DRAWING HELPERS
# ===========================================================================

def _clone(a: Optional[np.ndarray]) -> Optional[np.ndarray]:
    return a.copy() if a is not None else None


def _text(img, text, pos, font, scale, color, thick=1, center=False):
    if center:
        (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
        pos = (pos[0] - tw // 2, pos[1])
    cv2.putText(img, text, pos, font, scale, color, thick, cv2.LINE_AA)


def _rr(img, x1, y1, x2, y2, color, radius=0, fill=-1):
    """Rounded rectangle."""
    r = min(radius, (x2-x1)//2, (y2-y1)//2)
    if r <= 0:
        cv2.rectangle(img, (x1,y1),(x2,y2), color, fill)
        return
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, fill)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, fill)
    for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
        cv2.circle(img, (cx,cy), r, color, fill)


def _border_rect(img, x1, y1, x2, y2, color, thick=1, radius=0):
    if radius > 0:
        _rr(img, x1, y1, x2, y2, color, radius, thick)
    else:
        cv2.rectangle(img, (x1,y1),(x2,y2), color, thick)


def _draw_arc(img, cx, cy, r, start, end, color, thick):
    if start >= end:
        return
    pts = []
    for d in range(start, end+1, 2):
        a = math.radians(d)
        pts.append((int(cx+r*math.cos(a)), int(cy+r*math.sin(a))))
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts, np.int32)], False, color, thick, cv2.LINE_AA)


def _glow_line(img, x1, y1, x2, y2, color, thick=2, blur=3):
    ov = np.zeros_like(img)
    cv2.line(ov, (x1,y1),(x2,y2), color, thick)
    ov = cv2.GaussianBlur(ov, (blur*2+1,blur*2+1), 0)
    cv2.add(img, ov, img)


def _section_label(img, text, x, y):
    _text(img, text, (x, y+14), FONT, 0.36, C_DIMWHITE, 1)
    cv2.line(img, (x, y+18), (x+200, y+18), C_BORDER, 1)


def _card(img, x, y, w, h, label, value, val_col):
    _rr(img, x, y, x+w, y+h, C_BG, 6)
    _border_rect(img, x, y, x+w, y+h, C_BORDER, 1, 6)
    _text(img, label, (x+8, y+13), FONT, 0.32, C_DIMWHITE, 1)
    tw = cv2.getTextSize(value, FONT_BOLD, 0.50, 1)[0][0]
    _text(img, value, (x+w-tw-8, y+h-8), FONT_BOLD, 0.50, val_col, 1)


def _metric(img, x, y, label, value, color):
    _text(img, label, (x, y+12), FONT, 0.32, C_DIMWHITE, 1)
    _text(img, value, (x, y+36), FONT_BOLD, 0.68, color, 1)


def _nav_col(state):
    return {"ROUNDABOUT": C_ORANGE, "JUNCTION": C_YELLOW, "NORMAL": C_GREEN}.get(state, C_DIMWHITE)

def _behav_col(mode):
    return {"FULL_STOP": C_RED, "SLOW": C_YELLOW, "HIGHWAY": C_BLUE,
            "DETOUR": C_CYAN, "NORMAL": C_GREEN}.get(mode, C_DIMWHITE)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    from behavior_engine import TrafficLightState

    state = DashboardState()
    state.speed         = 72.0
    state.steer_angle   = -12.0
    state.nav_state     = "NORMAL"
    state.behavior_mode = "NORMAL"
    state.main_fps      = 28.0
    state.serial_connected = True
    state.traffic_light = TrafficLightState.RED

    dash   = Dashboard(state)
    canvas = dash._render()
    assert canvas.shape == (config.DASH_H, config.DASH_W, 3)
    print("dashboard smoke-test PASSED  shape:", canvas.shape)
    print("Traffic light rendered for RED state.")
