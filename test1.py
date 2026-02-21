"""
BFMC Hybrid Pilot - Version 3.2 (Debug Mode & RGB Visualization)
"""

import cv2
import numpy as np
import math
import time
import logging
import argparse
import sys

# ---------------------------------------------------------------------------
# Scene Analyzer Import (With detailed error reporting)
# ---------------------------------------------------------------------------
try:
    from bfmc_complete_detector import BFMCCompleteDetector
    _HAS_SCENE_ANALYZER = True
    print("SUCCESS: Loaded external BFMCCompleteDetector")
except ImportError as e:
    print(f"WARNING: Could not load external detector. Reason: {e}")
    print("Falling back to built-in Traffic Light detection only.")
    _HAS_SCENE_ANALYZER = False

# ---------------------------------------------------------------------------
# Serial handler
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, "..")
    from serial_handler import STM32_SerialHandler
    _SERIAL_AVAILABLE = True
except ImportError:
    _SERIAL_AVAILABLE = False
    class STM32_SerialHandler:
        def connect(self): return False
        def set_speed(self, s): pass
        def set_steering(self, s): pass
        def disconnect(self): pass

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
_CAM_AVAILABLE = False
try:
    from picamera2 import Picamera2
    _CAM_AVAILABLE = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ===========================================================================
# PHYSICAL CONSTANTS
# ===========================================================================
WHEELBASE_M          = 0.23
LANE_WIDTH_M         = 0.35
TARGET_FPS           = 30
FRAME_PERIOD         = 1.0 / TARGET_FPS
LOST_GRACE_FRAMES    = 8

SRC_PTS = np.float32([[200, 260], [440, 260], [40,  450], [600, 450]])
DST_PTS = np.float32([[150,   0], [490,   0], [150, 480], [490, 480]])

RIGHT_LANE_OFFSET_PX  = 70
DUAL_OFFSET_PX        = 0
SINGLE_DIV_OFFSET_PX  = 40
SINGLE_EDGE_OFFSET_PX = -40

# ===========================================================================
# BUILT-IN TRAFFIC LIGHT DETECTOR (With RGB Debug)
# ===========================================================================
class TrafficLightDetector:
    def __init__(self):
        self.red_low1 = np.array([0, 120, 70])
        self.red_high1 = np.array([10, 255, 255])
        self.red_low2 = np.array([160, 120, 70])
        self.red_high2 = np.array([180, 255, 255])
        self.red_count = 0
        self.RED_THRESHOLD_FRAMES = 2

    def update(self, frame):
        if frame is None or frame.size == 0:
            return False, None

        # Check only upper half
        roi = frame[0:frame.shape[0]//2, :].copy()
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        tl_dbg = roi.copy() # The RGB Debug window

        mask1 = cv2.inRange(hsv, self.red_low1, self.red_high1)
        mask2 = cv2.inRange(hsv, self.red_low2, self.red_high2)
        full_mask = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0)

        kernel = np.ones((5, 5), np.uint8)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found_red = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Draw ALL detected red blobs in thin blue
            cv2.drawContours(tl_dbg, [cnt], -1, (255, 0, 0), 1)
            cv2.rectangle(tl_dbg, (x, y), (x+w, y+h), (255, 0, 0), 1)
            
            aspect_ratio = float(w) / h if h > 0 else 0

            # --- Spatial & Shape Filters ---
            if 100 < area < 10000:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                # Center text
                cX, cY = x + w//2, y + h//2
                cv2.putText(tl_dbg, f"A:{int(area)} C:{circularity:.2f} AR:{aspect_ratio:.2f}", (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # Rules for a valid traffic light:
                # 1. Circularity > 0.65 OR Aspect Ratio between 0.3 and 1.2
                # 2. Not resting on the very bottom edge of the ROI (y > 200 implies ground level)
                if (circularity > 0.65 or (0.3 <= aspect_ratio <= 1.2)) and y < 200:
                    found_red = True
                    cv2.rectangle(tl_dbg, (x, y), (x+w, y+h), (0, 255, 0), 3) # Green = ACCEPTED
                    cv2.circle(tl_dbg, (cX, cY), 4, (0, 0, 255), -1)
                    break

        if found_red:
            self.red_count += 1
        else:
            self.red_count = 0

        is_red = self.red_count >= self.RED_THRESHOLD_FRAMES

        status_color = (0, 0, 255) if is_red else (0, 255, 0)
        cv2.putText(tl_dbg, f"RED LIGHT: {is_red}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        return is_red, tl_dbg

# ===========================================================================
# TRAFFIC DECISION MAKER
# ===========================================================================
class TrafficDecisionMaker:
    def __init__(self):
        self.stop_sign_timer = 0
        self.cleared_stop_sign = False

    def evaluate(self, base_speed, scene_results, built_in_red_light):
        # 1. Built-in Red Light detector takes absolute priority
        if built_in_red_light:
            return 0.0, "STOP: RED LIGHT (Built-in)"

        if not scene_results:
            return base_speed, ""

        # 2. External Detector Overrides
        if len(scene_results.get('pedestrians', [])) > 0:
            return 0.0, "YIELD: PEDESTRIAN"
            
        if len(scene_results.get('vehicles', [])) > 0:
            return 0.0, "YIELD: VEHICLE"

        for tl in scene_results.get('traffic_lights', []):
            val = tl['sign_type'] if isinstance(tl, dict) else getattr(tl.sign_type, 'value', str(tl))
            if 'RED' in str(val).upper():
                bbox = tl.get('bbox') if isinstance(tl, dict) else getattr(tl, 'bbox', None)
                if bbox is not None:
                    x, y, w, h = bbox
                    area = w * h
                    aspect_ratio = float(w) / h if h > 0 else 0
                    if y > 240 or not (0.3 <= aspect_ratio <= 1.2) or area < 100:
                        continue
                return 0.0, "STOP: RED LIGHT (External)"

        is_stop_sign = False
        for sign in scene_results.get('signs', []):
            val = sign['sign_type'] if isinstance(sign, dict) else getattr(sign.sign_type, 'value', str(sign))
            if 'STOP' in str(val).upper():
                is_stop_sign = True
                break

        if is_stop_sign:
            if not self.cleared_stop_sign:
                if self.stop_sign_timer == 0:
                    self.stop_sign_timer = time.time()
                elapsed = time.time() - self.stop_sign_timer
                if elapsed < 3.0:
                    return 0.0, f"STOP SIGN ({3.0 - elapsed:.1f}s)"
                else:
                    self.cleared_stop_sign = True
        else:
            self.stop_sign_timer = 0
            self.cleared_stop_sign = False

        if len(scene_results.get('crosswalks', [])) > 0:
            return base_speed * 0.5, "SLOW: CROSSWALK"

        if len(scene_results.get('parking_spots', [])) > 0:
            return base_speed * 0.7, "PARKING ZONE"

        return base_speed, ""

# ===========================================================================
# HYBRID LANE TRACKER
# ===========================================================================
class HybridLaneTracker:
    NWINDOWS         = 9
    SW_MARGIN        = 60
    MINPIX           = 50
    POLY_MARGIN_BASE = 60
    POLY_MARGIN_CURV = 120
    MIN_PIX_OK       = 200
    EMA_ALPHA        = 0.50
    STALE_FIT_FRAMES = 5

    def __init__(self, img_shape=(480, 640)):
        self.h, self.w = img_shape
        self.mode       = "SEARCH"
        self.left_fit   = None
        self.right_fit  = None
        self.sl         = None
        self.sr         = None
        self.left_conf  = 0
        self.right_conf = 0
        self.left_stale  = 0
        self.right_stale = 0

    def update(self, warped_binary):
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
            self.left_fit  = fl
            self.sl        = self._ema(self.sl, fl)
            self.left_stale = 0
        else:
            self.left_stale += 1
            if self.left_stale > self.STALE_FIT_FRAMES:
                self.left_fit = None
                self.sl       = None

        if has_r:
            fr = np.polyfit(nzy[ri], nzx[ri], 2)
            self.right_fit  = fr
            self.sr         = self._ema(self.sr, fr)
            self.right_stale = 0
        else:
            self.right_stale += 1
            if self.right_stale > self.STALE_FIT_FRAMES:
                self.right_fit = None
                self.sr        = None

        if has_l and has_r:
            if not self._width_sane(self.left_fit, self.right_fit):
                if self.left_conf < self.right_conf:
                    self.left_fit  = None
                    self.sl        = None
                    self.left_stale = self.STALE_FIT_FRAMES
                    has_l          = False
                else:
                    self.right_fit  = None
                    self.sr         = None
                    self.right_stale = self.STALE_FIT_FRAMES
                    has_r           = False

        self.mode = "TRACKING" if (has_l or has_r or self.sl is not None or self.sr is not None) else "SEARCH"
        return self.sl, self.sr, dbg, mode_label

    def get_target_x(self, y_eval, lane_width_px, extra_offset_px=0, nav_state="NORMAL"):
        sl = self.sl
        sr = self.sr
        hw = lane_width_px / 2.0
        def ev(fit): return float(np.polyval(fit, y_eval))

        if nav_state == "ROUNDABOUT":
            if sl is not None: return ev(sl) + hw + extra_offset_px, "RBT_INNER"
            if sr is not None: return ev(sr) - hw + extra_offset_px, "RBT_OUTER"
            return None, "RBT_LOST"

        if nav_state == "JUNCTION":
            if sr is not None: return ev(sr) - hw + extra_offset_px, "JCT_EDGE"
            if sl is not None: return ev(sl) + hw + extra_offset_px, "JCT_DIV"
            return None, "JCT_LOST"

        if sl is not None and sr is not None:
            return (ev(sl) + ev(sr)) / 2.0 + DUAL_OFFSET_PX, "DUAL"

        if sr is not None and sl is None:
            ghost_sl = sr - np.array([0.0, 0.0, float(lane_width_px)])
            return (ev(ghost_sl) + ev(sr)) / 2.0 + SINGLE_EDGE_OFFSET_PX, "GHOST_L"

        if sl is not None and sr is None:
            ghost_sr = sl + np.array([0.0, 0.0, float(lane_width_px)])
            return (ev(sl) + ev(ghost_sr)) / 2.0 + SINGLE_DIV_OFFSET_PX, "GHOST_R"

        return None, "LOST"

    def get_curvature(self, y_eval):
        fit = self.sr if self.sr is not None else self.sl
        if fit is None: return 0.0
        a, b = fit[0], fit[1]
        num   = abs(2.0 * a)
        denom = (1.0 + (2.0 * a * y_eval + b) ** 2) ** 1.5
        return num / max(denom, 1e-6)

    def _sliding_window(self, warped, nzx, nzy):
        dbg  = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        hist = np.sum(warped[self.h // 2:, :], axis=0)

        mid    = int(self.w * 0.40)
        margin = self.SW_MARGIN
        lb = int(np.argmax(hist[margin : mid - margin])) + margin
        rb = int(np.argmax(hist[mid + margin : self.w - margin])) + mid + margin

        if abs(rb - lb) < 100:
            smoothed = np.convolve(hist.astype(float), np.ones(20) / 20, mode='same')
            p1 = int(np.argmax(smoothed))
            tmp = smoothed.copy()
            tmp[max(0, p1-40):min(self.w, p1+40)] = 0
            p2 = int(np.argmax(tmp))
            lb, rb = (min(p1, p2), max(p1, p2))

        wh = self.h // self.NWINDOWS
        lx, rx = lb, rb
        li, ri = [], []

        for win in range(self.NWINDOWS):
            y_lo = self.h - (win + 1) * wh
            y_hi = self.h - win * wh
            xl0 = max(0, lx - self.SW_MARGIN)
            xl1 = min(self.w, lx + self.SW_MARGIN)
            xr0 = max(0, rx - self.SW_MARGIN)
            xr1 = min(self.w, rx + self.SW_MARGIN)

            cv2.rectangle(dbg, (xl0, y_lo), (xl1, y_hi), (0, 255, 0), 2)
            cv2.rectangle(dbg, (xr0, y_lo), (xr1, y_hi), (0, 255, 0), 2)

            gl = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xl0)  & (nzx < xl1)).nonzero()[0]
            gr = ((nzy >= y_lo) & (nzy < y_hi) & (nzx >= xr0)  & (nzx < xr1)).nonzero()[0]

            li.append(gl)
            ri.append(gr)

            if len(gl) > self.MINPIX: lx = int(np.mean(nzx[gl]))
            if len(gr) > self.MINPIX: rx = int(np.mean(nzx[gr]))

        li = np.concatenate(li)
        ri = np.concatenate(ri)

        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    def _poly_search(self, warped, nzx, nzy, curvature=0.0):
        dbg = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        m = (self.POLY_MARGIN_CURV if curvature > 0.0015 else self.POLY_MARGIN_BASE)
        def band(fit):
            cx = np.polyval(fit, nzy)
            return ((nzx > cx - m) & (nzx < cx + m)).nonzero()[0]
        li = band(self.sl) if self.sl is not None else np.array([], dtype=int)
        ri = band(self.sr) if self.sr is not None else np.array([], dtype=int)

        if len(li) < self.MIN_PIX_OK and len(ri) < self.MIN_PIX_OK:
            self.mode = "SEARCH"
            return self._sliding_window(warped, nzx, nzy)

        if len(li): dbg[nzy[li], nzx[li]] = [255, 80, 80]
        if len(ri): dbg[nzy[ri], nzx[ri]] = [80,  80, 255]
        return li, ri, dbg

    def _width_sane(self, lf, rf, y=400):
        w = np.polyval(rf, y) - np.polyval(lf, y)
        return 80 < w < 560

    def _ema(self, prev, new):
        if prev is None: return new.copy()
        return self.EMA_ALPHA * new + (1.0 - self.EMA_ALPHA) * prev

# ===========================================================================
# JUNCTION DETECTOR
# ===========================================================================
class JunctionDetector:
    ENTRY_FRAMES       = 5
    EXIT_FRAMES        = 8
    CROSS_ENERGY_RATIO = 1.4
    WIDTH_RATIO_HIGH   = 1.6
    MIN_BOT_ENERGY     = 500
    def __init__(self):
        self.state         = "NORMAL"
        self.entry_count   = 0
        self.exit_count    = 0
        self.frames_in_jct = 0
    def update(self, warped_binary, left_conf, right_conf, left_fit, right_fit, lane_width_px):
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
            if (rx - lx) > lane_width_px * self.WIDTH_RATIO_HIGH: wide_lane = True
        evidence = both_lost or cross_energy or wide_lane

        if self.state == "NORMAL":
            self.entry_count = self.entry_count + 1 if evidence else 0
            if self.entry_count >= self.ENTRY_FRAMES:
                self.state, self.exit_count, self.frames_in_jct = "JUNCTION", 0, 0
        elif self.state == "JUNCTION":
            self.frames_in_jct += 1
            self.exit_count = self.exit_count + 1 if not evidence else 0
            if self.exit_count >= self.EXIT_FRAMES and self.frames_in_jct > 15:
                self.state, self.entry_count = "NORMAL", 0
        return self.state

# ===========================================================================
# ROUNDABOUT NAVIGATOR
# ===========================================================================
class RoundaboutNavigator:
    ENTRY_WIDTH_RATIO  = 0.60
    EXIT_WIDTH_RATIO   = 0.82
    MIN_CIRCLE_FRAMES  = 25
    MAX_CIRCLE_FRAMES  = 120
    SPEED_SCALE        = 0.50
    LOOKAHEAD_SCALE    = 0.55
    def __init__(self):
        self.state  = "NORMAL"
        self.frames = 0
    def update(self, left_fit, right_fit, lane_width_px, img_h=480):
        y = img_h - 50
        if left_fit is not None and right_fit is not None:
            lx, rx = np.polyval(left_fit, y), np.polyval(right_fit, y)
            ratio = (rx - lx) / max(float(lane_width_px), 1.0)
            if self.state == "NORMAL":
                if ratio < self.ENTRY_WIDTH_RATIO:
                    self.state, self.frames = "ROUNDABOUT", 0
            elif self.state == "ROUNDABOUT":
                self.frames += 1
                if (self.frames > self.MIN_CIRCLE_FRAMES and ratio > self.EXIT_WIDTH_RATIO) or self.frames > self.MAX_CIRCLE_FRAMES:
                    self.state, self.frames = "NORMAL", 0
        elif self.state == "ROUNDABOUT":
            self.frames += 1
            if self.frames > self.MAX_CIRCLE_FRAMES:
                self.state, self.frames = "NORMAL", 0
        return self.state

# ===========================================================================
# DIVIDER GUARD
# ===========================================================================
class DividerGuard:
    DIVIDER_SAFE_PX = 55
    EDGE_SAFE_PX    = 50
    GAIN            = 0.09
    MAX_CORR        = 8.0
    DEADBAND_PX     = 5
    def apply(self, steer_angle, left_fit, right_fit, y_eval=440, car_x=320):
        correction, speed_scale, triggered   = 0.0, 1.0, False
        div_corr = 0.0
        if left_fit is not None:
            gap = car_x - float(np.polyval(left_fit, y_eval))
            if gap < self.DIVIDER_SAFE_PX - self.DEADBAND_PX:
                err = float(self.DIVIDER_SAFE_PX - gap)
                div_corr = min(self.GAIN * err, self.MAX_CORR)
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 120.0))
                triggered = True
        edge_corr = 0.0
        if right_fit is not None:
            gap = float(np.polyval(right_fit, y_eval)) - car_x
            if gap < self.EDGE_SAFE_PX - self.DEADBAND_PX:
                err = float(self.EDGE_SAFE_PX - gap)
                edge_corr = min(self.GAIN * err, self.MAX_CORR)
                speed_scale = min(speed_scale, max(0.5, 1.0 - err / 120.0))
                triggered = True
        correction = max(div_corr - edge_corr, self.DEADBAND_PX * self.GAIN) if div_corr > 0 and edge_corr > 0 else div_corr - edge_corr
        return steer_angle + correction, speed_scale, triggered

# ===========================================================================
# MAIN PILOT
# ===========================================================================
class BFMC_Pilot:
    STEER_EMA_SLOW = 0.25
    STEER_EMA_FAST = 0.50
    GUARD_EMA      = 0.55
    MAX_STEER      = 30.0
    MAX_STEER_RATE = 5.0
    HIGH_CURV_THRESH = 0.003
    MED_CURV_THRESH  = 0.0015
    HIGH_CURV_SCALE  = 0.60
    MED_CURV_SCALE   = 0.80
    DUAL_SPEED_SCALE = 1.15

    def __init__(self, sim_mode=False):
        self.sim_mode = sim_mode
        self.handler   = STM32_SerialHandler()
        self.connected = False if sim_mode else self.handler.connect()
        self.cam_ok = False
        
        if not sim_mode and _CAM_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                cfg = self.picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"})
                self.picam2.configure(cfg)
                self.picam2.start()
                self.cam_ok = True
            except Exception as e:
                log.warning(f"Camera init failed: {e}")

        self.M     = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        self.tracker        = HybridLaneTracker(img_shape=(480, 640))
        self.rbt            = RoundaboutNavigator()
        self.jct            = JunctionDetector()
        self.guard          = DividerGuard()
        self.decision_maker = TrafficDecisionMaker()
        self.builtin_tl     = TrafficLightDetector() # Built-in Backup
        
        if _HAS_SCENE_ANALYZER:
            self.scene_analyzer = BFMCCompleteDetector(use_templates=False)
        else:
            self.scene_analyzer = None

        self.smooth_steer  = 0.0
        self.smooth_guard  = 0.0
        self.prev_steer    = 0.0
        self.last_target   = 320.0 + RIGHT_LANE_OFFSET_PX
        self.lost_frames   = 0
        self._fps_t, self._fps = time.time(), 0.0

        cv2.namedWindow("BFMC_v3")
        cv2.createTrackbar("Look Ahead",    "BFMC_v3", 150, 300, lambda x: None)
        cv2.createTrackbar("Lane Width PX", "BFMC_v3", 280, 400, lambda x: None)
        cv2.createTrackbar("Fine Offset",   "BFMC_v3",  50, 100, lambda x: None)
        cv2.createTrackbar("Base Speed",    "BFMC_v3",  50, 150, lambda x: None)

    def _get_bev(self, frame):
        warped_colour = cv2.warpPerspective(frame, self.M, (640, 480))
        hls = cv2.cvtColor(warped_colour, cv2.COLOR_BGR2HLS)
        L   = self.clahe.apply(hls[:, :, 1])
        binary = cv2.adaptiveThreshold(L, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    def _pure_pursuit(self, target_x, look_ahead_px, lane_width_px):
        lane_width_px = max(lane_width_px, 50)
        ppm   = lane_width_px / LANE_WIDTH_M
        dx, dy = target_x - 320.0, max(float(look_ahead_px), 1.0)
        ld, alpha = math.sqrt(dx * dx + dy * dy), math.atan2(dx, dy)
        wb_px = WHEELBASE_M * ppm
        return math.degrees(math.atan2(2.0 * wb_px * math.sin(alpha), ld))

    def _draw_poly(self, img, fit, colour):
        if fit is None: return
        ploty = np.linspace(0, 479, 240).astype(np.float32)
        xs    = np.polyval(fit, ploty).astype(np.float32)
        pts = np.stack([xs, ploty], axis=1).reshape(-1, 1, 2).astype(np.int32)
        pts[:, 0, 0] = np.clip(pts[:, 0, 0], 0, 639)
        cv2.polylines(img, [pts], isClosed=False, color=colour, thickness=3)

    def _update_fps(self):
        now = time.time()
        dt, self._fps_t = now - self._fps_t, now
        self._fps = 0.9 * self._fps + 0.1 * (1.0 / max(dt, 1e-6))

    def run(self):
        print("BFMC Pilot v3.2: MULTI-SYSTEM PERCEPTION")
        try:
            while True:
                t_frame_start = time.time()
                look_ahead    = cv2.getTrackbarPos("Look Ahead",    "BFMC_v3")
                lane_width_px = cv2.getTrackbarPos("Lane Width PX", "BFMC_v3")
                fine_offset   = cv2.getTrackbarPos("Fine Offset",   "BFMC_v3")
                base_speed    = cv2.getTrackbarPos("Base Speed",    "BFMC_v3")
                total_offset = RIGHT_LANE_OFFSET_PX + ((fine_offset - 50) * 2)

                if self.cam_ok:
                    frame = self.picam2.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

                # --- 1. Perception Step ---
                # A: Built-in strict RGB Traffic Light Check
                builtin_is_red, tl_dbg_frame = self.builtin_tl.update(frame)
                
                # B: External Scene Analysis
                scene_results, viz_frame = None, None
                if self.scene_analyzer:
                    scene_results = self.scene_analyzer.analyze_scene(frame)
                    viz_frame = self.scene_analyzer.visualize_complete(frame, scene_results)

                # --- 2. Decision Maker ---
                decision_speed, decision_reason = self.decision_maker.evaluate(base_speed, scene_results, builtin_is_red)

                # --- 3. Lane tracking ---
                warped = self._get_bev(frame)
                sl, sr, dbg, detect_mode = self.tracker.update(warped)

                jct_state = self.jct.update(warped, self.tracker.left_conf, self.tracker.right_conf, self.tracker.left_fit, self.tracker.right_fit, lane_width_px)
                rbt_state = self.rbt.update(self.tracker.left_fit, self.tracker.right_fit, lane_width_px)
                nav_state = rbt_state if rbt_state == "ROUNDABOUT" else jct_state

                curvature_pre = self.tracker.get_curvature(self.tracker.h // 2)
                if nav_state == "ROUNDABOUT": eff_la = int(look_ahead * self.rbt.LOOKAHEAD_SCALE)
                elif nav_state == "JUNCTION": eff_la = int(look_ahead * 0.75)
                elif curvature_pre > self.HIGH_CURV_THRESH: eff_la = int(look_ahead * 0.60)
                elif curvature_pre > self.MED_CURV_THRESH: eff_la = int(look_ahead * 0.80)
                else: eff_la = look_ahead
                eff_la, y_eval = max(60, eff_la), max(0, 480 - eff_la)

                target_x, anchor = self.tracker.get_target_x(y_eval, lane_width_px, total_offset, nav_state)
                lost = target_x is None
                if lost: self.lost_frames += 1; target_x = self.last_target
                else: self.lost_frames = 0; self.last_target = target_x

                # --- 4. Steering ---
                raw_steer = self._pure_pursuit(target_x, eff_la, lane_width_px)
                alpha = (self.STEER_EMA_FAST if abs(raw_steer - self.smooth_steer) > 8.0 else self.STEER_EMA_SLOW)
                self.smooth_steer = alpha * raw_steer + (1.0 - alpha) * self.smooth_steer
                steer_angle = self.prev_steer + max(-self.MAX_STEER_RATE, min(self.MAX_STEER_RATE, self.smooth_steer - self.prev_steer))
                self.prev_steer = steer_angle

                guard_left  = (self.tracker.sl if self.tracker.left_stale  == 0 else None)
                guard_right = (self.tracker.sr if self.tracker.right_stale == 0 else None)
                raw_steer_guarded, guard_spd, guard_on = self.guard.apply(steer_angle, guard_left, guard_right, y_eval=y_eval)

                if lost: self.smooth_guard, guard_on = 0.0, False
                else: self.smooth_guard = (self.GUARD_EMA * (raw_steer_guarded - steer_angle) + (1.0 - self.GUARD_EMA) * self.smooth_guard)
                steer_angle = max(-self.MAX_STEER, min(self.MAX_STEER, steer_angle + self.smooth_guard))

                # --- 5. Final Speed ---
                curvature = self.tracker.get_curvature(y_eval)
                if decision_reason != "": speed = decision_speed
                elif self.lost_frames > LOST_GRACE_FRAMES or base_speed == 0: speed = 0.0
                elif nav_state == "ROUNDABOUT": speed = base_speed * self.rbt.SPEED_SCALE
                elif nav_state == "JUNCTION": speed = base_speed * 0.55
                elif curvature > self.HIGH_CURV_THRESH: speed = base_speed * self.HIGH_CURV_SCALE
                elif curvature > self.MED_CURV_THRESH: speed = base_speed * self.MED_CURV_SCALE
                elif anchor == "DUAL" and abs(steer_angle) < 10: speed = base_speed * self.DUAL_SPEED_SCALE
                elif abs(steer_angle) > 18: speed = base_speed * 0.60
                elif abs(steer_angle) > 10: speed = base_speed * 0.80
                else: speed = float(base_speed)

                if 0 < self.lost_frames <= LOST_GRACE_FRAMES: speed *= max(0.3, 1.0 - self.lost_frames / LOST_GRACE_FRAMES)
                if guard_on: speed *= guard_spd

                # --- 6. Actuate ---
                if self.connected: self.handler.set_speed(speed); self.handler.set_steering(steer_angle)

                # --- 7. Visualisation ---
                self._draw_poly(dbg, sl, (255, 220, 0)); self._draw_poly(dbg, sr, (0, 200, 255))
                cv2.circle(dbg, (int(target_x), y_eval), 8, (0, 255, 0), -1)
                cv2.line(dbg, (int(target_x), y_eval), (320, 470), (0, 255, 0), 2)
                cv2.line(dbg, (320, 450), (320, 480), (0, 0, 255), 3)

                if decision_reason != "": cv2.putText(dbg, decision_reason, (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if speed == 0 else (0, 255, 255), 3)
                if guard_on: cv2.putText(dbg, "! GUARD !", (230, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                self._update_fps()
                cv2.putText(dbg, f"{detect_mode} | {anchor} | {nav_state} | {self._fps:.0f}fps", (10,  26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2)
                cv2.putText(dbg, f"Steer:{steer_angle:.1f} Spd:{speed:.0f}", (10, 462), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 200), 2)

                cv2.imshow("BFMC_v3_BEV", dbg)
                if tl_dbg_frame is not None: cv2.imshow("RGB_TL_Debug", tl_dbg_frame)
                if viz_frame is not None: cv2.imshow("BFMC_v3_SCENE", viz_frame)

                wait_ms = max(1, int((FRAME_PERIOD - (time.time() - t_frame_start)) * 1000))
                if cv2.waitKey(wait_ms) == ord("q"): break

        finally: self.stop()

    def stop(self):
        if self.connected: self.handler.set_speed(0); self.handler.set_steering(0); self.handler.disconnect()
        if self.cam_ok: self.picam2.stop()
        cv2.destroyAllWindows()
        print("BFMC Pilot v3: STOPPED")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    args = parser.parse_args()
    BFMC_Pilot(sim_mode=args.sim).run()