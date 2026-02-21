"""
BFMC Hybrid Pilot - Version 2.3
FIXED: AttributeError: find_contours -> findContours
FIXED: Stability counters for Red Light detection
"""

import cv2
import numpy as np
import math
import time
import logging
import argparse
import sys

# ---------------------------------------------------------------------------
# Serial & Camera fallbacks
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

_CAM_AVAILABLE = False
try:
    from picamera2 import Picamera2
    _CAM_AVAILABLE = True
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ===========================================================================
# CONSTANTS
# ===========================================================================
WHEELBASE_M      = 0.23
LANE_WIDTH_M     = 0.35
TARGET_FPS       = 30
FRAME_PERIOD     = 1.0 / TARGET_FPS
LOST_GRACE_FRAMES = 8
RIGHT_LANE_OFFSET_PX = 70

SRC_PTS = np.float32([[200, 260], [440, 260], [40,  450], [600, 450]])
DST_PTS = np.float32([[150,   0], [490,   0], [150, 480], [490, 480]])

# ===========================================================================
# TRAFFIC LIGHT DETECTOR (Red Light Stop Logic)
# ===========================================================================
class TrafficLightDetector:
    def __init__(self):
        # Red spans two ranges in HSV (0-10 and 160-180)
        self.red_low1 = np.array([0, 120, 70])
        self.red_high1 = np.array([10, 255, 255])
        self.red_low2 = np.array([160, 120, 70])
        self.red_high2 = np.array([180, 255, 255])
        
        self.red_count = 0
        self.RED_STABILITY_THRESHOLD = 2 

    def update(self, frame):
        if frame is None: return False
        
        # Traffic lights are in the top half of the raw image
        roi = frame[0:frame.shape[0]//2, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, self.red_low1, self.red_high1)
        mask2 = cv2.inRange(hsv, self.red_low2, self.red_high2)
        full_mask = cv2.addWeighted(mask1, 1.0, mask2, 1.0, 0)

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)

        # CRITICAL FIX: Changed find_contours to findContours
        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        found_red_now = False
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 40 < area < 5000:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                # Circularity: 4*pi*Area / Perimeter^2
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if circularity > 0.65:
                    found_red_now = True
                    break

        if found_red_now:
            self.red_count += 1
        else:
            self.red_count = 0

        return self.red_count >= self.RED_STABILITY_THRESHOLD

# ===========================================================================
# LANE TRACKING & NAVIGATION
# ===========================================================================
class HybridLaneTracker:
    def __init__(self, img_shape=(480, 640)):
        self.h, self.w = img_shape
        self.sl = None
        self.sr = None

    def update(self, warped_binary):
        dbg = cv2.cvtColor(warped_binary, cv2.COLOR_GRAY2BGR)
        # Placeholder for your sliding window logic
        return self.sl, self.sr, dbg

    def get_target_x(self, offset_px):
        # Default positioning for lane keeping
        return 320 + offset_px

# ===========================================================================
# MAIN PILOT
# ===========================================================================
class BFMC_Pilot:
    def __init__(self, sim_mode=False):
        self.sim_mode = sim_mode
        self.handler = STM32_SerialHandler()
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
                log.error(f"Camera init failed: {e}")

        self.M = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)
        self.tracker = HybridLaneTracker()
        self.tl_detector = TrafficLightDetector()
        self.prev_steer = 0.0
        self.lost_frames = 0
        
        cv2.namedWindow("BFMC_v2.3")
        cv2.createTrackbar("Look Ahead", "BFMC_v2.3", 150, 300, lambda x: None)
        cv2.createTrackbar("Base Speed", "BFMC_v2.3", 40, 120, lambda x: None)

    def run(self):
        log.info("BFMC Pilot v2.3 Starting...")
        try:
            while True:
                t_frame = time.time()
                look_ahead = cv2.getTrackbarPos("Look Ahead", "BFMC_v2.3")
                base_speed = cv2.getTrackbarPos("Base Speed", "BFMC_v2.3")

                # 1. Image Capture
                frame = self.picam2.capture_array() if self.cam_ok else np.zeros((480, 640, 3), np.uint8)

                # 2. Traffic Light Check (Raw Frame)
                
                stop_condition = self.tl_detector.update(frame)

                # 3. Lane Detection (BEV)
                warped = cv2.warpPerspective(frame, self.M, (640, 480))
                gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
                
                sl, sr, dbg = self.tracker.update(binary)
                target_x = self.tracker.get_target_x(RIGHT_LANE_OFFSET_PX)

                # 4. Steering Logic
                dx = target_x - 320
                dy = max(look_ahead, 1)
                steer_angle = math.degrees(math.atan2(2.0 * WHEELBASE_M * dx, dy**2))
                steer_angle = np.clip(steer_angle, -30, 30)

                # 5. Speed Policy with Red Light Override
                if stop_condition:
                    speed = 0.0
                    cv2.putText(dbg, "STOP: RED LIGHT", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                else:
                    speed = float(base_speed)
                    if abs(steer_angle) > 15: speed *= 0.65

                # 6. Command STM32
                if self.connected:
                    self.handler.set_speed(speed)
                    self.handler.set_steering(steer_angle)

                # Visual Debug
                cv2.circle(dbg, (int(target_x), 480 - look_ahead), 8, (0, 255, 0), -1)
                cv2.imshow("BFMC_v2.3", dbg)
                
                if cv2.waitKey(1) == ord("q"): break
                
                # Keep FPS stable
                elapsed = time.time() - t_frame
                time.sleep(max(0, FRAME_PERIOD - elapsed))

        finally: self.stop()

    def stop(self):
        if self.connected:
            self.handler.set_speed(0)
            self.handler.set_steering(0)
            self.handler.disconnect()
        if self.cam_ok: self.picam2.stop()
        cv2.destroyAllWindows()
        log.info("Pilot Stopped Gracefully.")

if __name__ == "__main__":
    BFMC_Pilot(sim_mode=("--sim" in sys.argv)).run()