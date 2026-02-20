"""
vision_ai.py — BFMC Autonomous Car System
==========================================
AI inference module. Runs on raw forward-facing camera frames.

Four detectors, each in its own daemon thread:
  1. TrafficLightDetector  → VisionState.traffic_light  + bbox
  2. RoadSignDetector      → VisionState.sign           + bbox
  3. LaneDividerDetector   → VisionState.lane_divider   (advisory)
  4. ObstacleDetector      → VisionState.obstacle       + area filter

REAL-LIFE FIXES in this version
--------------------------------
  1. TL class-name detection: fuzzy matching works for any model naming convention.
  2. TL infer() returns (state, bbox) — bbox drawn on raw camera window.
  3. RoadSignDetector: uses model's own results.names (not config index assumption).
  4. Sign name is normalised to UPPER_CASE with underscores before storing.
  5. Obstacle: minimum bbox area filter — ignores tiny far detections.
  6. ObstacleDetector: correct best-detection selection after area filter.
  7. All loops paced at AI_FPS (8 Hz), not TARGET_FPS (30 Hz).
  8. AIOverlayState: thread-safe list of (label, conf, bbox) for raw window.
  9. TL + sign overlays rebuilt atomically after every detection update.
 10. Model class names logged at startup for debugging.
 11. Separate _tl_loop and _sign_loop (not generic) to support overlay rebuild.
 12. DividerGuard edge correction sign fixed (edge pushes LEFT, not right).
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from behavior_engine import (
    TrafficLightState,
    SignDetection,
    LaneDividerDetection,
    ObstacleDetection,
    ObstacleSide,
    VisionState,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ultralytics / YOLO graceful import
# ---------------------------------------------------------------------------
_YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO  # type: ignore
    _YOLO_AVAILABLE = True
    log.info("vision_ai: ultralytics YOLO available")
except ImportError:
    log.warning("vision_ai: ultralytics not installed — all AI detectors disabled")


# ===========================================================================
# HELPERS
# ===========================================================================

def _scale_bbox(bbox: tuple, scale_x: float, scale_y: float) -> tuple[int, int, int, int]:
    """Scale bounding box from inference size back to original frame size."""
    x1, y1, x2, y2 = bbox
    return (int(x1 * scale_x), int(y1 * scale_y),
            int(x2 * scale_x), int(y2 * scale_y))


def _load_model(path: str, name: str) -> Optional["YOLO"]:
    """
    Load a YOLO .pt model. Returns None if file missing, empty, or YOLO not installed.
    Logs the model's class names so engineers can verify the mapping.
    """
    if not _YOLO_AVAILABLE:
        return None
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        log.warning("vision_ai [%s]: model not found at '%s' — DISABLED", name, path)
        return None
    try:
        model = YOLO(str(p))
        mb = p.stat().st_size / 1e6
        log.info("vision_ai [%s]: loaded '%s'  (%.1f MB)", name, p.name, mb)
        # Print class names so engineer can verify — CRITICAL for debugging
        if hasattr(model, "names"):
            log.info("vision_ai [%s]: class names → %s", name, model.names)
        return model
    except Exception as exc:
        log.warning("vision_ai [%s]: failed to load: %s", name, exc)
        return None


def _norm_class_name(raw: str) -> str:
    """
    Normalise a raw class name from any model to one of the canonical tokens.
    Fix 4: Priority check on 'red/yellow/green' before generic 'traffic light'.
    Added 'dark', 'off', 'unlit' mappings.
    """
    s = raw.lower().strip()
    
    # Traffic light colours (Exact color detection takes priority over generic fixture)
    if any(k in s for k in ("red", "stop_light")):
        return "red"
    if any(k in s for k in ("yellow", "amber")):
        return "yellow"
    if any(k in s for k in ("green", "go_light")):
        return "green"
    
    # Generic fixture or dark states
    if any(k in s for k in ("traffic", "light", "signal", "tl", "fixture", "off", "dark", "unlit")):
        # Guard: check if it's just 'light' which might be too generic
        if s == "light" or s == "signal":
            return "traffic_light"
        return "traffic_light"

    # Sign / obstacle: return normalised upper_case
    return raw.upper().replace(" ", "_").replace("-", "_")


# ===========================================================================
# AI OVERLAY STATE  (thread-safe bbox list for raw camera window)
# ===========================================================================
class AIOverlayState:
    """
    Thread-safe store for the latest AI bounding boxes + labels.
    Written by detector threads; read by main loop for cv2.imshow overlay.
    Each entry: (label_str, confidence_float, bbox_tuple_or_None)
    """
    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._items: list[tuple[str, float, Optional[tuple[int,int,int,int]]]] = []

    def set(self, items: list) -> None:
        with self._lock:
            self._items = list(items)

    def get(self) -> list[tuple[str, float, Optional[tuple[int,int,int,int]]]]:
        with self._lock:
            return list(self._items)


# ===========================================================================
# DETECTOR 1 — TRAFFIC LIGHT
# ===========================================================================
class TrafficLightDetector:
    """
    Robust to any model class naming convention.
    Priority: RED > YELLOW > GREEN > DARK (Fix 5 reversed from G>Y>R).
    """

    COLOUR_CONF_MIN = 0.15

    def __init__(self) -> None:
        self._model   = _load_model(config.MODEL_TRAFFIC_LIGHT, "traffic_light")
        self._enabled = self._model is not None
        if self._enabled:
            log.info("TrafficLightDetector: ENABLED")

    def infer(self, frame: np.ndarray
              ) -> tuple[TrafficLightState, Optional[tuple[int,int,int,int]]]:
        null = (TrafficLightState.NONE, None)
        if not self._enabled:
            return null

        h, w = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (config.AI_INFER_W, config.AI_INFER_H))
        sx, sy  = w / config.AI_INFER_W, h / config.AI_INFER_H

        try:
            results = self._model(resized, verbose=False,
                                  conf=self.COLOUR_CONF_MIN)[0]
        except Exception:
            log.warning("TL infer error", exc_info=True)  # Fix 3
            return null

        names = results.names
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return null

        cls_arr  = boxes.cls.cpu().numpy().astype(int)
        conf_arr = boxes.conf.cpu().numpy()
        xyxy_arr = boxes.xyxy.cpu().numpy()

        mapped = []
        for cls_id, conf, xyxy in zip(cls_arr, conf_arr, xyxy_arr):
            token = _norm_class_name(str(names.get(int(cls_id), cls_id)))
            bbox  = _scale_bbox(tuple(xyxy), sx, sy)
            mapped.append((float(conf), token, bbox))

        best: dict[str, tuple[float, tuple]] = {}
        for conf, token, bbox in mapped:
            if token not in best or conf > best[token][0]:
                best[token] = (conf, bbox)

        # Priority: RED > YELLOW > GREEN (Fix 5: Safety priority)
        for token, state in [("red",          TrafficLightState.RED),
                              ("yellow",       TrafficLightState.YELLOW),
                              ("green",        TrafficLightState.GREEN)]:
            if token in best and best[token][0] >= config.CONF_TRAFFIC_LIGHT:
                conf, bbox = best[token]
                log.debug("TL: %s  conf=%.2f  bbox=%s", token, conf, bbox)
                return (state, bbox)

        # Fixture visible but no colour above threshold → DARK (treat as RED)
        if "traffic_light" in best:
            _, bbox = best["traffic_light"]
            log.debug("TL: fixture only → DARK")
            return (TrafficLightState.DARK, bbox)

        return null


# ===========================================================================
# DETECTOR 2 — ROAD SIGN
# ===========================================================================
class RoadSignDetector:
    """
    Fix 2: Uses model ensemble (v1 and v2) from config.
    """

    def __init__(self) -> None:
        self._models = []
        m1 = _load_model(config.MODEL_ROAD_SIGN, "road_sign_v1")
        m2 = _load_model(config.MODEL_ROAD_SIGN_V2, "road_sign_v2")
        if m1: self._models.append(m1)
        if m2: self._models.append(m2)
        
        self._enabled = len(self._models) > 0
        if self._enabled:
            log.info("RoadSignDetector: ENABLED (%d models)", len(self._models))

    def infer(self, frame: np.ndarray) -> Optional[SignDetection]:
        if not self._enabled:
            return None

        h, w = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (config.AI_INFER_W, config.AI_INFER_H))
        sx, sy  = w / config.AI_INFER_W, h / config.AI_INFER_H

        candidates: list[tuple[float, str, tuple]] = []

        for model in self._models:
            try:
                results = model(resized, verbose=False,
                                conf=config.CONF_ROAD_SIGN)[0]
            except Exception:
                log.warning("Sign ensemble infer error", exc_info=True) # Fix 3
                continue

            names = results.names
            boxes = results.boxes
            if boxes is None or len(boxes) == 0:
                continue

            confs = boxes.conf.cpu().numpy()
            clss  = boxes.cls.cpu().numpy().astype(int)
            xyxys = boxes.xyxy.cpu().numpy()

            best_i   = int(confs.argmax())
            cls_id   = int(clss[best_i])
            raw_name = str(names.get(cls_id, ""))
            sign_name = _norm_class_name(raw_name) if raw_name else f"SIGN_{cls_id}"
            bbox      = _scale_bbox(tuple(xyxys[best_i]), sx, sy)
            candidates.append((float(confs[best_i]), sign_name, bbox))

        if not candidates:
            return None

        # Winner = highest confidence across models
        best_conf, sign_name, bbox = max(candidates, key=lambda t: t[0])
        det = SignDetection(sign_type=sign_name, confidence=best_conf, bbox=bbox)
        return det


# ===========================================================================
# DETECTOR 3 — LANE DIVIDER
# ===========================================================================
class LaneDividerDetector:
    def __init__(self) -> None:
        self._model   = _load_model(config.MODEL_LANE_DIVIDER, "lane_divider")
        self._enabled = self._model is not None

    def infer(self, frame: np.ndarray) -> Optional[LaneDividerDetection]:
        if not self._enabled:
            return None

        h, w    = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (config.AI_INFER_W, config.AI_INFER_H))

        try:
            results = self._model(resized, verbose=False,
                                  conf=config.CONF_LANE_DIVIDER)[0]
        except Exception:
            log.warning("Divider infer error", exc_info=True) # Fix 3
            return None

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return None

        confs = boxes.conf.cpu().numpy()
        xyxys = boxes.xyxy.cpu().numpy()
        best  = int(confs.argmax())
        x1, _, x2, _ = xyxys[best]
        sx       = w / config.AI_INFER_W
        x_centre = float(((x1 + x2) / 2.0) * sx)
        x_centre = max(0.0, min(x_centre, float(w)))

        return LaneDividerDetection(x_position=x_centre,
                                    divider_type="unknown",
                                    confidence=float(confs[best]))


# ===========================================================================
# DETECTOR 4 — OBSTACLE
# ===========================================================================
class ObstacleDetector:
    MIN_AREA_FRAC = 0.008

    def __init__(self) -> None:
        self._model   = _load_model(config.MODEL_OBSTACLE, "obstacle")
        self._enabled = self._model is not None

    def infer(self, frame: np.ndarray) -> ObstacleDetection:
        _no = ObstacleDetection(present=False, bbox=None,
                                estimated_side=ObstacleSide.NONE)
        if not self._enabled:
            return _no

        h, w    = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (config.AI_INFER_W, config.AI_INFER_H))
        sx, sy  = w / config.AI_INFER_W, h / config.AI_INFER_H

        try:
            results = self._model(resized, verbose=False,
                                  conf=config.CONF_OBSTACLE)[0]
        except Exception:
            log.warning("Obstacle infer error", exc_info=True) # Fix 3
            return _no

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return _no

        confs = boxes.conf.cpu().numpy()
        xyxys = boxes.xyxy.cpu().numpy()

        min_area = self.MIN_AREA_FRAC * config.AI_INFER_W * config.AI_INFER_H
        valid = [(float(c), tuple(xy)) for c, xy in zip(confs, xyxys)
                 if (xy[2]-xy[0]) * (xy[3]-xy[1]) >= min_area]
        if not valid:
            return _no

        best_conf, best_xyxy = max(valid, key=lambda t: t[0])
        x1, y1, x2, y2 = best_xyxy
        bbox = _scale_bbox((x1, y1, x2, y2), sx, sy)

        x_centre = (x1 + x2) / 2.0
        third    = config.AI_INFER_W / 3.0
        side     = (ObstacleSide.LEFT  if x_centre < third else
                    ObstacleSide.RIGHT if x_centre > 2 * third else
                    ObstacleSide.CENTER)

        return ObstacleDetection(present=True, bbox=bbox, estimated_side=side)


# ===========================================================================
# VISION AI COORDINATOR
# ===========================================================================
class VisionAI:
    def __init__(self) -> None:
        self._running      = False
        self._threads: list = []
        self._shared: Optional[VisionState] = None

        self._frame_lock   = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

        self._tl_det   = TrafficLightDetector()
        self._sign_det = RoadSignDetector()
        self._div_det  = LaneDividerDetector()
        self._obs_det  = ObstacleDetector()

        self._fps_vals: dict[str, float] = {
            k: 0.0 for k in ("traffic_light", "sign", "divider", "obstacle")}

        self._overlay  = AIOverlayState()

        # Heartbeats (Fix 38)
        self._heartbeats: dict[str, float] = {
            k: 0.0 for k in ("traffic_light", "sign", "divider", "obstacle")}

        self._tl_lock  = threading.Lock()
        self._last_tl: tuple[TrafficLightState, Optional[tuple]] = \
            (TrafficLightState.NONE, None)

    def start(self, shared_state: VisionState) -> None:
        self._shared  = shared_state
        self._running = True
        specs = [
            ("traffic_light", self._tl_loop,   self._tl_det._enabled),
            ("sign",          self._sign_loop,  self._sign_det._enabled),
            ("divider",       self._div_loop,   self._div_det._enabled),
            ("obstacle",      self._obs_loop,   self._obs_det._enabled),
        ]
        for name, target, enabled in specs:
            t = threading.Thread(target=target,
                                 name=f"VisionAI-{name}", daemon=True)
            t.start()
            self._threads.append(t)
        log.info("VisionAI: all detector threads started")

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=1.0)
        log.info("VisionAI: stopped")

    def push_frame(self, frame: np.ndarray) -> None:
        """Inject latest camera frame. Fix 7: Copy frame to avoid mid-processing mutation."""
        with self._frame_lock:
            self._latest_frame = frame.copy()

    def get_fps(self) -> float:
        vals = [v for v in self._fps_vals.values() if v > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def get_fps_dict(self) -> dict[str, float]:
        return dict(self._fps_vals)

    def get_heartbeats(self) -> dict[str, float]:
        """Fix 38: Return last iteration timestamps for watchdog."""
        return dict(self._heartbeats)

    def get_overlays(self) -> list[tuple[str, float, Optional[tuple[int,int,int,int]]]]:
        return self._overlay.get()

    def last_tl_result(self) -> tuple[TrafficLightState, Optional[tuple[int,int,int,int]]]:
        with self._tl_lock:
            return self._last_tl

    def _pace(self, name: str, t0: float) -> None:
        period  = 1.0 / config.AI_FPS
        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, period - elapsed))
        total_dt = time.monotonic() - t0
        self._fps_vals[name] = (0.9 * self._fps_vals[name]
                                + 0.1 / max(total_dt, 1e-6))
        # Heartbeat (Fix 38)
        self._heartbeats[name] = time.monotonic()

    def _get_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame

    def _rebuild_overlays(self, tl_state: TrafficLightState,
                           tl_bbox: Optional[tuple]) -> None:
        """Rebuild the unified overlay list. Fix 8: added obstacles."""
        items: list = []
        if tl_bbox and tl_state != TrafficLightState.NONE:
            items.append((f"TL:{tl_state.name}", 1.0, tl_bbox))
        
        if self._shared is not None:
            with self._shared.lock:
                sign = self._shared.sign
                obs  = self._shared.obstacle
            
            if sign is not None and sign.bbox is not None:
                items.append((sign.sign_type, sign.confidence, sign.bbox))
            
            # Fix 8: Draw obstacle if present
            if obs is not None and obs.present and obs.bbox is not None:
                label = f"OBSTACLE:{obs.estimated_side.name}"
                items.append((label, 1.0, obs.bbox))
                
        self._overlay.set(items)

    def _tl_loop(self) -> None:
        while self._running:
            t0    = time.monotonic()
            frame = self._get_frame()
            if frame is not None and self._shared is not None:
                try:
                    tl_state, tl_bbox = self._tl_det.infer(frame)
                    with self._shared.lock:
                        self._shared.traffic_light = tl_state
                    with self._tl_lock:
                        self._last_tl = (tl_state, tl_bbox)
                    self._rebuild_overlays(tl_state, tl_bbox)
                except Exception:
                    log.warning("VisionAI [TL] loop error", exc_info=True)
            self._pace("traffic_light", t0)

    def _sign_loop(self) -> None:
        while self._running:
            t0    = time.monotonic()
            frame = self._get_frame()
            if frame is not None and self._shared is not None:
                try:
                    result = self._sign_det.infer(frame)
                    with self._shared.lock:
                        self._shared.sign = result
                    with self._tl_lock:
                        tl_s, tl_b = self._last_tl
                    self._rebuild_overlays(tl_s, tl_b)
                except Exception:
                    log.warning("VisionAI [Sign] loop error", exc_info=True)
            self._pace("sign", t0)

    def _div_loop(self) -> None:
        while self._running:
            t0    = time.monotonic()
            frame = self._get_frame()
            if frame is not None and self._shared is not None:
                try:
                    result = self._div_det.infer(frame)
                    with self._shared.lock:
                        self._shared.lane_divider = result
                except Exception:
                    log.warning("VisionAI [Divider] loop error", exc_info=True)
            self._pace("divider", t0)

    def _obs_loop(self) -> None:
        while self._running:
            t0    = time.monotonic()
            frame = self._get_frame()
            if frame is not None and self._shared is not None:
                try:
                    result = self._obs_det.infer(frame)
                    with self._shared.lock:
                        self._shared.obstacle = result
                except Exception:
                    log.warning("VisionAI [Obstacle] loop error", exc_info=True)
            self._pace("obstacle", t0)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print(f"\nSign Model paths:\n v1: {config.MODEL_ROAD_SIGN}\n v2: {config.MODEL_ROAD_SIGN_V2}\n")

    vs     = VisionState()
    vision = VisionAI()
    vision.start(vs)
    time.sleep(0.5)

    with vs.lock:
        print(f"TrafficLight  : {vs.traffic_light}")
        print(f"Sign          : {vs.sign}")
        print(f"Obstacle      : {vs.obstacle}")

    print(f"FPS dict      : {vision.get_fps_dict()}")
    print(f"Heartbeats    : {vision.get_heartbeats()}")
    print(f"Overlays      : {vision.get_overlays()}")
    vision.stop()
    print("vision_ai smoke-test DONE")
