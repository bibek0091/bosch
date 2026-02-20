"""
vision_ai.py — BFMC Autonomous Car System
==========================================
AI inference module. Runs on raw forward-facing camera frames.
Lane detection and BEV are handled strictly by image_processing.py / lane_detection.py.

Four detectors, each in its own daemon thread:
  1. TrafficLightDetector  → VisionState.traffic_light  (+bbox for overlay)
  2. RoadSignDetector      → VisionState.sign           (+bbox for overlay)
  3. LaneDividerDetector   → VisionState.lane_divider   (advisory only)
  4. ObstacleDetector      → VisionState.obstacle       (+min-area filter)

BUG FIXES in this version:
  - TrafficLightDetector: robust class-name detection (handles string names AND
    index-only models). Returns (state, bbox) so overlays work.
  - RoadSignDetector: uses model's own results.names dict — NOT config.SIGN_CLASSES
    index map (which assumes fixed order the model may not follow).
  - ObstacleDetector: minimum bbox area filter to ignore far-away tiny detections.
  - All detectors: paced at config.AI_FPS (8 Hz), not TARGET_FPS (30 Hz).
  - VisionState fields tl_bbox / obs_bbox added for raw-frame overlay.
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

def _scale_bbox(
    bbox: tuple,
    scale_x: float,
    scale_y: float,
) -> tuple[int, int, int, int]:
    """Scale bounding box from inference size back to original frame size."""
    x1, y1, x2, y2 = bbox
    return (
        int(x1 * scale_x),
        int(y1 * scale_y),
        int(x2 * scale_x),
        int(y2 * scale_y),
    )


def _load_model(path: str, name: str) -> Optional["YOLO"]:
    """
    Load a YOLO .pt model.
    Returns None if: file missing, empty path, YOLO not installed, or load fails.
    """
    if not _YOLO_AVAILABLE:
        return None
    if not path:
        log.debug("vision_ai [%s]: path is empty — detector DISABLED", name)
        return None
    p = Path(path)
    if not p.exists():
        log.warning("vision_ai [%s]: model not found at '%s' — detector DISABLED", name, path)
        return None
    try:
        model = YOLO(str(p))
        log.info("vision_ai [%s]: loaded '%s'  (%.1f MB)",
                 name, p.name, p.stat().st_size / 1e6)
        # Print model class names so user can verify mapping
        if hasattr(model, "names"):
            log.info("vision_ai [%s]: class names = %s", name, model.names)
        return model
    except Exception as exc:
        log.warning("vision_ai [%s]: failed to load '%s': %s", name, path, exc)
        return None


def _classify_name(name_str: str) -> Optional[str]:
    """
    Given a raw class name string from the model, return one of:
      'red', 'yellow', 'green', 'traffic_light', or None.
    Case-insensitive. Handles partial matches and common synonyms.
    """
    n = name_str.lower().strip()
    if any(k in n for k in ("red", "stop_light")):
        return "red"
    if any(k in n for k in ("yellow", "amber")):
        return "yellow"
    if any(k in n for k in ("green", "go")):
        return "green"
    if any(k in n for k in ("traffic", "light", "signal", "fixture", "tl")):
        return "traffic_light"
    return None


# ===========================================================================
# DETECTOR 1 — TRAFFIC LIGHT
# ===========================================================================
class TrafficLightDetector:
    """
    Model:  models/traffic_light.pt

    Robust class detection — works with:
      • Named classes: "red", "yellow", "green", "traffic_light"
      • Integer-indexed models where names may be "0", "1", etc.
      • Single-class models (only 'traffic_light' class)

    Returns (TrafficLightState, bbox_or_None) for overlay support.
    Priority: GREEN > YELLOW > RED > DARK (fixture seen, no colour).
    """

    # Minimum confidence for colour classes (can be lower than fixture conf)
    COLOUR_CONF_MIN = 0.35

    def __init__(self) -> None:
        self._model   = _load_model(config.MODEL_TRAFFIC_LIGHT, "traffic_light")
        self._enabled = self._model is not None
        if self._enabled:
            log.info("TrafficLightDetector: ENABLED")

    def infer(self, frame: np.ndarray) -> tuple[TrafficLightState, Optional[tuple[int,int,int,int]]]:
        """
        Returns (state, bbox).  bbox is in original frame coordinates.
        """
        null_result = (TrafficLightState.NONE, None)
        if not self._enabled:
            return null_result

        h_orig, w_orig = frame.shape[:2]
        resized = cv2.resize(frame, (config.AI_INFER_W, config.AI_INFER_H))
        sx = w_orig / config.AI_INFER_W
        sy = h_orig / config.AI_INFER_H

        try:
            results = self._model(resized, verbose=False,
                                  conf=self.COLOUR_CONF_MIN)[0]
        except Exception as exc:
            log.debug("TrafficLightDetector: inference error: %s", exc)
            return null_result

        names = results.names    # dict: {0: "red", 1: "yellow", ...}
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return null_result

        cls_arr   = boxes.cls.cpu().numpy().astype(int)
        conf_arr  = boxes.conf.cpu().numpy()
        xyxy_arr  = boxes.xyxy.cpu().numpy()

        # Map each detection to its normalised class token
        mapped: list[tuple[float, str, tuple]] = []
        for i, (cls_id, conf, xyxy) in enumerate(zip(cls_arr, conf_arr, xyxy_arr)):
            raw_name   = str(names.get(int(cls_id), cls_id))
            class_tok  = _classify_name(raw_name) or raw_name.lower()
            bbox       = _scale_bbox(tuple(xyxy), sx, sy)
            mapped.append((float(conf), class_tok, bbox))

        # Filter to those above the fixture confidence threshold
        mapped = [(c, t, b) for c, t, b in mapped if c >= self.COLOUR_CONF_MIN]
        if not mapped:
            return null_result

        # Pick best detection per colour class
        colours = {"red": None, "yellow": None, "green": None, "traffic_light": None}
        for conf, tok, bbox in mapped:
            if tok in colours:
                if colours[tok] is None or conf > colours[tok][0]:
                    colours[tok] = (conf, bbox)

        best_fixture = colours["traffic_light"]

        # Highest-confidence colour detection wins (above main threshold)
        for tok, state in [("green",  TrafficLightState.GREEN),
                           ("yellow", TrafficLightState.YELLOW),
                           ("red",    TrafficLightState.RED)]:
            if colours[tok] is not None:
                conf, bbox = colours[tok]
                if conf >= config.CONF_TRAFFIC_LIGHT:
                    log.debug("TL detected: %s  conf=%.2f", tok, conf)
                    return (state, bbox)

        # Fixture seen but no colour lit above threshold → DARK
        if best_fixture is not None:
            _, bbox = best_fixture
            log.debug("TL: fixture visible, no colour lit → DARK")
            return (TrafficLightState.DARK, bbox)

        return null_result


# ===========================================================================
# DETECTOR 2 — ROAD SIGN
# ===========================================================================
class RoadSignDetector:
    """
    Model:  models/road_sign.pt

    FIX: uses model's own results.names dict for class resolution,
    NOT config.SIGN_CLASSES (which assumes a fixed index order).

    Falls back to config mapping for indices the model doesn't name.
    """

    def __init__(self) -> None:
        self._model       = _load_model(config.MODEL_ROAD_SIGN, "road_sign")
        self._enabled     = self._model is not None
        # Reverse map from config as fallback (idx → name)
        self._fallback    = {v: k for k, v in config.SIGN_CLASSES.items()}
        if self._enabled:
            log.info("RoadSignDetector: ENABLED")

    def infer(self, frame: np.ndarray) -> Optional[SignDetection]:
        if not self._enabled:
            return None

        h_orig, w_orig = frame.shape[:2]
        resized = cv2.resize(frame, (config.AI_INFER_W, config.AI_INFER_H))
        sx = w_orig / config.AI_INFER_W
        sy = h_orig / config.AI_INFER_H

        try:
            results = self._model(resized, verbose=False,
                                  conf=config.CONF_ROAD_SIGN)[0]
        except Exception as exc:
            log.debug("RoadSignDetector: inference error: %s", exc)
            return None

        names = results.names   # use MODEL's own names dict — not config
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return None

        confs  = boxes.conf.cpu().numpy()
        clss   = boxes.cls.cpu().numpy().astype(int)
        xyxys  = boxes.xyxy.cpu().numpy()

        best_idx  = int(confs.argmax())
        cls_id    = clss[best_idx]

        # Resolve class name: model's names dict first, then config fallback
        raw_name  = str(names.get(int(cls_id), ""))
        sign_name = raw_name.upper().replace(" ", "_")
        if not sign_name:
            sign_name = self._fallback.get(int(cls_id), f"SIGN_{cls_id}")

        bbox = _scale_bbox(tuple(xyxys[best_idx]), sx, sy)  # type: ignore[arg-type]

        det = SignDetection(
            sign_type  = sign_name,
            confidence = float(confs[best_idx]),
            bbox       = bbox,
        )
        log.debug("Sign detected: %s  conf=%.2f", sign_name, det.confidence)
        return det


# ===========================================================================
# DETECTOR 3 — LANE DIVIDER  (AI supplement — CV is primary, this is advisory)
# ===========================================================================
class LaneDividerDetector:
    """
    Model:  models/lane_divider.pt   (optional)
    Supplements the CV lane tracker — does NOT replace it.
    """

    def __init__(self) -> None:
        self._model   = _load_model(config.MODEL_LANE_DIVIDER, "lane_divider")
        self._enabled = self._model is not None
        if self._enabled:
            log.info("LaneDividerDetector: ENABLED (advisory only — CV is primary)")

    def infer(self, frame: np.ndarray) -> Optional[LaneDividerDetection]:
        if not self._enabled:
            return None

        h_orig, w_orig = frame.shape[:2]
        resized = cv2.resize(frame, (config.AI_INFER_W, config.AI_INFER_H))

        try:
            results = self._model(resized, verbose=False,
                                  conf=config.CONF_LANE_DIVIDER)[0]
        except Exception as exc:
            log.debug("LaneDividerDetector: inference error: %s", exc)
            return None

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return None

        confs = boxes.conf.cpu().numpy()
        xyxys = boxes.xyxy.cpu().numpy()
        best  = int(confs.argmax())
        x1, _, x2, _ = xyxys[best]
        sx            = w_orig / config.AI_INFER_W
        x_centre      = float((x1 + x2) / 2.0 * sx)
        # Clamp to frame bounds
        x_centre = max(0.0, min(x_centre, float(w_orig)))

        return LaneDividerDetection(
            x_position   = x_centre,
            divider_type = "unknown",
            confidence   = float(confs[best]),
        )


# ===========================================================================
# DETECTOR 4 — OBSTACLE
# ===========================================================================
class ObstacleDetector:
    """
    Model:  models/obstacle.pt

    FIX: Minimum bbox area filter — ignores tiny far-away detections that
    cover < MIN_AREA_FRAC of the inference frame area.
    """

    MIN_AREA_FRAC = 0.008   # bbox must be ≥ 0.8% of inference frame area

    def __init__(self) -> None:
        self._model   = _load_model(config.MODEL_OBSTACLE, "obstacle")
        self._enabled = self._model is not None
        if self._enabled:
            log.info("ObstacleDetector: ENABLED")

    def infer(self, frame: np.ndarray) -> ObstacleDetection:
        _no_obs = ObstacleDetection(present=False, bbox=None,
                                    estimated_side=ObstacleSide.NONE)
        if not self._enabled:
            return _no_obs

        h_orig, w_orig = frame.shape[:2]
        resized = cv2.resize(frame, (config.AI_INFER_W, config.AI_INFER_H))
        sx = w_orig / config.AI_INFER_W
        sy = h_orig / config.AI_INFER_H

        try:
            results = self._model(resized, verbose=False,
                                  conf=config.CONF_OBSTACLE)[0]
        except Exception as exc:
            log.debug("ObstacleDetector: inference error: %s", exc)
            return _no_obs

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return _no_obs

        confs = boxes.conf.cpu().numpy()
        xyxys = boxes.xyxy.cpu().numpy()

        # Filter by minimum area in inference frame
        min_area = self.MIN_AREA_FRAC * config.AI_INFER_W * config.AI_INFER_H
        valid = []
        for i, (conf, xyxy) in enumerate(zip(confs, xyxys)):
            x1, y1, x2, y2 = xyxy
            area = (x2 - x1) * (y2 - y1)
            if area >= min_area:
                valid.append((float(conf), xyxy))

        if not valid:
            return _no_obs

        _, best_xyxy = max(valid, key=lambda t: t[0])
        x1, y1, x2, y2 = best_xyxy
        conf = float(confs[list(xyxys).index(best_xyxy)] if hasattr(xyxys, 'index') else confs[0])
        bbox = _scale_bbox((x1, y1, x2, y2), sx, sy)

        x_centre = float((x1 + x2) / 2.0)
        third    = config.AI_INFER_W / 3.0
        if x_centre < third:
            side = ObstacleSide.LEFT
        elif x_centre > 2 * third:
            side = ObstacleSide.RIGHT
        else:
            side = ObstacleSide.CENTER

        log.debug("Obstacle: side=%s", side.name)
        return ObstacleDetection(present=True, bbox=bbox, estimated_side=side)


# ===========================================================================
# EXTENDED VISION STATE with bboxes for overlays
# ===========================================================================
class AIOverlayState:
    """
    Thread-safe store for the latest AI bounding boxes + labels for overlay.
    Written by detector threads; read by main loop for cv2.imshow overlay.
    """
    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._overlays: list[tuple[str, float, Optional[tuple[int,int,int,int]]]] = []

    def update(self, items: list) -> None:
        with self._lock:
            self._overlays = list(items)

    def get(self) -> list:
        with self._lock:
            return list(self._overlays)


# ===========================================================================
# VISION AI COORDINATOR
# ===========================================================================
class VisionAI:
    """
    Co-ordinates four detector threads.

    Usage::
        vision = VisionAI()
        state  = VisionState()
        vision.start(state)         # spawns threads

        vision.push_frame(frame)    # call every control loop tick (30fps)
        tl_state, tl_bbox = vision.last_tl_result()
        overlays = vision.get_overlays()   # list of (label, conf, bbox)

        vision.stop()
    """

    def __init__(self) -> None:
        self._running       = False
        self._threads:  list = []
        self._shared: Optional[VisionState] = None

        # Thread-safe frame slot
        self._frame_lock    = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

        # Instantiate detectors
        self._tl_det   = TrafficLightDetector()
        self._sign_det = RoadSignDetector()
        self._div_det  = LaneDividerDetector()
        self._obs_det  = ObstacleDetector()

        # Per-detector EMA FPS tracking
        self._fps_vals: dict[str, float] = {
            "traffic_light": 0.0,
            "sign":          0.0,
            "divider":       0.0,
            "obstacle":      0.0,
        }

        # Overlay state (for raw camera visualization)
        self._overlay = AIOverlayState()
        # Last TL result for main-loop reads
        self._tl_lock   = threading.Lock()
        self._last_tl: tuple[TrafficLightState, Optional[tuple]] = (TrafficLightState.NONE, None)

    def start(self, shared_state: VisionState) -> None:
        self._shared  = shared_state
        self._running = True
        _enabled: list[str] = []

        specs = [
            ("traffic_light", self._tl_loop,      self._tl_det._enabled),
            ("sign",          self._sign_loop,     self._sign_det._enabled),
            ("divider",       self._div_loop,      self._div_det._enabled),
            ("obstacle",      self._obs_loop,      self._obs_det._enabled),
        ]
        for name, target, enabled in specs:
            t = threading.Thread(target=target, name=f"VisionAI-{name}", daemon=True)
            t.start()
            self._threads.append(t)
            if enabled:
                _enabled.append(name)

        if _enabled:
            log.info("VisionAI: active detectors: %s", ", ".join(_enabled))
        else:
            log.warning("VisionAI: no detectors active (place .pt files in models/ and install ultralytics)")
        log.info("VisionAI: all detector threads started")

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=1.0)
        log.info("VisionAI: stopped")

    def push_frame(self, frame: np.ndarray) -> None:
        """Called by main.py every control tick to inject the latest camera frame."""
        with self._frame_lock:
            self._latest_frame = frame

    def get_fps(self) -> float:
        """Returns average fps across all active detector threads."""
        vals = [v for v in self._fps_vals.values() if v > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def get_fps_dict(self) -> dict[str, float]:
        """Returns per-detector fps values."""
        return dict(self._fps_vals)

    def get_overlays(self) -> list[tuple[str, float, Optional[tuple[int,int,int,int]]]]:
        """Returns latest AI detections as [(label, conf, bbox), ...] for overlay."""
        return self._overlay.get()

    def last_tl_result(self) -> tuple[TrafficLightState, Optional[tuple[int,int,int,int]]]:
        """Returns last (TrafficLightState, bbox) from traffic light detector."""
        with self._tl_lock:
            return self._last_tl

    # ------------------------------------------------------------------
    # INTERNAL: generic detect loop
    # ------------------------------------------------------------------
    def _detect_loop(self, name: str, infer_fn, attr: str) -> None:
        """
        Generic detector loop.
        Paced at config.AI_FPS (8 Hz), NOT TARGET_FPS (30 Hz).
        YOLO cannot sustain 30fps inference on a Pi.
        """
        period = 1.0 / config.AI_FPS

        while self._running:
            t0 = time.monotonic()

            with self._frame_lock:
                frame = self._latest_frame

            if frame is not None and self._shared is not None:
                try:
                    result = infer_fn(frame)
                    with self._shared.lock:
                        setattr(self._shared, attr, result)
                except Exception as exc:
                    log.debug("VisionAI [%s] error: %s", name, exc)

            # Pace + FPS tracking
            elapsed  = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))
            total_dt = time.monotonic() - t0
            self._fps_vals[name] = (
                0.9 * self._fps_vals[name]
                + 0.1 * (1.0 / max(total_dt, 1e-6))
            )

    # ------------------------------------------------------------------
    # TL loop — special because it returns (state, bbox)
    # ------------------------------------------------------------------
    def _tl_loop(self) -> None:
        period = 1.0 / config.AI_FPS

        while self._running:
            t0 = time.monotonic()

            with self._frame_lock:
                frame = self._latest_frame

            if frame is not None and self._shared is not None:
                try:
                    tl_state, tl_bbox = self._tl_det.infer(frame)
                    with self._shared.lock:
                        self._shared.traffic_light = tl_state
                    with self._tl_lock:
                        self._last_tl = (tl_state, tl_bbox)
                    # Refresh overlay list to include TL bbox
                    self._rebuild_overlays(tl_state, tl_bbox)
                except Exception as exc:
                    log.debug("VisionAI [traffic_light] error: %s", exc)

            elapsed  = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))
            total_dt = time.monotonic() - t0
            self._fps_vals["traffic_light"] = (
                0.9 * self._fps_vals["traffic_light"]
                + 0.1 * (1.0 / max(total_dt, 1e-6))
            )

    def _rebuild_overlays(self, tl_state: TrafficLightState,
                           tl_bbox: Optional[tuple]) -> None:
        """Rebuild the overlay list after each TL or sign update."""
        items = []
        if tl_bbox and tl_state != TrafficLightState.NONE:
            label = f"TL:{tl_state.name}"
            color_conf = 1.0   # show with full "confidence" for overlay
            items.append((label, color_conf, tl_bbox))
        if self._shared is not None:
            with self._shared.lock:
                sign = self._shared.sign
            if sign is not None:
                items.append((sign.sign_type, sign.confidence, sign.bbox))
        self._overlay.update(items)

    def _sign_loop(self) -> None:
        period = 1.0 / config.AI_FPS

        while self._running:
            t0 = time.monotonic()

            with self._frame_lock:
                frame = self._latest_frame

            if frame is not None and self._shared is not None:
                try:
                    result = self._sign_det.infer(frame)
                    with self._shared.lock:
                        self._shared.sign = result
                    # Rebuild overlays so sign is included
                    with self._tl_lock:
                        tl_s, tl_b = self._last_tl
                    self._rebuild_overlays(tl_s, tl_b)
                except Exception as exc:
                    log.debug("VisionAI [sign] error: %s", exc)

            elapsed  = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))
            total_dt = time.monotonic() - t0
            self._fps_vals["sign"] = (
                0.9 * self._fps_vals["sign"]
                + 0.1 * (1.0 / max(total_dt, 1e-6))
            )

    def _div_loop(self)  -> None: self._detect_loop("divider",  self._div_det.infer,  "lane_divider")
    def _obs_loop(self)  -> None: self._detect_loop("obstacle", self._obs_det.infer,  "obstacle")


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # List models found
    print()
    models_dir = config.MODELS_DIR
    print(f"Models found in {models_dir}:")
    for f in sorted(models_dir.glob("*.pt")):
        print(f"  {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    print()

    vs     = VisionState()
    vision = VisionAI()
    vision.start(vs)

    time.sleep(0.5)

    with vs.lock:
        tl   = vs.traffic_light
        sign = vs.sign
        obs  = vs.obstacle

    print(f"TrafficLight  : {tl}")
    print(f"Sign          : {sign}")
    print(f"Obstacle      : {obs}")
    print(f"Detector FPS  : {vision.get_fps_dict()}")
    print(f"Overlays      : {vision.get_overlays()}")
    print()
    tl_s, tl_b = vision.last_tl_result()
    print(f"Last TL result: state={tl_s}  bbox={tl_b}")

    vision.stop()
    print("vision_ai smoke-test DONE")
