"""
vision_ai.py — BFMC Autonomous Car System
==========================================
AI inference module. Runs on raw forward-facing camera frames (NOT BEV —
lane detection and BEV are strictly handled by image_processing.py /
lane_detection.py using the original CV pipeline).

Four detectors, each in its own daemon thread:
  1. TrafficLightDetector   → VisionState.traffic_light
  2. RoadSignDetector       → VisionState.sign
  3. LaneDividerDetector    → VisionState.lane_divider  (AI supplement, CV is primary)
  4. ObstacleDetector       → VisionState.obstacle

Model files — place in  bosch/models/
  traffic_light.pt  — red / yellow / green + fixture class
  road_sign.pt      — highway_entry, zebra, stop, highway_exit, parking, one_way
  obstacle.pt       — pedestrian / object on crossing
  lane_divider.pt   — (optional) lane divider supplement

Any missing .pt file is silently disabled — system runs CV-only.
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
    bbox: tuple[float, float, float, float],
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
    Load a YOLO .pt model from models/ folder.
    Returns None if:  file missing, YOLO not installed, or load fails.
    """
    if not _YOLO_AVAILABLE:
        return None
    p = Path(path)
    if not path or not p.exists():
        log.warning("vision_ai [%s]: model not found at '%s' — detector DISABLED", name, path)
        return None
    try:
        model = YOLO(str(p))
        log.info("vision_ai [%s]: loaded '%s'  (%.1f MB)",
                 name, p.name, p.stat().st_size / 1e6)
        return model
    except Exception as exc:
        log.warning("vision_ai [%s]: failed to load '%s': %s", name, path, exc)
        return None


# ===========================================================================
# DETECTOR 1 — TRAFFIC LIGHT
# ===========================================================================
class TrafficLightDetector:
    """
    Model:  models/traffic_light.pt
    Classes (names dict in model):
        traffic_light   whole fixture
        red
        yellow
        green

    If fixture is detected but no colour sub-class → DARK (treated as RED).
    """

    CLASS_FIXTURE = "traffic_light"
    CLASS_RED     = "red"
    CLASS_YELLOW  = "yellow"
    CLASS_GREEN   = "green"

    def __init__(self) -> None:
        self._model   = _load_model(config.MODEL_TRAFFIC_LIGHT, "traffic_light")
        self._enabled = self._model is not None
        if self._enabled:
            log.info("TrafficLightDetector: ENABLED")

    def infer(self, frame: np.ndarray) -> TrafficLightState:
        if not self._enabled:
            return TrafficLightState.NONE

        resized = cv2.resize(frame, (config.AI_INFER_W, config.AI_INFER_H))
        try:
            results = self._model(resized, verbose=False,
                                  conf=config.CONF_TRAFFIC_LIGHT)[0]
        except Exception as exc:
            log.debug("TrafficLightDetector: inference error: %s", exc)
            return TrafficLightState.NONE

        names = results.names
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return TrafficLightState.NONE

        labels = [names[int(c)] for c in boxes.cls.cpu().numpy()]

        fixture_seen = self.CLASS_FIXTURE in labels
        has_green    = self.CLASS_GREEN  in labels
        has_yellow   = self.CLASS_YELLOW in labels
        has_red      = self.CLASS_RED    in labels

        if has_green:
            return TrafficLightState.GREEN
        if has_yellow:
            return TrafficLightState.YELLOW
        if has_red:
            return TrafficLightState.RED
        if fixture_seen:
            return TrafficLightState.DARK   # fixture seen, no colour lit → treat RED

        return TrafficLightState.NONE


# ===========================================================================
# DETECTOR 2 — ROAD SIGN
# ===========================================================================
class RoadSignDetector:
    """
    Model:  models/road_sign.pt
    Class indices defined in config.SIGN_CLASSES:
        0: HIGHWAY_ENTRY
        1: ZEBRA_CROSSING
        2: STOP_SIGN
        3: HIGHWAY_EXIT
        4: PARKING
        5: ONE_WAY
    Returns the highest-confidence detection above threshold, or None.
    """

    def __init__(self) -> None:
        self._model       = _load_model(config.MODEL_ROAD_SIGN, "road_sign")
        self._enabled     = self._model is not None
        self._idx_to_name = {v: k for k, v in config.SIGN_CLASSES.items()}
        if self._enabled:
            log.info("RoadSignDetector: ENABLED  classes=%s", list(config.SIGN_CLASSES.keys()))

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

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return None

        confs  = boxes.conf.cpu().numpy()
        clss   = boxes.cls.cpu().numpy().astype(int)
        xyxys  = boxes.xyxy.cpu().numpy()

        best_idx  = int(confs.argmax())
        cls_id    = clss[best_idx]
        sign_name = self._idx_to_name.get(cls_id, f"SIGN_{cls_id}")
        bbox      = _scale_bbox(tuple(xyxys[best_idx]), sx, sy)  # type: ignore[arg-type]

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
    Results are written to VisionState.lane_divider.
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

        return LaneDividerDetection(
            x_position   = x_centre,
            divider_type = "unknown",
            confidence   = float(confs[best]),
        )


# ===========================================================================
# DETECTOR 4 — OBSTACLE (pedestrian / object on crossing)
# ===========================================================================
class ObstacleDetector:
    """
    Model:  models/obstacle.pt
    Detects pedestrians / objects on the zebra crossing area.
    Estimates which third of the frame the obstacle occupies.
    """

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

        w_orig  = frame.shape[1]
        h_orig  = frame.shape[0]
        resized = cv2.resize(frame, (config.AI_INFER_W, config.AI_INFER_H))
        sx      = w_orig / config.AI_INFER_W
        sy      = h_orig / config.AI_INFER_H

        try:
            results = self._model(resized, verbose=False,
                                  conf=config.CONF_OBSTACLE)[0]
        except Exception as exc:
            log.debug("ObstacleDetector: inference error: %s", exc)
            return _no_obs

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return _no_obs

        confs        = boxes.conf.cpu().numpy()
        xyxys        = boxes.xyxy.cpu().numpy()
        best         = int(confs.argmax())
        x1, y1, x2, y2 = xyxys[best]
        bbox         = _scale_bbox((x1, y1, x2, y2), sx, sy)

        # Which third of the frame?
        x_centre = (x1 + x2) / 2.0
        third    = config.AI_INFER_W / 3.0
        if x_centre < third:
            side = ObstacleSide.LEFT
        elif x_centre > 2 * third:
            side = ObstacleSide.RIGHT
        else:
            side = ObstacleSide.CENTER

        log.debug("Obstacle: side=%s  conf=%.2f", side.name, float(confs[best]))
        return ObstacleDetection(present=True, bbox=bbox, estimated_side=side)


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

        vision.push_frame(frame)    # call every main-loop tick

        fps = vision.get_fps()      # float — average detector fps

        vision.stop()
    """

    # Frame is shared between main thread (writer) and detector threads (readers)
    _latest_frame: Optional[np.ndarray] = None
    _frame_lock = threading.Lock()

    def __init__(self) -> None:
        self._tl_det   = TrafficLightDetector()
        self._sign_det = RoadSignDetector()
        self._div_det  = LaneDividerDetector()
        self._obs_det  = ObstacleDetector()

        self._running = False
        self._threads: list[threading.Thread] = []
        self._shared:  Optional[VisionState]  = None

        # Per-detector FPS tracking
        self._fps_vals: dict[str, float] = {
            "traffic_light": 0.0,
            "sign":          0.0,
            "divider":       0.0,
            "obstacle":      0.0,
        }

    # ------------------------------------------------------------------
    def start(self, shared_state: VisionState) -> None:
        """Start all detector threads, writing results to shared_state."""
        self._shared  = shared_state
        self._running = True

        _enabled = []
        specs = [
            ("traffic_light", self._tl_loop,  self._tl_det._enabled),
            ("sign",          self._sign_loop, self._sign_det._enabled),
            ("divider",       self._div_loop,  self._div_det._enabled),
            ("obstacle",      self._obs_loop,  self._obs_det._enabled),
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
            log.warning("VisionAI: no detectors active (place .pt files in models/)")
        log.info("VisionAI: all detector threads started")

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=1.0)
        log.info("VisionAI: stopped")

    def push_frame(self, frame: np.ndarray) -> None:
        """Called by main.py every tick to inject the latest camera frame."""
        with self._frame_lock:
            self._latest_frame = frame

    def get_fps(self) -> float:
        """Returns average fps across all active detector threads."""
        vals = [v for v in self._fps_vals.values() if v > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def get_fps_dict(self) -> dict[str, float]:
        """Returns per-detector fps values."""
        return dict(self._fps_vals)

    # ------------------------------------------------------------------
    # INTERNAL: generic detect loop
    # ------------------------------------------------------------------
    def _detect_loop(self, name: str, infer_fn, attr: str) -> None:
        """
        Generic detector loop.
        - Reads latest frame (push_frame sets it)
        - Calls infer_fn(frame)
        - Writes result to self._shared.<attr>
        - Tracks per-detector fps
        """
        period = 1.0 / config.AI_FPS   # pace at AI rate, NOT control-loop rate

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
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))

            total_dt = time.monotonic() - t0
            self._fps_vals[name] = (
                0.9 * self._fps_vals[name]
                + 0.1 * (1.0 / max(total_dt, 1e-6))
            )

    def _tl_loop(self)   -> None: self._detect_loop("traffic_light", self._tl_det.infer,   "traffic_light")
    def _sign_loop(self) -> None: self._detect_loop("sign",          self._sign_det.infer,  "sign")
    def _div_loop(self)  -> None: self._detect_loop("divider",       self._div_det.infer,   "lane_divider")
    def _obs_loop(self)  -> None: self._detect_loop("obstacle",      self._obs_det.infer,   "obstacle")


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # List models found
    models_dir = Path(config.PROJECT_ROOT) / "models"
    found = list(models_dir.glob("*.pt"))
    if found:
        print(f"Models found in {models_dir}:")
        for m in found:
            print(f"  {m.name}  ({m.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"No .pt models in {models_dir}  (detectors will be disabled)")

    state  = VisionState()
    vision = VisionAI()
    vision.push_frame(np.zeros((config.CAM_H, config.CAM_W, 3), dtype=np.uint8))
    vision.start(state)

    time.sleep(1.0)

    with state.lock:
        tl   = state.traffic_light
        sign = state.sign
        obs  = state.obstacle

    print(f"\nTrafficLight  : {tl}")
    print(f"Sign          : {sign}")
    print(f"Obstacle      : {obs}")
    print(f"Detector FPS  : {vision.get_fps_dict()}")

    vision.stop()
    print("vision_ai smoke-test DONE")
