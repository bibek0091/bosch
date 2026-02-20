"""
vision_ai.py — BFMC Autonomous Car System
==========================================
AI inference module. Uses the raw forward-facing camera frame (NOT BEV).

Four detectors, each in its own thread:
  1. Traffic light  — TrafficLightState
  2. Road sign      — SignDetection
  3. Lane divider   — LaneDividerDetection  (CV supplement, not replacement)
  4. Obstacle       — ObstacleDetection     (on zebra crossing)

All results are written to a shared VisionState dataclass under a threading.Lock.
The main loop reads VisionState non-blocking — stale values are fine.

Model requirements:
  - YOLO v8 / ultralytics compatible .pt files
  - Paths configured in config.py (MODEL_TRAFFIC_LIGHT, etc.)
  - If a model file is missing: detector is silently disabled
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


def _load_model(path: str) -> Optional["YOLO"]:
    """Load a YOLO model; return None if file missing or YOLO unavailable."""
    if not _YOLO_AVAILABLE:
        return None
    if not path or not Path(path).exists():
        log.warning("vision_ai: model not found at '%s' — detector disabled", path)
        return None
    try:
        model = YOLO(path)
        log.info("vision_ai: loaded model '%s'", path)
        return model
    except Exception as exc:
        log.warning("vision_ai: failed to load '%s': %s", path, exc)
        return None


# ===========================================================================
# TRAFFIC LIGHT DETECTOR
# ===========================================================================
class TrafficLightDetector:
    """
    Detects traffic light state from raw BGR frame.

    Class names expected in the model:
        0: traffic_light   (whole fixture)
        1: red
        2: yellow
        3: green

    Special rule: if the fixture bounding box is detected but no colour
    sub-class is found → state = DARK (treat as RED).
    """

    CLASS_FIXTURE = "traffic_light"
    CLASS_RED     = "red"
    CLASS_YELLOW  = "yellow"
    CLASS_GREEN   = "green"

    def __init__(self) -> None:
        self._model = _load_model(config.MODEL_TRAFFIC_LIGHT)
        self._enabled = self._model is not None

    def infer(self, frame: np.ndarray) -> TrafficLightState:
        if not self._enabled:
            return TrafficLightState.NONE

        # Resize for speed
        resized = cv2.resize(frame, (config.AI_INFER_W, config.AI_INFER_H))
        try:
            results = self._model(resized, verbose=False,
                                  conf=config.CONF_TRAFFIC_LIGHT)[0]
        except Exception as exc:
            log.debug("TrafficLightDetector: inference error: %s", exc)
            return TrafficLightState.NONE

        names  = results.names
        boxes  = results.boxes
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
            # Light visible but no colour lit → treat as RED (safety)
            return TrafficLightState.DARK

        return TrafficLightState.NONE


# ===========================================================================
# ROAD SIGN DETECTOR
# ===========================================================================
class RoadSignDetector:
    """
    Detects road signs. Class indices defined in config.SIGN_CLASSES.
    Returns the highest-confidence detection above threshold, or None.
    """

    def __init__(self) -> None:
        self._model   = _load_model(config.MODEL_ROAD_SIGN)
        self._enabled = self._model is not None
        # Invert SIGN_CLASSES: index → name
        self._idx_to_name = {v: k for k, v in config.SIGN_CLASSES.items()}

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

        best_idx = int(confs.argmax())
        cls_id   = clss[best_idx]
        sign_name = self._idx_to_name.get(cls_id, f"SIGN_{cls_id}")
        bbox = _scale_bbox(tuple(xyxys[best_idx]), sx, sy)     # type: ignore[arg-type]

        return SignDetection(
            sign_type=sign_name,
            confidence=float(confs[best_idx]),
            bbox=bbox,
        )


# ===========================================================================
# LANE DIVIDER DETECTOR  (AI-assisted, CV primary)
# ===========================================================================
class LaneDividerDetector:
    """
    AI-assisted lane divider detection. Supplements CV — does NOT replace it.
    Returns estimated BEV x position and type.
    """

    def __init__(self) -> None:
        self._model   = _load_model(config.MODEL_LANE_DIVIDER)
        self._enabled = self._model is not None

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

        # Use the highest-confidence box; estimate x as bbox centre
        confs = boxes.conf.cpu().numpy()
        xyxys = boxes.xyxy.cpu().numpy()
        best  = int(confs.argmax())
        x1, _, x2, _ = xyxys[best]
        sx = w_orig / config.AI_INFER_W
        x_centre = float((x1 + x2) / 2.0 * sx)

        return LaneDividerDetection(
            x_position=x_centre,
            divider_type="unknown",
            confidence=float(confs[best]),
        )


# ===========================================================================
# OBSTACLE DETECTOR
# ===========================================================================
class ObstacleDetector:
    """
    Detects pedestrians / objects on the zebra crossing area.
    Uses the road sign model (or a dedicated obstacle model from config).

    Estimates which side of the frame the obstacle is on.
    """

    PERSON_CLASS = "person"

    def __init__(self) -> None:
        # Use dedicated model if it exists, otherwise fall back to sign model
        path          = config.MODEL_OBSTACLE
        self._model   = _load_model(path)
        self._enabled = self._model is not None

    def infer(self, frame: np.ndarray) -> ObstacleDetection:
        if not self._enabled:
            return ObstacleDetection(present=False, bbox=None,
                                     estimated_side=ObstacleSide.NONE)

        w_orig = frame.shape[1]
        resized = cv2.resize(frame, (config.AI_INFER_W, config.AI_INFER_H))
        sx = w_orig / config.AI_INFER_W
        sy = frame.shape[0] / config.AI_INFER_H

        try:
            results = self._model(resized, verbose=False,
                                  conf=config.CONF_OBSTACLE)[0]
        except Exception as exc:
            log.debug("ObstacleDetector: inference error: %s", exc)
            return ObstacleDetection(present=False, bbox=None,
                                     estimated_side=ObstacleSide.NONE)

        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return ObstacleDetection(present=False, bbox=None,
                                     estimated_side=ObstacleSide.NONE)

        confs  = boxes.conf.cpu().numpy()
        xyxys  = boxes.xyxy.cpu().numpy()
        best   = int(confs.argmax())
        x1, y1, x2, y2 = xyxys[best]
        bbox   = _scale_bbox((x1, y1, x2, y2), sx, sy)

        # Estimate which side the obstacle is on (thirds of frame)
        x_centre = (x1 + x2) / 2.0
        third    = config.AI_INFER_W / 3.0
        if x_centre < third:
            side = ObstacleSide.LEFT
        elif x_centre > 2 * third:
            side = ObstacleSide.RIGHT
        else:
            side = ObstacleSide.CENTER

        return ObstacleDetection(present=True, bbox=bbox, estimated_side=side)


# ===========================================================================
# VISION AI COORDINATOR
# ===========================================================================
class VisionAI:
    """
    Co-ordinates all four detectors. Each runs in a daemon thread.

    Usage::

        vision = VisionAI()
        shared_state = VisionState()
        vision.start(shared_state)

        # … main loop …
        with shared_state.lock:
            tl = shared_state.traffic_light

        vision.stop()
    """

    def __init__(self) -> None:
        self._tl_det  = TrafficLightDetector()
        self._sign_det = RoadSignDetector()
        self._div_det  = LaneDividerDetector()
        self._obs_det  = ObstacleDetector()

        self._running        = False
        self._threads: list[threading.Thread] = []
        self._shared: Optional[VisionState]   = None

        # FPS tracking per detector
        self._fps: dict[str, float] = {
            "traffic_light": 0.0,
            "sign":          0.0,
            "divider":       0.0,
            "obstacle":      0.0,
        }

    def start(self, shared_state: VisionState) -> None:
        """Start all detector threads, writing results to shared_state."""
        self._shared  = shared_state
        self._running = True

        specs = [
            ("traffic_light", self._tl_loop),
            ("sign",          self._sign_loop),
            ("divider",       self._div_loop),
            ("obstacle",      self._obs_loop),
        ]
        for name, target in specs:
            t = threading.Thread(target=target, name=f"VisionAI-{name}", daemon=True)
            t.start()
            self._threads.append(t)
        log.info("VisionAI: all detector threads started")

    def stop(self) -> None:
        """Signal all threads to exit."""
        self._running = False
        for t in self._threads:
            t.join(timeout=1.0)
        log.info("VisionAI: stopped")

    def get_fps(self) -> dict[str, float]:
        return dict(self._fps)

    # ------------------------------------------------------------------
    # DETECTOR THREADS
    # ------------------------------------------------------------------

    def _run_detector(
        self,
        name: str,
        infer_fn,
        write_fn,
    ) -> None:
        """Generic detector loop: infer → write to shared state → pace."""
        period = 1.0 / config.TARGET_FPS
        t_fps  = time.monotonic()
        while self._running:
            t0    = time.monotonic()
            frame = self._shared and self._shared  # access via camera below

            # NOTE: frames are injected via set_frame(). See _tl_loop etc.
            time.sleep(period)   # placeholder — real frame injection path below

        # (Real implementation uses self._frame set by VisionAI.push_frame)

    def push_frame(self, frame: np.ndarray) -> None:
        """
        Called by main.py each tick to push the latest raw frame into
        all detector threads.
        """
        self._latest_frame = frame

    def _detect_loop(self, name: str, infer_fn, attr: str) -> None:
        """
        Reusable detector loop. Runs infer_fn on the latest frame and
        writes the result to self._shared.<attr>.
        """
        period = 1.0 / config.TARGET_FPS
        while self._running:
            t0 = time.monotonic()
            frame = getattr(self, "_latest_frame", None)
            if frame is not None and self._shared is not None:
                try:
                    result = infer_fn(frame)
                    with self._shared.lock:
                        setattr(self._shared, attr, result)
                except Exception as exc:
                    log.debug("VisionAI [%s]: %s", name, exc)

            # Pace to FPS
            elapsed    = time.monotonic() - t0
            sleep_t    = max(0.0, period - elapsed)
            time.sleep(sleep_t)

            # EMA FPS
            dt = time.monotonic() - t0
            self._fps[name] = 0.9 * self._fps[name] + 0.1 * (1.0 / max(dt, 1e-6))

    def _tl_loop(self) -> None:
        self._detect_loop("traffic_light", self._tl_det.infer, "traffic_light")

    def _sign_loop(self) -> None:
        self._detect_loop("sign", self._sign_det.infer, "sign")

    def _div_loop(self) -> None:
        self._detect_loop("divider", self._div_det.infer, "lane_divider")

    def _obs_loop(self) -> None:
        self._detect_loop("obstacle", self._obs_det.infer, "obstacle")

    # Initial value so push_frame works before first call
    _latest_frame: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Smoke-test  (python vision_ai.py --sim)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, logging
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    args = parser.parse_args()

    state  = VisionState()
    vision = VisionAI()
    vision.push_frame(np.zeros((config.CAM_H, config.CAM_W, 3), dtype=np.uint8))
    vision.start(state)

    time.sleep(1.0)

    with state.lock:
        tl   = state.traffic_light
        sign = state.sign
        obs  = state.obstacle

    print(f"TrafficLight : {tl}")
    print(f"Sign         : {sign}")
    print(f"Obstacle     : {obs}")
    print(f"Detector FPS : {vision.get_fps()}")

    vision.stop()
    print("vision_ai smoke-test DONE (all detectors disabled without models — expected)")
