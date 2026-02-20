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
    Normalise a raw class name from any model to one of the canonical tokens:
      'red', 'yellow', 'green', 'traffic_light',
      or the upper-cased underscore version for signs.

    Handles: case differences, spaces, numbered IDs, partial matches.
    """
    s = raw.lower().strip()
    # Traffic light colours
    if any(k in s for k in ("red", "stop_light")):
        return "red"
    if any(k in s for k in ("yellow", "amber")):
        return "yellow"
    if any(k in s for k in ("green", "go_light")):
        return "green"
    if any(k in s for k in ("traffic", "light", "signal", "tl", "fixture")):
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
    Returns (TrafficLightState, bbox_or_None) — bbox in original frame coords.

    Priority: GREEN > YELLOW > RED > DARK (fixture with no lit colour).
    Min confidence for colour detection = COLOUR_CONF_MIN.
    """

    COLOUR_CONF_MIN = 0.30   # lower = more sensitive (trade: more false positives)

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
        resized = cv2.resize(frame, (config.AI_INFER_W, config.AI_INFER_H))
        sx, sy  = w / config.AI_INFER_W, h / config.AI_INFER_H

        try:
            results = self._model(resized, verbose=False,
                                  conf=self.COLOUR_CONF_MIN)[0]
        except Exception as exc:
            log.debug("TL infer error: %s", exc)
            return null

        names = results.names
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return null

        cls_arr  = boxes.cls.cpu().numpy().astype(int)
        conf_arr = boxes.conf.cpu().numpy()
        xyxy_arr = boxes.xyxy.cpu().numpy()

        # Build list of (conf, normalised_token, bbox)
        mapped = []
        for cls_id, conf, xyxy in zip(cls_arr, conf_arr, xyxy_arr):
            token = _norm_class_name(str(names.get(int(cls_id), cls_id)))
            bbox  = _scale_bbox(tuple(xyxy), sx, sy)
            mapped.append((float(conf), token, bbox))

        # Best detection per class (highest conf wins)
        best: dict[str, tuple[float, tuple]] = {}
        for conf, token, bbox in mapped:
            if token not in best or conf > best[token][0]:
                best[token] = (conf, bbox)

        # Priority: GREEN > YELLOW > RED > DARK
        for token, state in [("green",        TrafficLightState.GREEN),
                              ("yellow",       TrafficLightState.YELLOW),
                              ("red",          TrafficLightState.RED)]:
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
    Uses model's own results.names dict — NOT config.SIGN_CLASSES index map.
    Sign name is normalised to UPPER_CASE_WITH_UNDERSCORES.
    """

    def __init__(self) -> None:
        self._model    = _load_model(config.MODEL_ROAD_SIGN, "road_sign")
        self._enabled  = self._model is not None
        # Fallback: reverse of config map (index → name)
        self._fallback = {v: k for k, v in config.SIGN_CLASSES.items()}
        if self._enabled:
            log.info("RoadSignDetector: ENABLED")

    def infer(self, frame: np.ndarray) -> Optional[SignDetection]:
        if not self._enabled:
            return None

        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (config.AI_INFER_W, config.AI_INFER_H))
        sx, sy  = w / config.AI_INFER_W, h / config.AI_INFER_H

        try:
            results = self._model(resized, verbose=False,
                                  conf=config.CONF_ROAD_SIGN)[0]
        except Exception as exc:
            log.debug("Sign infer error: %s", exc)
            return None

        names = results.names      # MODEL's own dict — correct!
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return None

        confs  = boxes.conf.cpu().numpy()
        clss   = boxes.cls.cpu().numpy().astype(int)
        xyxys  = boxes.xyxy.cpu().numpy()

        best_i    = int(confs.argmax())
        cls_id    = int(clss[best_i])
        raw_name  = str(names.get(cls_id, ""))
        # Use model name first; fall back to config index map
        sign_name = _norm_class_name(raw_name) if raw_name else \
                    self._fallback.get(cls_id, f"SIGN_{cls_id}")
        bbox      = _scale_bbox(tuple(xyxys[best_i]), sx, sy)  # type: ignore[arg-type]

        det = SignDetection(sign_type=sign_name,
                            confidence=float(confs[best_i]),
                            bbox=bbox)
        log.debug("Sign: %s  conf=%.2f", sign_name, det.confidence)
        return det


# ===========================================================================
# DETECTOR 3 — LANE DIVIDER  (advisory — CV is primary)
# ===========================================================================
class LaneDividerDetector:
    def __init__(self) -> None:
        self._model   = _load_model(config.MODEL_LANE_DIVIDER, "lane_divider")
        self._enabled = self._model is not None
        if self._enabled:
            log.info("LaneDividerDetector: ENABLED (advisory only)")

    def infer(self, frame: np.ndarray) -> Optional[LaneDividerDetection]:
        if not self._enabled:
            return None

        h, w    = frame.shape[:2]
        resized = cv2.resize(frame, (config.AI_INFER_W, config.AI_INFER_H))

        try:
            results = self._model(resized, verbose=False,
                                  conf=config.CONF_LANE_DIVIDER)[0]
        except Exception as exc:
            log.debug("Divider infer error: %s", exc)
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
        x_centre = max(0.0, min(x_centre, float(w)))   # clamp to frame

        return LaneDividerDetection(x_position=x_centre,
                                    divider_type="unknown",
                                    confidence=float(confs[best]))


# ===========================================================================
# DETECTOR 4 — OBSTACLE
# ===========================================================================
class ObstacleDetector:
    """
    Minimum bbox area filter: box must cover >= MIN_AREA_FRAC of inference frame.
    This prevents far-away tiny objects from triggering a detour.
    """

    MIN_AREA_FRAC = 0.008   # 0.8% of AI_INFER_W * AI_INFER_H

    def __init__(self) -> None:
        self._model   = _load_model(config.MODEL_OBSTACLE, "obstacle")
        self._enabled = self._model is not None
        if self._enabled:
            log.info("ObstacleDetector: ENABLED")

    def infer(self, frame: np.ndarray) -> ObstacleDetection:
        _no = ObstacleDetection(present=False, bbox=None,
                                estimated_side=ObstacleSide.NONE)
        if not self._enabled:
            return _no

        h, w    = frame.shape[:2]
        resized = cv2.resize(frame, (config.AI_INFER_W, config.AI_INFER_H))
        sx, sy  = w / config.AI_INFER_W, h / config.AI_INFER_H

        try:
            results = self._model(resized, verbose=False,
                                  conf=config.CONF_OBSTACLE)[0]
        except Exception as exc:
            log.debug("Obstacle infer error: %s", exc)
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

        # FIX: pick highest-confidence valid detection
        best_conf, best_xyxy = max(valid, key=lambda t: t[0])
        x1, y1, x2, y2 = best_xyxy
        bbox = _scale_bbox((x1, y1, x2, y2), sx, sy)

        x_centre = (x1 + x2) / 2.0
        third    = config.AI_INFER_W / 3.0
        side     = (ObstacleSide.LEFT  if x_centre < third else
                    ObstacleSide.RIGHT if x_centre > 2 * third else
                    ObstacleSide.CENTER)

        log.debug("Obstacle: side=%s  conf=%.2f", side.name, best_conf)
        return ObstacleDetection(present=True, bbox=bbox, estimated_side=side)


# ===========================================================================
# VISION AI COORDINATOR
# ===========================================================================
class VisionAI:
    """
    Co-ordinates four detector threads at AI_FPS (not TARGET_FPS).

    Usage::
        vision = VisionAI()
        state  = VisionState()
        vision.start(state)

        # Every control tick:
        vision.push_frame(frame)

        # Read results:
        overlays = vision.get_overlays()          # for raw camera window
        tl_s, tl_b = vision.last_tl_result()
        fps_d = vision.get_fps_dict()

        vision.stop()
    """

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

        self._tl_lock  = threading.Lock()
        self._last_tl: tuple[TrafficLightState, Optional[tuple]] = \
            (TrafficLightState.NONE, None)

    # ------------------------------------------------------------------
    def start(self, shared_state: VisionState) -> None:
        self._shared  = shared_state
        self._running = True
        _enabled: list[str] = []

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
            if enabled:
                _enabled.append(name)

        if _enabled:
            log.info("VisionAI: active detectors: %s", ", ".join(_enabled))
        else:
            log.warning("VisionAI: no detectors active — check models/ and ultralytics install")
        log.info("VisionAI: all detector threads started")

    def stop(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=1.0)
        log.info("VisionAI: stopped")

    def push_frame(self, frame: np.ndarray) -> None:
        """Inject latest camera frame. Called every main-loop tick. Thread-safe."""
        with self._frame_lock:
            self._latest_frame = frame   # only store reference — no copy needed

    def get_fps(self) -> float:
        vals = [v for v in self._fps_vals.values() if v > 0]
        return sum(vals) / len(vals) if vals else 0.0

    def get_fps_dict(self) -> dict[str, float]:
        return dict(self._fps_vals)

    def get_overlays(self) -> list[tuple[str, float, Optional[tuple[int,int,int,int]]]]:
        """Returns latest AI detections as [(label, conf, bbox)] for raw-window overlay."""
        return self._overlay.get()

    def last_tl_result(self) -> tuple[TrafficLightState, Optional[tuple[int,int,int,int]]]:
        """Returns last (TrafficLightState, bbox). Thread-safe."""
        with self._tl_lock:
            return self._last_tl

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------
    def _pace(self, name: str, t0: float) -> None:
        """Sleep remainder of AI_FPS period and update EMA fps counter."""
        period  = 1.0 / config.AI_FPS
        elapsed = time.monotonic() - t0
        time.sleep(max(0.0, period - elapsed))
        total_dt = time.monotonic() - t0
        self._fps_vals[name] = (0.9 * self._fps_vals[name]
                                + 0.1 / max(total_dt, 1e-6))

    def _get_frame(self) -> Optional[np.ndarray]:
        """Thread-safe frame read. Returns None if no frame yet."""
        with self._frame_lock:
            return self._latest_frame

    def _rebuild_overlays(self, tl_state: TrafficLightState,
                           tl_bbox: Optional[tuple]) -> None:
        """Rebuild the unified overlay list from TL + sign state."""
        items: list = []
        if tl_bbox and tl_state != TrafficLightState.NONE:
            items.append((f"TL:{tl_state.name}", 1.0, tl_bbox))
        if self._shared is not None:
            with self._shared.lock:
                sign = self._shared.sign
            if sign is not None and sign.bbox is not None:
                label = sign.sign_type
                items.append((label, sign.confidence, sign.bbox))
        self._overlay.set(items)

    # ------------------------------------------------------------------
    # DETECTOR LOOPS
    # ------------------------------------------------------------------
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
                except Exception as exc:
                    log.debug("VisionAI [traffic_light] error: %s", exc)
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
                    # Rebuild overlays to include updated sign
                    with self._tl_lock:
                        tl_s, tl_b = self._last_tl
                    self._rebuild_overlays(tl_s, tl_b)
                except Exception as exc:
                    log.debug("VisionAI [sign] error: %s", exc)
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
                except Exception as exc:
                    log.debug("VisionAI [divider] error: %s", exc)
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
                except Exception as exc:
                    log.debug("VisionAI [obstacle] error: %s", exc)
            self._pace("obstacle", t0)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    print()
    print(f"Models found in {config.MODELS_DIR}:")
    for f in sorted(config.MODELS_DIR.glob("*.pt")):
        print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")
    print()

    vs     = VisionState()
    vision = VisionAI()
    vision.start(vs)
    time.sleep(0.5)

    with vs.lock:
        print(f"TrafficLight  : {vs.traffic_light}")
        print(f"Sign          : {vs.sign}")
        print(f"Obstacle      : {vs.obstacle}")

    print(f"FPS dict      : {vision.get_fps_dict()}")
    print(f"Overlays      : {vision.get_overlays()}")
    tl_s, tl_b = vision.last_tl_result()
    print(f"Last TL       : state={tl_s}  bbox={tl_b}")
    vision.stop()
    print("vision_ai smoke-test DONE")
