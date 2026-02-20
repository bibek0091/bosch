"""
camera_manager.py — BFMC Autonomous Car System
===============================================
Producer-consumer camera abstraction. One Picamera2 instance distributes
raw BGR frames to two independent consumers:

  Consumer 1 — image_processing.py  : BEV warp + lane detection pipeline
  Consumer 2 — vision_ai.py         : AI inference (natural perspective)

The frame is captured ONCE per tick and shared via a thread-safe buffer.
Both consumers call get_frame() and receive the same numpy array reference.

IMPORTANT: Picamera2 capture_array() ALWAYS returns RGB regardless of the
BGR888 format setting in create_video_configuration (that setting only
affects the DMA hardware buffer layout, not the numpy output). We convert
RGB→BGR once here so all downstream code (BEV, YOLO, dashboard) gets BGR.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful Picamera2 import
# ---------------------------------------------------------------------------
_CAM_AVAILABLE = False
try:
    from picamera2 import Picamera2          # type: ignore
    _CAM_AVAILABLE = True
except ImportError:
    log.warning("picamera2 not found — camera disabled, using blank frames")


class CameraManager:
    """
    Manages a single Picamera2 instance and exposes one shared frame buffer.

    Usage::

        cam = CameraManager(sim_mode=False)
        cam.start()
        frame = cam.get_frame()   # returns latest np.ndarray (H,W,3) BGR
        cam.stop()

    In sim_mode=True (or if camera init fails), get_frame() returns a
    blank (all-zero) BGR frame of size CAM_H × CAM_W.
    """

    def __init__(self, sim_mode: bool = False) -> None:
        self._sim_mode  = sim_mode
        self._cam_ok    = False
        self._running   = False
        self._lock      = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._thread: Optional[threading.Thread] = None

        # Pre-allocate blank fallback frame
        self._blank = np.zeros((config.CAM_H, config.CAM_W, 3), dtype=np.uint8)
        self._frame_count: int = 0
        self._heartbeat: float = 0.0  # Fix 38: Real-time watchdog heartbeat

        if not sim_mode:
            self._init_camera()

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCapture",
            daemon=True,
        )
        self._thread.start()
        log.info("CameraManager: capture thread started (sim=%s)", self._sim_mode)

    def stop(self) -> None:
        """Signal the capture thread to stop and release hardware resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cam_ok:
            try:
                self._picam2.stop()
                log.info("CameraManager: Picamera2 stopped")
            except Exception as exc:
                log.warning("CameraManager: error stopping camera: %s", exc)

    def get_frame(self) -> np.ndarray:
        """
        Return a copy of the latest captured frame (BGR numpy array).
        """
        with self._lock:
            if self._frame is None:
                return self._blank.copy()
            return self._frame.copy()

    @property
    def frame_count(self) -> int:
        """Monotonically increasing counter — use to detect stale frames."""
        return self._frame_count

    @property
    def heartbeat(self) -> float:
        """Fix 38: monotonic timestamp of last capture loop iteration."""
        return self._heartbeat

    @property
    def is_camera_ok(self) -> bool:
        """True if hardware camera is active (False in sim mode or on failure)."""
        return self._cam_ok

    # ------------------------------------------------------------------
    # PRIVATE
    # ------------------------------------------------------------------

    def _init_camera(self) -> None:
        if not _CAM_AVAILABLE:
            log.warning("CameraManager: picamera2 unavailable — falling back to blank frames")
            return
        try:
            self._picam2 = Picamera2()
            cfg = self._picam2.create_video_configuration(
                main={
                    "size":   (config.CAM_W, config.CAM_H),
                    "format": "BGR888",
                },
                controls={"FrameRate": config.TARGET_FPS},
            )
            self._picam2.configure(cfg)
            self._picam2.start()
            self._cam_ok = True
            log.info(
                "CameraManager: Picamera2 started at %dx%d @ %dfps",
                config.CAM_W, config.CAM_H, config.TARGET_FPS,
            )
        except Exception as exc:
            log.warning("CameraManager: camera init failed (%s) — using blank frames", exc)
            self._cam_ok = False

    def _capture_loop(self) -> None:
        """Background thread: continuously capture frames into the shared buffer."""
        period = config.FRAME_PERIOD
        while self._running:
            t0 = time.monotonic()
            self._heartbeat = t0  # Fix 38: Update heartbeat every iteration

            if self._cam_ok:
                try:
                    raw = self._picam2.capture_array()
                    bgr = cv2.cvtColor(np.array(raw, copy=True), cv2.COLOR_RGB2BGR)
                    with self._lock:
                        self._frame = bgr
                        self._frame_count += 1
                except Exception as exc:
                    log.warning("CameraManager: capture error: %s", exc)
                    with self._lock:
                        self._frame = self._blank.copy()
            else:
                # Sim mode or camera failure — produce blank frame
                with self._lock:
                    if self._frame is None:
                        self._frame = self._blank.copy()

            # Pace to TARGET_FPS
            elapsed = time.monotonic() - t0
            sleep_s = period - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# Smoke-test  (python camera_manager.py --sim)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, cv2

    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cam = CameraManager(sim_mode=args.sim)
    cam.start()

    print("CameraManager smoke-test — press 'q' to quit")
    while True:
        frame = cam.get_frame()
        cv2.putText(frame.copy(), f"shape={frame.shape}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("CameraManager test", frame)
        if cv2.waitKey(33) == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()
    print("Done.")
