"""
image_processing.py — BFMC Autonomous Car System
=================================================
Converts a raw BGR camera frame into a binary Bird's-Eye-View image
suitable for lane detection.

Pipeline:
  1. Trapezoidal ROI mask  — blacks out sky / car hood
  2. Perspective warp      — cv2.warpPerspective with M from config
  3. HLS conversion        — CLAHE on L channel
  4. Adaptive threshold    — picks bright lane lines under variable light
  5. Morphological close   — fills gaps in dashed/dotted lane markings

Returns both the binary BEV and the colour-warped BEV for visualisation.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)


class ImageProcessor:
    """
    Stateful image processor that caches the perspective matrix
    and CLAHE object across frames.

    Usage::

        proc = ImageProcessor()
        warped_binary, warped_colour = proc.process(frame)
    """

    def __init__(self) -> None:
        # Perspective transform matrix  (src → BEV)
        self._M = cv2.getPerspectiveTransform(config.SRC_PTS, config.DST_PTS)

        # CLAHE for L-channel enhancement
        self._clahe = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_GRID,
        )

        # Morphological close kernel
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, config.MORPH_KERNEL_SIZE
        )

        # ROI mask — only pixels inside SRC_PTS trapezoid are processed
        self._roi_mask = np.zeros((config.CAM_H, config.CAM_W), dtype=np.uint8)
        cv2.fillPoly(self._roi_mask, [config.SRC_PTS.astype(np.int32)], 255)

        log.info("ImageProcessor initialised (BEV %dx%d)", config.BEV_W, config.BEV_H)

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a raw BGR camera frame to a binary BEV image.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR frame from CameraManager (shape CAM_H × CAM_W × 3).
            This array is NOT modified.

        Returns
        -------
        warped_binary : np.ndarray
            Single-channel binary image (0 / 255) in BEV space.
            Used as input for lane_detection.HybridLaneTracker.update().
        warped_colour : np.ndarray
            Colour (BGR) warped frame — used by dashboard for visualisation.
        """
        # Step 1: Apply ROI mask — blacks out regions outside the road trapezoid
        masked = cv2.bitwise_and(frame, frame, mask=self._roi_mask)

        # Step 2: Warp the COLOUR frame first (avoid binary aliasing artefacts)
        warped_colour = cv2.warpPerspective(
            masked, self._M, (config.BEV_W, config.BEV_H)
        )

        # Step 3: Convert to HLS; enhance L channel with CLAHE
        hls = cv2.cvtColor(warped_colour, cv2.COLOR_BGR2HLS)
        L   = self._clahe.apply(hls[:, :, 1])
        S   = hls[:, :, 2]

        # Step 4a: Adaptive threshold on L — good for white lines
        binary_L = cv2.adaptiveThreshold(
            L,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            config.ADAPT_BLOCK_SIZE,
            config.ADAPT_C,
        )

        # Step 4b: Simple threshold on S — picks up yellow / coloured lines
        _, binary_S = cv2.threshold(
            S, config.S_THRESH_LOW, 255, cv2.THRESH_BINARY
        )

        # Combine both binary channels
        binary = cv2.bitwise_or(binary_L, binary_S)

        # Step 5: Morphological close — fills gaps in dashed lane lines
        warped_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._kernel)

        return warped_binary, warped_colour

    def get_perspective_matrix(self) -> np.ndarray:
        """Return the 3×3 perspective transform matrix (for external use)."""
        return self._M


# ---------------------------------------------------------------------------
# Smoke-test  (python image_processing.py --sim)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys
    import logging

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true",
                        help="Use a blank frame (no camera required)")
    args = parser.parse_args()

    # Local import to avoid circular dependency in production
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from camera_manager import CameraManager

    cam  = CameraManager(sim_mode=args.sim)
    cam.start()
    proc = ImageProcessor()

    print("ImageProcessor smoke-test — press 'q' to quit")
    while True:
        frame  = cam.get_frame()
        wb, wc = proc.process(frame)

        # Visualise ROI trapezoid on raw frame
        raw_viz = frame.copy()
        cv2.polylines(
            raw_viz,
            [config.SRC_PTS.astype(np.int32)[[0, 1, 3, 2]]],
            True, (0, 255, 255), 2,
        )

        cv2.imshow("Raw + ROI",      raw_viz)
        cv2.imshow("Warped Colour",  wc)
        cv2.imshow("Warped Binary",  wb)

        if cv2.waitKey(33) == ord("q"):
            break

    cam.stop()
    cv2.destroyAllWindows()
    print("Done.")
