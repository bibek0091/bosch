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

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a raw BGR camera frame to a binary BEV image.
        """
        # Step 1: Apply ROI mask
        masked = cv2.bitwise_and(frame, frame, mask=self._roi_mask)

        # Step 2: Warp the COLOUR frame
        warped_colour = cv2.warpPerspective(
            masked, self._M, (config.BEV_W, config.BEV_H)
        )

        # Step 3: Perspective-space Binary Pipeline (Fix 29, 30, 31)
        binary = self._get_binary(warped_colour)

        # Step 4: Morphological close
        warped_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._kernel)

        return warped_binary, warped_colour

    def _get_binary(self, warped_bgr: np.ndarray) -> np.ndarray:
        """
        Fix 29, 30, 31: Multi-channel binary extraction.
        Combines L-adaptive (whites), S-channel (yellows), and Sobel (edges).
        """
        hls = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HLS)
        L = hls[:, :, 1]
        S = hls[:, :, 2]

        # 1. L-channel Adaptive (Main White Line Detector)
        L_enhanced = self._clahe.apply(L)
        bin_l = cv2.adaptiveThreshold(
            L_enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, config.ADAPT_BLOCK_SIZE, config.ADAPT_C
        )

        # 2. S-channel Threshold (Yellow Line Detector - Fix 29)
        # Fix 31: Optional guard if S_THRESH_LOW is missing or None
        s_thresh = getattr(config, "S_THRESH_LOW", 100)
        _, bin_s = cv2.threshold(S, s_thresh, 255, cv2.THRESH_BINARY)

        # 3. Sobel Gradient Magnitude (Edge Detector - Fix 30)
        # Use L channel for gradient to avoid color noise
        sobelx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
        # Fix 30: Use absolute magnitude
        mag = np.sqrt(sobelx**2 + sobely**2)
        mag_norm = np.uint8(255 * mag / np.max(mag)) if np.max(mag) > 0 else np.zeros_like(L)
        _, bin_sobel = cv2.threshold(mag_norm, 40, 255, cv2.THRESH_BINARY) # fixed sensitivity

        # Combine: OR all signals
        combined = cv2.bitwise_or(bin_l, bin_s)
        combined = cv2.bitwise_or(combined, bin_sobel)

        return combined

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
