"""Court-plane projection, never a reconstruction of an airborne ball."""

import cv2
import numpy as np

from baseline_lab.contracts import Calibration, Point

WIDTH_M = 8.23
LENGTH_M = 23.77
COURT = np.array([[0, 0], [WIDTH_M, 0], [WIDTH_M, LENGTH_M], [0, LENGTH_M]], np.float32)


class Court:
    def __init__(self, calibration: Calibration):
        src = np.asarray(calibration.corners_px, dtype=np.float32)
        if not np.isfinite(src).all() or not cv2.isContourConvex(src):
            raise ValueError("Select four distinct corners in the indicated clockwise order")
        if abs(cv2.contourArea(src)) < 1000:
            raise ValueError("Calibration area is too small or degenerate")
        # Both left-to-right direction and near/far ordering must hold in this supported view.
        if src[0, 0] >= src[1, 0] or src[3, 0] >= src[2, 0]:
            raise ValueError("Corner order must be far-left, far-right, near-right, near-left")
        if src[:2, 1].mean() >= src[2:, 1].mean():
            raise ValueError("Near baseline must appear below far baseline")
        self.h = np.asarray(cv2.getPerspectiveTransform(src, COURT), dtype=np.float64)
        self.inverse = np.linalg.inv(self.h)
        if np.linalg.cond(self.h) > 1e8:
            raise ValueError("Ill-conditioned calibration")
        self.corners = calibration.corners_px
        roundtrip = [self.image(self.project(tuple(p))) for p in src]
        self.fit_residual_px = float(np.max(np.linalg.norm(np.asarray(roundtrip) - src, axis=1)))

    def project(self, p: Point) -> Point:
        return transform(self.h, p)

    def image(self, p: Point) -> Point:
        return transform(self.inverse, p)

    def details(self) -> dict:
        return {
            "source": "manual_four_corners",
            "corners_px": self.corners,
            "homography": self.h.tolist(),
            "fit_residual_px": self.fit_residual_px,
            "quality": "unvalidated_manual",
            "uncertainty_m": None,
            "note": "Four-point fit residual is not independent calibration error",
        }


def transform(h: np.ndarray, p: Point) -> Point:
    q = h @ np.array([*p, 1.0])
    if abs(q[2]) < 1e-10:
        raise ValueError("Point at projective infinity")
    return float(q[0] / q[2]), float(q[1] / q[2])


def line_call(p: Point, uncertainty_m: float | None) -> str:
    # Corners represent OUTSIDE edges of singles painted lines. Lines belong to court.
    # Ball radius + contact/annotation error must already be included in uncertainty_m.
    if uncertainty_m is None:
        return "uncertain"
    x, y = p
    signed_inside = min(x, WIDTH_M - x, y, LENGTH_M - y)
    if signed_inside > uncertainty_m:
        return "estimated_in"
    if signed_inside < -uncertainty_m:
        return "estimated_out"
    return "uncertain"


def letterbox(p: Point, source: Point, destination: Point) -> Point:
    scale = min(destination[0] / source[0], destination[1] / source[1])
    return (
        p[0] * scale + (destination[0] - source[0] * scale) / 2,
        p[1] * scale + (destination[1] - source[1] * scale) / 2,
    )
