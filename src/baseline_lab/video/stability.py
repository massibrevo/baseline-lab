"""Conservative camera-motion check against the first displayed frame."""

import cv2
import numpy as np


class CameraStability:
    def __init__(self, image: np.ndarray):
        self.width = image.shape[1]
        self.orb = cv2.ORB_create(nfeatures=500)  # type: ignore[attr-defined]
        self.keypoints, self.descriptors = self.orb.detectAndCompute(self.small(image), None)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    def small(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
            (640, round(image.shape[0] * 640 / image.shape[1])),
        )

    def mapping(self, image: np.ndarray):
        kp, desc = self.orb.detectAndCompute(self.small(image), None)
        if desc is None or self.descriptors is None:
            return None
        pairs = self.matcher.knnMatch(self.descriptors, desc, k=2)
        good = [
            pair[0]
            for pair in pairs
            if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance
        ]
        if len(good) < 15:
            return None
        src = np.asarray([self.keypoints[m.queryIdx].pt for m in good], dtype=np.float32)
        dst = np.asarray([kp[m.trainIdx].pt for m in good], dtype=np.float32)
        h, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2)
        if h is None or mask is None or int(mask.sum()) < 12:
            return None
        anchors = np.asarray([[[160, 120], [480, 120], [480, 300], [160, 300]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(anchors, h)
        projected = cv2.perspectiveTransform(src[None], h)[0]
        residual = float(
            np.median(np.linalg.norm(projected - dst, axis=1)[mask.ravel() > 0]) * self.width / 640
        )
        spread = np.ptp(src[mask.ravel() > 0], axis=0)
        if residual > 4 or spread[0] < 160 or spread[1] < 80:
            return None
        scale = np.diag([640 / self.width, 640 / self.width, 1.0])
        display_h = np.linalg.inv(scale) @ h @ scale
        drift = float(np.median(np.linalg.norm(transformed - anchors, axis=2)) * self.width / 640)
        return display_h, drift, residual

    def drift(self, image: np.ndarray) -> float | None:
        result = self.mapping(image)
        return result[1] if result is not None else None
