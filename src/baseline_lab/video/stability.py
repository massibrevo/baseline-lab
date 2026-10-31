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

    def drift(self, image: np.ndarray) -> float | None:
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
        return float(np.median(np.linalg.norm(transformed - anchors, axis=2)) * self.width / 640)
