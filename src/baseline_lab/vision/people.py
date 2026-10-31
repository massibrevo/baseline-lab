"""Detect every third frame, measure intervening box motion with checked LK flow.

This propagates only real feature measurements, never Kalman predictions. Pose
runs again on each current crop. Failure falls back immediately to detection.
"""

import cv2
import numpy as np

from baseline_lab.contracts import Observation


def flow_people(previous: np.ndarray, current: np.ndarray, observations: list[Observation]):
    before = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    after = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    output = []
    for observation in observations:
        if not observation.bbox_px:
            continue
        x1, y1, x2, y2 = observation.bbox_px
        anchors = [
            p.raw_xy_px or p.xy_px
            for p in observation.keypoints
            if p.xy_px
            and p.name
            in {
                "left_shoulder",
                "right_shoulder",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "nose",
            }
        ]
        if len(anchors) < 4:
            mask = np.zeros(before.shape, np.uint8)
            mask[
                max(0, int(y1)) : min(before.shape[0], int(y2)),
                max(0, int(x1)) : min(before.shape[1], int(x2)),
            ] = 255
            points = cv2.goodFeaturesToTrack(
                before, maxCorners=25, qualityLevel=0.03, minDistance=5, mask=mask
            )
        else:
            points = np.array(anchors, np.float32).reshape(-1, 1, 2)
        if points is None or len(points) < 4:
            continue
        moved, valid, _ = cv2.calcOpticalFlowPyrLK(
            before, after, points, points.copy(), winSize=(21, 21), maxLevel=3
        )
        if moved is None or valid is None:
            continue
        back, back_valid, _ = cv2.calcOpticalFlowPyrLK(
            after, before, moved, moved.copy(), winSize=(21, 21), maxLevel=3
        )
        if back is None or back_valid is None:
            continue
        keep = (
            (valid.ravel() > 0)
            & (back_valid.ravel() > 0)
            & (np.linalg.norm((points - back).reshape(-1, 2), axis=1) < 1.5)
        )
        if keep.sum() < 4:
            continue
        delta = np.median((moved - points).reshape(-1, 2)[keep], axis=0)
        if np.linalg.norm(delta) > max(20, (y2 - y1) * 0.3):
            continue
        dx, dy = (float(v) for v in delta)
        output.append(
            {
                "bbox": [x1 + dx, y1 + dy, x2 + dx, y2 + dy],
                "score": observation.score or 0.5,
                "source": "forward_backward_checked_optical_flow",
            }
        )
    return output


def player_regions(
    observations: list[Observation], width: int, height: int
) -> list[tuple[int, int, int, int]] | None:
    """Local detector searches; callers periodically refresh the full scene."""
    regions = []
    for observation in observations:
        if observation.state != "detected" or not observation.bbox_px:
            return None
        x1, y1, x2, y2 = observation.bbox_px
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = max(220, (x2 - x1) * 2.5), max(220, (y2 - y1) * 1.8)
        regions.append(
            (
                max(0, int(cx - w / 2)),
                max(0, int(cy - h / 2)),
                min(width, int(cx + w / 2)),
                min(height, int(cy + h / 2)),
            )
        )
    return regions
