"""RTMPose SimCC player skeletons in analysis pixels; local OpenCV DNN.

Original adapter, using the released OpenMMLab graph's documented input/output
contract. No 3D anatomy or calibrated model-probability claim is made.
"""

import hashlib
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from baseline_lab.contracts import Keypoint, Observation
from baseline_lab.geometry.court import Court
from baseline_lab.tracking.smoothing import OneEuro

JOINTS = [
    (0, "nose"),
    (2, "left_eye"),
    (5, "right_eye"),
    (7, "left_ear"),
    (8, "right_ear"),
    (11, "left_shoulder"),
    (12, "right_shoulder"),
    (13, "left_elbow"),
    (14, "right_elbow"),
    (15, "left_wrist"),
    (16, "right_wrist"),
    (23, "left_hip"),
    (24, "right_hip"),
    (25, "left_knee"),
    (26, "right_knee"),
    (27, "left_ankle"),
    (28, "right_ankle"),
]
BONES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_eye", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
]


def draw_pose(image: np.ndarray, observation: Observation):
    joints = {p.name: p.xy_px for p in observation.keypoints if p.state == "detected" and p.xy_px}
    for a, b in BONES:
        if a in joints and b in joints:
            cv2.line(
                image,
                tuple(round(v) for v in joints[a]),
                tuple(round(v) for v in joints[b]),
                (69, 239, 216),
                2,
                cv2.LINE_AA,
            )
    for point in joints.values():
        center = tuple(round(v) for v in point)
        cv2.circle(image, center, 4, (70, 230, 80), -1, cv2.LINE_AA)
        cv2.circle(image, center, 4, (245, 245, 235), 1, cv2.LINE_AA)


def far_player_regions(court: Court, width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Overlapping higher-resolution searches around the far singles baseline."""
    left, right = court.corners[:2]
    side = max(96, abs(right[0] - left[0]) * 0.85)
    regions = []
    for alpha in [0, 0.5, 1]:
        x = left[0] + alpha * (right[0] - left[0])
        y = left[1] + alpha * (right[1] - left[1])
        x1, y1 = max(0, int(x - side / 2)), max(0, int(y - side * 0.7))
        x2, y2 = min(width, int(x + side / 2)), min(height, int(y + side * 0.3))
        if x2 - x1 > 32 and y2 - y1 > 32:
            regions.append((x1, y1, x2, y2))
    return regions


POSE_FILE = "rtmpose-m-body7.onnx"
POSE_SHA = "5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c"
POSE_REVISION = "cd4d7095f5cfc9cfc4f46289bee91ea4a1e1d9fd"
POSE_ARCHIVE_SHA = "f7fbb6c5c11a1bb70f3d445e4ddec5d144ea89ad5649c081ef976f9b24a0b741"
POSE_URL = f"https://huggingface.co/Tau-J/RTMPose/resolve/{POSE_REVISION}/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"


def provision_pose(directory: Path) -> Path:
    import io
    import zipfile

    directory.mkdir(parents=True, exist_ok=True)
    target = directory / POSE_FILE
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == POSE_SHA:
        return target
    with urllib.request.urlopen(POSE_URL, timeout=90) as response:
        data = response.read(60_000_001)
    if hashlib.sha256(data).hexdigest() != POSE_ARCHIVE_SHA:
        raise ValueError("pose_archive_checksum_mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        item = next(a for a in archive.infolist() if a.filename.endswith("/end2end.onnx"))
        data = archive.read(item)
    if hashlib.sha256(data).hexdigest() != POSE_SHA:
        raise ValueError("pose_model_checksum_mismatch")
    temp = target.with_suffix(".download")
    temp.write_bytes(data)
    temp.replace(target)
    return target


def pose_crop(box):
    x1, y1, x2, y2 = box
    crop_w, crop_h = (x2 - x1) * 1.25, (y2 - y1) * 1.25
    crop_w = max(crop_w, crop_h * 0.75)
    crop_h = crop_w / 0.75
    left, top = (x1 + x2 - crop_w) / 2, (y1 + y2 - crop_h) / 2
    transform = np.array(
        [[192 / crop_w, 0, -left * 192 / crop_w], [0, 256 / crop_h, -top * 256 / crop_h]],
        np.float32,
    )
    return transform, crop_w, crop_h, left, top


class PoseEstimator:
    """RTMPose-m SimCC on an aspect-preserving 192x256 padded person crop.

    OpenMMLab/rtmlib's released graph expects BGR float, per-channel mean/std,
    NCHW, and a 2x SimCC split. No remote runtime code or 3D outputs.
    """

    def __init__(self, directory: Path):
        path = directory / POSE_FILE
        if not path.exists():
            raise ValueError("pose_model_missing: run baseline-lab setup-models")
        if hashlib.sha256(path.read_bytes()).hexdigest() != POSE_SHA:
            raise ValueError("pose_model_checksum_mismatch")
        cv2.setNumThreads(2)
        self.smoothers: dict[str, OneEuro] = {}
        self.net = cv2.dnn.readNetFromONNX(str(path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def estimate(
        self,
        image: np.ndarray,
        observation: Observation,
        court: Court | None,
        identity: str | None = None,
    ):
        if observation.bbox_px is None:
            observation.pose_reason = "person_detection_missing"
            return
        transform, crop_w, crop_h, left, top = pose_crop(observation.bbox_px)
        patch = cv2.warpAffine(image, transform, (192, 256))
        tensor = (
            patch.astype(np.float32) - np.array([123.675, 116.28, 103.53], np.float32)
        ) / np.array([58.395, 57.12, 57.375], np.float32)
        self.net.setInput(np.ascontiguousarray(tensor.transpose(2, 0, 1)[None]))
        sx, sy = self.net.forward(self.net.getUnconnectedOutLayersNames())
        scores = (sx[0].max(axis=1) + sy[0].max(axis=1)) / 2
        xy = np.column_stack([sx[0].argmax(axis=1) / 2, sy[0].argmax(axis=1) / 2])
        xy = xy / np.array([192, 256]) * np.array([crop_w, crop_h]) + np.array([left, top])
        points = []
        for (_, name), point, score in zip(JOINTS, xy, scores, strict=True):
            valid = (
                score >= 0.3 and 0 <= point[0] < image.shape[1] and 0 <= point[1] < image.shape[0]
            )
            raw_point = tuple(point) if valid else None
            key = f"{identity}/{name}"
            if identity and valid:
                point = np.array(
                    self.smoothers.setdefault(key, OneEuro()).update(point, observation.t_s)
                )
            elif identity:
                self.smoothers.pop(key, None)
            points.append(
                Keypoint(
                    name=name,
                    raw_xy_px=raw_point,
                    xy_px=tuple(point) if valid else None,
                    state="detected" if valid else "missing",
                    visibility=None,
                    presence=None,
                    score=float(score),
                    score_semantics="RTMPose mean SimCC response; not probability",
                )
            )
        observation.keypoints = points
        observation.pose_score = float(np.mean(scores))
        observation.pose_reason = (
            None if sum(p.xy_px is not None for p in points) >= 6 else "weak_pose_evidence"
        )
        ankles = [p.xy_px for p in points if p.name in {"left_ankle", "right_ankle"} and p.xy_px]
        if len(ankles) == 2:
            foot = ((ankles[0][0] + ankles[1][0]) / 2, max(ankles[0][1], ankles[1][1]))
            observation.xy_px = foot
            observation.court_m = court.project(foot) if court else None
            observation.footpoint_source = "rtmpose_ankles_approximate"
