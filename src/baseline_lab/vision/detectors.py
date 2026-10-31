"""Narrow detector adapters. No hidden provider calls or checkpoint execution."""

import hashlib
import urllib.request
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

MODEL_URL = "https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth"
MODEL_SHA = "a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2"
MODEL_FILE = "ssdlite320_mobilenet_v3_large_coco-a79551df.pth"


class BallDetector(Protocol):
    def detect(self, image: np.ndarray, previous: np.ndarray | None) -> list[dict]: ...


class PlayerDetector(Protocol):
    def detect(self, image: np.ndarray) -> list[dict]: ...


class CourtCalibrator(Protocol):
    def calibrate(self, image: np.ndarray) -> dict | None: ...


def provision(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / MODEL_FILE
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == MODEL_SHA:
        return target
    temp = target.with_suffix(".download")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, temp.open("wb") as out:
            while block := response.read(1 << 20):
                out.write(block)
        if hashlib.sha256(temp.read_bytes()).hexdigest() != MODEL_SHA:
            raise ValueError("Model checksum mismatch")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return target


class SSDPlayers:
    def __init__(self, directory: Path):
        import torch
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large

        target = directory / MODEL_FILE
        if not target.exists():
            raise ValueError("model_missing: run baseline-lab setup-models before analysis")
        if hashlib.sha256(target.read_bytes()).hexdigest() != MODEL_SHA:
            raise ValueError("model_checksum_mismatch")
        torch.set_num_threads(2)
        self.torch = torch
        self.model = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)
        self.model.load_state_dict(torch.load(target, map_location="cpu", weights_only=True))
        self.model.eval()

    def detect(self, image: np.ndarray) -> list[dict]:
        torch = self.torch
        height, width = image.shape[:2]
        # Overlapping tiles preserve small people; return every box in original pixels.
        if width >= 960:
            regions = [
                (x, y, min(width, x + int(width * 0.60)), min(height, y + int(height * 0.60)))
                for y in [int(height * 0.15), int(height * 0.40)]
                for x in [0, int(width * 0.40)]
            ]
        else:
            regions = [(0, 0, width, height)]
        tensors = [
            torch.from_numpy(cv2.cvtColor(image[y:y2, x:x2], cv2.COLOR_BGR2RGB))
            .permute(2, 0, 1)
            .float()
            / 255
            for x, y, x2, y2 in regions
        ]
        with torch.inference_mode():
            results = self.model(tensors)
        detections: list[dict] = []
        for result, (x, y, _, _) in zip(results, regions, strict=True):
            for box, score, label in zip(
                result["boxes"], result["scores"], result["labels"], strict=True
            ):
                if int(label) == 1 and float(score) >= 0.35:
                    b = box.tolist()
                    detections.append(
                        {"bbox": [b[0] + x, b[1] + y, b[2] + x, b[3] + y], "score": float(score)}
                    )
        # Greedy overlap suppression across tile boundaries.
        output: list[dict] = []
        for detection in sorted(detections, key=lambda d: d["score"], reverse=True):
            if not any(overlap(detection["bbox"], d["bbox"]) > 0.4 for d in output):
                output.append(detection)
        return output


def overlap(a: list[float], b: list[float]) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0


class HOGPlayers:
    def __init__(self):
        self.model = cv2.HOGDescriptor()  # type: ignore[attr-defined]
        self.model.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # type: ignore[attr-defined]

    def detect(self, image: np.ndarray) -> list[dict]:
        scale = min(1.0, 800 / image.shape[1])
        small = cv2.resize(image, None, fx=scale, fy=scale)
        boxes, scores = self.model.detectMultiScale(small, winStride=(8, 8), scale=1.08)
        return [
            {
                "bbox": [x / scale, y / scale, (x + w) / scale, (y + h) / scale],
                "score": float(score),
            }
            for (x, y, w, h), score in zip(boxes, scores, strict=False)
        ]


class ColorMotionBall:
    """A deliberately conservative yellow/white small moving-blob baseline.

    Score is heuristic shape/color/motion agreement, NOT calibrated probability.
    Operates at original display resolution; never silently upscales model coordinates.
    """

    def __init__(self):
        self.static_white: np.ndarray | None = None

    def detect(self, image: np.ndarray, previous: np.ndarray | None) -> list[dict]:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, np.array([20, 25, 180]), np.array([60, 255, 255]))
        white = cv2.inRange(hsv, np.array([0, 0, 205]), np.array([180, 60, 255]))
        if previous is None:
            self.static_white = cv2.dilate(white, np.ones((7, 7), np.uint8))
            return []
        difference = cv2.absdiff(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        )
        motion = cv2.threshold(difference, 22, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.bitwise_and(cv2.bitwise_or(yellow, white), motion)
        if self.static_white is not None:
            mask[self.static_white > 0] = 0
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[dict] = []
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if not (2 <= area <= 60 and 2 <= w <= 16 and 2 <= h <= 16 and 0.4 < w / h < 2.5):
                continue
            perimeter = cv2.arcLength(c, True)
            circularity = 4 * np.pi * area / max(perimeter * perimeter, 1)
            if circularity < 0.35:
                continue
            gray = cv2.cvtColor(
                image[max(0, y - 6) : y + h + 6, max(0, x - 6) : x + w + 6], cv2.COLOR_BGR2GRAY
            )
            contrast = float(hsv[y : y + h, x : x + w, 2].max()) - float(
                np.median(np.asarray(gray, dtype=np.float64))
            )
            if contrast < 35:
                continue
            score = min(1.0, 0.35 * circularity + 0.65 * min(contrast / 100, 1))
            candidates.append({"xy": (x + w / 2, y + h / 2), "score": score})
        return sorted(candidates, key=lambda c: c["score"], reverse=True)[:40]
