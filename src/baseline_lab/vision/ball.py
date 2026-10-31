"""WASB HRNet tennis heatmaps from three causal RGB frames, all inference local.

Architecture MIT nttcom/WASB-SBDT; exact released tennis checkpoint is fetched
separately and never redistributed. Original preprocessing/adapter/decoder here.
"""

import hashlib
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

from baseline_lab.vision.vendor.wasb_hrnet import HRNet

BALL_FILE = "wasb-tennis.pth.tar"
BALL_SHA = "9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb"
BALL_URL = "https://drive.usercontent.google.com/download?id=14AeyIOCQ2UaQmbZLNQJa1H_eSwxUXk7z&export=download&confirm=t"
BALL_REVISION = "923462cacdeb3353b84ddebdedb3f4b7a8553b0f"


def provision_ball(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / BALL_FILE
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == BALL_SHA:
        return target
    temp = target.with_suffix(".download")
    try:
        with urllib.request.urlopen(BALL_URL, timeout=90) as response, temp.open("wb") as out:
            count = 0
            while block := response.read(1 << 20):
                count += len(block)
                if count > 10_000_000:
                    raise ValueError("ball_model_download_limit")
                out.write(block)
        if hashlib.sha256(temp.read_bytes()).hexdigest() != BALL_SHA:
            raise ValueError("ball_model_checksum_mismatch")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def apparent_diameter(image: np.ndarray, xy) -> float | None:
    x, y = (round(v) for v in xy)
    height, width = image.shape[:2]
    left, top = max(0, x - 14), max(0, y - 14)
    patch = image[top : min(height, y + 15), left : min(width, x + 15)]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    color = ((hsv[:, :, 0] >= 18) & (hsv[:, :, 0] <= 75) & (hsv[:, :, 1] > 10)) | (
        hsv[:, :, 1] < 90
    )
    mask = (
        color
        & (
            hsv[:, :, 2]
            > max(140, float(np.median(np.asarray(hsv[:, :, 2], dtype=np.float64))) + 20)
        )
    ).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    options = []
    for contour in contours:
        a, b, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if not 2 <= area <= 150 or min(w, h) < 2 or max(w, h) > 24 or max(w, h) / min(w, h) > 4:
            continue
        distance = cv2.pointPolygonTest(contour, (float(x - left), float(y - top)), True)
        if distance < -5:
            continue
        diameter = float(min(w, h))
        if len(contour) >= 5:
            diameter = float(min(cv2.fitEllipse(contour)[1]))
        if 1.5 <= diameter <= 20:
            options.append((abs(distance), diameter))
    return min(options)[1] if options else None


def add_diameters(image, candidates):
    for candidate in candidates:
        candidate["diameter_px"] = apparent_diameter(image, candidate["xy"])
    return candidates


class Config(dict):
    def __getattr__(self, name):
        return self[name]


def config_tree(value):
    return (
        Config({k: config_tree(v) for k, v in value.items()}) if isinstance(value, dict) else value
    )


class TemporalBall:
    def __init__(self, directory: Path):
        import torch

        path = directory / BALL_FILE
        if not path.exists():
            raise ValueError("ball_model_missing: run baseline-lab setup-models")
        if hashlib.sha256(path.read_bytes()).hexdigest() != BALL_SHA:
            raise ValueError("ball_model_checksum_mismatch")
        self.torch = torch
        torch.set_num_threads(2)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        cfg = config_tree(
            yaml.safe_load((Path(__file__).parent / "vendor" / "wasb.yaml").read_text())
        )
        self.model = HRNet(cfg)
        state = torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"]
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        self.model.eval().to(self.device)
        self.history: deque[np.ndarray] = deque(maxlen=3)

    def reset(self):
        self.history.clear()

    def prepare(self, image):
        height, width = image.shape[:2]
        scale = 512 / max(width, height)
        offset = (256 - width * scale / 2, 144 - height * scale / 2)
        patch = cv2.warpAffine(
            image, np.array([[scale, 0, offset[0]], [0, scale, offset[1]]], np.float32), (512, 288)
        )
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
            [0.229, 0.224, 0.225], np.float32
        )
        return rgb.transpose(2, 0, 1), (scale, offset, width, height)

    def decode(self, heat, geometry):
        scale, offset, width, height = geometry
        count, labels = cv2.connectedComponents((heat > 0.5).astype(np.uint8))
        output: list[dict] = []
        for index in range(1, count):
            ys, xs = np.where(labels == index)
            weights = heat[ys, xs]
            x = (float(np.average(xs, weights=weights)) - offset[0]) / scale
            y = (float(np.average(ys, weights=weights)) - offset[1]) / scale
            if 0 <= x < width and 0 <= y < height:
                output.append(
                    {"xy": (x, y), "score": float(weights.max()), "source": "wasb_triplet_heatmap"}
                )
        return sorted(output, key=lambda p: p["score"], reverse=True)[:10]

    def detect_batch(self, images: list[np.ndarray]) -> list[list[dict]]:
        output = []
        for start in range(0, len(images), 3):
            group = images[start : start + 3]
            prepared = [self.prepare(image) for image in group]
            channels = [item[0] for item in prepared]
            channels += [channels[-1]] * (3 - len(channels))
            tensor = self.torch.from_numpy(
                np.ascontiguousarray(np.concatenate(channels, axis=0)[None])
            ).to(self.device)
            with self.torch.inference_mode():
                heat = self.model(tensor)[0][0].sigmoid().cpu().numpy()
            output += [
                add_diameters(group[i], self.decode(heat[i], geometry))
                for i, (_, geometry) in enumerate(prepared)
            ]
        return output

    def detect(self, image: np.ndarray, previous: np.ndarray | None) -> list[dict]:
        if previous is None:
            self.reset()
        prepared, geometry = self.prepare(image)
        self.history.append(prepared)
        if len(self.history) < 3:
            return []
        tensor = self.torch.from_numpy(
            np.ascontiguousarray(np.concatenate(list(self.history), axis=0)[None])
        ).to(self.device)
        with self.torch.inference_mode():
            heat = self.model(tensor)[0][0, -1].sigmoid().cpu().numpy()
        return add_diameters(image, self.decode(heat, geometry))
