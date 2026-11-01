"""Local learned stroke recognition. No human annotation is an inference input."""

import hashlib
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from baseline_lab.contracts import Metadata, Observation, ShotSignal
from baseline_lab.video.media import frames

STROKE_SHA = "74539f80559615312aa3ad2631bfe96be4eff3f713dd05797340a9676c8711b8"
STROKE_FILE = "f3ed-tennis-049.pt"
STROKE_URL = (
    "https://drive.usercontent.google.com/download?"
    "id=1QM9FSHpZ0e6WWaNepUdIsORc56B4GPEK&export=download"
)
ELEMENTS = [
    "near",
    "far",
    "deuce",
    "middle",
    "ad",
    "serve",
    "return",
    "stroke",
    "forehand",
    "backhand",
    "groundstroke",
    "slice",
    "volley",
    "smash",
    "drop",
    "lob",
    "T",
    "B",
    "W",
    "CC",
    "DL",
    "DM",
    "II",
    "IO",
    "approach",
    "in",
    "winner",
    "forced_error",
    "unforced_error",
]


def provision_strokes(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / STROKE_FILE
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == STROKE_SHA:
        return target
    temp = target.with_suffix(".download")
    try:
        size = 0
        with urllib.request.urlopen(STROKE_URL, timeout=90) as response, temp.open("wb") as out:
            while block := response.read(1 << 20):
                size += len(block)
                if size > 22649709:
                    raise ValueError("stroke_model_size_exceeded")
                out.write(block)
        if hashlib.sha256(temp.read_bytes()).hexdigest() != STROKE_SHA:
            raise ValueError("stroke_model_checksum_mismatch")
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    return target


def thumbnail(image: np.ndarray) -> np.ndarray:
    """Training geometry: preserve aspect, resize height to 224, center crop/pad."""
    height, width = image.shape[:2]
    resized = cv2.resize(image, (max(1, round(width * 224 / height)), 224))
    if resized.shape[1] < 224:
        pad = 224 - resized.shape[1]
        resized = cv2.copyMakeBorder(resized, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_CONSTANT)
    left = (resized.shape[1] - 224) // 2
    return cv2.cvtColor(resized[:, left : left + 224], cv2.COLOR_BGR2RGB)


class StrokeRecognizer:
    def __init__(self, directory: Path):
        import torch

        from baseline_lab.vision.stroke_network import StrokeNetwork

        target = directory / STROKE_FILE
        if not target.exists():
            raise ValueError("stroke_model_missing")
        if hashlib.sha256(target.read_bytes()).hexdigest() != STROKE_SHA:
            raise ValueError("stroke_model_checksum_mismatch")
        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        torch.set_num_threads(2)
        self.model = StrokeNetwork()
        checkpoint = torch.load(target, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval().to(self.device)

    def predict(self, images: np.ndarray, checkpoint=lambda *_: None):
        """Overlapping temporal windows, mirror averaging, no handedness assumption."""
        torch = self.torch
        x = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).float() / 255
        x = (x - torch.tensor([0.485, 0.456, 0.406])[:, None, None]) / torch.tensor(
            [0.229, 0.224, 0.225]
        )[:, None, None]
        coarse = np.zeros((len(x), 2), np.float32)
        fine = np.zeros((len(x), 29), np.float32)
        support = np.zeros(len(x), np.float32)
        starts = list(range(-48, len(x), 48))
        with torch.inference_mode():
            for window, start in enumerate(starts):
                checkpoint("recognizing_strokes", window, len(starts))
                indices = np.arange(start, start + 96)
                valid = (indices >= 0) & (indices < len(x))
                batch = torch.zeros(96, 3, 224, 224)
                batch[valid] = x[indices[valid]]
                # Averaging is symmetric: flipping the input swaps these two passes.
                pair = torch.stack([batch, batch.flip(-1)]).to(self.device)
                c, f = self.model(pair)
                c = c.softmax(-1).mean(0).cpu().numpy()
                f = f.sigmoid().mean(0).cpu().numpy()
                coarse[indices[valid]] += c[valid]
                fine[indices[valid]] += f[valid]
                support[indices[valid]] += 1
        return coarse / support[:, None], fine / support[:, None]


def wing_type(attributes: np.ndarray) -> tuple[str, float, float]:
    """Attribute scores are uncalibrated. Ambiguous wings remain unknown."""
    wing = attributes[8:10]
    winner = int(wing.argmax())
    score, margin = float(wing[winner]), float(abs(wing[0] - wing[1]))
    category = attributes[5:8]
    if category[0] >= 0.8 and category[0] - max(category[1:]) >= 0.4:
        return "serve", float(category[0]), float(category[0] - max(category[1:]))
    if score < 0.7 or margin < 0.4:
        return "unknown", score, margin
    kind = ["forehand", "backhand"][winner]
    technique = attributes[10:16]
    if technique[3] >= 0.75 and technique[3] - np.partition(technique, -2)[-2] >= 0.35:
        kind = "overhead"
    elif technique[2] >= 0.75 and technique[2] - np.partition(technique, -2)[-2] >= 0.35:
        kind += "_volley"
    return kind, score, margin


def edge_swing_times(observations: list[Observation], duration: float) -> list[float]:
    """Recover a visible swing cut off by a clip edge, without inventing a hit."""
    samples = []
    for p in observations:
        joints = {k.name: k.xy_px for k in p.keypoints if k.xy_px and k.state == "detected"}
        names = [
            "left_wrist",
            "right_wrist",
            "left_hip",
            "right_hip",
            "left_shoulder",
            "right_shoulder",
        ]
        if not all(n in joints for n in names):
            continue
        a, b, lh, rh, ls, rs = [np.array(joints[n]) for n in names]
        center = (lh + rh) / 2
        scale = np.linalg.norm((ls + rs) / 2 - center) * 2.8
        if scale < 15:
            continue
        samples.append((p.t_s, np.concatenate([(a - center) / scale, (b - center) / scale])))
    peaks = []
    for index, (t, _value) in enumerate(samples):
        if 0.6 < t < duration - 0.6:
            continue
        before = [
            (tt, v) for tt, v in samples[max(0, index - 12) : index] if 0.04 <= t - tt <= 0.15
        ]
        after = [(tt, v) for tt, v in samples[index + 1 : index + 13] if 0.04 <= tt - t <= 0.15]
        if len(before) < 2 or len(after) < 2:
            continue
        velocity = (
            np.median([v for _, v in after], axis=0) - np.median([v for _, v in before], axis=0)
        ) / (np.mean([tt for tt, _ in after]) - np.mean([tt for tt, _ in before]))
        speed = max(np.linalg.norm(velocity[:2]), np.linalg.norm(velocity[2:]))
        if 1.5 <= speed <= 12:
            peaks.append((float(speed), t))
    selected: list[float] = []
    for _, t in sorted(peaks, reverse=True):
        if all(abs(t - existing) > 0.5 for existing in selected):
            selected.append(t)
    return selected


def decode_strokes(
    times: np.ndarray,
    coarse: np.ndarray,
    fine: np.ndarray,
    metadata: Metadata,
    players: dict[str, list[Observation]],
) -> list[ShotSignal]:
    candidates: list[tuple[int, str, str | None]] = []
    for i in range(len(times)):
        if coarse[i, 1] >= 0.3 and coarse[i, 1] >= max(coarse[max(0, i - 3) : i + 4, 1]):
            candidates.append((i, "neural_impact_candidate", None))
    for track_id, observations in players.items():
        for t in edge_swing_times(observations, metadata.duration_s):
            if t < times[0] or t > times[-1]:
                continue
            i = int(abs(times - t).argmin())
            if not any(abs(times[j] - t) < 0.65 for j, _, _ in candidates):
                candidates.append((i, "clip_boundary_swing_not_verified_contact", track_id))
    output: list[ShotSignal] = []
    for i, basis, identity in sorted(candidates, key=lambda c: times[c[0]]):
        attributes = fine[i]
        player = ["near", "far"][int(attributes[:2].argmax())]
        if identity:
            # A visible tracked skeleton supplies identity at a clip boundary.
            # Veto a strong contradictory neural identity, otherwise use the pose.
            if identity != player and max(attributes[:2]) >= 0.8:
                continue
            player = identity
        kind, score, margin = wing_type(attributes)
        if basis.startswith("clip_boundary") and (
            kind in {"unknown", "serve", "overhead"} or score < 0.8
        ):
            continue
        t = float(times[i])
        if output and t - output[-1].t_s < 0.55:
            continue
        frame = min(range(metadata.frames), key=lambda j: abs(metadata.timestamps_s[j] - t))
        technique = ELEMENTS[10 + int(attributes[10:16].argmax())]
        output.append(
            ShotSignal.model_validate(
                {
                    "id": f"auto-stroke-{frame}",
                    "t_s": t,
                    "frame": frame,
                    "player": player,
                    "kind": kind,
                    "status": "estimated",
                    "evidence_score": score,
                    "score_semantics": "neural attribute score; uncalibrated; not accuracy",
                    "method": "F3ED LCL/MLC + bidirectional GRU; mirror average; no human labels",
                    "context_start_s": max(0, t - 0.7),
                    "context_end_s": min(metadata.duration_s, t + 0.45),
                    "recognition_basis": basis,
                    "technique": technique
                    if kind != "unknown" and max(attributes[10:16]) >= 0.65
                    else None,
                    "model_scores": {
                        "event": float(coarse[i, 1]),
                        "forehand": float(attributes[8]),
                        "backhand": float(attributes[9]),
                        "serve": float(attributes[5]),
                        "attribute": score,
                        "attribute_margin": margin,
                    },
                }
            )
        )
    return output


def recognize_strokes(source: Path, metadata: Metadata, players, directory: Path, checkpoint):
    """Inputs deliberately exclude events, stroke reviews and user annotations."""
    if not (directory / STROKE_FILE).exists():
        return [], {"status": "unavailable", "reason": "run baseline-lab setup-models"}
    started = time.monotonic()
    model = StrokeRecognizer(directory)
    times = np.arange(0, metadata.timestamps_s[-1] + 0.001, 0.08)
    source_times = np.array(metadata.timestamps_s)
    indices = [int(abs(source_times - t).argmin()) for t in times]
    wanted = set(indices)
    images = {
        i: thumbnail(image) for i, (_, _, _, _, image) in enumerate(frames(source)) if i in wanted
    }
    coarse, fine = model.predict(np.stack([images[i] for i in indices]), checkpoint)
    output = decode_strokes(times, coarse, fine, metadata, players)
    info = {
        "status": "experimental",
        "model": "F3ED event localizer and multi-label classifier",
        "sha256": STROKE_SHA,
        "device": model.device,
        "sample_interval_s": 0.08,
        "temporal_window_frames": 96,
        "mirror_averaging": True,
        "human_annotations_used": False,
        "contextual_refinement": "disabled: no verified handedness metadata",
        "elapsed_s": round(time.monotonic() - started, 3),
        "limitations": (
            "Broadcast-trained; viewpoints, occlusion and clipped swings can fail. "
            "Scores are not accuracy. Edge swings are not verified contacts."
        ),
    }
    return output, info


class LiveStrokeRecognizer:
    """Bounded causal image history. Sparse capture gaps never become motion evidence."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.model: StrokeRecognizer | None = None
        self.history: list[tuple[float, np.ndarray]] = []
        self.last_run = -10.0
        self.shots: list[ShotSignal] = []

    def update(self, images, metadata: Metadata, players, checkpoint):
        self.history.extend(images)
        now = metadata.timestamps_s[-1]
        self.history = [(t, image) for t, image in self.history if now - t < 7.68][-384:]
        if not (self.directory / STROKE_FILE).exists():
            return self.shots, {"status": "unavailable", "reason": "stroke model missing"}
        times = np.arange(max(0, now - 7.6), now + 0.001, 0.08)
        capture = np.array([t for t, _ in self.history])
        indices = [int(abs(capture - t).argmin()) for t in times]
        supported = abs(capture[indices] - times) <= 0.065
        coverage = float(supported.mean())
        info = {
            "status": "collecting",
            "causal": True,
            "motion_context_coverage": coverage,
            "reason": "Need contiguous motion; capture gaps are excluded from classification",
        }
        if now < 1.2 or coverage < 0.8 or now - self.last_run < 0.8:
            return self.shots, info
        if self.model is None:
            self.model = StrokeRecognizer(self.directory)
        batch = np.stack([self.history[i][1] for i in indices])
        batch[~supported] = 0
        coarse, fine = self.model.predict(batch, checkpoint)
        candidates = decode_strokes(times, coarse, fine, metadata, players)
        for shot in candidates:
            existing = next(
                (s for s in self.shots if s.player == shot.player and abs(s.t_s - shot.t_s) < 0.55),
                None,
            )
            if existing:
                if shot.model_scores.get("event", 0) > existing.model_scores.get("event", 0):
                    self.shots.remove(existing)
                else:
                    continue
            self.shots.append(shot)
        self.shots.sort(key=lambda s: s.t_s)
        self.last_run = now
        return self.shots, {
            **info,
            "status": "experimental",
            "sha256": STROKE_SHA,
            "human_annotations_used": False,
            "mirror_averaging": True,
        }
