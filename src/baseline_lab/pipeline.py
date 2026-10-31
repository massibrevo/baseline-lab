"""CLI and worker share this sequential pipeline; HTTP never runs inference."""

import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import cv2

from baseline_lab.analytics.events import candidates
from baseline_lab.analytics.live import live_metrics
from baseline_lab.contracts import Calibration, Result
from baseline_lab.geometry.court import Court
from baseline_lab.rendering.export import annotated, csvs
from baseline_lab.settings import Settings
from baseline_lab.tracking.association import BallAssociation, Players, interpolate, summarize
from baseline_lab.video.media import frames, inspect, preview
from baseline_lab.video.stability import CameraStability
from baseline_lab.vision.detectors import MODEL_SHA, ColorMotionBall, HOGPlayers, SSDPlayers


def analyze(
    source: Path,
    output: Path,
    cfg: Settings,
    calibration: Calibration | None = None,
    checkpoint=lambda stage, done, total: None,
) -> Result:
    start = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint("validating", 0, None)
    metadata = inspect(source, cfg)
    court = Court(calibration) if calibration else None
    checkpoint("loading_model", 0, None)
    detector = (
        SSDPlayers(cfg.data_dir / "models") if cfg.player_model == "ssdlite" else HOGPlayers()
    )
    ball_detector = ColorMotionBall()
    association = BallAssociation(math.hypot(metadata.width, metadata.height))
    players = Players(court, metadata.width, metadata.height)
    ball = []
    raw = []
    previous = None
    warnings = [
        "Ball: experimental color/motion baseline, not validated for arbitrary footage",
        "Bounce candidates are not confirmed contact events",
        "Player footpoints and measured contiguous-path distance are approximate",
        "Line calls remain uncertain: independent calibration error is not measured",
    ]
    if court is None:
        warnings.append("No calibration: court meters and distances unavailable")
    previous_hist = None
    cuts = []
    stability = None
    calibration_invalidated = False
    motion_checks = []
    for i, (t, _, _, _, image) in enumerate(frames(source)):
        checkpoint("detecting_and_tracking", i, metadata.frames)
        hist = cv2.calcHist([image], [0, 1], None, [24, 24], [0, 256, 0, 256])
        cv2.normalize(hist, hist)
        if (
            previous_hist is not None
            and cv2.compareHist(hist, previous_hist, cv2.HISTCMP_BHATTACHARYYA) > 0.65
        ):
            cuts.append(t)
            # Refuse stale geometry/identities rather than fabricating a full match.
            raise ValueError(f"camera_cut_at_{t:.3f}s: trim to one rally and recalibrate")
        previous_hist = hist
        if stability is None:
            stability = CameraStability(image)
        if i % 15 == 0:
            drift = stability.drift(image)
            motion_checks.append({"t_s": t, "drift_px": drift})
            if (drift is None or drift > 8) and court is not None:
                calibration_invalidated = True
                court = None
                players.court = None
                warnings.append(
                    f"Calibration invalidated at {t:.3f}s: camera moved or could not be verified"
                )
        player_detections = detector.detect(image)
        detections = ball_detector.detect(image, previous)
        detections = [
            d
            for d in detections
            if not any(
                p["bbox"][0] - 8 <= d["xy"][0] <= p["bbox"][2] + 8
                and p["bbox"][1] - 8 <= d["xy"][1] <= p["bbox"][3] + 8
                for p in player_detections
            )
        ]
        detections = [
            d for d in detections if 0.08 * metadata.height < d["xy"][1] < 0.98 * metadata.height
        ]
        if court:
            import numpy as np

            polygon = np.asarray(court.corners, np.float32)
            detections = [
                d for d in detections if cv2.pointPolygonTest(polygon, d["xy"], True) >= -50
            ]
        raw.append({"frame": i, "t_s": t, "candidates": detections})
        ball.append(association.update(detections, t, i))
        players.update(player_detections, t, i)
        previous = image
    if calibration_invalidated:
        for observations in players.points.values():
            for observation in observations:
                observation.court_m = None
    interpolate(ball)
    checkpoint("deriving_events", 0, None)
    events = candidates(ball, list(players.points.values()))
    tracks = [summarize("ball", ball, False)]
    tracks += [summarize(k, v, court is not None) for k, v in players.points.items()]
    result = Result(
        metadata=metadata,
        tracks=tracks,
        events=events,
        calibration=court.details() if court else None,
        warnings=warnings,
        provenance={
            "pipeline": "cpu-baseline-v1",
            "player_detector": cfg.player_model,
            "weights_sha256": MODEL_SHA if cfg.player_model == "ssdlite" else None,
            "ball_detector": "color-motion-v1",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "device": "cpu",
            "dependencies": {
                name: importlib.metadata.version(name)
                for name in ["torch", "torchvision", "opencv-python-headless", "av", "numpy"]
            },
            "camera_cuts_s": cuts,
            "motion_checks": motion_checks,
            "calibration_invalidated": calibration_invalidated,
            "score_semantics": "detector scores and heuristic scores are not probabilities",
        },
        elapsed_s=0,
    )
    result.live_metrics = live_metrics(result)
    (output / "raw-detections.json").write_text(json.dumps(raw))
    annotated(source, output / "annotated.mp4", result, checkpoint)
    checkpoint("preparing_playback", 0, None)
    preview(source, output / "review.mp4")
    result.elapsed_s = round(time.monotonic() - start, 3)
    (output / "result.json").write_text(result.model_dump_json())
    csvs(result, output)
    return result
