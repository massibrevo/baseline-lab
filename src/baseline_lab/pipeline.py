"""CLI and worker share this sequential pipeline; HTTP never runs inference."""

import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from baseline_lab.analytics.events import candidates
from baseline_lab.analytics.live import live_metrics
from baseline_lab.analytics.quality import update_quality
from baseline_lab.analytics.shots import shot_signals
from baseline_lab.analytics.speed import research_speed
from baseline_lab.contracts import Calibration, Result
from baseline_lab.geometry.court import Court
from baseline_lab.rendering.export import annotated, csvs
from baseline_lab.rendering.trail import visual_trail
from baseline_lab.settings import Settings
from baseline_lab.tracking.association import BallAssociation, Players, interpolate, summarize
from baseline_lab.video.media import frames, inspect, preview
from baseline_lab.video.stability import CameraStability
from baseline_lab.vision.ball import BALL_SHA, TemporalBall
from baseline_lab.vision.detectors import MODEL_SHA, ColorMotionBall, HOGPlayers, SSDPlayers
from baseline_lab.vision.people import flow_people, player_regions
from baseline_lab.vision.pose import POSE_SHA, PoseEstimator, far_player_regions


def analyze(
    source: Path,
    output: Path,
    cfg: Settings,
    calibration: Calibration | None = None,
    checkpoint=lambda stage, done, total: None,
    real_time_playback: bool = False,
) -> Result:
    start = time.monotonic()
    timings: dict[str, float] = {}
    output.mkdir(parents=True, exist_ok=True)
    checkpoint("validating", 0, None)
    metadata = inspect(source, cfg)
    reference_court = Court(calibration) if calibration else None
    court = reference_court
    frame_calibrations = []
    checkpoint("loading_model", 0, None)
    detector = (
        SSDPlayers(cfg.data_dir / "models") if cfg.player_model == "ssdlite" else HOGPlayers()
    )
    pose = PoseEstimator(cfg.data_dir / "models") if cfg.player_model == "ssdlite" else None
    if court and isinstance(detector, SSDPlayers):
        detector.focus_regions = far_player_regions(court, metadata.width, metadata.height)
    ball_detector = (
        TemporalBall(cfg.data_dir / "models")
        if cfg.player_model == "ssdlite"
        else ColorMotionBall()
    )
    association = BallAssociation(
        math.hypot(metadata.width, metadata.height), neural=cfg.player_model == "ssdlite"
    )
    players = Players(court, metadata.width, metadata.height)
    ball = []
    raw = []
    previous = None
    warnings = [
        "Ball: temporal tennis heatmap model; arbitrary-footage accuracy is not validated",
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
    reference_image = None
    neural_detections = None
    heat_start = time.monotonic()
    if isinstance(ball_detector, TemporalBall):
        neural_detections = []
        batch = []
        for _, _, _, _, image in frames(source, cfg.max_pixels, cfg.max_source_pixels):
            batch.append(image)
            if len(batch) == 3:
                neural_detections.extend(ball_detector.detect_batch(batch))
                batch = []
                checkpoint("ball_heatmaps", len(neural_detections), metadata.frames)
        if batch:
            neural_detections.extend(ball_detector.detect_batch(batch))
    timings["ball_heatmaps_s"] = time.monotonic() - heat_start
    tracking_start = time.monotonic()
    for i, (t, _, _, _, image) in enumerate(frames(source, cfg.max_pixels, cfg.max_source_pixels)):
        checkpoint("detecting_and_tracking", i, metadata.frames)
        if reference_image is None:
            reference_image = image.copy()
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
        mapping = stability.mapping(image) if reference_court else None
        court = None
        details = None
        if mapping is not None and reference_court:
            transform, drift, residual = mapping
            if drift < metadata.width * 0.15:
                transformed = cv2.perspectiveTransform(
                    np.array([reference_court.corners], np.float32), transform
                )[0]
                try:
                    court = Court(Calibration(corners_px=transformed.tolist()))
                    details = court.details()
                    details.update({"registration_residual_px": residual, "camera_drift_px": drift})
                except ValueError:
                    pass
        if i % 15 == 0:
            motion_checks.append(
                {
                    "t_s": t,
                    "drift_px": mapping[1] if mapping else None,
                    "registration_residual_px": mapping[2] if mapping else None,
                }
            )
        if reference_court and court is None:
            calibration_invalidated = True
        frame_calibrations.append(details)
        players.court = court
        player_detections: list[dict] = []
        if previous is not None and i % 6 and cfg.player_model == "ssdlite":
            player_detections = flow_people(
                previous, image, [v[-1] for v in players.points.values()]
            )
        if len(player_detections) < 2:
            regions: list[tuple[int, int, int, int]] | None = (
                player_regions(
                    [v[-1] for v in players.points.values()], metadata.width, metadata.height
                )
                if i % 30 and previous is not None and isinstance(detector, SSDPlayers)
                else None
            )
            player_detections = (
                detector.detect(image, regions)
                if isinstance(detector, SSDPlayers)
                else detector.detect(image)
            )
            if regions and len(player_detections) < 2:
                player_detections = detector.detect(image)
        detections = (
            neural_detections[i]
            if neural_detections is not None
            else ball_detector.detect(image, previous)
        )
        # Keep candidates near a racket/player: blanket person-box suppression
        # removed the actual ball around impacts. Temporal association rejects clutter.
        detections = [
            d for d in detections if 0.08 * metadata.height < d["xy"][1] < 0.98 * metadata.height
        ]
        if court:
            # Image ROI includes airborne space above the far baseline. A ground
            # polygon is not a valid exclusion mask for an airborne ball.
            top = min(y for _, y in court.corners) - 0.3 * metadata.height
            left = min(x for x, _ in court.corners) - 0.2 * metadata.width
            right = max(x for x, _ in court.corners) + 0.2 * metadata.width
            detections = [
                d for d in detections if d["xy"][1] >= top and left <= d["xy"][0] <= right
            ]
        raw.append({"frame": i, "t_s": t, "candidates": detections})
        ball.append(association.update(detections, t, i))
        players.update(player_detections, t, i)
        if pose:
            for identity, observations in players.points.items():
                pose.estimate(image, observations[-1], court, identity=identity)
        previous = image
    timings["tracking_and_pose_s"] = time.monotonic() - tracking_start
    if calibration_invalidated:
        warnings.append(
            "Camera registration failed on some frames; ground metrics missing for those frames"
        )
    interpolate(ball)
    checkpoint("deriving_events", 0, None)
    events = candidates(ball, list(players.points.values()))
    tracks = [summarize("ball", ball, False)]
    tracks += [summarize(k, v, court is not None) for k, v in players.points.items()]
    result = Result(
        metadata=metadata,
        tracks=tracks,
        events=events,
        calibration=reference_court.details() if reference_court else None,
        frame_calibrations=frame_calibrations,
        warnings=warnings,
        provenance={
            "pipeline": "neural-tracking-v2",
            "player_detector": cfg.player_model,
            "pose_model": "rtmpose-m-body7-simcc-17" if pose else None,
            "pose_sha256": POSE_SHA if pose else None,
            "weights_sha256": MODEL_SHA if cfg.player_model == "ssdlite" else None,
            "ball_detector": "WASB-HRNet-tennis-batched-triplets"
            if cfg.player_model == "ssdlite"
            else "color-motion-v2-temporal",
            "ball_weights_sha256": BALL_SHA if cfg.player_model == "ssdlite" else None,
            "ball_device": ball_detector.device
            if isinstance(ball_detector, TemporalBall)
            else "cpu",
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "device": detector.device if isinstance(detector, SSDPlayers) else "cpu",
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
    update_quality(result)
    result.live_metrics = live_metrics(result)
    result.shots = shot_signals(result)
    if cfg.player_model == "ssdlite":
        from baseline_lab.vision.strokes import recognize_strokes

        result.automatic_shots, result.stroke_analysis = recognize_strokes(
            source, metadata, players.points, cfg.data_dir / "models", checkpoint
        )
    result.display_ball_trail = visual_trail(ball, metadata.width, metadata.height)
    result.provenance["visual_trail"] = (
        "display-only shape-preserving bridges and damped end predictions; not measurements"
    )
    result.ball_speed_estimates, result.speed_analysis = research_speed(
        result, reference_image, real_time_playback
    )
    (output / "raw-detections.json").write_text(json.dumps(raw))
    render_start = time.monotonic()
    annotated(source, output / "annotated.mp4", result, checkpoint)
    timings["annotated_export_s"] = time.monotonic() - render_start
    checkpoint("preparing_playback", 0, None)
    preview(source, output / "review.mp4", (metadata.width, metadata.height))
    result.provenance["stage_timings_s"] = {k: round(v, 3) for k, v in timings.items()}
    result.elapsed_s = round(time.monotonic() - start, 3)
    (output / "result.json").write_text(result.model_dump_json())
    csvs(result, output)
    return result
