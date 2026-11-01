"""Camera job in the existing isolated single-worker slot, never in HTTP handlers."""

import base64
import io
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from baseline_lab.analytics.events import candidates
from baseline_lab.analytics.live import live_metrics
from baseline_lab.analytics.quality import update_quality
from baseline_lab.analytics.shots import shot_signals
from baseline_lab.analytics.speed import research_speed
from baseline_lab.contracts import (
    BallSpeedSample,
    Calibration,
    CameraBatch,
    CameraFrame,
    Correction,
    Event,
    Metadata,
    Observation,
    Result,
)
from baseline_lab.geometry.court import Court
from baseline_lab.rendering.export import annotated, csvs
from baseline_lab.rendering.trail import visual_trail
from baseline_lab.settings import Settings
from baseline_lab.tracking.association import BallAssociation, Players, summarize
from baseline_lab.video.camera_recording import CameraRecording
from baseline_lab.video.stability import CameraStability
from baseline_lab.vision.ball import BALL_SHA, TemporalBall
from baseline_lab.vision.detectors import ColorMotionBall, HOGPlayers, SSDPlayers
from baseline_lab.vision.people import flow_people
from baseline_lab.vision.pose import POSE_SHA, PoseEstimator, far_player_regions
from baseline_lab.vision.strokes import LiveStrokeRecognizer, thumbnail


def atomic_json(path: Path, value: dict):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value))
    temp.replace(path)


def run_camera(directory: Path, output: Path, payload: dict, cfg: Settings, checkpoint):
    detector = (
        SSDPlayers(cfg.data_dir / "models") if cfg.player_model == "ssdlite" else HOGPlayers()
    )
    calibration = (
        Calibration.model_validate(payload["calibration"]) if payload.get("calibration") else None
    )
    court = Court(calibration) if calibration else None
    reference_court = court
    calibration_frames = []
    players = Players(court, payload["width"], payload["height"])
    pose = PoseEstimator(cfg.data_dir / "models") if cfg.player_model == "ssdlite" else None
    if court and isinstance(detector, SSDPlayers):
        detector.focus_regions = far_player_regions(court, payload["width"], payload["height"])
    ball_detector = (
        TemporalBall(cfg.data_dir / "models")
        if cfg.player_model == "ssdlite"
        else ColorMotionBall()
    )
    association = BallAssociation(
        math.hypot(payload["width"], payload["height"]), neural=cfg.player_model == "ssdlite"
    )
    stroke_engine = LiveStrokeRecognizer(cfg.data_dir / "models")
    ball: list[Observation] = []
    timestamps = []
    raw_timestamps = []
    capture_origin = None
    recording = CameraRecording(output / "review.mp4", payload["width"], payload["height"])
    previous = None
    stability = None
    reference_image = None
    reference = directory / "camera-reference.jpg"
    if reference.exists():
        with Image.open(reference) as header:
            if header.format != "JPEG" or header.size != (payload["width"], payload["height"]):
                raise ValueError("invalid_calibration_reference")
        reference_image = cv2.imread(str(reference))
        if reference_image is None:
            raise ValueError("invalid_calibration_reference")
        stability = CameraStability(reference_image)
    sequence = -1
    skipped = 0
    bursts = 0
    last_capture = -1.0
    started = time.monotonic()
    last_input = started
    reviewed: list[Event] = []
    consumed = set()
    warnings = [
        "Sparse live batches; missed frames are not measurements",
        "Ball image motion is not physical 3D speed",
        "Manual contact depth is approximate; all line calls uncertain",
    ]
    result = None
    speed_samples: list[BallSpeedSample] = []
    while time.monotonic() - started < 70:
        checkpoint("live_waiting_for_frames", len(ball), None)
        path = directory / "camera-input.json"
        if path.exists():
            batch = CameraBatch.model_validate_json(path.read_text())
            if batch.sequence > sequence:
                stage_start = time.monotonic()
                skipped += max(0, batch.sequence - sequence - 1)
                bursts += 1
                sequence = batch.sequence
                last_input = stage_start
                last_image = None
                decoded: list[tuple[CameraFrame, np.ndarray]] = []
                for captured in batch.frames:
                    if captured.t_s <= last_capture or len(ball) + len(decoded) >= 1800:
                        continue
                    binary = base64.b64decode(captured.jpeg_base64, validate=True)
                    with Image.open(io.BytesIO(binary)) as header:
                        if header.format != "JPEG" or header.size != (
                            payload["width"],
                            payload["height"],
                        ):
                            raise ValueError("camera_dimensions_or_format_changed")
                    image = cv2.imdecode(np.frombuffer(binary, np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError("invalid_camera_jpeg")
                    decoded.append((captured, image))
                batch_detections = (
                    ball_detector.detect_batch([image for _, image in decoded])
                    if isinstance(ball_detector, TemporalBall) and decoded
                    else None
                )
                for index, (captured, image) in enumerate(decoded):
                    if reference_image is None:
                        reference_image = image.copy()
                    if stability is None:
                        stability = CameraStability(image)
                    details = None
                    if reference_court:
                        court = None
                        mapping = stability.mapping(image)
                        if mapping is not None and mapping[1] < payload["width"] * 0.15:
                            transformed = cv2.perspectiveTransform(
                                np.array([reference_court.corners], np.float32), mapping[0]
                            )[0]
                            try:
                                court = Court(Calibration(corners_px=transformed.tolist()))
                                details = court.details()
                                details.update(
                                    {
                                        "camera_drift_px": mapping[1],
                                        "registration_residual_px": mapping[2],
                                    }
                                )
                            except ValueError:
                                pass
                        players.court = court
                    calibration_frames.append(details)
                    if capture_origin is None:
                        capture_origin = captured.t_s
                    sample_t = captured.t_s - capture_origin
                    raw_timestamps.append(sample_t)
                    recording.write(image, sample_t)
                    detections = (
                        batch_detections[index]
                        if batch_detections is not None
                        else ball_detector.detect(image, previous)
                    )
                    if court:
                        top = min(y for _, y in court.corners) - 0.3 * payload["height"]
                        left = min(x for x, _ in court.corners) - 0.2 * payload["width"]
                        right = max(x for x, _ in court.corners) + 0.2 * payload["width"]
                        detections = [
                            d
                            for d in detections
                            if d["xy"][1] >= top and left <= d["xy"][0] <= right
                        ]
                    ball.append(association.update(detections, sample_t, len(ball)))
                    people: list[dict] = []
                    if previous is not None and captured.t_s - last_capture <= 0.1:
                        people = flow_people(
                            previous, image, [v[-1] for v in players.points.values()]
                        )
                    if index in (0, len(decoded) - 1) or len(people) < 2:
                        people = detector.detect(image)
                    players.update(people, sample_t, len(ball) - 1)
                    if pose and people:
                        for identity, observations in players.points.items():
                            pose.estimate(image, observations[-1], court, identity=identity)
                    timestamps.append(sample_t)
                    last_capture = captured.t_s
                    previous = image
                    last_image = image
                if last_image is not None:
                    for contact in directory.glob("camera-contact-*.json"):
                        if contact.name in consumed:
                            continue
                        body = Correction.model_validate_json(contact.read_text())
                        contact_frame = min(
                            range(len(timestamps)), key=lambda i: abs(timestamps[i] - body.t_s)
                        )
                        contact_calibration = calibration_frames[contact_frame]
                        contact_court = (
                            Court(Calibration(corners_px=contact_calibration["corners_px"]))
                            if contact_calibration
                            else None
                        )
                        position = (
                            contact_court.project(body.xy_px)
                            if contact_court and body.status != "rejected"
                            else None
                        )
                        reviewed.append(
                            Event(
                                id=contact.stem,
                                t_s=body.t_s,
                                frame=min(
                                    range(len(timestamps)),
                                    key=lambda i: abs(timestamps[i] - body.t_s),
                                ),
                                xy_px=body.xy_px,
                                status="manual",
                                court_m=position,
                                shot_depth_m=min(position[1], 23.77 - position[1])
                                if position
                                else None,
                                notes=body.notes,
                                shot_evidence=body.shot_evidence,
                            )
                        )
                        consumed.add(contact.name)
                    meta = Metadata(
                        width=payload["width"],
                        height=payload["height"],
                        duration_s=timestamps[-1],
                        codec="camera_jpeg_bursts",
                        frames=len(ball),
                        nominal_fps=None,
                        variable_frame_rate=True,
                        rotation_deg=0,
                        timestamps_s=timestamps,
                        original_pts=[round(t * 1e6) for t in raw_timestamps],
                        time_base="1/1000000",
                        warnings=warnings,
                    )
                    tracks = [summarize("ball", ball, False)] + [
                        summarize(k, v, court is not None) for k, v in players.points.items()
                    ]
                    result = Result(
                        metadata=meta,
                        tracks=tracks,
                        events=candidates(ball, list(players.points.values())) + reviewed,
                        calibration=reference_court.details() if reference_court else None,
                        frame_calibrations=calibration_frames,
                        warnings=warnings,
                        elapsed_s=time.monotonic() - started,
                        provenance={
                            "source": "local_camera",
                            "pose_sha256": POSE_SHA if pose else None,
                            "ball_weights_sha256": BALL_SHA
                            if cfg.player_model == "ssdlite"
                            else None,
                            "timestamps": "browser performance clock relative to capture start",
                            "sampling": (
                                "up to 8 sequential frames per latest-only burst; "
                                "person detections with checked optical flow between"
                            ),
                            "device": "cpu",
                        },
                    )
                    update_quality(result)
                    result.live_metrics = live_metrics(result)
                    result.shots = shot_signals(result)
                    if cfg.player_model == "ssdlite":
                        assert capture_origin is not None
                        result.automatic_shots, result.stroke_analysis = stroke_engine.update(
                            [
                                (captured.t_s - capture_origin, thumbnail(image))
                                for captured, image in decoded
                            ],
                            meta,
                            players.points,
                            checkpoint,
                        )
                    result.display_ball_trail = visual_trail(
                        ball, payload["width"], payload["height"], offline=False
                    )
                    result.ball_speed_estimates, result.speed_analysis = research_speed(
                        result, reference_image, True, offline=False
                    )
                    for i, sample in enumerate(speed_samples):
                        result.ball_speed_estimates[i] = sample
                    speed_samples = result.ball_speed_estimates
                    # Sparse person observations require a clearly labeled inter-burst estimate.
                    speeds = {}
                    for identity in ["near", "far"]:
                        observed = [
                            p
                            for p in players.points[identity]
                            if p.court_m and p.state == "detected"
                        ]
                        if len(observed) >= 2 and observed[-1].frame == len(ball) - 1:
                            a, b = observed[-2:]
                            assert a.court_m is not None and b.court_m is not None
                            dt = b.t_s - a.t_s
                            speed = math.dist(a.court_m, b.court_m) / dt if dt > 0 else None
                            speeds[identity] = (
                                speed if speed is not None and dt <= 1 and speed <= 12 else None
                            )
                    temp_image = directory / "camera-frame.tmp.jpg"
                    cv2.imwrite(str(temp_image), last_image)
                    temp_image.replace(directory / f"camera-frame-{sequence}.jpg")
                    for old in directory.glob("camera-frame-*.jpg"):
                        if int(old.stem.split("-")[-1]) < sequence - 2:
                            old.unlink(missing_ok=True)
                    atomic_json(
                        directory / "camera-state.json",
                        {
                            "sequence": sequence,
                            "skipped_batches": skipped,
                            "processed_bursts": bursts,
                            "capture_t_s": last_capture,
                            "processing_ms": round((time.monotonic() - stage_start) * 1000),
                            "metrics": result.live_metrics[-1].model_dump(),
                            "shots": [s.model_dump() for s in result.shots[-5:]],
                            "automatic_shots": [
                                s.model_dump() for s in result.automatic_shots[-10:]
                            ],
                            "stroke_analysis": result.stroke_analysis,
                            "ball_speed": result.ball_speed_estimates[-1].model_dump(),
                            "player_speed_m_s": speeds,
                            "tracks": [
                                {"id": t.id, "observation": t.observations[-1].model_dump()}
                                for t in tracks
                            ],
                            "events": [e.model_dump() for e in result.events[-20:]],
                            "calibration_valid": court is not None,
                            "calibration": calibration_frames[-1],
                            "display_ball_trail": [
                                p.model_dump() for p in result.display_ball_trail[-180:]
                            ],
                            "ball_trail": [
                                p.model_dump() for p in ball[-120:] if ball[-1].t_s - p.t_s <= 2
                            ],
                            "warnings": list(dict.fromkeys(warnings)),
                        },
                    )
                    last_input = time.monotonic()
                    checkpoint("live_processing", len(ball), None)
        if (directory / "camera-stop").exists() or len(ball) >= 1800:
            break
        if time.monotonic() - last_input > 10:
            warnings.append("Camera input idle timeout")
            break
        time.sleep(0.03)
    recording.close()
    if result is None:
        raise ValueError("no_camera_frames_received")
    annotated(output / "review.mp4", output / "annotated.mp4", result, checkpoint)
    (output / "result.json").write_text(result.model_dump_json())
    csvs(result, output)
    (directory / "camera-input.json").unlink(missing_ok=True)
    (directory / "camera-stop").unlink(missing_ok=True)
