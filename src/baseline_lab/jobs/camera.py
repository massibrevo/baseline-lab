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
from baseline_lab.contracts import (
    Calibration,
    CameraBatch,
    Correction,
    Event,
    Metadata,
    Observation,
    Result,
)
from baseline_lab.geometry.court import Court
from baseline_lab.rendering.export import annotated, csvs
from baseline_lab.settings import Settings
from baseline_lab.tracking.association import BallAssociation, Players, summarize
from baseline_lab.video.camera_recording import CameraRecording
from baseline_lab.video.stability import CameraStability
from baseline_lab.vision.detectors import ColorMotionBall, HOGPlayers, SSDPlayers


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
    players = Players(court, payload["width"], payload["height"])
    ball_detector = ColorMotionBall()
    association = BallAssociation(math.hypot(payload["width"], payload["height"]))
    ball: list[Observation] = []
    timestamps = []
    raw_timestamps = []
    capture_origin = None
    recording = CameraRecording(output / "review.mp4", payload["width"], payload["height"])
    previous = None
    stability = None
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
    reviewed = []
    consumed = set()
    warnings = [
        "Sparse live batches; missed frames are not measurements",
        "Ball image motion is not physical 3D speed",
        "Manual contact depth is approximate; all line calls uncertain",
    ]
    result = None
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
                for index, captured in enumerate(batch.frames):
                    if captured.t_s <= last_capture:
                        continue
                    if len(ball) >= 1800:
                        break
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
                    if stability is None:
                        stability = CameraStability(image)
                    if index == len(batch.frames) - 1 and court is not None:
                        drift = stability.drift(image)
                        if drift is None or drift > 8:
                            court = None
                            players.court = None
                            warnings.append(
                                "Camera motion invalidated calibration; stop and recalibrate"
                            )
                            for points in players.points.values():
                                for point in points:
                                    point.court_m = None
                    if capture_origin is None:
                        capture_origin = captured.t_s
                    sample_t = captured.t_s - capture_origin
                    raw_timestamps.append(sample_t)
                    recording.write(image, sample_t)
                    detections = ball_detector.detect(image, previous)
                    if court:
                        polygon = np.asarray(court.corners, np.float32)
                        detections = [
                            d
                            for d in detections
                            if cv2.pointPolygonTest(polygon, d["xy"], True) >= -50
                        ]
                    ball.append(association.update(detections, sample_t, len(ball)))
                    # Full person inference is one frame per burst, explicitly missing elsewhere.
                    people = detector.detect(image) if index == len(batch.frames) - 1 else []
                    players.update(people, sample_t, len(ball) - 1)
                    timestamps.append(sample_t)
                    last_capture = captured.t_s
                    previous = image
                    last_image = image
                if last_image is not None:
                    for contact in directory.glob("camera-contact-*.json"):
                        if contact.name in consumed:
                            continue
                        body = Correction.model_validate_json(contact.read_text())
                        position = (
                            court.project(body.xy_px)
                            if court and body.status != "rejected"
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
                        calibration=court.details() if court else None,
                        warnings=warnings,
                        elapsed_s=time.monotonic() - started,
                        provenance={
                            "source": "local_camera",
                            "timestamps": "browser performance clock relative to capture start",
                            "sampling": (
                                "up to 8 sequential frames per latest-only burst; "
                                "one person inference per burst"
                            ),
                            "device": "cpu",
                        },
                    )
                    result.live_metrics = live_metrics(result)
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
                            "player_speed_m_s": speeds,
                            "tracks": [
                                {"id": t.id, "observation": t.observations[-1].model_dump()}
                                for t in tracks
                            ],
                            "events": [e.model_dump() for e in result.events[-20:]],
                            "calibration_valid": court is not None,
                            "warnings": list(dict.fromkeys(warnings)),
                        },
                    )
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
