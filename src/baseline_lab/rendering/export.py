"""Timestamp-preserving H.264 exports and explicit-coordinate CSVs."""

import csv
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from baseline_lab.contracts import BallSpeedSample, Result
from baseline_lab.rendering.trail import draw_broadcast_trail, visual_trail
from baseline_lab.video.media import frames
from baseline_lab.vision.pose import draw_pose


def csvs(result: Result, directory: Path) -> None:
    with (directory / "visual-trail.csv").open("w", newline="") as trail_file:
        writer = csv.writer(trail_file)
        writer.writerow(
            [
                "time_s",
                "frame",
                "display_x_px",
                "display_y_px",
                "display_state",
                "reason",
                "purpose",
            ]
        )
        for visual_point in result.display_ball_trail:
            writer.writerow(
                [
                    visual_point.t_s,
                    visual_point.frame,
                    *(visual_point.xy_px or (None, None)),
                    visual_point.state,
                    visual_point.reason,
                    "visualization_only_not_measurement",
                ]
            )
    with (directory / "ball-speed.csv").open("w", newline="") as speeds:
        writer = csv.writer(speeds)
        fields = list(BallSpeedSample.model_fields)
        writer.writerow(fields)
        writer.writerows(
            [sample.model_dump()[key] for key in fields] for sample in result.ball_speed_estimates
        )
    with (directory / "shots.csv").open("w", newline="") as shots:
        writer = csv.writer(shots)
        writer.writerow(
            [
                "id",
                "time_s",
                "frame",
                "player",
                "stroke_type",
                "image_speed_px_s",
                "status",
                "method",
                "label_source",
                "context_start_s",
                "context_end_s",
                "recognition_basis",
            ]
        )
        for shot in sorted(result.automatic_shots + result.shots, key=lambda s: s.t_s):
            writer.writerow(
                [
                    shot.id,
                    shot.t_s,
                    shot.frame,
                    shot.player,
                    shot.kind,
                    shot.image_speed_px_s,
                    shot.status,
                    shot.method,
                    "automatic"
                    if shot in result.automatic_shots
                    else "human_review"
                    if shot.status == "reviewed"
                    else "unclassified_candidate",
                    shot.context_start_s,
                    shot.context_end_s,
                    shot.recognition_basis,
                ]
            )
    with (directory / "keypoints.csv").open("w", newline="") as poses:
        writer = csv.writer(poses)
        writer.writerow(
            [
                "player",
                "frame",
                "time_s",
                "joint",
                "state",
                "x_px",
                "y_px",
                "visibility",
                "presence",
                "pose_score",
                "footpoint_source",
                "joint_score",
                "joint_score_semantics",
            ]
        )
        for track in result.tracks:
            for observation in track.observations:
                for point in observation.keypoints:
                    writer.writerow(
                        [
                            track.id,
                            observation.frame,
                            observation.t_s,
                            point.name,
                            point.state,
                            *(point.xy_px or (None, None)),
                            point.visibility,
                            point.presence,
                            observation.pose_score,
                            observation.footpoint_source,
                            point.score,
                            point.score_semantics,
                        ]
                    )
    with (directory / "shot-quality.csv").open("w", newline="") as shots:
        writer = csv.writer(shots)
        writer.writerow(
            [
                "event_id",
                "time_s",
                "quality_0_10",
                "status",
                "available_factors",
                "weighted_evidence",
                "factor",
                "value",
                "unit",
                "factor_score_0_10",
                "weight",
                "source",
                "error_attribution",
            ]
        )
        for event in result.events:
            quality = event.shot_quality
            if quality:
                for name, factor in quality.factors.items():
                    source = (
                        "'" + factor.source
                        if factor.source[:1] in ("=", "+", "-", "@")
                        else factor.source
                    )
                    writer.writerow(
                        [
                            event.id,
                            event.t_s,
                            quality.score,
                            quality.status,
                            quality.available_factors,
                            quality.evidence_coverage,
                            name,
                            factor.value,
                            factor.unit,
                            factor.score,
                            factor.weight,
                            source,
                            quality.error_attribution,
                        ]
                    )
    with (directory / "live-metrics.csv").open("w", newline="") as live:
        writer = csv.writer(live)
        fields = list(result.live_metrics[0].model_dump()) if result.live_metrics else []
        writer.writerow(fields)
        for sample in result.live_metrics:
            writer.writerow(sample.model_dump().values())
    with (directory / "tracks.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "track_id",
                "frame",
                "time_s",
                "state",
                "x_px",
                "y_px",
                "court_x_m",
                "court_y_m",
                "score",
                "reason",
            ]
        )
        for track in result.tracks:
            for p in track.observations:
                writer.writerow(
                    [
                        track.id,
                        p.frame,
                        p.t_s,
                        p.state,
                        *(p.xy_px or (None, None)),
                        *(p.court_m or (None, None)),
                        p.score,
                        p.reason,
                    ]
                )
    with (directory / "events.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "event_id",
                "time_s",
                "frame",
                "x_px",
                "y_px",
                "status",
                "classification",
                "court_x_m",
                "court_y_m",
                "shot_depth_m",
                "evidence_score",
                "notes",
            ]
        )
        for e in result.events:
            # Prefix user text that spreadsheet programs might interpret as formulae.
            notes = "'" + e.notes if e.notes[:1] in ("=", "+", "-", "@") else e.notes
            writer.writerow(
                [
                    e.id,
                    e.t_s,
                    e.frame,
                    *e.xy_px,
                    e.status,
                    e.classification,
                    *(e.court_m or (None, None)),
                    e.shot_depth_m,
                    e.evidence_score,
                    notes,
                ]
            )


def annotated(source: Path, target: Path, result: Result, checkpoint) -> None:
    meta = result.metadata
    display = result.display_ball_trail or visual_trail(
        result.tracks[0].observations, meta.width, meta.height
    )
    with av.open(str(target), "w") as out:
        stream = out.add_stream(
            "libx264", rate=Fraction(str(meta.nominal_fps or 30)).limit_denominator()
        )
        stream.width, stream.height = meta.width, meta.height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, 90000)
        stream.codec_context.time_base = Fraction(1, 90000)
        stream.options = {"crf": "23", "preset": "fast", "threads": "2"}
        for i, (t, _, _, _, image) in enumerate(frames(source)):
            if image.shape[1::-1] != (meta.width, meta.height):
                image = cv2.resize(image, (meta.width, meta.height), interpolation=cv2.INTER_AREA)
            checkpoint("rendering", i, meta.frames)
            calibration = (
                result.frame_calibrations[i] if result.frame_calibrations else result.calibration
            )
            if calibration:
                pts = np.array(calibration["corners_px"], np.int32)
                cv2.polylines(image, [pts], True, (210, 230, 225), 1, cv2.LINE_AA)
            for track in result.tracks:
                if track.id == "ball":
                    draw_broadcast_trail(image, display[: i + 1], t)
                point = track.observations[i]
                draw_pose(image, point)
                if point.xy_px:
                    x, y = (int(v) for v in point.xy_px)
                    if track.id != "ball" and point.bbox_px:
                        x1, y1, x2, y2 = (int(v) for v in point.bbox_px)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (165, 142, 108), 2)
                        cv2.putText(
                            image,
                            track.id.upper(),
                            (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            (245, 240, 230),
                            1,
                        )
            cv2.putText(
                image,
                f"BASELINE LAB | {t:.3f}s | estimates",
                (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (245, 240, 230),
                1,
                cv2.LINE_AA,
            )
            if result.ball_speed_estimates:
                speed = result.ball_speed_estimates[i]
                if speed.estimated_kmh is not None:
                    cv2.putText(
                        image,
                        f"BALL ~{speed.estimated_kmh:.0f} km/h | experimental fit",
                        (15, meta.height - 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (245, 240, 230),
                        1,
                        cv2.LINE_AA,
                    )
            active = [
                s
                for s in result.automatic_shots
                if s.context_start_s is not None
                and s.context_end_s is not None
                and s.context_start_s <= t <= s.context_end_s
            ]
            if active:
                shot = min(active, key=lambda s: abs(s.t_s - t))
                label = shot.kind.replace("_", " ").upper()
                cv2.putText(
                    image,
                    f"AUTO: {label} | {shot.player} | provisional",
                    (15, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (69, 239, 216),
                    1,
                    cv2.LINE_AA,
                )
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = round(t * 90000)
            frame.time_base = Fraction(1, 90000)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
