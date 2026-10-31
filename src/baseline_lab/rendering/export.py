"""Timestamp-preserving H.264 exports and explicit-coordinate CSVs."""

import csv
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np

from baseline_lab.contracts import Result
from baseline_lab.video.media import frames


def csvs(result: Result, directory: Path) -> None:
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
                    e.evidence_score,
                    notes,
                ]
            )


def annotated(source: Path, target: Path, result: Result, checkpoint) -> None:
    meta = result.metadata
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
            checkpoint("rendering", i, meta.frames)
            if result.calibration:
                pts = np.array(result.calibration["corners_px"], np.int32)
                cv2.polylines(image, [pts], True, (210, 230, 225), 1, cv2.LINE_AA)
            for track in result.tracks:
                point = track.observations[i]
                if point.xy_px:
                    x, y = (int(v) for v in point.xy_px)
                    if track.id == "ball":
                        cv2.circle(image, (x, y), 6, (69, 239, 216), 2, cv2.LINE_AA)
                    elif point.bbox_px:
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
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = round(t * 90000)
            frame.time_base = Fraction(1, 90000)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
