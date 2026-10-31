"""Decode in presentation order with real PTS; rotation normalized once."""

import json
import subprocess
from pathlib import Path

import av
import cv2
import imageio_ffmpeg
import numpy as np

from baseline_lab.contracts import Metadata
from baseline_lab.settings import Settings


def analysis_size(width: int, height: int, max_pixels: int = 1920 * 1080) -> tuple[int, int]:
    scale = min(1.0, (max_pixels / (width * height)) ** 0.5, 1920 / max(width, height))
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


def frames(path: Path, max_pixels: int = 1920 * 1080, max_source_pixels: int = 4096 * 2160):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        origin = None
        for frame in container.decode(stream):
            if (
                frame.width * frame.height > max_source_pixels
                or max(frame.width, frame.height) > 7680
            ):
                raise ValueError("source_pixel_limit_exceeded")
            if frame.pts is None:
                raise ValueError("missing_presentation_timestamp")
            timestamp = float(frame.pts * frame.time_base)
            if origin is None:
                origin = timestamp
            rotation = int(round(float(getattr(frame, "rotation", 0) or 0))) % 360
            if rotation not in (0, 90, 180, 270):
                raise ValueError("unsupported_non_right_angle_rotation")
            # Display-matrix rotation reported by PyAV is counterclockwise.
            rgb: np.ndarray = frame.to_ndarray(format="bgr24")
            if rotation:
                rgb = np.ascontiguousarray(np.rot90(rgb, rotation // 90))
            height, width = rgb.shape[:2]
            size = analysis_size(width, height, max_pixels)
            if size != (width, height):
                rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
            yield timestamp - origin, frame.pts, str(frame.time_base), rotation, rgb


def inspect(path: Path, cfg: Settings, thumbnail: Path | None = None) -> Metadata:
    pts: list[int] = []
    times: list[float] = []
    width = height = rotation = 0
    time_base = ""
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError("no_video_stream")
        stream = container.streams.video[0]
        codec = stream.codec_context.name
        if stream.sample_aspect_ratio and stream.sample_aspect_ratio != 1:
            raise ValueError("unsupported_non_square_pixels")
        source_width, source_height = stream.codec_context.width, stream.codec_context.height
        if (
            source_width * source_height > cfg.max_source_pixels
            or max(source_width, source_height) > 7680
        ):
            raise ValueError("source_pixel_limit_exceeded")
        nominal = float(stream.average_rate) if stream.average_rate else None
    for timestamp, p, tb, r, bgr in frames(path, cfg.max_pixels, cfg.max_source_pixels):
        if not times:
            height, width = bgr.shape[:2]
            rotation, time_base = r, tb
            if thumbnail:
                cv2.imwrite(str(thumbnail), bgr)
        if times and timestamp <= times[-1]:
            raise ValueError("non_monotonic_timestamps")
        pts.append(p)
        times.append(timestamp)
        if len(times) > cfg.max_frames or timestamp > cfg.max_duration_s:
            raise ValueError("duration_or_frame_limit_exceeded")
    if not times:
        raise ValueError("no_decodable_frames")
    delta = float(np.median(np.diff(times))) if len(times) > 1 else 0
    duration = times[-1] + delta
    if not cfg.min_duration_s <= duration <= cfg.max_duration_s + 0.1:
        raise ValueError("duration_outside_supported_range")
    vfr = bool(len(times) > 2 and np.max(np.abs(np.diff(times) - delta)) > 0.002)
    warnings = ["Manual review of camera position, lighting and singles suitability is required"]
    if rotation:
        warnings.append("Rotation normalized into displayed pixels")
    if rotation in (90, 270):
        source_width, source_height = source_height, source_width
    if (width, height) != (source_width, source_height):
        warnings.append(
            f"Source {source_width}×{source_height} resized to {width}×{height}; "
            "coordinates use analysis pixels"
        )
    return Metadata(
        width=width,
        height=height,
        duration_s=duration,
        codec=codec,
        frames=len(times),
        nominal_fps=nominal,
        variable_frame_rate=vfr,
        rotation_deg=rotation,
        timestamps_s=times,
        original_pts=pts,
        time_base=time_base,
        source_width=source_width,
        source_height=source_height,
        analysis_scale=width / source_width,
        warnings=warnings,
    )


def isolated_probe(source: Path, destination: Path, cfg: Settings) -> Metadata:
    # Independent decoder process, killed on timeout; never shell-interpolate user data.
    import os
    import sys

    env = dict(
        os.environ,
        BASELINE_DATA_DIR=str(cfg.data_dir),
        BASELINE_MAX_PIXELS=str(cfg.max_pixels),
        BASELINE_MAX_SOURCE_PIXELS=str(cfg.max_source_pixels),
        BASELINE_MAX_FRAMES=str(cfg.max_frames),
        BASELINE_MAX_DURATION_S=str(cfg.max_duration_s),
        BASELINE_MIN_DURATION_S=str(cfg.min_duration_s),
    )
    process = subprocess.run(
        [sys.executable, "-m", "baseline_lab.cli", "probe", str(source), str(destination)],
        capture_output=True,
        timeout=120,
        env=env,
    )
    if process.returncode:
        try:
            code = json.loads(process.stderr.decode().splitlines()[-1])["error"]
        except (ValueError, KeyError, IndexError):
            code = "invalid_or_unsupported_video"
        raise ValueError(code if code in MEDIA_ERRORS else "invalid_or_unsupported_video")
    return Metadata.model_validate_json((destination / "metadata.json").read_text())


MEDIA_ERRORS = {
    "source_pixel_limit_exceeded": "Above the 4K import limit. Export at 1080p and retry.",
    "duration_or_frame_limit_exceeded": "Trim to 5–30 seconds, at most 1,800 frames.",
    "duration_outside_supported_range": "Trim the recording to 5–30 seconds and try again.",
    "unsupported_non_square_pixels": "Export this recording with square pixels (MP4/H.264).",
    "non_monotonic_timestamps": "This recording has invalid timestamps. Re-export it as MP4/H.264.",
    "missing_presentation_timestamp": "Missing timestamps. Re-export as MP4/H.264.",
    "unsupported_non_right_angle_rotation": "Export with rotation applied before import.",
    "no_video_stream": "The file contains no video stream. Choose an MP4 or MOV video.",
    "no_decodable_frames": "No video frames could be decoded. Try an MP4/H.264 export.",
    "invalid_or_unsupported_video": "This video could not be read. Try an MP4/H.264 export.",
    "media_import_timeout": "Import took too long. Try a shorter clip or a 1080p MP4/H.264 export.",
}


def preview(source: Path, target: Path, size: tuple[int, int] | None = None) -> None:
    # ffmpeg honors display matrices and preserves VFR, with a bounded decode thread count.
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-threads",
            "2",
            "-i",
            str(source),
            "-an",
            *(["-vf", f"scale={size[0]}:{size[1]}"] if size else []),
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-fps_mode",
            "vfr",
            "-enc_time_base:v",
            "1:90000",
            "-video_track_timescale",
            "90000",
            "-movflags",
            "+faststart",
            str(target),
        ],
        check=True,
        timeout=90,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def write_metadata(meta: Metadata, directory: Path) -> None:
    (directory / "metadata.json").write_text(json.dumps(meta.model_dump()))
