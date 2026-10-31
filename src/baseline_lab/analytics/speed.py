"""Experimental metric speed from calibrated rays and a constrained 3D flight.

This is NOT a validated speed gun. Normal 1x timing, square pixels, centered
principal point, negligible distortion, and constant gravity/no drag or Magnus
are assumed. Unsupported/unstable fits remain unavailable. No airborne point
is treated as a ground contact. See docs/research-speed.md.
"""

import math

import cv2
import numpy as np

from baseline_lab.contracts import BallSpeedSample, Calibration, Observation, Result
from baseline_lab.geometry.court import COURT, Court

GRAVITY = 9.81


def refine_court(image: np.ndarray, court: Court) -> Court:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 90, 255]))
    ys, xs = np.where(mask > 0)
    points = np.column_stack([xs, ys]).astype(np.float32)
    corners = np.array(court.corners, np.float32)
    lines = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        vector = b - a
        length = np.linalg.norm(vector)
        unit = vector / length
        relative = points - a
        along = relative @ unit
        distance = abs(relative[:, 0] * unit[1] - relative[:, 1] * unit[0])
        corridor = (8 if i == 0 else 25) * image.shape[1] / 1920
        selected = points[(distance < corridor) & (along > length * 0.03) & (along < length * 0.97)]
        if len(selected) < max(40, length * 0.25):
            return court
        vx, vy, x, y = cv2.fitLine(selected, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        lines.append(np.array([vy, -vx, vx * y - vy * x]))
    refined = []
    for i in range(4):
        point = np.cross(lines[(i - 1) % 4], lines[i])
        if abs(point[2]) < 1e-8:
            return court
        refined.append(point[:2] / point[2])
    if max(np.linalg.norm(np.array(refined) - corners, axis=1)) > image.shape[1] * 0.025:
        return court
    try:
        return Court(Calibration(corners_px=np.array(refined).tolist()))
    except ValueError:
        return court


def camera_intrinsics(court: Court, width: int, height: int) -> np.ndarray:
    h = court.inverse
    cx, cy = width / 2, height / 2
    a = h[:, 0] - np.array([cx, cy, 0]) * h[2, 0]
    b = h[:, 1] - np.array([cx, cy, 0]) * h[2, 1]
    q = np.array([a[0] * b[0] + a[1] * b[1], a[0] ** 2 + a[1] ** 2 - b[0] ** 2 - b[1] ** 2])
    z = np.array([a[2] * b[2], a[2] ** 2 - b[2] ** 2])
    denominator = float(q @ z)
    if abs(denominator) < 1e-12:
        raise ValueError("camera_intrinsics_ambiguous")
    f2 = -float(q @ q) / denominator
    if not (width * 0.3) ** 2 < f2 < (width * 5) ** 2:
        raise ValueError("camera_intrinsics_ambiguous")
    focal = math.sqrt(f2)
    return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1.0]])


def camera_projection(court: Court, intrinsics: np.ndarray):
    b = np.linalg.inv(intrinsics) @ court.inverse
    scale = 2 / (np.linalg.norm(b[:, 0]) + np.linalg.norm(b[:, 1]))
    for sign in [1, -1]:
        r1, r2, t = sign * scale * b[:, 0], sign * scale * b[:, 1], sign * scale * b[:, 2]
        if t[2] <= 0:
            continue
        for normal in [1, -1]:
            raw = np.column_stack([r1, r2, normal * np.cross(r1, r2)])
            u, _, v = np.linalg.svd(raw)
            rotation = u @ v
            center = -rotation.T @ t
            if not 1 < center[2] < 15:
                continue
            projection = intrinsics @ np.column_stack([rotation, t])
            ground = np.column_stack([COURT, np.zeros(4), np.ones(4)]) @ projection.T
            image = ground[:, :2] / ground[:, 2, None]
            error = float(np.max(np.linalg.norm(image - np.array(court.corners), axis=1)))
            if error > max(2, intrinsics[0, 2] * 2 / 384):
                raise ValueError("camera_reprojection_inconsistent")
            return projection, center, error
    raise ValueError("camera_pose_ambiguous")


def fit_flight(
    points: list[Observation], projections: dict[int, np.ndarray], focal_px: float | None = None
):
    diameter_support = sum(p.apparent_diameter_px is not None for p in points)
    if len(points) < 8 or points[-1].t_s - points[0].t_s < (
        0.22 if diameter_support >= 6 else 0.35
    ):
        return None
    if max(np.diff([p.t_s for p in points])) > 0.12:
        return None
    origin = points[0].t_s
    rows = []
    targets = []
    matrices = []
    for point in points:
        if point.frame not in projections or point.xy_px is None:
            return None
        p = projections[point.frame]
        matrices.append(p)
        dt = point.t_s - origin
        for coordinate, index in zip(point.xy_px, [0, 1], strict=True):
            a = p[index, :3] - coordinate * p[2, :3]
            rows.append(np.r_[a, a * dt])
            targets.append(0.5 * GRAVITY * dt * dt * a[2] - (p[index, 3] - coordinate * p[2, 3]))
    design = np.array(rows)
    target = np.array(targets)
    parameters = np.linalg.lstsq(design, target, rcond=None)[0]
    for _ in range(4):
        weights = []
        for point, p in zip(points, matrices, strict=True):
            dt = point.t_s - origin
            position = parameters[:3] + parameters[3:] * dt - [0, 0, 0.5 * GRAVITY * dt * dt]
            depth = abs((p @ np.r_[position, 1])[2])
            weights.extend([1 / max(depth, 0.01)] * 2)
        w = np.array(weights)
        residual = (design @ parameters - target) * w
        w *= np.sqrt(np.minimum(1, 4 / np.maximum(abs(residual), 0.01)))
        weighted = design * w[:, None]
        weighted_target = target * w
        prior_rows = []
        prior_targets = []
        for point, p in zip(points, matrices, strict=True):
            if point.apparent_diameter_px is None or focal_px is None:
                continue
            # Known nominal ball diameter provides a noisy, independent ray-depth
            # constraint. Blur/color segmentation uncertainty is deliberately large.
            depth = focal_px * 0.067 / point.apparent_diameter_px
            dt = point.t_s - origin
            weight = 2 / max(2.0, depth * 0.35)
            prior_rows.append(np.r_[p[2, :3], p[2, :3] * dt] * weight)
            prior_targets.append((depth - p[2, 3] + 0.5 * GRAVITY * dt * dt * p[2, 2]) * weight)
        if prior_rows:
            weighted = np.vstack([weighted, prior_rows])
            weighted_target = np.r_[weighted_target, prior_targets]
        parameters = np.linalg.lstsq(weighted, weighted_target, rcond=None)[0]
    if np.linalg.cond(weighted) > 5000:
        return None
    errors = []
    positions = []
    for point, p in zip(points, matrices, strict=True):
        dt = point.t_s - origin
        position = parameters[:3] + parameters[3:] * dt - [0, 0, 0.5 * GRAVITY * dt * dt]
        projected = p @ np.r_[position, 1]
        if projected[2] <= 0.25:
            return None
        positions.append(position)
        errors.append(np.linalg.norm(projected[:2] / projected[2] - point.xy_px))
    positions_array = np.array(positions)
    rms = float(np.sqrt(np.mean(np.square(errors))))
    if (
        rms > 6
        or max(errors) > 16
        or min(positions_array[:, 2]) < -0.3
        or max(positions_array[:, 2]) > 8
        or min(positions_array[:, 0]) < -6
        or max(positions_array[:, 0]) > 14
        or min(positions_array[:, 1]) < -10
        or max(positions_array[:, 1]) > 35
    ):
        return None
    dt = points[-1].t_s - origin
    velocity = parameters[3:] - [0, 0, GRAVITY * dt / 2]
    speed = float(np.linalg.norm(velocity) * 3.6)
    if not 15 < speed < 250:
        return None
    pixel_sigma = max(2.0, rms)
    covariance = np.linalg.pinv(weighted.T @ weighted) * pixel_sigma**2
    gradient = np.r_[np.zeros(3), velocity / max(np.linalg.norm(velocity), 1e-9) * 3.6]
    sigma = math.sqrt(max(0, float(gradient @ covariance @ gradient)))
    if sigma > speed * 0.4:
        return None
    # This floor is a sensitivity allowance, not a calibrated confidence interval.
    sensitivity = max(speed * 0.25, 2 * sigma)
    return {
        "start": origin,
        "end": points[-1].t_s,
        "speed": speed,
        "sensitivity": sensitivity,
        "rms": rms,
        "n": len(points),
    }


def research_speed(
    result: Result, reference_image: np.ndarray | None, timing_confirmed: bool, offline: bool = True
):
    ball = next(t for t in result.tracks if t.id == "ball").observations
    samples = [
        BallSpeedSample(t_s=p.t_s, frame=p.frame, reason="insufficient_supported_flight")
        for p in ball
    ]
    details = {
        "method": "constrained_3d_ballistic_fit_v1",
        "validated_with_radar": False,
        "timing_confirmed": timing_confirmed,
        "assumptions": [
            "normal 1x action timing",
            "square pixels and centered principal point",
            "negligible lens distortion",
            "gravity 9.81 m/s²; drag and Magnus omitted",
        ],
        "mode": "offline_window_fit" if offline else "causal_window_fit",
    }
    if not timing_confirmed or not result.calibration:
        reason = (
            "normal_1x_timing_not_confirmed"
            if not timing_confirmed
            else "court_calibration_missing"
        )
        for sample in samples:
            sample.reason = reason
        return samples, details
    original = Court(Calibration(corners_px=result.calibration["corners_px"]))
    refined = refine_court(reference_image, original) if reference_image is not None else original
    try:
        intrinsics = camera_intrinsics(refined, result.metadata.width, result.metadata.height)
        _, center, error = camera_projection(refined, intrinsics)
    except ValueError as exc:
        for sample in samples:
            sample.reason = str(exc)
        return samples, details
    details.update(
        {
            "refined_corners_px": refined.corners,
            "focal_px": float(intrinsics[0, 0]),
            "camera_height_m": float(center[2]),
            "ground_reprojection_error_px": error,
            "sensitivity_is_not_confidence_interval": True,
        }
    )
    projections = {}
    for i in range(len(ball)):
        calibration = (
            result.frame_calibrations[i] if result.frame_calibrations else result.calibration
        )
        if not calibration:
            continue
        transform = np.linalg.inv(calibration["homography"]) @ original.h
        current = refined
        try:
            # Reconstruct current corners without treating airborne pixels as ground.
            corners = cv2.perspectiveTransform(np.array([refined.corners], np.float32), transform)[
                0
            ]
            current = Court(Calibration(corners_px=corners.tolist()))
            projections[i] = camera_projection(current, intrinsics)[0]
        except ValueError:
            continue
    models = []
    candidates = enumerate(ball) if offline else [(len(ball) - 1, ball[-1])]
    for i, point in candidates:
        if point.state != "detected":
            continue
        for span in [0.3, 0.45, 0.65, 0.85]:
            window = [
                p
                for p in ball[max(0, i - 60) : i + 1]
                if point.t_s - p.t_s <= span and p.state == "detected" and p.xy_px
            ]
            model = fit_flight(window, projections, float(intrinsics[0, 0]))
            if model:
                models.append(model)
    for sample in samples if offline else samples[-1:]:
        supported = [
            m
            for m in models
            if m["start"] <= sample.t_s <= m["end"] and (offline or m["end"] <= sample.t_s + 1e-5)
        ]
        if not supported:
            continue
        best = min(
            supported,
            key=lambda m: (
                m["rms"] + m["sensitivity"] / m["speed"] * 4 - min(1, m["end"] - m["start"])
            ),
        )
        sample.estimated_kmh = round(best["speed"], 1)
        sample.low_kmh = round(max(0, best["speed"] - best["sensitivity"]), 1)
        sample.high_kmh = round(best["speed"] + best["sensitivity"], 1)
        sample.status = "experimental"
        sample.reason = None
        sample.support_start_s = best["start"]
        sample.support_end_s = best["end"]
        sample.reprojection_rms_px = best["rms"]
    return samples, details
