"""Time-indexed playback metrics, with explicit physical and evidential limits.

Image speed = norm(delta display pixels)/delta real PTS seconds. It is NOT km/h
or 3D speed. Player speeds use the court-plane approximate footpoint track only.
"""

import math

import numpy as np

from baseline_lab.contracts import LiveMetrics, Observation, Result
from baseline_lab.tracking.association import distance


def knee_angle(observation: Observation, side: str) -> float | None:
    joints = {p.name: p.xy_px for p in observation.keypoints if p.xy_px and p.state == "detected"}
    names = [f"{side}_{joint}" for joint in ["hip", "knee", "ankle"]]
    if not all(name in joints for name in names):
        return None
    a, b, c = (np.asarray(joints[name], dtype=float) for name in names)
    u, v = a - b, c - b
    denominator = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denominator < 1:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denominator, -1, 1))))


def velocity(observations: list[Observation], frame: int, metric: bool) -> float | None:
    current = observations[frame]
    if current.state != "detected":
        return None
    point = current.court_m if metric else current.xy_px
    if point is None:
        return None
    if metric:
        points = []
        previous_t = current.t_s
        for p in reversed(observations[max(0, frame - 35) : frame + 1]):
            if current.t_s - p.t_s > 0.5:
                break
            if p.state == "detected" and p.court_m:
                if previous_t - p.t_s > 0.10:
                    break
                points.append(p)
                previous_t = p.t_s
        if len(points) < 3 or current.t_s - points[-1].t_s < 0.05:
            return None
        times = np.array([p.t_s - current.t_s for p in points])
        positions = np.array([p.court_m for p in points], float)
        design = np.column_stack([times, np.ones(len(times))])
        fit, *_ = np.linalg.lstsq(design, positions, rcond=None)
        residual = np.linalg.norm(design @ fit - positions, axis=1)
        keep = residual <= max(0.12, float(np.median(residual)) * 2.5)
        if keep.sum() >= 3:
            fit, *_ = np.linalg.lstsq(design[keep], positions[keep], rcond=None)
        speed = float(np.linalg.norm(fit[0]))
        return speed if speed <= 12 else None
    window = [current]
    # A trailing 50–150 ms window suppresses single-frame quantization. Never
    # bridge a missing/interpolated observation, even within this small window.
    for i in range(frame - 1, max(-1, frame - 10), -1):
        previous = observations[i]
        if previous.state != "detected":
            break
        dt = current.t_s - previous.t_s
        if dt > 0.15:
            break
        window.append(previous)
        if dt < 0.05:
            continue
        prior = previous.court_m if metric else previous.xy_px
        if prior is None:
            return None
        if metric:
            speed = distance(point, prior) / dt
            return speed if speed <= 12 else None
    if metric or len(window) < 3 or current.t_s - window[-1].t_s < 0.05:
        return None
    times = np.array([p.t_s - current.t_s for p in window])
    positions = np.array([p.xy_px for p in window], dtype=float)
    design = np.column_stack([times, np.ones(len(times))])
    fit, *_ = np.linalg.lstsq(design, positions, rcond=None)
    speed = float(np.linalg.norm(fit[0]))
    rms = float(np.sqrt(np.mean(np.sum((design @ fit - positions) ** 2, axis=1))))
    # Reject switches/outliers and sharp turns. Use observed positions only;
    # fitting does not manufacture a position or bridge missing observations.
    if rms > max(2.0, speed * (current.t_s - window[-1].t_s) * 0.12):
        return None
    return speed


def live_metrics(result: Result) -> list[LiveMetrics]:
    by_id = {track.id: track for track in result.tracks}
    ball = by_id["ball"].observations
    observed = 0
    cumulative = {"near": 0.0, "far": 0.0}
    segments = {"near": 0, "far": 0}
    samples: list[LiveMetrics] = []
    filtered_speeds: dict[str, tuple[float, float]] = {}
    for i, point in enumerate(ball):
        observed += point.state == "detected"
        speeds = {}
        for identity in ["near", "far"]:
            points = by_id[identity].observations
            raw_speed = velocity(points, i, True)
            if raw_speed is not None:
                last = filtered_speeds.get(identity)
                if last and 0 < point.t_s - last[0] <= 0.15:
                    dt = point.t_s - last[0]
                    alpha = 1 - math.exp(-dt / 0.25)
                    raw_speed = last[1] + max(-4 * dt, min(4 * dt, alpha * (raw_speed - last[1])))
                filtered_speeds[identity] = (point.t_s, raw_speed)
            speeds[identity] = raw_speed
            if i:
                a, b = points[i - 1], points[i]
                dt = b.t_s - a.t_s
                if a.state == b.state == "detected" and a.court_m and b.court_m and 0 < dt <= 0.10:
                    prior_speed = (
                        getattr(samples[-1], f"{identity}_speed_m_s", None) if samples else None
                    )
                    if raw_speed is not None and prior_speed is not None:
                        cumulative[identity] += (prior_speed + raw_speed) * 0.5 * dt
                        segments[identity] += 1
        image_speed = velocity(ball, i, False)
        samples.append(
            LiveMetrics(
                t_s=point.t_s,
                frame=i,
                ball_image_speed_px_s=image_speed,
                ball_image_speed_reason=None
                if image_speed is not None
                else "insufficient_contiguous_observations",
                near_speed_m_s=speeds["near"],
                far_speed_m_s=speeds["far"],
                near_observed_distance_m=cumulative["near"] if segments["near"] else None,
                far_observed_distance_m=cumulative["far"] if segments["far"] else None,
                ball_observation_coverage_so_far=observed / (i + 1),
                contact_candidates_so_far=sum(
                    e.t_s <= point.t_s and e.status != "rejected" for e in result.events
                ),
                near_visible_joints=sum(
                    p.xy_px is not None for p in by_id["near"].observations[i].keypoints
                ),
                far_visible_joints=sum(
                    p.xy_px is not None for p in by_id["far"].observations[i].keypoints
                ),
                **{
                    f"{identity}_{side}_knee_angle_deg_2d": knee_angle(
                        by_id[identity].observations[i], side
                    )
                    for identity in ["near", "far"]
                    for side in ["left", "right"]
                },
            )
        )
    result.provenance.setdefault(
        "raw_footpoint_path_m", {t.id: t.distance_m for t in result.tracks if t.id != "ball"}
    )
    result.provenance["player_path_method"] = (
        "integral of robust smoothed ground speed over adjacent observed links; "
        "excludes warmup and missing geometry/observations; approximate"
    )
    for track in result.tracks:
        if track.id in cumulative:
            track.distance_m = cumulative[track.id] if segments[track.id] else None
            track.distance_reason = (
                None if segments[track.id] else "insufficient_contiguous_velocity_support"
            )
    return samples
