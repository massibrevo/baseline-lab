"""Time-indexed playback metrics, with explicit physical and evidential limits.

Image speed = norm(delta display pixels)/delta real PTS seconds. It is NOT km/h
or 3D speed. Player speeds use the court-plane approximate footpoint track only.
"""

from baseline_lab.contracts import LiveMetrics, Observation, Result
from baseline_lab.tracking.association import distance


def velocity(observations: list[Observation], frame: int, metric: bool) -> float | None:
    current = observations[frame]
    if current.state != "detected":
        return None
    point = current.court_m if metric else current.xy_px
    if point is None:
        return None
    # A trailing 50–150 ms window suppresses single-frame quantization. Never
    # bridge a missing/interpolated observation, even within this small window.
    for i in range(frame - 1, max(-1, frame - 10), -1):
        previous = observations[i]
        if previous.state != "detected":
            return None
        dt = current.t_s - previous.t_s
        if dt > 0.15:
            return None
        if dt < 0.05:
            continue
        prior = previous.court_m if metric else previous.xy_px
        if prior is None:
            return None
        speed = distance(point, prior) / dt
        return speed if not metric or speed <= 12 else None
    return None


def live_metrics(result: Result) -> list[LiveMetrics]:
    by_id = {track.id: track for track in result.tracks}
    ball = by_id["ball"].observations
    observed = 0
    cumulative = {"near": 0.0, "far": 0.0}
    segments = {"near": 0, "far": 0}
    samples = []
    for i, point in enumerate(ball):
        observed += point.state == "detected"
        speeds = {}
        for identity in ["near", "far"]:
            points = by_id[identity].observations
            speeds[identity] = velocity(points, i, True)
            if i:
                a, b = points[i - 1], points[i]
                dt = b.t_s - a.t_s
                if a.state == b.state == "detected" and a.court_m and b.court_m and 0 < dt <= 0.10:
                    displacement = distance(a.court_m, b.court_m)
                    if displacement / dt <= 12:
                        cumulative[identity] += displacement
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
            )
        )
    return samples
