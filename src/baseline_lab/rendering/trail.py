"""Display-only reconstruction. Never consumed by measurement or event inference."""

import bisect
import math

from baseline_lab.contracts import DisplayTrailPoint, Observation


def visual_trail(
    observations: list[Observation], width: int, height: int, *, offline: bool = True
) -> list[DisplayTrailPoint]:
    anchors = [
        p for p in observations if p.xy_px is not None and p.state in {"detected", "interpolated"}
    ]
    times = [p.t_s for p in anchors]
    output = []
    for p in observations:
        if p.xy_px is not None and p.state in {"detected", "interpolated"}:
            output.append(
                DisplayTrailPoint(
                    t_s=p.t_s,
                    frame=p.frame,
                    xy_px=p.xy_px,
                    state="observed" if p.state == "detected" else "interpolated",
                )
            )
            continue
        index = bisect.bisect_left(times, p.t_s)
        xy = None
        reason = "no_ball_anchor"
        if 0 < index < len(anchors):
            a, b = anchors[index - 1], anchors[index]
            assert a.xy_px is not None and b.xy_px is not None
            dt = b.t_s - a.t_s
            u = (p.t_s - a.t_s) / dt
            chord = [(b.xy_px[d] - a.xy_px[d]) / dt for d in [0, 1]]
            va, vb = chord.copy(), chord.copy()
            if dt < 1.0:
                if index >= 2 and a.t_s - anchors[index - 2].t_s <= 0.15:
                    prior = anchors[index - 2]
                    assert prior.xy_px is not None
                    va = [(a.xy_px[d] - prior.xy_px[d]) / (a.t_s - prior.t_s) for d in [0, 1]]
                if index + 1 < len(anchors) and anchors[index + 1].t_s - b.t_s <= 0.15:
                    following = anchors[index + 1]
                    assert following.xy_px is not None
                    vb = [
                        (following.xy_px[d] - b.xy_px[d]) / (following.t_s - b.t_s) for d in [0, 1]
                    ]
            point = []
            for d in [0, 1]:
                # Shape-preserving tangent bounds prevent invented overshoot.
                ma = (
                    math.copysign(min(abs(va[d]), 3 * abs(chord[d])), chord[d])
                    if va[d] * chord[d] > 0
                    else 0
                )
                mb = (
                    math.copysign(min(abs(vb[d]), 3 * abs(chord[d])), chord[d])
                    if vb[d] * chord[d] > 0
                    else 0
                )
                point.append(
                    (2 * u**3 - 3 * u * u + 1) * a.xy_px[d]
                    + (u**3 - 2 * u * u + u) * dt * ma
                    + (-2 * u**3 + 3 * u * u) * b.xy_px[d]
                    + (u**3 - u * u) * dt * mb
                )
            xy = tuple(point)
            reason = "display_bridge_between_observations"
        elif anchors and (index > 0 or offline):
            edge = anchors[-1] if index > 0 else anchors[0]
            assert edge.xy_px is not None
            neighbor = (
                anchors[-2]
                if index > 0 and len(anchors) > 1
                else anchors[1]
                if len(anchors) > 1
                else edge
            )
            assert neighbor.xy_px is not None
            dt = edge.t_s - neighbor.t_s
            velocity = [
                (edge.xy_px[d] - neighbor.xy_px[d]) / dt if dt and abs(dt) <= 0.15 else 0
                for d in [0, 1]
            ]
            elapsed = p.t_s - edge.t_s
            damping = math.copysign(0.35 * (1 - math.exp(-abs(elapsed) / 0.35)), elapsed)
            xy = tuple(edge.xy_px[d] + velocity[d] * damping for d in [0, 1])
            reason = (
                "display_prediction_after_last_observation"
                if index
                else "display_backfill_before_first_observation"
            )
        if xy is not None:
            xy = (min(max(0, xy[0]), width - 1), min(max(0, xy[1]), height - 1))
        output.append(
            DisplayTrailPoint(
                t_s=p.t_s,
                frame=p.frame,
                xy_px=xy,
                state="inferred" if xy else "unavailable",
                reason=reason,
            )
        )
    return output
