"""Conservative association and bounded gaps; no invented measurements."""

import math

from baseline_lab.contracts import Observation, Point, Track
from baseline_lab.geometry.court import Court


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class BallAssociation:
    def __init__(self, diagonal: float):
        self.last: list[Observation] = []
        self.diagonal = diagonal

    def update(self, candidates: list[dict], t: float, frame: int) -> Observation:
        recent = self.last[-1] if self.last else None
        if recent and t - recent.t_s > 0.20:
            self.last = []
            recent = None
        prediction = recent.xy_px if recent else None
        if len(self.last) >= 2:
            a, b = self.last[-2:]
            if a.xy_px and b.xy_px and b.t_s > a.t_s:
                dt = (t - b.t_s) / (b.t_s - a.t_s)
                prediction = (
                    b.xy_px[0] + (b.xy_px[0] - a.xy_px[0]) * dt,
                    b.xy_px[1] + (b.xy_px[1] - a.xy_px[1]) * dt,
                )
        chosen = None
        if prediction and recent:
            gate = self.diagonal * (0.02 + 2.0 * (t - recent.t_s))
            viable = [c for c in candidates if distance(c["xy"], prediction) <= gate]
            if viable:
                chosen = min(viable, key=lambda c: distance(c["xy"], prediction))
        elif candidates:
            chosen = candidates[0]
        if not chosen:
            return Observation(t_s=t, frame=frame, state="missing", reason="no_plausible_link")
        obs = Observation(
            t_s=t,
            frame=frame,
            state="detected",
            xy_px=chosen["xy"],
            raw_xy_px=chosen["xy"],
            score=chosen["score"],
        )
        self.last.append(obs)
        self.last = self.last[-2:]
        return obs


def interpolate(points: list[Observation], max_gap_s: float = 0.10) -> None:
    observed = [i for i, p in enumerate(points) if p.state == "detected"]
    for first, last in zip(observed, observed[1:], strict=False):
        a, b = points[first], points[last]
        if last - first <= 1 or last - first > 4 or b.t_s - a.t_s > max_gap_s:
            continue
        assert a.xy_px and b.xy_px
        for i in range(first + 1, last):
            weight = (points[i].t_s - a.t_s) / (b.t_s - a.t_s)
            points[i].xy_px = (
                a.xy_px[0] * (1 - weight) + b.xy_px[0] * weight,
                a.xy_px[1] * (1 - weight) + b.xy_px[1] * weight,
            )
            points[i].state = "interpolated"
            points[i].reason = "linear_gap_at_most_3_frames_and_100ms"


class Players:
    def __init__(self, court: Court | None, width: int, height: int):
        self.court, self.width, self.height = court, width, height
        self.points: dict[str, list[Observation]] = {"near": [], "far": []}

    def update(self, candidates: list[dict], t: float, frame: int) -> None:
        for identity in self.points:
            viable = []
            for c in candidates:
                x1, y1, x2, y2 = c["bbox"]
                foot = ((x1 + x2) / 2, y2)
                metric = self.court.project(foot) if self.court else None
                if metric:
                    x, y = metric
                    if not (-3 < x < 11.23 and -5 < y < 28.77):
                        continue
                    side = "near" if y >= 23.77 / 2 else "far"
                else:
                    if not (0.12 * self.width < foot[0] < 0.88 * self.width):
                        continue
                    if not (0.20 * self.height < foot[1] < 0.98 * self.height):
                        continue
                    side = "near" if foot[1] >= 0.53 * self.height else "far"
                if side == identity:
                    viable.append((c, foot, metric))
            previous = next((p for p in reversed(self.points[identity]) if p.xy_px), None)
            if previous and t - previous.t_s <= 0.3:
                viable = [c for c in viable if self._plausible(previous, c[1], c[2], t)]
            if viable:
                chosen, foot, metric = min(
                    viable,
                    key=lambda c: (
                        distance(c[1], previous.xy_px)
                        if previous and previous.xy_px
                        else -c[0]["score"]
                    ),
                )
                # Light exponential smoothing; output remains explicitly estimated.
                if previous and t - previous.t_s < 0.10 and previous.xy_px:
                    foot = tuple(
                        0.65 * a + 0.35 * b for a, b in zip(foot, previous.xy_px, strict=False)
                    )
                    metric = self.court.project(foot) if self.court else None
                self.points[identity].append(
                    Observation(
                        t_s=t,
                        frame=frame,
                        state="detected",
                        xy_px=foot,
                        raw_xy_px=((chosen["bbox"][0] + chosen["bbox"][2]) / 2, chosen["bbox"][3]),
                        bbox_px=chosen["bbox"],
                        court_m=metric,
                        score=chosen["score"],
                    )
                )
            else:
                self.points[identity].append(
                    Observation(t_s=t, frame=frame, state="missing", reason="no_plausible_player")
                )

    def _plausible(self, previous: Observation, foot: Point, metric: Point | None, t: float):
        if metric and previous.court_m:
            return distance(metric, previous.court_m) <= 12 * (t - previous.t_s) + 0.4
        assert previous.xy_px
        return distance(foot, previous.xy_px) <= self.width * (t - previous.t_s) + 20


def summarize(identity: str, observations: list[Observation], calibrated: bool) -> Track:
    measured = sum(p.state == "detected" for p in observations)
    coverage = measured / len(observations) if observations else 0
    valid_distance = 0.0
    segments = 0
    for a, b in zip(observations, observations[1:], strict=False):
        dt = b.t_s - a.t_s
        if a.court_m and b.court_m and 0 < dt <= 0.10:
            d = distance(a.court_m, b.court_m)
            if d / dt <= 12:
                valid_distance += d
                segments += 1
    reason = None if calibrated and segments else "calibration_or_contiguous_coverage_missing"
    return Track(
        id=identity,
        observations=observations,
        coverage=coverage,
        distance_m=valid_distance if reason is None else None,
        distance_reason=reason,
    )
