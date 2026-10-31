"""Conservative association and bounded gaps; no invented measurements."""

import math

from baseline_lab.contracts import Observation, Point, Track
from baseline_lab.geometry.court import Court


def distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class BallAssociation:
    """Online beam of short tracklets; detections are never replaced by predictions.

    New tracks need three temporally coherent samples. Competing moving blobs
    remain hypotheses, with velocity residuals and stationary clutter penalties.
    Real timestamp gaps expire hypotheses rather than inventing camera samples.
    """

    def __init__(self, diagonal: float, neural: bool = False):
        self.diagonal = diagonal
        self.minimum_travel = 0.003 if neural else 0.012
        self.minimum_quality = 0.45 if neural else 0.6
        self.hypotheses: list[tuple[list[Observation], float]] = []
        self.emitted: Observation | None = None

    def update(self, candidates: list[dict], t: float, frame: int) -> Observation:
        next_paths: list[tuple[list[Observation], float]] = []
        for candidate in candidates[:40]:
            point = Observation(
                t_s=t,
                frame=frame,
                state="detected",
                xy_px=candidate["xy"],
                raw_xy_px=candidate["xy"],
                apparent_diameter_px=candidate.get("diameter_px"),
                score=candidate["score"],
            )
            next_paths.append(([point], candidate["score"] - 0.8))
            for history, quality in self.hypotheses:
                last = history[-1]
                dt = t - last.t_s
                if not 0 < dt <= 0.10 or last.xy_px is None:
                    continue
                step = distance(last.xy_px, candidate["xy"])
                if step > self.diagonal * (0.006 + 3 * dt):
                    continue
                prediction = last.xy_px
                if len(history) >= 2:
                    before = history[-2]
                    assert before.xy_px is not None
                    ratio = dt / (last.t_s - before.t_s)
                    prediction = (
                        last.xy_px[0] + (last.xy_px[0] - before.xy_px[0]) * ratio,
                        last.xy_px[1] + (last.xy_px[1] - before.xy_px[1]) * ratio,
                    )
                gate = self.diagonal * (0.008 + 0.8 * dt)
                residual = distance(prediction, candidate["xy"])
                if len(history) >= 2 and residual > gate:
                    continue
                path = (history + [point])[-5:]
                assert path[0].xy_px is not None
                travel = distance(path[0].xy_px, candidate["xy"])
                stationary = len(path) >= 3 and travel < self.diagonal * self.minimum_travel
                score = 0.8 * quality + 0.2 * candidate["score"] + 0.18 - 0.35 * residual / gate
                if stationary:
                    score -= 1.0
                next_paths.append((path, score))
        # Retain one best history per endpoint and starting frame, plus fresh starts.
        ranked = sorted(next_paths, key=lambda item: item[1], reverse=True)
        kept: list[tuple[list[Observation], float]] = []
        signatures = set()
        for path, quality in ranked:
            signature = (path[-1].xy_px, min(3, len(path)))
            if signature not in signatures:
                signatures.add(signature)
                kept.append((path, quality))
            if len(kept) >= 32:
                break
        if not candidates:
            kept = [(p, q - 0.2) for p, q in self.hypotheses if t - p[-1].t_s < 0.10]
        self.hypotheses = kept
        viable = [
            (p, q)
            for p, q in kept
            if len(p) >= 3 and p[-1].frame == frame and q > self.minimum_quality
        ]
        if self.emitted and self.emitted.xy_px and t - self.emitted.t_s <= 0.10:
            viable = [
                (p, q)
                for p, q in viable
                if p[-1].xy_px
                and distance(p[-1].xy_px, self.emitted.xy_px)
                <= self.diagonal * (0.01 + 3 * (t - self.emitted.t_s))
            ]
        if viable:
            self.emitted = viable[0][0][-1]
            return self.emitted
        return Observation(t_s=t, frame=frame, state="missing", reason="unconfirmed_temporal_track")


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
                    if not (-1.5 < x < 9.73 and -5 < y < 28.77):
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
                        distance(c[1], previous.raw_xy_px or previous.xy_px)
                        if previous and previous.xy_px
                        else -c[0]["score"] + (0.12 * abs(c[2][0] - 4.115) if c[2] else 0)
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
                        reason=chosen.get("source", "person_detector"),
                    )
                )
            else:
                self.points[identity].append(
                    Observation(t_s=t, frame=frame, state="missing", reason="no_plausible_player")
                )

    def _plausible(self, previous: Observation, foot: Point, metric: Point | None, t: float):
        # Compare like-for-like detector box footpoints for association. Pose
        # ankles refine output geometry afterwards and have a different bias.
        prior_metric = (
            self.court.project(previous.raw_xy_px)
            if self.court and previous.raw_xy_px
            else previous.court_m
        )
        if metric and prior_metric:
            return distance(metric, prior_metric) <= 12 * (t - previous.t_s) + 0.6
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
