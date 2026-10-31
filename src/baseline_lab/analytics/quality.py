"""Transparent shot-pressure index. It cannot label an error forced/unforced.

Placement comes only from reviewed grounded contacts. Physical speed, spin and
outgoing vertical bounce angle require independently measured, attributed input.
No image-plane velocity/angle or missing value is substituted for that evidence.
The profile favors deep, wide, fast, high-spin, steep-bouncing groundstrokes.
It is not appropriate as a universal rating of serves, dropshots or slices.
"""

from baseline_lab.contracts import Event, QualityFactor, Result, ShotQuality

WEIGHTS = {"speed": 0.30, "spin": 0.15, "depth": 0.25, "width": 0.20, "bounce_angle": 0.10}


def clamp(value: float) -> float:
    return min(10.0, max(0.0, value))


def quality(event: Event) -> ShotQuality:
    evidence = event.shot_evidence
    grounded = event.status in {"manual", "confirmed"} and event.court_m is not None
    x, y = event.court_m if grounded and event.court_m else (0.0, 0.0)
    depth = min(y, 23.77 - y) if grounded else None
    width = abs(x - 8.23 / 2) if grounded else None
    in_bounds = grounded and 0 <= x <= 8.23 and 0 <= y <= 23.77
    event.shot_depth_m, event.shot_width_m = depth, width
    measurements = {
        "speed": (evidence.speed_kmh if evidence else None, "km/h", lambda v: clamp((v - 40) / 14)),
        "spin": (evidence.spin_rpm if evidence else None, "rpm", lambda v: clamp(v / 400)),
        "depth": (
            depth,
            "m inside baseline",
            lambda v: clamp(10 * (1 - v / 11.885)) if in_bounds else 0.0,
        ),
        "width": (width, "m from center", lambda v: clamp(10 * v / 4.115) if in_bounds else 0.0),
        "bounce_angle": (
            evidence.bounce_angle_deg if evidence else None,
            "degrees above court",
            lambda v: clamp(v / 6),
        ),
    }
    factors = {}
    for name, (value, unit, normalize) in measurements.items():
        source = (
            "reviewed ground contact + manual calibration"
            if name in {"depth", "width"}
            else (evidence.source if evidence else "unavailable: independent measurement required")
        )
        factors[name] = QualityFactor(
            value=value,
            unit=unit,
            score=round(normalize(value), 3) if value is not None and grounded else None,
            weight=WEIGHTS[name],
            source=source,
        )
    available = [f for f in factors.values() if f.score is not None]
    coverage = sum(f.weight for f in available)
    # A candidate or a single factor never gets a convincing-looking score.
    score = (
        sum(f.score * f.weight for f in available if f.score is not None) / coverage
        if len(available) >= 2
        else None
    )
    return ShotQuality(
        score=round(score, 2) if score is not None else None,
        available_factors=len(available),
        evidence_coverage=round(coverage, 3),
        status="complete"
        if len(available) == 5
        else "partial"
        if score is not None
        else "unavailable",
        factors=factors,
    )


def update_quality(result: Result) -> None:
    for event in result.events:
        event.shot_quality = quality(event)
