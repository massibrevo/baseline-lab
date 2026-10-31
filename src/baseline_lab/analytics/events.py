"""Candidates demand temporal support; the user establishes ground contact."""

from baseline_lab.contracts import Event, Observation
from baseline_lab.tracking.association import distance


def candidates(ball: list[Observation], players: list[list[Observation]]) -> list[Event]:
    result: list[Event] = []
    for i in range(3, len(ball) - 3):
        window = ball[i - 3 : i + 4]
        if any(p.state != "detected" or p.xy_px is None for p in window):
            continue
        a, p, b = ball[i - 2], ball[i], ball[i + 2]
        assert a.xy_px and p.xy_px and b.xy_px
        if b.t_s - a.t_s > 0.25:
            continue
        # Strong downward then upward change, stable horizontal continuity.
        before, after = p.xy_px[1] - a.xy_px[1], b.xy_px[1] - p.xy_px[1]
        if before < 3 or after > -3:
            continue
        if abs(b.xy_px[0] - a.xy_px[0]) > 160:
            continue
        # Nearby person could be a hit; trajectory reversal alone is insufficient.
        footpoints = [ps[i].xy_px for ps in players]
        near_player = any(foot and distance(foot, p.xy_px) < 65 for foot in footpoints)
        if near_player or (result and p.t_s - result[-1].t_s < 0.35):
            continue
        result.append(
            Event(
                id=f"candidate-{i}",
                t_s=p.t_s,
                frame=i,
                xy_px=p.xy_px,
                evidence_score=min(0.75, 0.3 + min(before, -after) / 60),
                notes="Temporal reversal with 7 observed frames; ground contact unproven",
            )
        )
    return result
