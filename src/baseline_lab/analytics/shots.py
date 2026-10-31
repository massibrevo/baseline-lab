"""Candidate impacts from ball/wrist proximity. Stroke type requires review."""

import math

from baseline_lab.contracts import Result, ShotSignal


def shot_signals(result: Result) -> list[ShotSignal]:
    tracks = {t.id: t for t in result.tracks}
    output: list[ShotSignal] = []
    for i, ball in enumerate(tracks["ball"].observations):
        if ball.state != "detected" or not ball.xy_px:
            continue
        if output and ball.t_s - output[-1].t_s < 0.55:
            continue
        options = []
        for identity in ["near", "far"]:
            player = tracks[identity].observations[i]
            if not player.bbox_px:
                continue
            joints = {
                p.name: p.xy_px for p in player.keypoints if p.xy_px and p.state == "detected"
            }
            names = ["right_wrist", "right_shoulder", "left_shoulder", "right_hip", "left_hip"]
            if not all(name in joints for name in names):
                continue
            wrist = joints["right_wrist"]
            height = player.bbox_px[3] - player.bbox_px[1]
            proximity = math.dist(ball.xy_px, wrist)
            if proximity > max(20, height * 0.30):
                continue
            # A single 2D pose cannot reliably determine stroke type or handedness.
            kind = "unknown"
            options.append((proximity / max(height, 1), identity, kind))
        if options:
            relative, identity, kind = min(options)
            sample = result.live_metrics[i] if result.live_metrics else None
            output.append(
                ShotSignal.model_validate(
                    {
                        "id": f"shot-{i}",
                        "t_s": ball.t_s,
                        "frame": i,
                        "player": identity,
                        "kind": kind,
                        "image_speed_px_s": sample.ball_image_speed_px_s if sample else None,
                        "evidence_score": max(0, 1 - relative / 0.3),
                    }
                )
            )
    return output
