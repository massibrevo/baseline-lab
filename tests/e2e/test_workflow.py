"""Bounded real API → durable worker → playable artifacts → review → deletion."""

from pathlib import Path

import av
import pytest
from conftest import make_clip
from fastapi.testclient import TestClient

from baseline_lab.api.app import create_app
from baseline_lab.jobs.worker import run_worker
from baseline_lab.pipeline import analyze
from baseline_lab.storage.database import Database
from baseline_lab.video.media import inspect

H = {"X-Baseline-Request": "1"}


def test_complete_workflow(config, clip):
    c = TestClient(create_app(config))
    c.get("/api/v1/identity")
    with clip.open("rb") as f:
        s = c.post("/api/v1/sessions", headers=H, files={"file": ("rally.mp4", f, "video/mp4")})
    assert s.status_code == 201, s.text
    sid = s.json()["id"]
    db = Database(config.data_dir)
    db.beat()
    j = c.post(
        f"/api/v1/sessions/{sid}/jobs",
        headers=H,
        json={
            "idempotency_key": "smoke",
            "calibration": {"corners_px": [[210, 90], [430, 90], [550, 430], [90, 430]]},
        },
    )
    assert j.status_code == 202, j.text
    jid = j.json()["id"]
    # Child reads centralized settings through environment, just like deployment.
    import os
    from unittest.mock import patch

    with patch.dict(
        os.environ, {"BASELINE_DATA_DIR": str(config.data_dir), "BASELINE_PLAYER_MODEL": "hog"}
    ):
        run_worker(config, once=True)
    status = c.get(f"/api/v1/sessions/{sid}/jobs/{jid}").json()
    assert status["state"] == "completed_with_warnings", status
    base = f"/api/v1/sessions/{sid}/jobs/{jid}"
    result = c.get(base + "/artifacts/result.json").json()
    assert result["ball_speed_m_s"] is None
    assert len(result["live_metrics"]) == result["metadata"]["frames"]
    assert result["schema_version"] == "1.4"
    assert c.get(base + "/result").json()["live_metrics"] == result["live_metrics"]
    for artifact in [
        "annotated.mp4",
        "review.mp4",
        "tracks.csv",
        "events.csv",
        "live-metrics.csv",
        "ball-speed.csv",
        "visual-trail.csv",
        "shot-quality.csv",
    ]:
        response = c.get(base + "/artifacts/" + artifact)
        assert response.status_code == 200 and len(response.content) > 10
    video = config.data_dir / "sessions" / sid / jid / "annotated.mp4"
    with av.open(str(video)) as container:
        assert container.streams.video[0].codec_context.name == "h264"
        frames = list(container.decode(video=0))
        assert len(frames) == 160
        assert float(frames[-1].time) == pytest.approx(159 / 30, abs=1 / 90000)
    correction = {
        "status": "manual",
        "t_s": 1.0,
        "xy_px": [320, 300],
        "notes": "Reviewed test contact",
        "shot_evidence": {
            "speed_kmh": 120,
            "spin_rpm": 2400,
            "bounce_angle_deg": 45,
            "source": "test sensor fixture",
        },
    }
    corrected = c.post(base + "/events", headers=H, json=correction)
    assert corrected.status_code == 200, corrected.text
    event = corrected.json()["events"][-1]
    assert event["court_m"] is not None and event["classification"] == "uncertain"
    assert event["shot_quality"]["status"] == "complete"
    assert event["shot_quality"]["available_factors"] == 5
    assert event["shot_quality"]["error_attribution"] == "undetermined"
    assert event["shot_depth_m"] == pytest.approx(
        min(event["court_m"][1], 23.77 - event["court_m"][1])
    )
    rejected = c.patch(
        base + "/events/" + event["id"], headers=H, json={**correction, "status": "rejected"}
    )
    assert rejected.status_code == 200
    event = rejected.json()["events"][-1]
    assert event["original_prediction"]["status"] == "manual"
    assert event["court_m"] is None and event["shot_depth_m"] is None
    assert event["shot_quality"]["score"] is None
    # Review a candidate stroke without overwriting its original prediction.
    import json

    result_file = config.data_dir / "sessions" / sid / jid / "result.json"
    saved = json.loads(result_file.read_text())
    saved["shots"] = [
        {
            "id": "shot-test",
            "t_s": 1,
            "frame": 30,
            "player": "near",
            "kind": "unknown",
            "evidence_score": 0.5,
        }
    ]
    saved["automatic_shots"] = [
        {
            "id": "auto-fixture",
            "t_s": 3.12,
            "frame": 94,
            "player": "near",
            "kind": "backhand",
            "evidence_score": 0.83,
            "method": "neural test fixture",
            "recognition_basis": "neural_impact_candidate",
        }
    ]
    result_file.write_text(json.dumps(saved))
    auto_review = c.patch(base + "/shots/auto-fixture", headers=H, json={"kind": "forehand"})
    assert auto_review.status_code == 200
    assert auto_review.json()["automatic_shots"][0]["kind"] == "backhand"
    copied = next(s for s in auto_review.json()["shots"] if s["id"] == "auto-fixture")
    assert copied["status"] == "reviewed" and copied["kind"] == "forehand"
    assert copied["original_prediction"]["kind"] == "backhand"
    other = TestClient(create_app(config))
    other.get("/api/v1/identity")
    assert (
        other.patch(base + "/shots/shot-test", headers=H, json={"kind": "forehand"}).status_code
        == 404
    )
    reviewed = c.patch(base + "/shots/shot-test", headers=H, json={"kind": "forehand"})
    assert reviewed.status_code == 200
    stroke = reviewed.json()["shots"][0]
    assert stroke["kind"] == "forehand" and stroke["status"] == "reviewed"
    assert stroke["original_prediction"]["kind"] == "unknown"
    assert (
        c.patch(base + "/shots/shot-test", headers=H, json={"kind": "fabricated"}).status_code
        == 422
    )
    assert (
        other.post(
            base + "/shots", headers=H, json={"t_s": 2.1, "player": "near", "kind": "forehand"}
        ).status_code
        == 404
    )
    annotated = c.post(
        base + "/shots", headers=H, json={"t_s": 2.1, "player": "near", "kind": "forehand"}
    )
    assert annotated.status_code == 200
    stroke = next(s for s in annotated.json()["shots"] if s["id"].startswith("manual-stroke-"))
    assert stroke["t_s"] == 2.1 and stroke["kind"] == "forehand" and stroke["status"] == "reviewed"
    assert stroke["frame"] == 63
    assert (
        c.post(
            base + "/shots", headers=H, json={"t_s": 100, "player": "near", "kind": "backhand"}
        ).status_code
        == 422
    )
    assert c.delete("/api/v1/sessions/" + sid, headers=H).status_code == 204
    assert not (config.data_dir / "sessions" / sid).exists()
    assert c.get(base + "/artifacts/annotated.mp4").status_code == 404


def test_vfr_roundtrip_export_and_camera_cut(tmp_path, config):
    source = make_clip(tmp_path / "vfr.mp4", variable=True)
    result = analyze(source, tmp_path / "output", config)
    encoded = inspect(tmp_path / "output" / "annotated.mp4", config)
    assert encoded.variable_frame_rate
    assert encoded.timestamps_s == pytest.approx(result.metadata.timestamps_s, abs=1 / 90000)
    source = make_clip(tmp_path / "cut.mp4", cut=True)
    with pytest.raises(ValueError, match="camera_cut"):
        analyze(source, tmp_path / "cut-output", config)


def test_active_cancellation_cleans_partial_output(config, clip):
    import os
    import threading
    import time
    from unittest.mock import patch

    c = TestClient(create_app(config))
    c.get("/api/v1/identity")
    sid = c.post(
        "/api/v1/sessions", headers=H, files={"file": ("cancel.mp4", clip.read_bytes())}
    ).json()["id"]
    db = Database(config.data_dir)
    db.beat()
    jid = c.post(
        f"/api/v1/sessions/{sid}/jobs", headers=H, json={"idempotency_key": "cancel"}
    ).json()["id"]
    with patch.dict(
        os.environ, {"BASELINE_DATA_DIR": str(config.data_dir), "BASELINE_PLAYER_MODEL": "hog"}
    ):
        worker = threading.Thread(target=run_worker, args=(config, True))
        worker.start()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            job = c.get(f"/api/v1/sessions/{sid}/jobs/{jid}").json()
            if job["state"] == "running":
                break
            time.sleep(0.02)
        assert job["state"] == "running"
        c.post(f"/api/v1/sessions/{sid}/jobs/{jid}/cancel", headers=H)
        worker.join(15)
    assert not worker.is_alive()
    job = c.get(f"/api/v1/sessions/{sid}/jobs/{jid}").json()
    assert job["state"] == "cancelled"
    assert not list((config.data_dir / "sessions" / sid).glob(".*.partial"))
    assert not (config.data_dir / "sessions" / sid / jid).exists()


def test_worker_timeout_is_reported(config, clip):
    import os
    from unittest.mock import patch

    config.job_timeout_s = 0
    c = TestClient(create_app(config))
    c.get("/api/v1/identity")
    sid = c.post(
        "/api/v1/sessions", headers=H, files={"file": ("timeout.mp4", clip.read_bytes())}
    ).json()["id"]
    Database(config.data_dir).beat()
    jid = c.post(
        f"/api/v1/sessions/{sid}/jobs", headers=H, json={"idempotency_key": "timeout"}
    ).json()["id"]
    with patch.dict(
        os.environ, {"BASELINE_DATA_DIR": str(config.data_dir), "BASELINE_PLAYER_MODEL": "hog"}
    ):
        run_worker(config, once=True)
    job = c.get(f"/api/v1/sessions/{sid}/jobs/{jid}").json()
    assert job["state"] == "failed" and job["error"] == "job_timeout"
    assert not (config.data_dir / "sessions" / sid / jid).exists()


@pytest.mark.model
def test_real_weights_and_real_clip(tmp_path):
    from baseline_lab.settings import Settings

    cfg = Settings()
    from baseline_lab.video.media import frames
    from baseline_lab.vision.detectors import SSDPlayers

    detector = SSDPlayers(cfg.data_dir / "models")
    image = next(frames(Path("examples/commons-stabilized.mp4")))[-1]
    people = detector.detect(image)
    assert people and all(p["score"] >= 0.35 for p in people)
    assert any(100 < p["bbox"][0] < 250 and p["bbox"][3] > 550 for p in people)
    from baseline_lab.contracts import Observation
    from baseline_lab.vision.pose import PoseEstimator

    near = max(people, key=lambda p: p["score"])
    point = Observation(t_s=0, frame=0, state="detected", bbox_px=near["bbox"])
    PoseEstimator(cfg.data_dir / "models").estimate(image, point, None)
    assert point.pose_score > 0.5
    assert len([p for p in point.keypoints if p.xy_px]) >= 10
