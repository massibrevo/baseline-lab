"""Real camera transport/worker smoke using timestamped JPEGs from a known fixture.

This verifies the streaming protocol, not a physical webcam permission or lens.
"""

import base64
import os
import threading
import time
from pathlib import Path
from unittest.mock import patch

import av
import cv2
import pytest
from fastapi.testclient import TestClient

from baseline_lab.api.app import create_app
from baseline_lab.jobs.worker import run_worker
from baseline_lab.storage.database import Database

HEAD = {"X-Baseline-Request": "1"}


def test_camera_bursts_ownership_review_recording_and_stop(config, clip):
    client = TestClient(create_app(config))
    client.get("/api/v1/identity")
    other = TestClient(create_app(config))
    other.get("/api/v1/identity")
    db = Database(config.data_dir)
    db.beat()
    with av.open(str(clip)) as source:
        width = source.streams.video[0].codec_context.width
        height = source.streams.video[0].codec_context.height
    corners = [[210, 90], [430, 90], [550, 430], [90, 430]]
    if width == 1280:
        import json

        corners = json.loads(Path("examples/commons-calibration.json").read_text())["corners_px"]
    with av.open(str(clip)) as input_clip:
        first_image = next(input_clip.decode(video=0)).to_ndarray(format="bgr24")
    reference = base64.b64encode(cv2.imencode(".jpg", first_image)[1]).decode()
    created = client.post(
        "/api/v1/camera",
        headers=HEAD,
        json={
            "width": width,
            "height": height,
            "suitability_confirmed": True,
            "reference_jpeg_base64": reference,
            "calibration": {"corners_px": corners},
        },
    )
    assert created.status_code == 201, created.text
    sid, jid = created.json()["id"], created.json()["job_id"]
    assert other.get(f"/api/v1/camera/{sid}/state").status_code == 404
    assert other.post(f"/api/v1/camera/{sid}/stop", headers=HEAD, json={}).status_code == 404
    with av.open(str(clip)) as container:
        images = [
            f.to_ndarray(format="bgr24") for i, f in enumerate(container.decode(video=0)) if i < 16
        ]

    def batch(sequence, first):
        return {
            "sequence": sequence,
            "frames": [
                {
                    "t_s": i / 30,
                    "jpeg_base64": base64.b64encode(cv2.imencode(".jpg", images[i])[1]).decode(),
                }
                for i in range(first, first + 8)
            ],
        }

    assert (
        client.post(f"/api/v1/camera/{sid}/frames", headers=HEAD, json=batch(0, 0)).status_code
        == 202
    )
    with patch.dict(
        os.environ,
        {"BASELINE_DATA_DIR": str(config.data_dir), "BASELINE_PLAYER_MODEL": config.player_model},
    ):
        thread = threading.Thread(target=run_worker, args=(config, True))
        thread.start()
        deadline = time.monotonic() + 20
        sample = None
        while time.monotonic() < deadline:
            state = client.get(f"/api/v1/camera/{sid}/state").json()
            sample = state["sample"]
            if sample:
                break
            time.sleep(0.05)
        assert sample is not None, state
        assert client.get(f"/api/v1/camera/{sid}/frame?sequence=0").status_code == 200
        assert other.get(f"/api/v1/camera/{sid}/frame?sequence=0").status_code == 404
        reviewed = client.post(
            f"/api/v1/camera/{sid}/contacts",
            headers=HEAD,
            json={
                "status": "manual",
                "t_s": 0.2,
                "xy_px": [320, 300],
                "notes": "Synthetic contact review",
            },
        )
        assert reviewed.status_code == 201, reviewed.text
        client.post(f"/api/v1/camera/{sid}/frames", headers=HEAD, json=batch(1, 8))
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = client.get(f"/api/v1/camera/{sid}/state").json()
            if state["sample"] and state["sample"]["sequence"] == 1:
                break
            time.sleep(0.05)
        assert state["sample"]["sequence"] == 1, state
        assert state["sample"]["events"][-1]["shot_depth_m"] is not None
        client.post(f"/api/v1/camera/{sid}/stop", headers=HEAD, json={})
        thread.join(20)
    assert not thread.is_alive()
    job = client.get(f"/api/v1/sessions/{sid}/jobs/{jid}").json()
    assert job["state"] == "completed_with_warnings", job
    base = f"/api/v1/sessions/{sid}/jobs/{jid}/artifacts/"
    for name in ["result.json", "live-metrics.csv", "annotated.mp4", "review.mp4"]:
        assert client.get(base + name).status_code == 200
    result = client.get(base + "result.json").json()
    assert len(result["live_metrics"]) == 16
    with av.open(str(config.data_dir / "sessions" / sid / jid / "review.mp4")) as c:
        frames = list(c.decode(video=0))
        assert len(frames) == 16
        assert abs(float(frames[-1].time) - 0.5) < 1 / 90000
    client.delete(f"/api/v1/sessions/{sid}", headers=HEAD)
    assert not (config.data_dir / "sessions" / sid).exists()


@pytest.mark.model
def test_real_tennis_camera_transport(config):
    from baseline_lab.settings import settings
    from baseline_lab.vision.detectors import MODEL_FILE

    config.player_model = "ssdlite"
    target = config.data_dir / "models"
    target.mkdir(parents=True)
    os.link(settings().data_dir / "models" / MODEL_FILE, target / MODEL_FILE)
    test_camera_bursts_ownership_review_recording_and_stop(
        config, Path("examples/commons-stabilized.mp4")
    )
