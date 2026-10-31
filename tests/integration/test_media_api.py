import pytest
from conftest import make_clip
from fastapi.testclient import TestClient

from baseline_lab.api.app import create_app
from baseline_lab.storage.database import Database
from baseline_lab.video.media import inspect

HEAD = {"X-Baseline-Request": "1"}


def client(config):
    c = TestClient(create_app(config))
    assert c.get("/api/v1/identity").status_code == 200
    return c


def test_vfr_timestamps_preserved_and_limits(tmp_path, config):
    path = make_clip(tmp_path / "vfr.mp4", variable=True)
    m = inspect(path, config)
    assert m.variable_frame_rate
    assert m.timestamps_s[5] - m.timestamps_s[4] == pytest.approx(0.05, abs=0.001)
    assert len(m.original_pts) == m.frames
    config.max_pixels = 100
    with pytest.raises(ValueError, match="pixel_limit"):
        inspect(path, config)


def test_api_upload_ownership_limits_and_delete(config, clip):
    a, b = client(config), client(config)
    assert (
        a.post("/api/v1/sessions", files={"file": ("rally.mp4", clip.read_bytes())}).status_code
        == 403
    )
    s = a.post("/api/v1/sessions", headers=HEAD, files={"file": ("rally.mp4", clip.read_bytes())})
    assert s.status_code == 201, s.text
    sid = s.json()["id"]
    assert b.get("/api/v1/sessions/" + sid).status_code == 404
    assert b.get("/api/v1/sessions/" + sid + "/source").status_code == 404
    assert b.delete("/api/v1/sessions/" + sid, headers=HEAD).status_code == 404
    assert (
        a.post(
            "/api/v1/sessions", headers=HEAD, files={"file": ("bad.mp4", b"not video")}
        ).status_code
        == 422
    )
    assert (
        a.post("/api/v1/sessions", headers=HEAD, files={"file": ("bad.txt", b"x")}).status_code
        == 415
    )
    assert (
        a.post(
            "/api/v1/sessions/" + sid + "/jobs", headers=HEAD, json={"idempotency_key": "a"}
        ).status_code
        == 503
    )
    assert a.delete("/api/v1/sessions/" + sid, headers=HEAD).status_code == 204
    assert not (config.data_dir / "sessions" / sid).exists()
    assert a.get("/api/v1/sessions/" + sid).status_code == 404


def test_idempotency_atomic_claim_restart_and_cancel(config, clip):
    a = client(config)
    sid = a.post(
        "/api/v1/sessions", headers=HEAD, files={"file": ("rally.mp4", clip.read_bytes())}
    ).json()["id"]
    db = Database(config.data_dir)
    db.beat()
    first = a.post(
        f"/api/v1/sessions/{sid}/jobs", headers=HEAD, json={"idempotency_key": "first"}
    ).json()
    second = a.post(
        f"/api/v1/sessions/{sid}/jobs", headers=HEAD, json={"idempotency_key": "first"}
    ).json()
    assert first["id"] == second["id"]
    job = db.claim()
    assert job["id"] == first["id"]
    assert db.claim() is None
    db.recover()
    assert db.claim()["id"] == first["id"]
    j = a.post(f"/api/v1/sessions/{sid}/jobs/{first['id']}/cancel", headers=HEAD)
    assert j.json()["cancel"] == 1
    db.recover()
    assert a.get(f"/api/v1/sessions/{sid}/jobs/{first['id']}").json()["state"] == "cancelled"


def test_mov_display_rotation(config, clip, tmp_path):
    target = make_clip(tmp_path / "rotated.mov", rotation=90)
    original = inspect(clip, config)
    rotated = inspect(target, config)
    assert rotated.rotation_deg in (90, 270)
    assert (rotated.width, rotated.height) == (original.height, original.width)
    assert rotated.timestamps_s == pytest.approx(original.timestamps_s, abs=1 / 90000)
