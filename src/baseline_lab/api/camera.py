"""Bounded local camera transport. Requests never run vision inference."""

import base64
import binascii
import json
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse

from baseline_lab.api.auth import owner
from baseline_lab.contracts import CameraBatch, Correction, LiveSessionInput
from baseline_lab.geometry.court import Court
from baseline_lab.settings import Settings
from baseline_lab.storage.database import Database


def camera_router(cfg: Settings, db: Database) -> APIRouter:
    router = APIRouter(prefix="/api/v1/camera")

    def owned(sid: str, identity: str):
        row = db.session(sid, identity)
        if not row or json.loads(row["metadata"]).get("kind") != "camera":
            raise HTTPException(404, "camera_session_not_found")
        return row, cfg.data_dir / "sessions" / sid

    @router.post("", status_code=201)
    def create(body: LiveSessionInput, identity: str = Depends(owner)):
        if body.width % 2 or body.height % 2:
            raise HTTPException(422, "even_camera_dimensions_required")
        if not db.worker_alive():
            raise HTTPException(503, "worker_unavailable")
        with db.connect() as c:
            if c.execute("SELECT 1 FROM jobs WHERE state IN ('queued','running')").fetchone():
                raise HTTPException(409, "worker_busy: stop or finish the current job first")
        if body.calibration and not body.reference_jpeg_base64:
            raise HTTPException(422, "calibration_reference_frame_required")
        if body.calibration:
            try:
                Court(body.calibration)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            if any(
                not (0 <= x < body.width and 0 <= y < body.height)
                for x, y in body.calibration.corners_px
            ):
                raise HTTPException(422, "calibration_outside_image")
        sid = uuid.uuid4().hex
        directory = cfg.data_dir / "sessions" / sid
        directory.mkdir(parents=True, mode=0o700)
        if body.reference_jpeg_base64:
            try:
                (directory / "camera-reference.jpg").write_bytes(
                    base64.b64decode(body.reference_jpeg_base64, validate=True)
                )
            except (ValueError, binascii.Error) as exc:
                directory.rmdir()
                raise HTTPException(422, "invalid_reference_base64") from exc
        meta = {"kind": "camera", "width": body.width, "height": body.height, "duration_s": 60}
        with db.connect() as c:
            c.execute(
                "INSERT INTO sessions(id,owner,title,metadata,created) VALUES(?,?,?,?,?)",
                (sid, identity, "Live camera", json.dumps(meta), time.time()),
            )
        try:
            job = db.submit(
                sid,
                "camera",
                {"kind": "camera", **body.model_dump(exclude={"reference_jpeg_base64"})},
            )
        except ValueError as exc:
            with db.connect() as c:
                c.execute("DELETE FROM sessions WHERE id=?", (sid,))
            (directory / "camera-reference.jpg").unlink(missing_ok=True)
            directory.rmdir()
            raise HTTPException(409, str(exc)) from exc
        return {
            "id": sid,
            "job_id": job["id"],
            "max_duration_s": 60,
            "max_batch_frames": 8,
            "transport": "latest batch replaces stale batch; variable processing rate",
        }

    @router.post("/{sid}/frames", status_code=202)
    def frames(sid: str, body: CameraBatch, identity: str = Depends(owner)):
        _, directory = owned(sid, identity)
        if any(b.t_s <= a.t_s for a, b in zip(body.frames, body.frames[1:], strict=False)):
            raise HTTPException(422, "non_monotonic_capture_timestamps")
        with db.connect() as c:
            job = c.execute("SELECT state FROM jobs WHERE session_id=?", (sid,)).fetchone()
        if not job or job[0] not in {"queued", "running"}:
            raise HTTPException(409, "camera_not_running")
        # Unique temp names + atomic replace protect worker reads from partial JSON.
        temporary = directory / ("camera-input-" + uuid.uuid4().hex + ".tmp")
        temporary.write_text(body.model_dump_json())
        temporary.replace(directory / "camera-input.json")
        return {"accepted_sequence": body.sequence}

    @router.get("/{sid}/state")
    def state(sid: str, identity: str = Depends(owner)):
        _, directory = owned(sid, identity)
        path = directory / "camera-state.json"
        with db.connect() as c:
            job = dict(c.execute("SELECT * FROM jobs WHERE session_id=?", (sid,)).fetchone())
        return {"job": job, "sample": json.loads(path.read_text()) if path.exists() else None}

    @router.get("/{sid}/frame")
    def image(sid: str, request: Request, identity: str = Depends(owner)):
        _, directory = owned(sid, identity)
        sequence = request.query_params.get("sequence", "")
        if not sequence.isdigit() or int(sequence) > 1800:
            raise HTTPException(422, "valid_sequence_required")
        path = directory / f"camera-frame-{sequence}.jpg"
        if not path.exists():
            raise HTTPException(404, "no_processed_frame")
        return FileResponse(path, media_type="image/jpeg")

    @router.post("/{sid}/contacts", status_code=201)
    def contact(sid: str, body: Correction, identity: str = Depends(owner)):
        row, directory = owned(sid, identity)
        meta = json.loads(row["metadata"])
        if not (
            0 <= body.xy_px[0] < meta["width"]
            and 0 <= body.xy_px[1] < meta["height"]
            and body.t_s <= 60
        ):
            raise HTTPException(422, "contact_outside_camera_capture")
        if len(list(directory.glob("camera-contact-*.json"))) >= 200:
            raise HTTPException(429, "live_contact_limit")
        # One file per manual review is idempotently consumed by the worker.
        eid = uuid.uuid4().hex
        temporary = directory / ("camera-contact-" + eid + ".tmp")
        temporary.write_text(body.model_dump_json())
        temporary.replace(directory / ("camera-contact-" + eid + ".json"))
        return {"id": eid, "status": "pending_worker_review", "classification": "uncertain"}

    @router.post("/{sid}/stop", status_code=202)
    def stop(sid: str, identity: str = Depends(owner)):
        _, directory = owned(sid, identity)
        (directory / "camera-stop").touch()
        return {"status": "stopping"}

    return router
