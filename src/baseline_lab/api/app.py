"""Versioned local API. Artifacts always pass session ownership checks."""

import json
import secrets
import shutil
import time
import uuid
from pathlib import Path
from subprocess import TimeoutExpired

from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from baseline_lab.api.auth import mint, owner, secret
from baseline_lab.contracts import Correction, Event, Result, Submission
from baseline_lab.geometry.court import Court
from baseline_lab.rendering.export import csvs
from baseline_lab.settings import ROOT, Settings, settings
from baseline_lab.storage.database import Database
from baseline_lab.video.media import isolated_probe

ARTIFACTS = {
    "annotated.mp4",
    "review.mp4",
    "result.json",
    "events.csv",
    "tracks.csv",
    "raw-detections.json",
}


def create_app(cfg: Settings | None = None) -> FastAPI:
    cfg = cfg or settings()
    cfg.prepare()
    db = Database(cfg.data_dir)
    app = FastAPI(title="Baseline Lab", version="1.0.0")
    app.state.key = secret(cfg.data_dir)
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Content-Type", "X-Baseline-Request"],
    )

    @app.middleware("http")
    async def guard(request: Request, call_next):
        if request.method == "POST" and request.url.path == "/api/v1/sessions":
            length = request.headers.get("content-length")
            if length is None:
                return JSONResponse({"detail": "content_length_required"}, 411)
            if not length.isdigit() or int(length) > (cfg.max_upload_mb + 1) * 1024 * 1024:
                return JSONResponse({"detail": "request_body_limit_exceeded"}, 413)
        if request.method in {"POST", "PATCH", "DELETE"}:
            origin = request.headers.get("origin")
            if origin and origin not in cfg.allowed_origins:
                return JSONResponse({"detail": "origin_not_allowed"}, 403)
            if request.headers.get("X-Baseline-Request") != "1":
                return JSONResponse({"detail": "csrf_header_required"}, 403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    def session(sid: str, identity: str) -> dict:
        row = db.session(sid, identity)
        if not row:
            raise HTTPException(404, "session_not_found")
        return row

    def job_for(sid: str, jid: str):
        with db.connect() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=? AND session_id=?", (jid, sid)).fetchone()
        if not row:
            raise HTTPException(404, "job_not_found")
        return dict(row)

    def result_path(sid: str, jid: str) -> Path:
        job = job_for(sid, jid)
        if job["state"] not in {"completed", "completed_with_warnings"}:
            raise HTTPException(409, "result_not_ready")
        return cfg.data_dir / "sessions" / sid / jid

    @app.get("/api/v1/identity")
    def identity(request: Request, response: Response):
        try:
            owner(request)
        except HTTPException:
            response.set_cookie(
                "baseline_owner",
                mint(app.state.key),
                httponly=True,
                secure=cfg.secure_cookies,
                samesite="strict",
                max_age=30 * 86400,
            )
        return {
            "deployment": "local",
            "private_processing": True,
            "max_upload_mb": cfg.max_upload_mb,
            "max_duration_s": cfg.max_duration_s,
            "worker_alive": db.worker_alive(),
            "csrf_header": "X-Baseline-Request: 1",
        }

    @app.get("/api/v1/health")
    def health():
        return {"status": "ok", "worker_alive": db.worker_alive()}

    @app.get("/api/v1/sessions")
    def sessions(offset: int = 0, limit: int = 20, identity: str = Depends(owner)):
        with db.connect() as c:
            rows = c.execute(
                "SELECT * FROM sessions WHERE owner=? AND deleted=0 "
                "ORDER BY created DESC LIMIT ? OFFSET ?",
                (identity, min(max(limit, 1), 100), max(offset, 0)),
            ).fetchall()
        return [{**dict(r), "metadata": json.loads(r["metadata"])} for r in rows]

    @app.post("/api/v1/sessions", status_code=201)
    def upload(file: UploadFile, identity: str = Depends(owner)):
        if Path(file.filename or "").suffix.lower() not in {".mp4", ".mov"}:
            raise HTTPException(415, "mp4_or_mov_required")
        sid = uuid.uuid4().hex
        directory = cfg.data_dir / "sessions" / sid
        directory.mkdir(parents=True, mode=0o700)
        size = 0
        try:
            with (directory / "source").open("wb") as out:
                while block := file.file.read(1 << 20):
                    size += len(block)
                    if size > cfg.max_upload_mb * 1024 * 1024:
                        raise HTTPException(413, "upload_limit_exceeded")
                    out.write(block)
            metadata = isolated_probe(directory / "source", directory, cfg)
            title = Path(file.filename or "Private clip").name[:150]
            with db.connect() as c:
                c.execute(
                    "INSERT INTO sessions(id,owner,title,metadata,created) VALUES(?,?,?,?,?)",
                    (sid, identity, title, metadata.model_dump_json(), time.time()),
                )
            return {"id": sid, "title": title, "metadata": metadata.model_dump(), "jobs": []}
        except HTTPException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        except (ValueError, OSError, TimeoutError, TimeoutExpired) as exc:
            shutil.rmtree(directory, ignore_errors=True)
            raise HTTPException(422, f"media_validation_failed: {str(exc)[-500:]}") from exc

    @app.get("/api/v1/sessions/{sid}")
    def get_session(sid: str, identity: str = Depends(owner)):
        row = session(sid, identity)
        with db.connect() as c:
            jobs = c.execute(
                "SELECT * FROM jobs WHERE session_id=? ORDER BY updated DESC", (sid,)
            ).fetchall()
        return {**row, "metadata": json.loads(row["metadata"]), "jobs": [dict(j) for j in jobs]}

    @app.get("/api/v1/sessions/{sid}/thumbnail")
    def thumbnail(sid: str, identity: str = Depends(owner)):
        session(sid, identity)
        return FileResponse(cfg.data_dir / "sessions" / sid / "thumbnail.jpg")

    @app.get("/api/v1/sessions/{sid}/source")
    def source(sid: str, identity: str = Depends(owner)):
        session(sid, identity)
        return FileResponse(cfg.data_dir / "sessions" / sid / "source", media_type="video/mp4")

    @app.post("/api/v1/sessions/{sid}/jobs", status_code=202)
    def submit(sid: str, body: Submission, identity: str = Depends(owner)):
        row = session(sid, identity)
        if body.calibration:
            meta = json.loads(row["metadata"])
            if any(
                not (0 <= x < meta["width"] and 0 <= y < meta["height"])
                for x, y in body.calibration.corners_px
            ):
                raise HTTPException(422, "calibration_outside_image")
            try:
                Court(body.calibration)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        with db.connect() as c:
            existing = c.execute(
                "SELECT 1 FROM jobs WHERE session_id=? AND key=?", (sid, body.idempotency_key)
            ).fetchone()
        if existing:
            try:
                return db.submit(sid, body.idempotency_key, body.model_dump())
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
        if not db.worker_alive():
            raise HTTPException(503, "worker_unavailable: start baseline-lab worker")
        try:
            return db.submit(sid, body.idempotency_key, body.model_dump())
        except ValueError as exc:
            raise HTTPException(
                429 if str(exc) == "local_queue_limit_5" else 409, str(exc)
            ) from exc

    @app.get("/api/v1/sessions/{sid}/jobs/{jid}")
    def get_job(sid: str, jid: str, identity: str = Depends(owner)):
        session(sid, identity)
        return {**job_for(sid, jid), "worker_alive": db.worker_alive()}

    @app.post("/api/v1/sessions/{sid}/jobs/{jid}/cancel")
    def cancel(sid: str, jid: str, identity: str = Depends(owner)):
        session(sid, identity)
        job = job_for(sid, jid)
        if job["state"] in {"queued", "running"}:
            db.update(
                jid,
                cancel=1,
                **(
                    {"state": "cancelled", "stage": "cancelled"} if job["state"] == "queued" else {}
                ),
            )
        return job_for(sid, jid)

    @app.get("/api/v1/sessions/{sid}/jobs/{jid}/artifacts/{name}")
    def artifact(sid: str, jid: str, name: str, identity: str = Depends(owner)):
        session(sid, identity)
        if name not in ARTIFACTS:
            raise HTTPException(404, "artifact_not_found")
        path = result_path(sid, jid) / name
        if not path.exists():
            raise HTTPException(404, "artifact_not_found")
        return FileResponse(
            path,
            filename=name,
            content_disposition_type="inline" if name.endswith("mp4") else "attachment",
        )

    def edit_event(sid: str, jid: str, eid: str | None, body: Correction, identity: str):
        row = session(sid, identity)
        directory = result_path(sid, jid)
        metadata = json.loads(row["metadata"])
        x, y = body.xy_px
        if not (
            body.t_s <= metadata["duration_s"]
            and 0 <= x < metadata["width"]
            and 0 <= y < metadata["height"]
        ):
            raise HTTPException(422, "event_outside_clip")
        # Serialize read/modify/replace across requests and preserve original predictions.
        import fcntl

        with (directory / ".edit.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            result = Result.model_validate_json((directory / "result.json").read_text())
            frame = min(
                range(len(metadata["timestamps_s"])),
                key=lambda i: abs(metadata["timestamps_s"][i] - body.t_s),
            )
            if eid:
                event = next((e for e in result.events if e.id == eid), None)
                if not event:
                    raise HTTPException(404, "event_not_found")
                if event.original_prediction is None:
                    event.original_prediction = event.model_dump(exclude={"original_prediction"})
                event.t_s, event.xy_px = body.t_s, body.xy_px
                event.status, event.notes, event.frame = body.status, body.notes, frame
            else:
                event = Event(
                    id="manual-" + secrets.token_hex(6),
                    t_s=body.t_s,
                    frame=frame,
                    xy_px=body.xy_px,
                    status="manual",
                    notes=body.notes,
                )
                result.events.append(event)
            event.court_m = None
            if event.status in {"confirmed", "manual"} and result.calibration:
                from baseline_lab.contracts import Calibration

                court = Court(Calibration(corners_px=result.calibration["corners_px"]))
                event.court_m = court.project(event.xy_px)
            event.classification = "uncertain"
            temp = directory / "result.edit.json"
            temp.write_text(result.model_dump_json())
            temp.replace(directory / "result.json")
            csvs(result, directory)
        return result

    @app.patch("/api/v1/sessions/{sid}/jobs/{jid}/events/{eid}")
    def correct(sid: str, jid: str, eid: str, body: Correction, identity: str = Depends(owner)):
        return edit_event(sid, jid, eid, body, identity)

    @app.post("/api/v1/sessions/{sid}/jobs/{jid}/events")
    def add_event(sid: str, jid: str, body: Correction, identity: str = Depends(owner)):
        return edit_event(sid, jid, None, body, identity)

    @app.delete("/api/v1/sessions/{sid}", status_code=204)
    def delete(sid: str, identity: str = Depends(owner)):
        session(sid, identity)
        with db.connect() as c:
            c.execute(
                "UPDATE sessions SET deleted=1, owner='', title='', metadata='{}' WHERE id=?",
                (sid,),
            )
            c.execute("UPDATE jobs SET payload='{}', error=NULL WHERE session_id=?", (sid,))
            c.execute(
                "UPDATE jobs SET cancel=1 WHERE session_id=? AND state IN ('queued','running')",
                (sid,),
            )
            running = c.execute(
                "SELECT 1 FROM jobs WHERE session_id=? AND state='running'", (sid,)
            ).fetchone()
            c.execute(
                "UPDATE jobs SET state='cancelled',stage='cancelled' "
                "WHERE session_id=? AND state='queued'",
                (sid,),
            )
        if not running:
            shutil.rmtree(cfg.data_dir / "sessions" / sid, ignore_errors=True)
        return Response(status_code=204)

    dist = ROOT / "apps" / "web" / "dist"
    if dist.exists():
        app.mount("/", StaticFiles(directory=dist, html=True), name="web")
    return app
