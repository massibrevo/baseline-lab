"""A single durable worker; one isolated inference child, cooperative cancellation."""

import fcntl
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time

from baseline_lab.settings import Settings
from baseline_lab.storage.database import Database

logger = logging.getLogger(__name__)


def run_worker(cfg: Settings, once: bool = False):
    cfg.prepare()
    db = Database(cfg.data_dir)
    lock = (cfg.data_dir / "worker.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Only one worker may use this data directory") from exc
    db.recover()
    with db.connect() as c:
        tombstones = c.execute("SELECT id FROM sessions WHERE deleted=1").fetchall()
    for row in tombstones:
        shutil.rmtree(cfg.data_dir / "sessions" / row["id"], ignore_errors=True)
    for partial in (cfg.data_dir / "sessions").glob("*/.*.partial"):
        shutil.rmtree(partial, ignore_errors=True)
    stop = threading.Event()

    def heartbeat():
        while not stop.is_set():
            db.beat()
            stop.wait(2)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    child = None
    try:
        while True:
            job = db.claim()
            if not job:
                if once:
                    return
                stop.wait(0.5)
                continue
            sid, jid = job["session_id"], job["id"]
            directory = cfg.data_dir / "sessions" / sid
            temp = directory / f".{jid}.partial"
            target = directory / jid
            shutil.rmtree(temp, ignore_errors=True)
            temp.mkdir(parents=True)
            config = temp / "job.json"
            config.write_text(json.dumps(json.loads(job["payload"])))
            log = (temp / "worker.log").open("w")
            child = subprocess.Popen(
                [sys.executable, "-m", "baseline_lab.cli", "process-job", sid, jid, str(config)],
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            start = time.monotonic()
            timed_out = False
            while child.poll() is None:
                with db.connect() as c:
                    cancelled = c.execute("SELECT cancel FROM jobs WHERE id=?", (jid,)).fetchone()
                if cancelled and cancelled[0]:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                elif time.monotonic() - start > cfg.job_timeout_s:
                    timed_out = True
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                stop.wait(0.3)
            log.close()
            with db.connect() as c:
                row = c.execute("SELECT cancel FROM jobs WHERE id=?", (jid,)).fetchone()
                deleted = c.execute("SELECT deleted FROM sessions WHERE id=?", (sid,)).fetchone()
            if not row or row[0] or (deleted and deleted[0]):
                db.update(jid, state="cancelled", stage="cancelled")
                shutil.rmtree(temp, ignore_errors=True)
            elif child.returncode or timed_out:
                error = "job_timeout" if timed_out else "analysis_failed"
                error_file = temp / "error.json"
                if error_file.exists() and not timed_out:
                    error = json.loads(error_file.read_text())["error"]
                db.update(jid, state="failed", stage="failed", error=error[:1000])
                shutil.rmtree(temp, ignore_errors=True)
            else:
                for internal in [config, temp / "worker.log"]:
                    internal.unlink(missing_ok=True)
                shutil.rmtree(target, ignore_errors=True)
                temp.replace(target)
                db.update(jid, state="completed_with_warnings", stage="completed", done=1, total=1)
            if deleted and deleted[0]:
                shutil.rmtree(directory, ignore_errors=True)
            logger.info(json.dumps({"event": "job_finished", "job_id": jid}))
            child = None
            if once:
                return
    finally:
        stop.set()
        if child and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=10)
        thread.join(timeout=3)
        lock.close()
