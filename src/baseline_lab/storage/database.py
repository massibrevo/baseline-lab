"""SQLite transactions, durable jobs and owner filtering in one small repository."""

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class Database:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory = directory
        self.path = directory / "jobs.sqlite3"
        with self.connect() as con:
            con.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS sessions (
              id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
              metadata TEXT NOT NULL, created REAL NOT NULL, deleted INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, session_id TEXT NOT NULL, key TEXT NOT NULL,
              payload TEXT NOT NULL, state TEXT NOT NULL, stage TEXT NOT NULL,
              done INTEGER, total INTEGER, error TEXT, cancel INTEGER DEFAULT 0,
              updated REAL NOT NULL, UNIQUE(session_id,key));
            CREATE TABLE IF NOT EXISTS worker (id INTEGER PRIMARY KEY, heartbeat REAL NOT NULL);
            """)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        except BaseException:
            con.rollback()
            raise
        finally:
            con.close()

    def session(self, sid: str, owner: str):
        with self.connect() as c:
            r = c.execute(
                "SELECT * FROM sessions WHERE id=? AND owner=? AND deleted=0", (sid, owner)
            ).fetchone()
            return dict(r) if r else None

    def submit(self, sid: str, key: str, payload: dict) -> dict:
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            existing = c.execute(
                "SELECT * FROM jobs WHERE session_id=? AND key=?", (sid, key)
            ).fetchone()
            if existing:
                if json.loads(existing["payload"]) != payload:
                    raise ValueError("idempotency_key_payload_conflict")
                return dict(existing)
            active = c.execute(
                "SELECT 1 FROM jobs WHERE session_id=? AND state IN ('queued','running')", (sid,)
            ).fetchone()
            if active:
                raise ValueError("session_already_processing")
            queued = c.execute(
                "SELECT count(*) FROM jobs WHERE state IN ('queued','running')"
            ).fetchone()[0]
            if payload.get("kind") == "camera" and queued:
                raise ValueError("worker_busy")
            if queued >= 5:
                raise ValueError("local_queue_limit_5")
            jid = uuid.uuid4().hex
            c.execute(
                "INSERT INTO jobs(id,session_id,key,payload,state,stage,updated) "
                "VALUES(?,?,?,?,?,?,?)",
                (jid, sid, key, json.dumps(payload), "queued", "queued", time.time()),
            )
            return dict(c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone())

    def claim(self):
        with self.connect() as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute(
                "SELECT j.* FROM jobs j JOIN sessions s ON s.id=j.session_id "
                "WHERE j.state='queued' AND s.deleted=0 AND j.cancel=0 ORDER BY j.updated LIMIT 1"
            ).fetchone()
            if row:
                c.execute(
                    "UPDATE jobs SET state='running',stage='starting',updated=? WHERE id=?",
                    (time.time(), row["id"]),
                )
                return dict(row)
            return None

    def update(self, jid: str, **values):
        allowed = {"state", "stage", "done", "total", "error", "cancel"}
        if not values or not values.keys() <= allowed:
            raise ValueError("invalid_job_fields")
        with self.connect() as c:
            c.execute(
                "UPDATE jobs SET " + ",".join(f"{k}=?" for k in values) + ",updated=? WHERE id=?",
                (*values.values(), time.time(), jid),
            )

    def beat(self):
        with self.connect() as c:
            c.execute("INSERT OR REPLACE INTO worker VALUES(1,?)", (time.time(),))

    def worker_alive(self) -> bool:
        with self.connect() as c:
            row = c.execute("SELECT heartbeat FROM worker WHERE id=1").fetchone()
            return bool(row and time.time() - row[0] < 15)

    def recover(self):
        # Caller holds exclusive OS worker lock, so no live worker can own these jobs.
        with self.connect() as c:
            c.execute(
                "UPDATE jobs SET state='failed',stage='failed',error='live_capture_interrupted' "
                "WHERE state='running' AND json_extract(payload,'$.kind')='camera'"
            )
            c.execute(
                "UPDATE jobs SET state='queued',stage='recovered_after_restart',updated=? "
                "WHERE state='running' AND cancel=0",
                (time.time(),),
            )
            c.execute(
                "UPDATE jobs SET state='cancelled',stage='cancelled' WHERE cancel=1 "
                "AND state IN ('queued','running')"
            )
