"""Reproducible command-line entrypoints for the same pipeline as the API."""

import argparse
import json
import logging
from pathlib import Path

from baseline_lab.settings import settings


def main():
    parser = argparse.ArgumentParser(prog="baseline-lab")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("setup-models")
    worker = sub.add_parser("worker")
    worker.add_argument("--once", action="store_true")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8000)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("source", type=Path)
    analyze_parser.add_argument("output", type=Path)
    analyze_parser.add_argument("--calibration", type=Path)
    analyze_parser.add_argument("--real-time-playback", action="store_true")
    probe = sub.add_parser("probe")
    probe.add_argument("source", type=Path)
    probe.add_argument("directory", type=Path)
    job = sub.add_parser("process-job")
    job.add_argument("sid")
    job.add_argument("jid")
    job.add_argument("config", type=Path)
    args = parser.parse_args()
    cfg = settings()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "setup-models":
        from baseline_lab.vision.ball import provision_ball
        from baseline_lab.vision.detectors import provision
        from baseline_lab.vision.pose import provision_pose
        from baseline_lab.vision.strokes import provision_strokes

        print(provision(cfg.data_dir / "models"))
        print(provision_pose(cfg.data_dir / "models"))
        print(provision_ball(cfg.data_dir / "models"))
        print(provision_strokes(cfg.data_dir / "models"))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run(
            "baseline_lab.api.app:create_app", factory=True, host="127.0.0.1", port=args.port
        )
    elif args.command == "worker":
        from baseline_lab.jobs.worker import run_worker

        run_worker(cfg, args.once)
    elif args.command == "probe":
        from baseline_lab.video.media import inspect, preview, write_metadata

        try:
            meta = inspect(args.source, cfg, args.directory / "thumbnail.jpg")
            preview(args.source, args.directory / "ingest.mp4", (meta.width, meta.height))
            write_metadata(meta, args.directory)
        except Exception as exc:
            import sys

            print(json.dumps({"error": str(exc)}), file=sys.stderr)
            raise SystemExit(1) from None
    elif args.command == "analyze":
        from baseline_lab.contracts import Calibration
        from baseline_lab.pipeline import analyze

        calibration = (
            Calibration.model_validate_json(args.calibration.read_text())
            if args.calibration
            else None
        )
        result = analyze(
            args.source,
            args.output,
            cfg,
            calibration,
            lambda s, d, t: (
                print(json.dumps({"stage": s, "done": d, "total": t}))
                if d == 0 or (isinstance(d, int) and d % 60 == 0)
                else None
            ),
            real_time_playback=args.real_time_playback,
        )
        print(json.dumps({"elapsed_s": result.elapsed_s, "events": len(result.events)}))
    elif args.command == "process-job":
        from baseline_lab.contracts import Submission
        from baseline_lab.pipeline import analyze
        from baseline_lab.storage.database import Database

        db = Database(cfg.data_dir)
        raw_payload = json.loads(args.config.read_text())
        payload = (
            Submission.model_validate(raw_payload) if raw_payload.get("kind") != "camera" else None
        )

        def checkpoint(stage, done, total):
            with db.connect() as c:
                row = c.execute("SELECT cancel FROM jobs WHERE id=?", (args.jid,)).fetchone()
            if not row or row[0]:
                raise InterruptedError("cancelled")
            db.update(args.jid, stage=stage, done=done, total=total)

        try:
            directory = cfg.data_dir / "sessions" / args.sid
            if raw_payload.get("kind") == "camera":
                from baseline_lab.jobs.camera import run_camera

                run_camera(directory, args.config.parent, raw_payload, cfg, checkpoint)
            else:
                assert payload is not None
                analyze(
                    directory / "source",
                    args.config.parent,
                    cfg,
                    payload.calibration,
                    checkpoint,
                    real_time_playback=payload.real_time_playback,
                )
            checkpoint("finishing", 1, 1)
        except (ValueError, RuntimeError, OSError, InterruptedError) as exc:
            (args.config.parent / "error.json").write_text(json.dumps({"error": str(exc)}))
            raise


if __name__ == "__main__":
    main()
