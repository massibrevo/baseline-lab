"""Small runnable examples accompanying docs/learning (no model download)."""

import argparse
from pathlib import Path

import numpy as np

from baseline_lab.contracts import Calibration
from baseline_lab.geometry.court import Court, line_call
from baseline_lab.storage.database import Database

parser = argparse.ArgumentParser()
parser.add_argument("chapter", type=int, choices=range(1, 8))
args = parser.parse_args()
if args.chapter == 1:
    times = np.array([0, 0.0333, 0.0833, 0.1166])
    print("PTS seconds:", times, "intervals:", np.diff(times))
    print("A frame-index/30 shortcut gives", 3 / 30, "s instead of", times[3])
elif args.chapter == 2:
    from baseline_lab.tracking.association import BallAssociation

    association = BallAssociation(1000)
    for i, (time, points) in enumerate(
        [(0, [(20, 30)]), (0.033, [(25, 32)]), (0.066, [(950, 950)])]
    ):
        print(association.update([{"xy": p, "score": 0.8} for p in points], time, i).model_dump())
elif args.chapter == 3:
    c = Court(Calibration(corners_px=[(210, 90), (430, 90), (550, 430), (90, 430)]))
    print("Court meters (4.115,11.885) → image:", c.image((4.115, 11.885)))
    print("Roundtrip:", c.project(c.image((4.115, 11.885))))
elif args.chapter == 4:
    # Scalar Kalman measurement update, an educational comparison, not the production tracker.
    estimate, variance, measurement, noise = 10.0, 4.0, 14.0, 9.0
    gain = variance / (variance + noise)
    print(
        "Kalman gain:",
        gain,
        "posterior position:",
        estimate + gain * (measurement - estimate),
        "posterior variance:",
        (1 - gain) * variance,
    )
elif args.chapter == 5:
    print([(x, line_call((x, 5), 0.25)) for x in [-0.5, -0.1, 0, 0.1, 0.5]])
    print("Unvalidated uncertainty:", line_call((3, 5), None))
elif args.chapter == 6:
    # Same coverage, very different localization; missing frames never equal zero error.
    print("Two tracks cover 90/100 frames, with 80/90 vs 10/90 localizations in tolerance.")
    print("Coverage:", 90 / 100, "localization recall:", 80 / 100, "vs", 10 / 100)
elif args.chapter == 7:
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        db = Database(Path(directory))
        with db.connect() as c:
            c.execute("INSERT INTO sessions VALUES('example','owner','test','{}',0,0)")
        a = db.submit("example", "same-key", {"preset": "example"})
        b = db.submit("example", "same-key", {"preset": "example"})
        print("Idempotent:", a["id"] == b["id"], "claimed:", db.claim()["id"], "next:", db.claim())
