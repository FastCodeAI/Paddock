"""Stand-in for `alpasim_wizard.main`, used only by the adapter's tests.

Synthesised, not recorded: it exists so the launch path can be exercised
without Docker, a GPU or a 20 GB checkpoint. Behaviour is chosen by argv so one
script can play every failure the harness must distinguish.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    mode = os.environ.get("FAKE_WIZARD_MODE", "ok")
    out_dir = Path(next(a.split("=", 1)[1] for a in sys.argv if a.startswith("hydra.run.dir=")))
    scene = next(a.split("=", 1)[1] for a in sys.argv if a.startswith("scenes.scene_ids=")).strip("[]")

    if mode == "hang":
        time.sleep(3600)
        return 0
    if mode == "exit3":
        print("simulated failure", file=sys.stderr)
        return 3
    if mode == "no_summary":
        return 0

    aggregate = out_dir / "aggregate"
    aggregate.mkdir(parents=True, exist_ok=True)
    score = None if mode == "unscored" else 0.75
    (aggregate / "results-summary.json").write_text(
        json.dumps(
            {
                "rollouts": [
                    {
                        "clipgt_id": scene,
                        "score": score,
                        "failure_reason": None,
                        "status": "complete",
                        "passed": True,
                        "metrics": {"progress": 0.9, "collision_any": 0.0, "offroad": 0.0},
                    }
                ],
                "scene_score_enabled": True,
            }
        )
    )
    marker = out_dir / "rollouts" / scene / "session-0"
    marker.mkdir(parents=True, exist_ok=True)
    (marker / "_complete").write_text("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
