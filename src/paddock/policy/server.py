"""Serve any `Policy` as the HTTP node closed-loop benchmarks drive.

A model implements `Policy` once. Open-loop harnesses call `infer` directly;
closed-loop simulators need it behind HTTP, and this supplies that half. No
model should contain its own FastAPI app -- four of them did, before paddock,
and each drifted a little from the others.

    python -m paddock.policy.server --model openpilot --port 9001

The endpoints are NeuroNCAP's: `/alive`, `/infer`, `/reset`, plus
`/diagnostics`, which is where model-specific reporting goes (openpilot's
field-of-view margin check, for instance) rather than into a bespoke route.
"""

from __future__ import annotations

import argparse
import importlib
import logging

from paddock.policy.protocol import Observation, Policy

logger = logging.getLogger(__name__)


def build_app(policy: Policy):
    """Wrap a loaded Policy in the endpoints a simulator expects."""
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/alive")
    async def alive() -> bool:
        return True

    @app.post("/infer")
    async def infer(body: dict) -> dict:
        # Decode only the cameras this model declared. Decoding is a torch
        # unpickle per camera and the simulator sends what it rendered, so a
        # single-camera model should not pay for six.
        obs = Observation.from_wire(body, cameras=policy.cameras)
        return policy.infer(obs).to_wire()

    @app.post("/infer_sequence")
    async def infer_sequence(body: dict) -> dict:
        """Score one open-loop sample: a sequence of frames, last one scored.

        The model runs in its own environment, so the sequence crosses a process
        boundary. That is deliberate -- open and closed loop share one execution
        path, and a second in-process path is where the two would diverge.
        """
        observations = [Observation.from_wire(item, cameras=policy.cameras) for item in body["observations"]]
        trajectory = policy.infer_sequence(observations)
        return {"trajectory": trajectory.waypoints.tolist(), "dt": trajectory.dt}

    @app.post("/reset")
    async def reset(body: dict | None = None) -> bool:
        policy.reset()
        return True

    @app.get("/diagnostics")
    async def diagnostics() -> dict:
        return policy.diagnostics()

    return app


def load_policy(model: str) -> Policy:
    """Import `adapters/models/<model>/policy.py` and instantiate its POLICY."""
    module = importlib.import_module(f"paddock.adapters.models.{model}.policy")
    try:
        factory = module.POLICY
    except AttributeError as exc:
        raise SystemExit(
            f"adapters/models/{model}/policy.py defines no POLICY. "
            f"Set `POLICY = YourPolicyClass` at module scope."
        ) from exc
    return factory()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a paddock model as an HTTP policy node.")
    parser.add_argument("--model", required=True, help="directory name under adapters/models/")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    import uvicorn

    policy = load_policy(args.model)
    # Load before serving, never lazily on first request: the simulator's
    # health check would pass while a 20 GB checkpoint was still being read, and
    # the first inference would time out with nothing explaining why.
    logger.info("loading %s", args.model)
    policy.load()
    logger.info("%s ready on port %d", args.model, args.port)

    uvicorn.run(build_app(policy), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
