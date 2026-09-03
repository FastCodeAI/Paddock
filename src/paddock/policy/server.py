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
import time

from paddock.policy.protocol import Observation, Policy

logger = logging.getLogger(__name__)

#: Log one line per this many /infer calls. At CARLA's 20 Hz a route is ~4000
#: calls; a line per 100 is a heartbeat, not a flood.
LOG_EVERY = 100


def _plan_reach(result: dict) -> tuple[float, float]:
    """(|first waypoint|, |last waypoint|) in metres from a wire response; a
    plan whose first point never leaves the origin is a car that never engages,
    and that is exactly what a node log needs to show."""
    try:
        wps = result["trajectory"]
        first, last = wps[0], wps[-1]
        return float((first[0] ** 2 + first[1] ** 2) ** 0.5), float((last[0] ** 2 + last[1] ** 2) ** 0.5)
    except (KeyError, IndexError, TypeError, ValueError):
        return float("nan"), float("nan")


def build_app(policy: Policy, model: str | None = None, nonce: str | None = None):
    """Wrap a loaded Policy in the endpoints a simulator expects."""
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/alive")
    async def alive() -> bool:
        return True

    @app.get("/whoami")
    async def whoami() -> dict:
        """Which model this node serves.

        `/alive` says only that *a* paddock node is answering. On a shared
        machine, with several runs choosing ports from the same range, that is
        not enough: a run whose own node was still loading once passed its
        health check against a sibling's node on the same port and scored the
        wrong model's plans under its name. The launcher checks the model,
        and the nonce it gave this launch -- two concurrent runs of the *same*
        model would otherwise pass each other's check.
        """
        return {"model": model or type(policy).__name__, "nonce": nonce}

    calls = {"n": 0, "ms": 0.0}

    @app.post("/infer")
    async def infer(body: dict) -> dict:
        # Decode only the cameras this model declared. Decoding is a torch
        # unpickle per camera and the simulator sends what it rendered, so a
        # single-camera model should not pay for six.
        t0 = time.monotonic()
        obs = Observation.from_wire(body, cameras=policy.cameras)
        result = policy.infer(obs).to_wire()
        calls["n"] += 1
        calls["ms"] += (time.monotonic() - t0) * 1000.0
        # A node that answers but never plans motion used to leave an empty log
        # and a route scored "blocked" with nothing to say why. First call and
        # every LOG_EVERY after: cadence, cost and how far the plan reaches.
        if calls["n"] == 1 or calls["n"] % LOG_EVERY == 0:
            first, last = _plan_reach(result)
            logger.info(
                "infer #%d: mean %.0f ms, plan reach %.2f m (first) / %.2f m (last)",
                calls["n"],
                calls["ms"] / calls["n"],
                first,
                last,
            )
        return result

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
    parser.add_argument(
        "--nonce", default=None, help="echoed by /whoami so the launcher knows this node is its own"
    )
    args = parser.parse_args()

    # Without a handler at INFO the "loading"/"ready" lines below and the infer
    # heartbeat are dropped, and the node log the launcher captures is empty --
    # which is how a model that never engaged went undiagnosed at one site.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import uvicorn

    policy = load_policy(args.model)
    # Load before serving, never lazily on first request: the simulator's
    # health check would pass while a 20 GB checkpoint was still being read, and
    # the first inference would time out with nothing explaining why.
    logger.info("loading %s", args.model)
    policy.load()
    logger.info("%s ready on port %d", args.model, args.port)

    app = build_app(policy, model=args.model, nonce=args.nonce)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
