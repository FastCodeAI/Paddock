"""Serve any `Policy` as the HTTP node closed-loop benchmarks drive.

A model implements `Policy` once. Open-loop harnesses call `infer` directly;
closed-loop simulators need it behind HTTP, and this supplies that half. No
model should contain its own FastAPI app -- four of them did, before paddock,
and each drifted a little from the others.

    python -m paddock.policy.server --model openpilot --port 9001

The endpoints are NeuroNCAP's: `/alive`, `/infer`, `/reset`, plus
`/diagnostics`, which is where model-specific reporting goes (openpilot's
field-of-view margin check, for instance) rather than into a bespoke route,
and `/observe`, which hands the model a frame of history without asking it to
plan.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import time
from typing import Any, Optional

from paddock.policy.protocol import Observation, Policy

logger = logging.getLogger(__name__)

#: Log one line per this many /infer calls. At CARLA's 20 Hz a route is ~4000
#: calls; a line per 100 is a heartbeat, not a flood.
LOG_EVERY = 100


#: Seconds ahead the heartbeat reads a plan at: the first and last of the six points 0.5 s
#: apart that a closed-loop simulator drives, so the line means the same for every model.
REACH_AT_S = (0.5, 3.0)


def _plan_reach(result: dict) -> list[tuple[float, float]]:
    """How far a wire response's plan reaches: (metres, seconds ahead) at each of REACH_AT_S.

    A plan whose 0.5 s point never leaves the origin is a car that never engages,
    and that is exactly what a node log needs to show. Each is the plan's own point
    nearest that time, read from ``plan``, every point at the model's own spacing; the
    seconds say which point it was, should a plan end sooner. Not from ``trajectory``:
    that holds a plan's first six points whatever their spacing, which is 3 s of a plan
    0.5 s apart and 0.6 s of one 0.1 s apart: alpamayo2edge's (#200), and qwendrive's and
    alpamayo15's on AlpaSim. For a plan 0.5 s apart the two points are the first and last
    ``trajectory`` holds, as before.
    """
    try:
        plan = result["plan"]
        waypoints, dt = plan["waypoints"], float(plan["dt"])
        reach = []
        for seconds in REACH_AT_S:
            i = min(len(waypoints), max(1, round(seconds / dt))) - 1
            x, y = float(waypoints[i][0]), float(waypoints[i][1])
            reach.append(((x**2 + y**2) ** 0.5, (i + 1) * dt))
        return reach
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return [(float("nan"), seconds) for seconds in REACH_AT_S]


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
            (near, near_s), (far, far_s) = _plan_reach(result)
            logger.info(
                "infer #%d: mean %.0f ms, plan reach %.2f m at %.1f s / %.2f m at %.1f s",
                calls["n"],
                calls["ms"] / calls["n"],
                near,
                near_s,
                far,
                far_s,
            )
        return result

    @app.post("/observe")
    async def observe(body: dict) -> dict:
        """Take in one frame of history; plan nothing the car could drive.

        The same body as `/infer`, passed to `Policy.observe`. A model that
        does not override it runs `infer` and drops the plan, which is exactly
        what a client sending `/infer` and ignoring the answer got before. A
        model that only records the frame (Drive-JEPA) then has no plan from
        that frame -- so its momentum stage has no previous plan to compare the
        next one with, as when the frame was never planned from.
        """
        obs = Observation.from_wire(body, cameras=policy.cameras)
        policy.observe(obs)
        return {"observed": True}

    @app.post("/infer_sequence")
    async def infer_sequence(body: dict) -> dict:
        """Score one open-loop sample: a sequence of frames, last one scored.

        The model runs in its own environment, so the sequence crosses a process
        boundary. That is deliberate -- open and closed loop share one execution
        path, and a second in-process path is where the two would diverge.

        `extras` crosses with the trajectory for the same reason the closed-loop
        path forwards it: the auxiliary heads are predictions the benchmark can
        score against its own recorded truth, and dropping them here would make
        open loop the one path where a model is judged on less than it produced.
        """
        observations = [Observation.from_wire(item, cameras=policy.cameras) for item in body["observations"]]
        trajectory = policy.infer_sequence(observations)
        result: dict[str, Any] = {"trajectory": trajectory.waypoints.tolist(), "dt": trajectory.dt}
        if trajectory.extras:
            result["extras"] = trajectory.extras
        return result

    @app.post("/reset")
    # `Optional[dict]`, not `dict | None`, and it is not a style choice. This
    # module runs inside the *model's* interpreter, which is as old as the model
    # needs -- Drive-JEPA's is Python 3.9. `from __future__ import annotations`
    # makes the annotation a string, FastAPI asks pydantic to evaluate it to build
    # the route, and on 3.9 pydantic cannot evaluate PEP 604 without the
    # `eval_type_backport` package. The node then dies at import, before load(),
    # with a pydantic traceback that names neither paddock nor the model. Every
    # other annotation here is either a builtin or behind the `__future__` import
    # and never evaluated, so this is the only one that has to be spelled the old
    # way.
    async def reset(body: Optional[dict] = None) -> bool:  # noqa: UP045
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
