#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Serve Drive-JEPA to AlpaSim as a native gRPC driver, adapter version 0.1.7.

    # in AlpaSim's own interpreter, from the paddock checkout:
    python tools/models/drivejepa/alpasim_driver/serve.py \\
        --host 0.0.0.0 --port 9149 --device cpu --control-timestep-us 50000

This is the driver that produced the 30-scene numbers in
docs/models/drivejepa.md. It is NOT the paddock bridge: AlpaSim's runtime talks
to this process directly, and this process owns the model. The environment it
needs (four variables) is listed in README.md beside this file.

**What 0.1.7 is.** 0.1.4b -- the 20 Hz divider plus paper §3.5 momentum-aware
selection -- driving the retrained checkpoint, with the two input changes that
checkpoint was trained with (a 37-row bonnet mask; acceleration zeroed in the
model input and in the momentum re-score) and the batch fix on. Those are not
switches: they are what 0.1.7 is.

**The referee clock (inherited from 0.1.4).** The referee runs every model on
one clock: control_timestep_us == frame_interval_us == 50_000, n_sim_steps ==
400. Drive-JEPA is still a 2 Hz agent, so the adapter divides -- see
`divider.py`. Three things this launcher does on purpose:

  1. It builds a `ReplanDivider` around the planner. The divider owns the
     2 Hz grid; the planner stays a pure function of one observation.
  2. It pins `--image-history` to K.DEFAULT_IMAGE_HISTORY (16). At 50 ms
     pushes, 8 frames span 350 ms and the 0.5 s context pair cannot exist.
  3. It REFUSES `--target-frame-spacing-us` rather than ignoring it, so a run
     cannot be launched with a frame pairing nobody actually asked for.

Three things enforce provenance, rather than leaving it to convention:

  1. ``_resolve_planner`` refuses to serve a planner, divider, bridge or
     momentum module whose real path is not this directory -- a shadowing copy
     elsewhere on the path fails at startup instead of quietly winning.
  2. The version gate reads ``constants.ADAPTER_VERSION`` from the *imported*
     module, so a stale package cannot claim to be this one.
  3. ``log_provenance`` prints the resolved path into the run's driver.log, so
     every run directory records which file actually drove it.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LIVE_PACKAGE = _HERE
# tools/models/drivejepa/alpasim_driver/serve.py -> the checkout root.
_REPO_SRC = _HERE.parents[3] / "src"
assert (_REPO_SRC / "paddock" / "adapters" / "benchmarks" / "alpasim" / "scaffold.py").is_file(), _REPO_SRC

# tools/models FIRST so this package resolves as `drivejepa.alpasim_driver` --
# never the bare `alpasim_driver`, which is AlpaSim's own package and must stay
# importable. paddock's src after it for the one scaffold every driver shares.
sys.path.insert(0, str(_HERE.parents[1]))
if str(_REPO_SRC) not in sys.path:
    sys.path.append(str(_REPO_SRC))

#: What this launcher is for. The *expectation*; the reported value always
#: comes out of the imported module, never from this literal.
EXPECTED_VERSION = "0.1.7"

#: AlpaSim logical_id of the front camera. Same as serve_planner's default.
FRONT_CAMERA_ID = "camera_front_wide_120fov"

logger = logging.getLogger("serve_drivejepa_alpasim")


def _resolve_planner():
    """Import the planner the way the server will, and prove where it came from.

    Returns ``(planner_module, divider_module, constants_module, real_path)``.
    Raises SystemExit -- at startup, before any GPU or scene time -- if the
    package on the path is not the live one, or is not the version this
    launcher is for.
    """
    # Import the planner FIRST and check where it came from before importing
    # anything else. A wrong `drivejepa` package on the path may not even have a
    # divider module, and a ModuleNotFoundError is a much worse thing to read
    # at 3am than the refusal below.
    import drivejepa.alpasim_driver.planner as planner

    # realpath, not __file__: the question worth answering in a log is which
    # *source file* ran.
    real_path = Path(os.path.realpath(planner.__file__))

    if real_path.parent != _LIVE_PACKAGE.resolve():
        raise SystemExit(
            f"refusing to serve: planner resolved to {real_path}\n"
            f"expected a file under {_LIVE_PACKAGE.resolve()}\n"
            "Another drivejepa.alpasim_driver package is ahead of this one on sys.path. "
            "The test suite would still be green against this file while the "
            "process served a different one."
        )

    import drivejepa.alpasim_driver.constants as constants
    import drivejepa.alpasim_driver.divider as divider

    divider_path = Path(os.path.realpath(divider.__file__))
    if divider_path.parent != _LIVE_PACKAGE.resolve():
        raise SystemExit(
            f"refusing to serve: divider resolved to {divider_path}\n"
            f"expected a file under {_LIVE_PACKAGE.resolve()}\n"
            "The divider decides which ticks run inference and how a cached "
            "plan is re-anchored. A shadowed copy would be invisible in the "
            "logs and would change every tick's output."
        )

    # The two files that carry the momentum path. bridge.py decides which
    # model class the worker builds; momentum.py is that class. Either one
    # shadowed would change every replan's pick, or silently drop §3.5.
    import drivejepa.alpasim_driver.bridge as bridge

    for label, path in (
        ("bridge", Path(os.path.realpath(bridge.__file__))),
        ("momentum", Path(os.path.realpath(Path(planner.__file__).parent / "momentum.py"))),
    ):
        if path.parent != _LIVE_PACKAGE.resolve() or not path.is_file():
            raise SystemExit(
                f"refusing to serve: {label} resolved to {path}\n"
                f"expected a file under {_LIVE_PACKAGE.resolve()}\n"
                "The momentum path lives in bridge.py (worker model class) "
                "and momentum.py (upstream's previous-plan keys). A shadowed or "
                "missing copy would drive on plain S while claiming §3.5."
            )

    version = getattr(constants, "ADAPTER_VERSION", None)
    if version != EXPECTED_VERSION:
        raise SystemExit(
            f"refusing to serve: resolved ADAPTER_VERSION {version!r}, "
            f"expected {EXPECTED_VERSION!r}.\n"
            f"drivejepa package came from {real_path}\n"
            "Check sys.path ordering."
        )
    return planner, divider, constants, real_path


def log_provenance() -> str:
    """Emit the one startup line that says what is about to drive. Returns it.

    Called by :func:`main` before the port is bound, so it lands in the run's
    ``driver.log``. Kept separate from ``main`` so the entrypoint-resolution
    test can assert on the real thing rather than a re-implementation of it.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s.%(msecs)03d %(levelname)s:\t%(message)s",
        datefmt="%H:%M:%S",
    )
    planner, divider, constants, real_path = _resolve_planner()
    message = (
        f"provenance: adapter_version={constants.ADAPTER_VERSION} "
        f"version_id={planner.DriveJEPAPlanner.version_id} "
        f"planner_module={real_path} "
        f"imported_via={planner.__file__} "
        f"divider_module={Path(os.path.realpath(divider.__file__))} "
        f"replan_period_us={constants.REPLAN_PERIOD_US}"
    )
    logger.info(message)
    # The second provenance line. The switch is resolved by the same
    # function the planner uses, and the vendored model file is hashed from
    # disk so the line names the calibrate_score that will actually run.
    import hashlib

    mode = planner.momentum_mode("auto")
    # The upstream checkout the worker will import (real_model.NAVSIM_V2).
    model_file = (
        Path(os.environ.get("NAVSIM_DEVKIT_ROOT", "NAVSIM_DEVKIT_ROOT-is-not-set")) / "navsim" / "agents"
        / "drive_jepa_perception_based" / "drive_jepa_model.py"
    )
    model_sha = (
        hashlib.sha256(model_file.read_bytes()).hexdigest()
        if model_file.is_file() else "MISSING"
    )
    line2 = (
        f"{constants.MOMENTUM_PROVENANCE} | "
        f"switch={'on' if mode == 'on' else 'OFF (exact 0.1.4 selection)'} | "
        f"previous plan dropped above {constants.MOMENTUM_MAX_AGE_S} s | "
        f"drive_jepa_model.py sha256={model_sha} | "
        f"momentum_module={Path(os.path.realpath(_LIVE_PACKAGE / 'momentum.py'))}"
    )
    logger.info(line2)
    return message + "\n" + line2


def build_parser() -> argparse.ArgumentParser:
    """The CLI."""
    import drivejepa.alpasim_driver.constants as K

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Interface to bind.")
    parser.add_argument(
        "--port",
        type=int,
        default=9149,
        help="Port to bind. Must sit outside the block the AlpaSim wizard hands "
        "out from its baseport (12 ports from 9100 by default).",
    )
    parser.add_argument(
        "--front-camera",
        default=FRONT_CAMERA_ID,
        help="AlpaSim logical_id of the front camera.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the Drive-JEPA worker.",
    )
    parser.add_argument(
        "--image-history",
        type=int,
        default=K.DEFAULT_IMAGE_HISTORY,
        help="Frames kept per camera. Must be long enough to reach one full "
        f"{K.FRAME_SPACING_US} us step back at the scenario's push rate; "
        f"default {K.DEFAULT_IMAGE_HISTORY}.",
    )
    parser.add_argument(
        "--control-timestep-us",
        type=int,
        default=50_000,
        help="The scenario's control_timestep_us, which must equal its "
        "frame_interval_us. Checked against the observed frame push interval "
        "on the first observation that carries two frames; a mismatch fails "
        "the rollout. Nothing in the runtime validates this. 0 disables.",
    )
    parser.add_argument(
        "--replan-period-us",
        type=int,
        default=K.REPLAN_PERIOD_US,
        help="Divider grid. Drive-JEPA's native period; changing it means "
        "asking the model for a cadence it was not trained at.",
    )
    parser.add_argument(
        "--target-frame-spacing-us",
        type=int,
        default=None,
        help=argparse.SUPPRESS,  # accepted only so it can be REFUSED, below
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Build the planner, report readiness, and exit WITHOUT binding a "
        "port. The desk-check that keeps a broken adapter out of the queue.",
    )
    return parser


def build_driver(args: argparse.Namespace):
    """Planner + divider + scaffold. Separated from main() so tests can call it."""
    import drivejepa.alpasim_driver.constants as K
    from drivejepa.alpasim_driver.bridge import BridgedDriveJEPA, BridgeError
    from drivejepa.alpasim_driver.divider import ReplanDivider
    from drivejepa.alpasim_driver.planner import DriveJEPAPlanner

    from paddock.adapters.benchmarks.alpasim.scaffold import DriverScaffold

    # Refused, not ignored: silently dropping a flag someone passed would hand
    # them a run whose frame pairing is not the one they asked for.
    if args.target_frame_spacing_us:
        raise SystemExit(
            "--target-frame-spacing-us is not accepted for drivejepa 0.1.7 "
            f"(got {args.target_frame_spacing_us}).\n"
            "Drive-JEPA's context pair is fixed at its trained spacing, "
            f"{K.FRAME_SPACING_US} us, and the divider replans on that same "
            "grid. The scenario cadence is set by the driver YAML "
            "(control_timestep_us / frame_interval_us), not by this flag."
        )

    span_us = args.image_history * args.control_timestep_us
    if args.control_timestep_us > 0 and span_us <= K.FRAME_SPACING_US:
        raise SystemExit(
            f"--image-history {args.image_history} at a "
            f"{args.control_timestep_us} us control timestep spans only "
            f"{span_us} us, which cannot hold the {K.FRAME_SPACING_US} us "
            "context pair the model needs.\n"
            f"Use at least {K.FRAME_SPACING_US // args.control_timestep_us + 1} "
            "frames. Refusing at startup rather than failing every tick."
        )

    # Two-process bridge: the model lives in the 3.9 conda env because torch 2.1
    # + mmcv-full are cp39 wheels, while this process needs 3.12 for
    # alpasim_utils/utils_rs. Construction blocks until the worker has the model
    # loaded, so the port still binds only when we are ready to drive.
    try:
        # The worker's stderr (which carries the batch-fix line) goes to the
        # run's own folder when the runner names one (DRIVEJEPA_WORKER_LOG).
        model = BridgedDriveJEPA(device=args.device)
    except BridgeError as exc:
        raise SystemExit(
            f"Drive-JEPA bridge could not start: {exc}\n"
            "See tools/models/drivejepa/alpasim_driver/README.md for the environment."
        ) from exc

    # Resolve the switch here, once, and hand the planner the result,
    # so the provenance line, the planner and the log below cannot disagree.
    from drivejepa.alpasim_driver.planner import momentum_mode

    mode = momentum_mode("auto")
    if mode == "off":
        logger.warning(
            "momentum OFF by %s=off: this process selects exactly as 0.1.4",
            K.MOMENTUM_ENV,
        )
    else:
        logger.info("momentum on: previous plan fed to the model at every replan")

    planner = DriveJEPAPlanner(
        model_fn=model,
        front_camera_id=args.front_camera,
        expected_frame_interval_us=(args.control_timestep_us or None),
        momentum=mode,
    )
    divider = ReplanDivider(planner, replan_period_us=args.replan_period_us)
    driver = DriverScaffold(
        planner=divider,
        keep_images=True,
        image_history=args.image_history,
    )
    return divider, driver


def main() -> None:
    log_provenance()
    args = build_parser().parse_args()

    divider, driver = build_driver(args)
    logger.info(
        "built ok: version_id=%s image_history=%d control_timestep_us=%d "
        "replan_period_us=%d",
        divider.version_id,
        args.image_history,
        args.control_timestep_us,
        args.replan_period_us,
    )

    if args.check_only:
        logger.info("--check-only: not binding a port. READY.")
        return

    from paddock.adapters.benchmarks.alpasim.scaffold import serve

    logger.info("serving on %s:%d", args.host, args.port)
    serve(args.host, args.port, driver)


if __name__ == "__main__":
    main()
