"""Build the environment a model subprocess runs in.

Each model has its own interpreter, its own dependencies and its own idea of
where data lives. `model.yaml` declares that shape; this turns it into a real
environment, resolving template variables against the site's `paddock.toml`.

Deliberately *not* inherited wholesale from paddock's own environment: a stray
PYTHONPATH or CONDA_PREFIX leaking into a model's subprocess is a genuinely
confusing failure, because the model then half-imports paddock's dependencies.
Only what is named here is passed through.
"""

from __future__ import annotations

import os
from pathlib import Path

from paddock.config import REPO_ROOT, SiteConfig
from paddock.registry import ModelSpec

#: Host variables a subprocess needs regardless of model.
PASSTHROUGH = (
    "PATH",
    "HOME",
    "USER",
    "LANG",
    "LC_ALL",
    "TERM",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "LD_LIBRARY_PATH",
    "HF_HOME",
    "HF_TOKEN",
    "TORCH_HOME",
    "TMPDIR",
)


def build(spec: ModelSpec, cfg: SiteConfig, extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment for one model's node process."""
    context = spec.context(cfg)
    env = {k: os.environ[k] for k in PASSTHROUGH if k in os.environ}

    # paddock's own source, so `python -m paddock.policy.server` resolves inside
    # the model's interpreter. Paddock is not installed into model environments:
    # they pin their own dependency sets and must not be disturbed.
    pythonpath = [str(REPO_ROOT / "src")]
    pythonpath += [cfg.resolve(p, **context) for p in spec.runtime.pythonpath]
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)

    for key, value in spec.runtime.env.items():
        env[key] = cfg.resolve(value, **context)

    # Site-local overrides win over the model's declared defaults. `paddock.toml`
    # already parsed these into ModelOverride.env and nothing applied them, so an
    # `[models.<name>] env` block was silently doing nothing -- the worst kind of
    # configuration, because it looks like it took effect.
    override = cfg.models.get(spec.name)
    if override is not None:
        for key, value in override.env.items():
            env[key] = cfg.resolve(value, **context)

    # Paths a model's own setup.sh may want, named consistently.
    env.setdefault("PADDOCK_REPO", str(cfg.repo(spec.repo)))
    env.setdefault("PADDOCK_WEIGHTS", str(cfg.weights))
    env.setdefault("PADDOCK_RUNS", str(cfg.runs))
    env["PADDOCK_SITE"] = cfg.site

    if extra:
        env.update(extra)
    return env


def working_dir(spec: ModelSpec, cfg: SiteConfig) -> Path:
    """Where the process starts.

    Load-bearing for several repos: Drive-JEPA's agent does
    `np.load("./data/8192.npy")`, and VisionPilot resolves its ONNX weights
    relative to the working directory.
    """
    if spec.runtime.cwd:
        return Path(cfg.resolve(spec.runtime.cwd, **spec.context(cfg)))
    return cfg.repo(spec.repo)
