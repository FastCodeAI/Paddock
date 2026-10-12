"""The re-ranker's features and its weights file: paddock's half of the contract.

Drive-JEPA proposes 32 trajectories and a scorer trained on NAVSIM's PDM score
picks one. Off NAVSIM that scorer ranks them close to random: on KITScenes the
best of the 32 is 0.37 m ADE and the pick 2.36 m. A re-ranker replaces only the
choice. It scores each candidate from what the model already produced -- its
six scorer logits and the trajectory's shape -- plus ego speed, and the
highest score is driven.

Gallops trains it (`gallops.train.rerank`) and writes `reranker.npz` beside a
`checkpoint.json`. Neither repo imports the other, so the file carries what
each side must agree on and this module checks it at load:

- ``feature_names`` must equal `FEATURE_NAMES` here, in order. A mismatch means
  the weights were fitted to a different definition.
- ``probe_x`` / ``probe_scores`` are inputs and the scores gallops's own forward
  pass gave them. `load` recomputes them with `Reranker.score`; a difference
  means the two forward passes disagree, and the file is refused.

Gallops builds the training data, so it computes the features with its own
copy of `candidate_features` (`gallops.data.rerank`). The probe cannot catch a
difference there, since it replays stored features. What holds the copies
equal is a test on each side against the same input and expected values:
`test_features_match_gallops` here, `GOLDEN` in gallops'
tests/test_rerank_data.py. Change one, change both.

numpy only, and Python 3.9-compatible: this runs inside Drive-JEPA's own env.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: The weights-file format this reader understands. Gallops writes the same
#: string; bump both together if a key changes meaning.
FORMAT = "paddock-listwise-mlp/1"

#: Waypoints the features look at: 3 s at 0.5 s, the horizon the open-loop
#: benchmarks score.
HORIZON = 6

FEATURE_NAMES = (
    "logit_0",
    "logit_1",
    "logit_2",
    "logit_3",
    "logit_4",
    "logit_5",
    "travel_1s",
    "travel_2s",
    "travel_3s",
    "lateral_3s",
    "heading_3s",
    "max_abs_lateral",
    "step_mean",
    "step_min",
    "step_max",
    "step_std",
    "total_turning",
    "ego_speed",
)


def candidate_features(proposals: np.ndarray, logits: np.ndarray, speed: float) -> np.ndarray:
    """(K, F) features for K candidates, in `FEATURE_NAMES` order.

    Nothing here looks at the logged future -- that is the training label. Ego
    speed is in on purpose even though "pick the candidate closest to constant
    velocity" is a shortcut it enables: the reports compare against exactly that
    rule, so a re-ranker that collapsed to it would show.
    """
    proposals = np.asarray(proposals, dtype=np.float64)
    xy = proposals[:, :HORIZON, :2]
    heading = proposals[:, :HORIZON, 2]
    path = np.concatenate([np.zeros((len(xy), 1, 2)), xy], axis=1)
    steps = np.linalg.norm(np.diff(path, axis=1), axis=2)  # metres per 0.5 s
    dist = np.linalg.norm(xy, axis=2)  # straight-line distance from ego

    features = np.column_stack(
        [
            np.asarray(logits, dtype=np.float64),
            dist[:, 1],
            dist[:, 3],
            dist[:, 5],
            xy[:, 5, 1],
            heading[:, 5],
            np.abs(xy[:, :, 1]).max(axis=1),
            steps.mean(axis=1),
            steps.min(axis=1),
            steps.max(axis=1),
            steps.std(axis=1),
            np.abs(np.diff(heading, axis=1)).sum(axis=1),
            np.full(len(xy), float(speed)),
        ]
    )
    if features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"built {features.shape[1]} features, FEATURE_NAMES lists {len(FEATURE_NAMES)}")
    return features


class RerankerError(Exception):
    """A weights file this reader cannot trust."""


class Reranker:
    """A small MLP over standardised features: relu(z @ w1 + b1) @ w2 + b2.

    ``w1``/``b1`` are absent for a linear ranker.
    """

    def __init__(self, arrays: dict[str, np.ndarray], sidecar: dict | None = None) -> None:
        #: The gallops checkpoint.json beside the weights: what they were trained
        #: and selected on. `load` requires it; the train/test guard in
        #: `tools/models/drivejepa/rerank.py report` and the run record read it.
        self.sidecar = sidecar or {}
        self.mean = arrays["mean"]
        self.std = arrays["std"]
        self.w1 = arrays.get("w1")
        self.b1 = arrays.get("b1")
        self.w2 = arrays["w2"]
        self.b2 = arrays["b2"]

    def score(self, features: np.ndarray) -> np.ndarray:
        """Features (..., F) -> one score per candidate (...)."""
        z = (np.asarray(features, dtype=np.float64) - self.mean) / self.std
        if self.w1 is not None:
            z = np.maximum(z @ self.w1 + self.b1, 0.0)
        return (z @ self.w2 + self.b2)[..., 0]

    def choose(self, proposals: np.ndarray, logits: np.ndarray, speed: float) -> int:
        """Index of the candidate to drive."""
        return int(np.argmax(self.score(candidate_features(proposals, logits, speed))))


def load(path: Path | str) -> Reranker:
    """Read and verify a `reranker.npz` and its `checkpoint.json`.

    Raises `RerankerError` naming the file. The sidecar is required: without it
    nothing says what the weights were trained on, and a guard that refuses to
    score on training data would silently have nothing to check against.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "reranker.npz"
    if not path.exists():
        raise RerankerError(f"no re-ranker weights at {path}")
    sidecar_path = path.parent / "checkpoint.json"
    if not sidecar_path.exists():
        raise RerankerError(
            f"no checkpoint.json beside {path}; copy the whole checkpoint directory, "
            "since the sidecar records what these weights were trained on"
        )
    sidecar = json.loads(sidecar_path.read_text())
    with np.load(path, allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}

    found = str(arrays.get("format", ""))
    if found != FORMAT:
        raise RerankerError(f"{path}: format {found!r}, this reader understands {FORMAT!r}")
    names = tuple(str(n) for n in arrays["feature_names"])
    if names != FEATURE_NAMES:
        raise RerankerError(
            f"{path}: trained on features {names}, but candidate_features builds {FEATURE_NAMES}"
        )

    reranker = Reranker(arrays, sidecar)
    replayed = reranker.score(arrays["probe_x"])
    if not np.allclose(replayed, arrays["probe_scores"], atol=1e-6, rtol=1e-6):
        worst = float(np.abs(replayed - arrays["probe_scores"]).max())
        raise RerankerError(
            f"{path}: this forward pass disagrees with the trainer's on its own probe "
            f"(max |diff| {worst:.2e}); the two implementations have drifted"
        )
    return reranker
