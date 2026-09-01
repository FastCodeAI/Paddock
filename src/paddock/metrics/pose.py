"""Pose transforms shared by every model and benchmark.

`to_relative_poses` defines the frame the KITScenes ground truth is expressed
in, so every model must use exactly this one: a second implementation that
differs by a sign convention would move every model's score without any of them
changing.

It lives here rather than beside one model because it was previously reached by
extracting its source with `ast` and `exec` -- the open-loop openpilot harness
needed it, but importing the module it lived in would have pulled in torch,
which openpilot's environment deliberately does not have. Being pure numpy, it
just needed a torch-free home.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["to_relative_poses"]


def to_relative_poses(
    origin: npt.NDArray[np.float64], poses: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Express absolute (x, y, yaw) poses in the frame of *origin*.

    Byte-for-byte the Drive-JEPA function, because the ground truth it produces
    is what every model is scored against.
    """
    cos_o, sin_o = np.cos(origin[2]), np.sin(origin[2])
    delta = poses[:, :2] - origin[:2]
    rotated = np.stack(
        [cos_o * delta[:, 0] + sin_o * delta[:, 1], -sin_o * delta[:, 0] + cos_o * delta[:, 1]],
        axis=-1,
    )
    heading = np.arctan2(np.sin(poses[:, 2] - origin[2]), np.cos(poses[:, 2] - origin[2]))
    return np.concatenate([rotated, heading[:, None]], axis=-1)
