"""route_video draws onto the front camera through the rig the route was driven with."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("cv2")  # route_video draws with OpenCV: the [analysis] extra

from paddock.adapters.benchmarks.bench2drive import rig  # noqa: E402
from tools.benchmarks.bench2drive import route_video  # noqa: E402


def test_a_route_recorded_before_the_rig_was_is_drawn_through_paddocks(tmp_path) -> None:
    assert route_video.front_camera(tmp_path) == route_video.FrontCamera(800, 450, 70.0, 1.5, 1.5)


def test_a_route_driven_on_another_rig_is_drawn_through_it(tmp_path) -> None:
    front = rig.cameras("bench2drive")["CAM_FRONT"]
    (tmp_path / rig.RECORD).write_text(json.dumps({"name": "bench2drive", "cameras": {"CAM_FRONT": front}}))
    cam = route_video.front_camera(tmp_path)
    assert cam == route_video.FrontCamera(1600, 900, 70.0, 0.8, 1.6)
    ((u, v),) = route_video.front_project(np.array([[20.0, 0.0]]), cam)
    assert u == pytest.approx(800.0), "straight ahead is the image's centre column"
    f = 800.0 / np.tan(np.radians(35.0))
    assert v == pytest.approx(450.0 + f * 1.6 / (20.0 - 0.8)), "below the horizon by the mount's height"
