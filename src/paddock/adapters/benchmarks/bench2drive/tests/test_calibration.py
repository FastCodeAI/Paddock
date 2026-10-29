"""The calibration the CARLA client sends is each camera's mount, in paddock's frames.

A multi-camera model lifts what it sees through `camera2ego` and `camera2image`
(METEOR's BEV grid, AutoE2E's projections), so a sign or axis slip here moves what
every such model is shown without failing anything. The agent builds both in
`setup()` from CARLA's own transform; this checks them against the conversion
written out one frame change at a time:

    camera2ego = FLIP_Y . M . CV_TO_CARLA

M is the mount as `carla.Transform.get_matrix()` gives it (Bench2Drive writes the
same formula out as `tools/utils.py:get_matrix` at 7ec25d1), CV_TO_CARLA takes
OpenCV camera axes (x right, y down, z forward) to CARLA's (x forward, y right,
z up), and FLIP_Y takes CARLA's left-handed ego frame (y right) to paddock's
(y left). `camera2image` is the pinhole of the rendered view, Bench2Drive's
`tools/utils.py:build_projection_matrix`. And whatever the conventions, a point on
the road to the ego's left has to land left of centre in the front camera, and in
front of the left cameras and behind the right ones.

carla is stood in for (conftest.py), and so is the HTTP session: `setup()` POSTs
/reset to the node, and on a shared machine one may be listening on the default port.
Only the agent's own code is real.
"""

from __future__ import annotations

import math
import types

import numpy as np
import pytest

from paddock.adapters.benchmarks.bench2drive import rig
from paddock.conformance import check_camera_mounting

FLIP_Y = np.diag([1.0, -1.0, 1.0, 1.0])
CV_TO_CARLA = np.array(
    [[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
)
#: What setup() reads besides the cameras. Unset, it records nothing and stops nothing.
ROUTE_ENV = (
    "PADDOCK_RECORD_DIR",
    "PADDOCK_RECORD_INPUTS",
    "PADDOCK_METRICS_DIR",
    "PADDOCK_STOP_STUCK_S",
    "PADDOCK_STOP_AFTER_S",
    "PADDOCK_STOP_ON_COLLISION",
    "PADDOCK_STOP_REQUEST",
    "PADDOCK_STOP_RECORD",
)


def carla_matrix(x, y, z, roll, pitch, yaw) -> np.ndarray:
    """CARLA's local -> parent transform: degrees, left-handed, as get_matrix() builds it."""
    cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    cr, sr = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    cp, sp = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
    return np.array(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x],
            [cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y],
            [sp, -cp * sr, cp * cr, z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


class _Location:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class _Transform:
    def __init__(self, location, rotation):
        self.location, self.rotation = location, rotation

    def get_matrix(self):
        loc, rot = self.location, self.rotation
        return carla_matrix(loc.x, loc.y, loc.z, rot.roll, rot.pitch, rot.yaw).tolist()


class _Session:
    """A requests.Session that reaches no node."""

    def post(self, *args, **kwargs):
        return types.SimpleNamespace(status_code=200)

    def close(self):
        pass


@pytest.fixture
def sent(load_agent, monkeypatch):
    """``sent(rig_name)``: every camera of that rig, and the camera2ego and camera2image setup() builds."""

    def build(name: str):
        for key in ROUTE_ENV:
            monkeypatch.delenv(key, raising=False)
        cameras = rig.cameras(name)
        agent = load_agent(PADDOCK_CAMERAS=",".join(cameras), PADDOCK_CAMERA_RIG=name)
        for attr, stand_in in (("Transform", _Transform), ("Location", _Location), ("Rotation", _Rotation)):
            monkeypatch.setattr(agent.carla, attr, stand_in, raising=False)
        monkeypatch.setattr(agent, "requests", types.SimpleNamespace(Session=_Session))
        client = agent.PaddockB2DAgent()
        client.setup("")
        cam2ego = {cid: np.array(m) for cid, m in client._cam2ego.items()}
        cam2img = {cid: np.array(k) for cid, k in client._cam2img.items()}
        return cameras, cam2ego, cam2img

    return build


@pytest.mark.parametrize("name", sorted(rig.RIGS))
def test_camera2ego_is_carlas_mount_in_paddocks_frames(sent, name) -> None:
    cameras, cam2ego, _ = sent(name)
    assert set(cam2ego) == set(cameras)
    for cid, cam in cameras.items():
        want = FLIP_Y @ carla_matrix(*cam["pose"][:3], *cam["pose"][3:]) @ CV_TO_CARLA
        np.testing.assert_allclose(cam2ego[cid], want, atol=1e-12, err_msg=f"{name} {cid}")
        finding = check_camera_mounting(cam2ego[cid], cid)
        assert finding.ok, finding.detail


@pytest.mark.parametrize("name", sorted(rig.RIGS))
def test_camera2image_is_the_pinhole_of_the_rendered_view(sent, name) -> None:
    cameras, _, cam2img = sent(name)
    for cid, cam in cameras.items():
        w, h = cam["size"]
        f = w / (2.0 * math.tan(math.radians(cam["fov"]) / 2.0))
        want = [[f, 0.0, w / 2], [0.0, f, h / 2], [0.0, 0.0, 1.0]]
        np.testing.assert_allclose(cam2img[cid], want, rtol=1e-12)


def test_a_road_point_on_the_egos_left_is_seen_left_of_centre(sent) -> None:
    """10 m ahead and 2 m to the left, on the road: in front of CAM_FRONT, left of its centre
    and below its horizon (it lands near u 266, v 326 of 800x450), and behind CAM_BACK."""
    cameras, cam2ego, cam2img = sent(rig.DEFAULT)
    point = np.array([10.0, 2.0, 0.0, 1.0])
    x, y, depth = (np.linalg.inv(cam2ego["CAM_FRONT"]) @ point)[:3]
    u, v = (cam2img["CAM_FRONT"] @ np.array([x / depth, y / depth, 1.0]))[:2]
    w, h = cameras["CAM_FRONT"]["size"]
    assert depth > 0 and u < w / 2 and v > h / 2, (depth, u, v)
    assert (np.linalg.inv(cam2ego["CAM_BACK"]) @ point)[2] < 0


@pytest.mark.parametrize("name", sorted(rig.RIGS))
def test_a_road_point_on_the_egos_left_is_in_front_of_the_left_cameras(sent, name) -> None:
    """5 m to the left, on the road: in front of both left cameras and behind both right ones."""
    # CAM_FRONT and CAM_BACK sit on the centre line, where the y-flip moves nothing, so the test
    # above passes without it; a side camera is where a left/right slip shows.
    _, cam2ego, _ = sent(name)
    point = np.array([0.0, 5.0, 0.0, 1.0])
    for left, right in (("CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"), ("CAM_BACK_LEFT", "CAM_BACK_RIGHT")):
        assert (np.linalg.inv(cam2ego[left]) @ point)[2] > 0, f"{name}: behind {left}"
        assert (np.linalg.inv(cam2ego[right]) @ point)[2] < 0, f"{name}: in front of {right}"
