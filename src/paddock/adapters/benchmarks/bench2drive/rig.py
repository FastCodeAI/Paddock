"""The camera rigs the Bench2Drive agent can render, by name.

One definition, read by three processes: the agent (inside the CARLA client's
interpreter, so keep the syntax to what Python 3.8 accepts), the harness (which
checks a site's ``camera_rig`` and puts it on the run's record) and
``tools/benchmarks/bench2drive/route_video.py`` (which draws plans onto the front
camera through the rig the route was driven with: the agent writes it beside the
route's record as :data:`RECORD`).

``paddock`` is the default and the rig a ranked run uses: every model on this
benchmark is compared through the same cameras. The others ask how much a model's
drive depends on its cameras, and a run with one is ``valid: false``:

* ``bench2drive``: Bench2Drive's own rig, the one Think2Drive's dataset was recorded
  with (``tools/data_collect.py`` at 7ec25d1) -- the six nuScenes cameras at
  nuScenes' 1600x900, where paddock renders 800x450 for speed;
* ``kitscenes``: KITScenes' roof pod as paddock's calibration notes record it
  (docs/benchmarks/kitscenes.md): the 88 x 54 deg base front camera ~1.8 m over the
  road and the ring's 87 x 63 deg side and rear cameras ~2.0 m over it, each at half
  its native size or more. Where the pod sits on the car is not recorded; here it is
  over the ego origin. Its front differs in size from its ring, so it suits a model
  that packs each view on its own (AutoE2E) and not one that stacks the raw frames.

A camera is its placement in the CARLA ego frame -- ``pose`` (x, y, z, roll, pitch,
yaw): x forward, y RIGHT, z up, metres and degrees, as ``sensors()`` declares it --
its horizontal field of view ``fov`` in degrees and its ``size`` (width, height) in
pixels. A rig lists only the cameras it moves; every other camera is paddock's.
"""

from __future__ import annotations

DEFAULT = "paddock"

#: Beside a route's record: the rig's name and every camera the agent rendered (``agent.py``).
RECORD = "rig.json"

#: The leaderboard's limit on a sensor's distance from the ego origin, metres.
MAX_RADIUS_M = 3.0


def _camera(x, y, z, yaw, fov, width, height, pitch=0.0):
    return {"pose": [x, y, z, 0.0, pitch, yaw], "fov": fov, "size": [width, height]}


_PADDOCK = {
    # The six nuScenes cameras. One size for all six: model feature builders np.stack
    # the frames, so unequal sizes raise. Half nuScenes' 1600x900 is a big render and
    # payload win, and the models crop or resize the frames downstream anyway.
    "CAM_FRONT": _camera(1.5, 0.0, 1.5, 0.0, 70.0, 800, 450),
    "CAM_FRONT_LEFT": _camera(1.0, -0.5, 1.5, -55.0, 70.0, 800, 450),
    "CAM_FRONT_RIGHT": _camera(1.0, 0.5, 1.5, 55.0, 70.0, 800, 450),
    "CAM_BACK": _camera(-1.5, 0.0, 1.5, 180.0, 110.0, 800, 450),
    "CAM_BACK_LEFT": _camera(0.0, -0.6, 1.5, -110.0, 70.0, 800, 450),
    "CAM_BACK_RIGHT": _camera(0.0, 0.6, 1.5, 110.0, 70.0, 800, 450),
    # openpilot's comma-three rig, reproduced: a narrow road camera (1928x1208, focal
    # 2648 -> 40 deg) and a wide camera (1928x1208, focal 567 -> ~119 deg, modelled as a
    # pinhole, as openpilot does), both at the windshield mount (~1.3 m). Each feeds its
    # own model slot (img / big_img) instead of both views being cut from one 800x450 frame.
    "CAM_OP_ROAD": _camera(1.0, 0.0, 1.3, 0.0, 40.0, 1928, 1208),
    "CAM_OP_WIDE": _camera(1.0, 0.0, 1.3, 0.0, 119.0, 1928, 1208),
    # SimLingo's one camera, as its agent mounts it (team_code/config_simlingo.py at 743b243:
    # camera_pos_0 [-1.5, 0.0, 2.0], rot [0, 0, 0], 1024x512, fov 110). Trained on it alone; the
    # 70 deg CAM_FRONT would show it a narrower, magnified view. Rendered only for a model that
    # declares it, like openpilot's pair.
    "CAM_SIMLINGO": _camera(-1.5, 0.0, 2.0, 0.0, 110.0, 1024, 512),
}

_BENCH2DRIVE = {
    "CAM_FRONT": _camera(0.80, 0.0, 1.60, 0.0, 70.0, 1600, 900),
    "CAM_FRONT_LEFT": _camera(0.27, -0.55, 1.60, -55.0, 70.0, 1600, 900),
    "CAM_FRONT_RIGHT": _camera(0.27, 0.55, 1.60, 55.0, 70.0, 1600, 900),
    "CAM_BACK": _camera(-2.0, 0.0, 1.60, 180.0, 110.0, 1600, 900),
    "CAM_BACK_LEFT": _camera(-0.32, -0.55, 1.60, -110.0, 70.0, 1600, 900),
    "CAM_BACK_RIGHT": _camera(-0.32, 0.55, 1.60, 110.0, 70.0, 1600, 900),
}

_KITSCENES = {
    # camera_base_front_center: 5856x3104 at f 3022 px (88 x 54 deg), 0.42 m ahead of
    # the pod reference and 0.2 m below the ring; rendered at 0.35x, the same aspect.
    "CAM_FRONT": _camera(0.42, 0.0, 1.8, 0.0, 88.0, 2048, 1086),
    # The ring: 3504x2272 at f ~1844 px (87 x 63 deg) on a 0.21 m radius, rendered at
    # half size. Yaws as calib.json measures them (61 left, 58 right, 179), the rear
    # pair at their nominal 120; CARLA's yaw is positive to the right.
    "CAM_FRONT_LEFT": _camera(0.105, -0.18, 2.0, -61.0, 87.0, 1752, 1136),
    "CAM_FRONT_RIGHT": _camera(0.105, 0.18, 2.0, 58.0, 87.0, 1752, 1136),
    "CAM_BACK": _camera(-0.21, 0.0, 2.0, 179.0, 87.0, 1752, 1136),
    "CAM_BACK_LEFT": _camera(-0.105, -0.18, 2.0, -120.0, 87.0, 1752, 1136),
    "CAM_BACK_RIGHT": _camera(-0.105, 0.18, 2.0, 120.0, 87.0, 1752, 1136),
}

RIGS = {DEFAULT: _PADDOCK, "bench2drive": _BENCH2DRIVE, "kitscenes": _KITSCENES}


def cameras(name: str) -> dict:
    """Every camera the agent can render, placed as rig ``name`` places it.

    Raises KeyError naming the rigs there are: a typo in a site's ``camera_rig``
    must fail before a route is driven, not drive it through paddock's cameras.
    """
    if name not in RIGS:
        raise KeyError(f"no camera rig {name!r}; the rigs are {', '.join(sorted(RIGS))}")
    out = {cid: dict(cam) for cid, cam in _PADDOCK.items()}
    out.update({cid: dict(cam) for cid, cam in RIGS[name].items()})
    return out
