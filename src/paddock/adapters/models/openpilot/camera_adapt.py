"""Map a KITScenes ring camera onto the view openpilot's supercombo expects.

Unlike the Alpamayo adaptation, this one is geometrically exact. The network never
sees a raw camera -- it sees two warped 512x256 views:

    img      (medmodel  frame)  31.4 deg H x 15.9 deg V
    big_img  (sbigmodel frame)  58.7 deg H x 31.3 deg V

Both fit *inside* KITScenes' 87 x 63 deg source, so no field of view has to be
invented and there is no fill/letterbox compromise. A single `camera_ring_front`
frame feeds both slots, warped twice.

Two constraints are load-bearing:

*Resolution is not free.* ModelState does ``self.warp = jits[(cam_w, cam_h)]`` and
only (1928,1208) and (1344,760) are compiled, so the source must be resized to one
of them and the intrinsics scaled to match.

*The residual mismatch is distortion, not FOV.* comma's wide camera is a fisheye
that openpilot models as a pinhole (the source admits "magnification is not
consistent across frame"), so `big_img` was trained on distorted pixels that a
rectified KITScenes pinhole does not have. Second order, but real, and it affects
only the wide slot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openpilot.common.transformations.camera import view_frame_from_device_frame
from openpilot.common.transformations.model import MEDMODEL_INPUT_SIZE, SBIGMODEL_INPUT_SIZE, get_warp_matrix
from openpilot.common.transformations.orientation import rot2euler
from PIL import Image

# the tici geometry ModelState has a compiled warp for
TARGET_W, TARGET_H = 1928, 1208
TARGET_ASPECT = TARGET_W / TARGET_H

# KITScenes `reference` (base_frame == lidar_top) is x-forward, y-left, z-up;
# openpilot's device frame is x-forward, y-right, z-down.
DEVICE_FROM_REFERENCE = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class AdaptedCamera:
    """A KITScenes camera expressed the way openpilot wants it."""

    intrinsics: np.ndarray  # 3x3 for the resized TARGET_W x TARGET_H frame
    rpy: np.ndarray  # device_from_calib euler, openpilot's rpyCalib analogue
    crop: tuple[int, int, int, int]  # x0, y0, x1, y1 in source pixels
    src_size: tuple[int, int]

    def warp_matrix(self, bigmodel_frame: bool) -> np.ndarray:
        return get_warp_matrix(self.rpy, self.intrinsics, bigmodel_frame)

    def prepare(self, img: Image.Image) -> Image.Image:
        """Source frame -> the TARGET_W x TARGET_H frame `intrinsics` describes."""
        return img.crop(self.crop).resize((TARGET_W, TARGET_H), Image.BILINEAR)


def load_camera(scene: Path, sensor: str = "camera_ring_front") -> AdaptedCamera:
    calib = json.loads((scene / "calibration" / "calib.json").read_text())[sensor]
    W = int(calib["resolution"]["width"])
    H = int(calib["resolution"]["height"])
    K = calib["intrinsics"]
    f, cu, cv = float(K["focal_length"]), float(K["principal_point_u"]), float(K["principal_point_v"])

    # Crop to the target aspect, anchored on the PRINCIPAL POINT (not the image
    # centre -- they differ by 3 px vertically here, and it is free to get right).
    if W / H > TARGET_ASPECT:  # source too wide -> trim width
        cw, ch = int(round(H * TARGET_ASPECT)), H
    else:  # source too tall -> trim height
        cw, ch = W, int(round(W / TARGET_ASPECT))
    x0 = int(round(np.clip(cu - cw / 2, 0, W - cw)))
    y0 = int(round(np.clip(cv - ch / 2, 0, H - ch)))
    crop = (x0, y0, x0 + cw, y0 + ch)

    s = TARGET_W / cw
    intrinsics = np.array([[f * s, 0.0, (cu - x0) * s], [0.0, f * s, (cv - y0) * s], [0.0, 0.0, 1.0]])

    # Mounting rotation. Columns of T_to_reference[:3,:3] are the camera's optical
    # axes in the reference frame; openpilot wants the residual rotation that makes
    #   view_frame_from_device_frame @ rot_from_euler(rpy) == R_cam_from_device
    R_ref_cam = np.array(calib["T_to_reference"])[:3, :3]
    R_dev_cam = DEVICE_FROM_REFERENCE @ R_ref_cam
    R_cam_dev = R_dev_cam.T
    rpy = rot2euler(view_frame_from_device_frame.T @ R_cam_dev)

    return AdaptedCamera(
        intrinsics=intrinsics, rpy=np.asarray(rpy, dtype=np.float32), crop=crop, src_size=(W, H)
    )


def warp_to_model_frame(frame: np.ndarray, M: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour perspective warp, matching openpilot's own sampler.

    `M` maps model-frame coords -> source pixels (it is used as M_inv in the JIT).
    """
    w, h = size
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(w * h)])
    src = M @ pts
    su = np.round(src[0] / src[2]).astype(int)
    sv = np.round(src[1] / src[2]).astype(int)
    ok = (su >= 0) & (su < frame.shape[1]) & (sv >= 0) & (sv < frame.shape[0]) & (src[2] > 0)
    out = np.zeros((h * w,) + frame.shape[2:], dtype=frame.dtype)
    out[ok] = frame[sv[ok], su[ok]]
    return out.reshape((h, w) + frame.shape[2:])


MODEL_VIEWS = {"img": (MEDMODEL_INPUT_SIZE, False), "big_img": (SBIGMODEL_INPUT_SIZE, True)}
