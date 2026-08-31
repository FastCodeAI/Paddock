"""Map KITScenes ring cameras onto the slots Alpamayo 1.5 expects.

Alpamayo's canonical ring is seven cameras; the trajectory task uses four of them:
``cross_left_120fov`` (0), ``front_wide_120fov`` (1), ``cross_right_120fov`` (2) and
``front_tele_30fov`` (6).  KITScenes ships six ring cameras, each a 3504x2272 pinhole
at ~87 deg hFOV.  Reconciling the two is a cropping problem with one asymmetry that
decides the whole design:

* The **telephoto** slot wants 30 deg, *narrower* than the 87 deg source, so a centre
  crop reproduces the training intrinsics exactly.  That slot is geometrically perfect.
* The **120 deg** slots want *wider* than the source.  No crop can invent field of
  view, so something has to give, and the two honest options trade off against each
  other:

  ``fill``       use the whole frame.  Image statistics and resolution are preserved
                 and there are no black pixels, but objects subtend 1.82x the angle
                 the model expects (focal/width 0.526 vs 0.289), biasing depth.
  ``letterbox``  rescale so a given bearing lands on the pixel the model expects, and
                 pad the rest with black.  Projectively faithful, but ~70% of the
                 frame carries no signal, which is its own kind of off-distribution.

``fill`` is the default: it is what a naive deployment would do, and a black border is
the more violent distribution shift.  ``letterbox`` is kept as a labelled ablation.

Every crop is anchored on the **principal point**, not the image centre -- for
``camera_ring_front`` those differ by 11.5 px.  Anchoring on the optical centre is what
makes the wide and telephoto views mutually consistent, and it is free to get right.

Only the 16:9 aspect ratio is load-bearing downstream: the Qwen3-VL processor resizes
to 320x576 regardless of what it is handed (``smart_resize(1080, 1920, factor=32,
max_pixels=196608)`` lands exactly there).  The 1920x1080 intermediate is a convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

TARGET_W, TARGET_H = 1920, 1080
TARGET_ASPECT = 9.0 / 16.0

# Alpamayo's nominal geometry for the two slot kinds, used by ``letterbox``.
WIDE_HFOV_DEG = 120.0
TELE_HFOV_DEG = 30.0


@dataclass(frozen=True)
class Pinhole:
    """Source camera intrinsics, in original image pixels."""

    focal: float
    cu: float
    cv: float
    width: int
    height: int

    @classmethod
    def from_calib(cls, entry: dict, width: int, height: int) -> "Pinhole":
        intrinsics = entry["intrinsics"]
        return cls(
            focal=float(intrinsics["focal_length"]),
            cu=float(intrinsics["principal_point_u"]),
            cv=float(intrinsics["principal_point_v"]),
            width=int(width),
            height=int(height),
        )

    def hfov_deg(self) -> float:
        """Horizontal field of view of the full frame."""
        return 2.0 * math.degrees(math.atan(0.5 * self.width / self.focal))


def crop_box(src: Pinhole, target_hfov_deg: float | None = None) -> tuple[int, int, int, int]:
    """A 16:9 crop window centred on the principal point.

    ``target_hfov_deg=None`` takes the widest symmetric window the sensor allows
    (the ``fill`` mode); a value narrows it to that exact field of view (``zoom``).
    Returns ``(left, top, right, bottom)`` with the width/height ratio exactly 16:9
    after integer rounding, which keeps the principal point at the output centre.
    """
    if target_hfov_deg is None:
        half_w = min(src.cu, src.width - 1 - src.cu)
    else:
        half_w = src.focal * math.tan(math.radians(0.5 * target_hfov_deg))

    width = int(round(2.0 * half_w))
    height = int(round(width * TARGET_ASPECT))
    left = int(round(src.cu - 0.5 * width))
    top = int(round(src.cv - 0.5 * height))
    box = (left, top, left + width, top + height)

    if box[0] < 0 or box[1] < 0 or box[2] > src.width or box[3] > src.height:
        raise ValueError(
            f"crop {box} escapes the {src.width}x{src.height} frame "
            f"(principal point {src.cu},{src.cv}, target hFOV {target_hfov_deg})"
        )
    return box


def crop_hfov_deg(src: Pinhole, box: tuple[int, int, int, int]) -> float:
    """Field of view actually delivered by *box* -- what the model really sees."""
    return 2.0 * math.degrees(math.atan(0.5 * (box[2] - box[0]) / src.focal))


def principal_point_in_output(src: Pinhole, box: tuple[int, int, int, int]) -> tuple[float, float]:
    """Where the optical centre lands in the 1920x1080 output, for assertions."""
    left, top, right, bottom = box
    return (
        (src.cu - left) * TARGET_W / (right - left),
        (src.cv - top) * TARGET_H / (bottom - top),
    )


def _to_array(image: Image.Image) -> npt.NDArray[np.uint8]:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def warp_fill(image: Image.Image, src: Pinhole) -> npt.NDArray[np.uint8]:
    """Widest symmetric 16:9 crop, rescaled to 1920x1080."""
    box = crop_box(src, None)
    return _to_array(image.crop(box).resize((TARGET_W, TARGET_H), Image.BICUBIC))


def warp_zoom(
    image: Image.Image, src: Pinhole, hfov_deg: float = TELE_HFOV_DEG
) -> npt.NDArray[np.uint8]:
    """Digital telephoto: an exact *hfov_deg* centre crop, rescaled to 1920x1080.

    Because the target is narrower than the source this reproduces the telephoto
    intrinsics the model was trained on, with no approximation beyond resampling.
    """
    box = crop_box(src, hfov_deg)
    return _to_array(image.crop(box).resize((TARGET_W, TARGET_H), Image.BICUBIC))


def warp_letterbox(
    image: Image.Image, src: Pinhole, hfov_deg: float = WIDE_HFOV_DEG
) -> npt.NDArray[np.uint8]:
    """Preserve projective geometry at the cost of a black border.

    Scales the fill crop so its focal length matches a *hfov_deg* camera at
    1920 px wide, then centres it on a black canvas.  A bearing then lands on the
    pixel Alpamayo expects; the frame is simply missing outside +/- 43.4 deg.
    """
    box = crop_box(src, None)
    target_focal = 0.5 * TARGET_W / math.tan(math.radians(0.5 * hfov_deg))
    scale = target_focal / src.focal

    width = max(1, int(round((box[2] - box[0]) * scale)))
    height = max(1, int(round((box[3] - box[1]) * scale)))
    if width > TARGET_W or height > TARGET_H:
        raise ValueError(f"letterbox {width}x{height} exceeds the {TARGET_W}x{TARGET_H} canvas")

    inner = image.crop(box).resize((width, height), Image.BICUBIC)
    canvas = Image.new("RGB", (TARGET_W, TARGET_H), (0, 0, 0))
    canvas.paste(inner, ((TARGET_W - width) // 2, (TARGET_H - height) // 2))
    return _to_array(canvas)


WARPS = {"fill": warp_fill, "zoom": warp_zoom, "letterbox": warp_letterbox}


def load_and_warp(path: Path, src: Pinhole, mode: str) -> npt.NDArray[np.uint8]:
    """Read one JPEG and apply *mode*, returning a (1080, 1920, 3) uint8 array."""
    with Image.open(path) as image:
        image.load()
        return WARPS[mode](image, src)
