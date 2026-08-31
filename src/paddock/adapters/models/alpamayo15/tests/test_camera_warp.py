"""The crop_box unification.

Two implementations of this existed: the open-loop harness's, and a forked copy
in the closed-loop node that added a vertical clamp. They also disagreed on
rounding, so the same KITScenes frame cropped one pixel differently depending on
which loop it went through -- small, invisible, and exactly the kind of drift
that makes an ADE and an NCAP score stop describing the same system.

These tests pin the resolution: rounding as the open-loop version had it, so
published KITScenes numbers are unchanged; clamping as the node had it, so a
render wider than 16:9 no longer escapes the frame.
"""

from __future__ import annotations

import math

import pytest

from paddock.adapters.models.alpamayo15.camera_warp import TARGET_ASPECT, Pinhole, crop_box


def kitscenes_like() -> Pinhole:
    """A KITScenes ring camera: 3504x2272, roughly 4:3."""
    return Pinhole(focal=1800.0, cu=1752.0, cv=1136.0, width=3504, height=2272)


def neurad_render() -> Pinhole:
    """A NeuRAD render of a nuScenes camera: 1599x900, already wider than 16:9."""
    return Pinhole(focal=1200.0, cu=799.5, cv=450.0, width=1599, height=900)


def test_kitscenes_crop_is_unchanged_by_the_clamp() -> None:
    """The value the published n=223 KITScenes runs used.

    On a 4:3-ish sensor the horizontal margin binds, so the vertical clamp is
    inert and this must equal what the original produced.
    """
    assert crop_box(kitscenes_like()) == (1, 151, 3503, 2121)


def test_neurad_render_no_longer_escapes_the_frame() -> None:
    """Without the vertical clamp this raised: the box ran off the bottom."""
    box = crop_box(neurad_render())
    left, top, right, bottom = box
    src = neurad_render()
    assert left >= 0 and top >= 0
    assert right <= src.width and bottom <= src.height


@pytest.mark.parametrize("src", [kitscenes_like(), neurad_render()])
def test_principal_point_stays_centred(src: Pinhole) -> None:
    """The property the whole camera adaptation rests on."""
    left, top, right, bottom = crop_box(src)
    assert abs((left + right) / 2 - src.cu) <= 1.0
    assert abs((top + bottom) / 2 - src.cv) <= 1.0


@pytest.mark.parametrize("src", [kitscenes_like(), neurad_render()])
def test_aspect_is_16_by_9(src: Pinhole) -> None:
    left, top, right, bottom = crop_box(src)
    assert (bottom - top) == pytest.approx((right - left) * TARGET_ASPECT, abs=1.0)


def test_zoom_is_clamped_to_the_sensor() -> None:
    """A requested field of view wider than the sensor must not sample past its edge.

    The open-loop version computed the zoom box from focal length alone, with no
    sensor clamp, so an over-wide request produced a box outside the frame.
    """
    src = neurad_render()
    absurd_hfov = 179.0
    left, top, right, bottom = crop_box(src, target_hfov_deg=absurd_hfov)
    assert left >= 0 and top >= 0 and right <= src.width and bottom <= src.height


def test_narrow_zoom_still_honours_the_requested_fov() -> None:
    """Clamping must not silently narrow a request the sensor can satisfy."""
    src = kitscenes_like()
    left, _, right, _ = crop_box(src, target_hfov_deg=30.0)
    expected = 2.0 * src.focal * math.tan(math.radians(15.0))
    assert (right - left) == pytest.approx(expected, abs=2.0)
