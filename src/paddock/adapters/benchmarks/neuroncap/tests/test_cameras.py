"""Declared cameras are filtered to what NeuRAD (nuScenes) can render."""

from paddock.adapters.benchmarks.neuroncap.harness import renderable_cameras


def test_comma_rig_falls_back_to_cam_front():
    # openpilot declares its comma rig plus a CAM_FRONT fallback; nuScenes has
    # neither comma camera, so only CAM_FRONT survives.
    assert renderable_cameras(["CAM_OP_ROAD", "CAM_OP_WIDE", "CAM_FRONT"]) == ["CAM_FRONT"]


def test_only_simulator_cameras_still_yields_cam_front():
    assert renderable_cameras(["CAM_OP_ROAD", "CAM_OP_WIDE"]) == ["CAM_FRONT"]


def test_none_and_empty_default_to_cam_front():
    assert renderable_cameras(None) == ["CAM_FRONT"]
    assert renderable_cameras([]) == ["CAM_FRONT"]


def test_ring_cameras_pass_through_in_order():
    ring = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_BACK"]
    assert renderable_cameras(ring) == ring
