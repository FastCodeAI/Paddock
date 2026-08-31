"""CPU-only checks that the KITScenes adaptation is correct.  No GPU, no model.

The expensive part of this evaluation is a 594 GB download, so everything that can be
falsified without a GPU is falsified first.  The load-bearing check is GT parity: the
ground truth recomputed here from ``poses.txt`` must reproduce the ``human_poses`` the
reference run pickled.  If that holds, the frame indexing, the pose parsing and the
keyframe arithmetic are all provably aligned with the run we are comparing against; if
it fails, every number downstream is meaningless.

    python eval_kitscenes/selftest.py [--contact-sheet out.png]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_kitscenes import manifest as manifest_mod  # noqa: E402
from eval_kitscenes import metrics as metrics_mod  # noqa: E402
from eval_kitscenes.camera_warp import (  # noqa: E402
    TELE_HFOV_DEG,
    crop_box,
    crop_hfov_deg,
    principal_point_in_output,
)
from eval_kitscenes.kitscenes_alpamayo_loader import (  # noqa: E402
    CAMERA_SLOTS,
    KITScenesAlpamayoScene,
)

PASS, FAIL = "  ok  ", " FAIL "


class Checker:
    """Collects results so every check runs even after one fails."""

    def __init__(self) -> None:
        self.failures = 0

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        print(f"[{PASS if ok else FAIL}] {label}{('  ' + detail) if detail else ''}")
        self.failures += 0 if ok else 1


def on_disk_scenes(root: Path) -> dict[str, Path]:
    """Scene id -> path, for whatever is currently extracted."""
    found = {}
    for split in ("val", "overlap_train_val"):
        split_dir = root / "data" / split
        if split_dir.is_dir():
            for path in sorted(split_dir.iterdir()):
                if path.is_dir() and (path / "poses.txt").exists():
                    found[path.name] = path
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitscenes-root", type=Path, default=manifest_mod.KITSCENES_ROOT)
    parser.add_argument("--drive-jepa-pkl", type=Path, default=manifest_mod.DRIVE_JEPA_PKL)
    parser.add_argument("--contact-sheet", type=Path, default=None)
    args = parser.parse_args()

    checker = Checker()
    samples = manifest_mod.samples_from_pickle(args.drive_jepa_pkl)
    print(f"reference run: {len(samples)} samples, {len({s.scene_id for s in samples})} scenes\n")
    checker.check(len(samples) == 223, "manifest has 223 reference samples", f"n={len(samples)}")

    available = on_disk_scenes(args.kitscenes_root)
    local = [s for s in samples if s.scene_id in available]
    print(f"on disk: {len(available)} scenes -> {len(local)} testable samples\n")
    if not local:
        print("no scenes on disk; cannot run the parity checks")
        return 1

    # ---- geometry ---------------------------------------------------------- #
    scene = KITScenesAlpamayoScene(available[local[0].scene_id])
    for slot, sensor, kind in CAMERA_SLOTS:
        pinhole = scene._pinhole(sensor)
        target = TELE_HFOV_DEG if kind == "zoom" else None
        box = crop_box(pinhole, target)
        pu, pv = principal_point_in_output(pinhole, box)
        hfov = crop_hfov_deg(pinhole, box)
        centred = abs(pu - 959.5) < 1.5 and abs(pv - 539.5) < 1.5
        checker.check(
            centred,
            f"slot {slot} ({kind:5s}) principal point centred",
            f"({pu:.1f}, {pv:.1f}) hFOV {hfov:.2f} deg box {box}",
        )
        if kind == "zoom":
            checker.check(
                abs(hfov - TELE_HFOV_DEG) < 0.05,
                "telephoto crop is exactly 30 deg",
                f"{hfov:.3f} deg",
            )
    print()

    # ---- keyframe rule ----------------------------------------------------- #
    for scene_id, path in available.items():
        expected = sorted(s.keyframe for s in samples if s.scene_id == scene_id)
        if not expected:
            continue
        derived = KITScenesAlpamayoScene(path).valid_keyframes()
        checker.check(
            derived == expected, f"{scene_id[:8]} keyframes match", f"{derived} vs {expected}"
        )
    print()

    # ---- the parity checks ------------------------------------------------- #
    worst_gt = worst_speed = worst_frame = 0.0
    for sample in local:
        scene = KITScenesAlpamayoScene(available[sample.scene_id])
        data = scene.build_sample(sample.keyframe, load_images=False)

        gt_delta = float(np.abs(data["kits_planar_gt"] - sample.reference["human_poses"]).max())
        speed_delta = abs(data["kits_ego_speed"] - sample.reference["ego_speed"])

        # The full-SO(3) future, rotated into the yaw-only frame, must reproduce the
        # planar ground truth; uncorrected it should differ only by the pod's tilt.
        correction = metrics_mod.yaw_frame_correction(data["kits_t0_rot"], data["kits_t0_yaw"])
        future = data["ego_future_xyz"][0, 0].numpy()
        corrected = metrics_mod.apply_correction(future[None], correction)[0]
        grid = metrics_mod.GRID_IDX[: len(data["kits_planar_gt"])]
        grid = grid[grid < len(future)]
        agree = float(
            np.abs(corrected[grid, :2] - data["kits_planar_gt"][: len(grid), :2]).max()
        )
        tilt = float(np.abs(corrected[:, :2] - future[:, :2]).max())

        worst_gt = max(worst_gt, gt_delta)
        worst_speed = max(worst_speed, speed_delta)
        worst_frame = max(worst_frame, tilt)

        checker.check(
            gt_delta < 1e-9,
            f"{sample.scene_id[:8]} k={sample.keyframe:<4d} ground truth matches reference",
            f"max|d| {gt_delta:.2e}",
        )
        checker.check(
            speed_delta < 1e-6,
            f"{sample.scene_id[:8]} k={sample.keyframe:<4d} ego speed matches reference",
            f"{data['kits_ego_speed']:.3f} vs {sample.reference['ego_speed']:.3f}",
        )
        # 1e-4 m, not 0: ``ego_future_xyz`` is float32 and the future reaches ~85 m at
        # highway speed, so a float32 round trip costs ~1e-5 m against the float64 GT.
        checker.check(
            agree < 1e-4,
            f"{sample.scene_id[:8]} k={sample.keyframe:<4d} SO(3) future -> planar frame",
            f"max|d| {agree:.2e}, tilt effect {tilt * 100:.2f} cm",
        )

    print(
        f"\nworst over {len(local)} samples: gt {worst_gt:.2e} m, "
        f"speed {worst_speed:.2e} m/s, frame tilt {worst_frame * 100:.2f} cm"
    )
    checker.check(worst_frame < 0.05, "yaw-only vs SO(3) frame delta under 5 cm")

    # ---- images ------------------------------------------------------------ #
    sample = local[0]
    scene = KITScenesAlpamayoScene(available[sample.scene_id])
    data = scene.build_sample(sample.keyframe, load_images=True)
    frames = data["image_frames"]
    checker.check(
        tuple(frames.shape) == (len(CAMERA_SLOTS), 4, 3, 1080, 1920),
        "image tensor shape",
        str(tuple(frames.shape)),
    )
    checker.check(str(frames.dtype) == "torch.uint8", "image dtype", str(frames.dtype))
    checker.check(
        data["camera_indices"].tolist() == [0, 1, 2, 6],
        "camera indices",
        str(data["camera_indices"].tolist()),
    )
    # The prompt builder re-derives grouping from camera_indices, so the flattened
    # order must be camera-major; a transpose here is invisible downstream.
    flat = frames.flatten(0, 1)
    checker.check(
        bool((flat[0] == frames[0, 0]).all() and (flat[4] == frames[1, 0]).all()),
        "flattened image order is camera-major",
    )
    # Frames must run oldest -> newest; consecutive frames differ but are not unrelated.
    diffs = [
        float((frames[1, i + 1].float() - frames[1, i].float()).abs().mean()) for i in range(3)
    ]
    checker.check(all(d > 0.5 for d in diffs), "temporal frames differ", f"mean |d| {diffs}")

    if args.contact_sheet:
        write_contact_sheet(frames, data, args.contact_sheet)
        print(f"\ncontact sheet -> {args.contact_sheet}")

    print(f"\n{'ALL CHECKS PASSED' if not checker.failures else f'{checker.failures} CHECKS FAILED'}")
    return 1 if checker.failures else 0


def write_contact_sheet(frames, data, path: Path) -> None:
    """4 slots x 4 frames, for the one check no assertion can make: looking at it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from alpamayo1_5.helper import CAMERA_DISPLAY_NAMES

    slots = data["camera_indices"].tolist()
    fig, axes = plt.subplots(len(slots), 4, figsize=(16, 3 * len(slots)))
    for row, slot in enumerate(slots):
        for col in range(4):
            ax = axes[row, col]
            ax.imshow(frames[row, col].permute(1, 2, 0).numpy())
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(CAMERA_DISPLAY_NAMES.get(slot, str(slot)), fontsize=9)
            ax.set_title(f"t-{3 - col}", fontsize=8)
    fig.suptitle(f"{data['clip_id']}  fov={data['kits_fov_mode']}  v0={data['kits_ego_speed']:.1f} m/s")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
