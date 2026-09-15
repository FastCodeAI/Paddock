"""Draw the NAVSIM comparison grid. Runs in the devkit's interpreter.

Invoked by ``render.py`` with one payload file, because the map API and the
scene loader live in the benchmark's environment and paddock's has neither.
Reads the payload, loads each scenario, and draws the map, the annotated
agents, the recorded human future and every model's scored plan on one panel.

    python draw_scenes.py <payload.json>

Prints the files it wrote, one per line, which is what ``render.py`` returns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from navsim.common.dataclasses import SceneFilter, SensorConfig  # noqa: E402
from navsim.common.dataloader import SceneLoader  # noqa: E402
from navsim.visualization.bev import add_configured_bev_on_ax  # noqa: E402
from navsim.visualization.plots import configure_ax, configure_bev_ax  # noqa: E402

from paddock.adapters.benchmarks.navsim.render import caption  # noqa: E402
from paddock.viz import palette  # noqa: E402

#: Half-width of a panel in metres. NAVSIM's own default is 64 m across, which
#: leaves a 4 s plan at speed as a short stub in the middle; 40 m fills it.
VIEW_M = 40.0


def main(payload_path: str) -> int:
    payload = json.loads(Path(payload_path).read_text())
    tokens: list[str] = payload["tokens"]
    root = Path(payload["data_root"])
    split = payload["data_split"]

    # The split's own filter, as the scorer used it (render.py reads it from the
    # devkit's scene_filter yaml); only the frame geometry matters here.
    spec = payload.get("scene_filter") or {}
    history = int(spec.get("num_history_frames", 4))
    scene_filter = SceneFilter(
        num_history_frames=history,
        num_future_frames=int(spec.get("num_future_frames", 10)),
        frame_interval=int(spec.get("frame_interval", 1)),
        has_route=bool(spec.get("has_route", True)),
        tokens=tokens,
    )
    loader = SceneLoader(
        data_path=root / "navsim_logs" / split,
        sensor_blobs_path=root / "sensor_blobs" / split,
        synthetic_scenes_path=root / "navsim_logs" / split,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )

    found = [t for t in tokens if t in set(loader.tokens)]
    if not found:
        print("none of the selected scenarios loaded", file=sys.stderr)
        return 1

    cols = 4 if len(found) > 4 else len(found)
    rows = (len(found) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 4.7 * rows), squeeze=False)

    for index, token in enumerate(found):
        ax = axes[index // cols][index % cols]
        scene = loader.get_scene_from_token(token)
        # The scored frame is the last of the history: what the model saw.
        frame = scene.frames[history - 1]
        add_configured_bev_on_ax(ax, scene.map_api, frame)

        human = scene.get_future_trajectory(payload["n_poses"])
        _line(ax, human.poses, color=palette.GROUND_TRUTH, lw=6.5, alpha=0.30, zorder=3)
        _line(ax, human.poses, color=palette.GROUND_TRUTH, lw=1.4, zorder=4)

        for model in payload["models"]:
            poses = payload["plans"].get(token, {}).get(model)
            if poses:
                _line(ax, poses, **palette.style(model, lw=2.0, zorder=5))

        configure_bev_ax(ax)
        ax.set_xlim(-VIEW_M / 2, VIEW_M / 2)
        ax.set_ylim(-VIEW_M / 2, VIEW_M / 2)
        configure_ax(ax)
        ax.set_title(caption(payload, token), fontsize=7.5, loc="left", pad=4)

    for spare in range(len(found), rows * cols):
        axes[spare // cols][spare % cols].axis("off")

    handles = [plt.Line2D([], [], color=palette.GROUND_TRUTH, lw=2.4, label="human (recorded)")]
    handles += [
        plt.Line2D([], [], color=palette.color(m), lw=2.4, label=payload["labels"].get(m, m))
        for m in payload["models"]
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), frameon=False, fontsize=9)
    title = (
        f"NAVSIM navtest: the {len(found)} scenarios where these models' 4 s plans diverge most, "
        "one per log. Ego at centre, x forward; grey is drivable area, dashed lines lane centres."
    )
    if payload.get("omitted"):
        title += (
            "\n" + ", ".join(payload["omitted"]) + " omitted: their runs' plan dumps carry no "
            "scenario token (recorded before the proxy labelled them), so they cannot be "
            "joined to a scenario."
        )
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0.055, 1, 0.955), h_pad=2.2)

    out = Path(payload["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(out)
    return 0


def _line(ax, poses, **style) -> None:
    """One plan, in the devkit's BEV convention: y across, x up, origin first."""
    xy = np.concatenate([np.zeros((1, 2)), np.asarray(poses, dtype=float)[:, :2]])
    ax.plot(xy[:, 1], xy[:, 0], **{"solid_capstyle": "round", **style})


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
