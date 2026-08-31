"""Launch NeuRAD's closed-loop renderer against a nuScenes install without sweeps.

This nuScenes copy was fetched with the ``*_keyframes.tgz`` blobs, so
``samples/`` exists but ``sweeps/`` does not.  NeuRAD's dataparser reads every
lidar file eagerly -- keyframes *and* sweeps -- and dies on the first missing
one, even though rendering a camera image never touches the points: the scene
lives in the trained hash grids, and the closed-loop server only reads poses,
intrinsics, transforms and actor trajectories out of the dataparser.

So missing point clouds are substituted with empty ones.  That is a real
deviation and is not taken on faith -- ``validate_render.py`` re-renders a
keyframe at its original pose and compares it against the recorded image.  If
the points had mattered, that comparison falls apart.

Camera sweeps are *not* needed on disk: image files are opened lazily, and the
closed-loop path never reads a training image.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/mnt/p4/naren/neuroncap/neurad-studio")

from nerfstudio.data.dataparsers import nuscenes_dataparser as ndp
from nerfstudio.scripts.closed_loop import server as cl_server_mod

_MISSING: list[str] = []
_original_read_lidars = ndp.NuScenes._read_lidars


def _read_lidars_tolerant(self, lidars, filepaths):
    """_read_lidars, but a missing .pcd.bin becomes an empty sweep."""
    real = [p for p in filepaths if Path(p).exists()]
    _MISSING.extend(str(p) for p in filepaths if not Path(p).exists())
    if not real:
        raise FileNotFoundError(
            "no lidar files found at all -- check the nuScenes path, this is not the sweeps issue"
        )
    clouds = _original_read_lidars(self, lidars, real)
    template = clouds[0]
    empty = torch.zeros((0, template.shape[1]), dtype=template.dtype)

    out, it = [], iter(clouds)
    for path in filepaths:
        out.append(next(it) if Path(path).exists() else empty)
    return out


ndp.NuScenes._read_lidars = _read_lidars_tolerant


# This neurad-studio is newer than the released checkpoints, and its dataparser
# now hands back 4x4 transforms where ``ClosedLoopServer.main`` still expects
# 3x4: ``to4x4`` then appends a row and produces a 5x4, which blows up on the
# ``.inverse()`` in /get_actors ("A must be batches of square matrices, but they
# are 5 by 4"). Rendering happens to survive it -- the first four rows of the
# product are still right -- but the actor endpoints do not, and NeuroNCAP calls
# those before it calls anything else. Pad only what actually needs padding.
_original_to4x4 = cl_server_mod.to4x4


def _to4x4_idempotent(pose):
    return pose if pose.shape[-2] == pose.shape[-1] == 4 else _original_to4x4(pose)


cl_server_mod.to4x4 = _to4x4_idempotent


# Same version drift, second symptom: the dataparser's transforms and the actor
# poses no longer share a dtype, so the matmul in get_actor_trajectories fails
# with "expected mat1 and mat2 to have the same dtype, but got: float != double".
# Do the chain in float64 -- these are metre-scale world poses that NeuroNCAP
# will difference to get velocities, and float32 world coordinates in the
# hundreds of metres only carry centimetre resolution.
def _get_actor_trajectories(self):
    poses = _to4x4_idempotent(self.model.dynamic_actors.get_poses_3x4()).to(torch.float64)
    world_inverse = self.world_transform.to(torch.float64).inverse().view(1, 4, 4)
    actor_transform_inv = self.actor_transform.to(torch.float64).inverse()
    timestamps = (self.model.dynamic_actors.unique_timestamps.to(torch.float64) + self.min_time) * 1e6
    return [
        {
            "poses": (world_inverse @ poses[:, i] @ actor_transform_inv).cpu(),
            "timestamps": timestamps.cpu().long(),
            "dims": self.model.dynamic_actors.actor_sizes[i].cpu(),
            "uuid": self.actor_uuids[i],
        }
        for i in range(poses.shape[1])
    ]


cl_server_mod.ClosedLoopServer.get_actor_trajectories = torch.no_grad()(_get_actor_trajectories)


# /get_actors must always describe the *original* scene, because that is what
# NeuroNCAP builds each scenario from -- it looks up every overridden actor's
# dimensions there. But /update_actors replaces the renderer's actor set with
# just the scenario's actors, so after one scenario has run, /get_actors reports
# only that scenario's target and the next scenario dies with a KeyError on a
# uuid the renderer no longer knows.
#
# Upstream never sees this: the release scripts start a fresh renderer per
# sequence. Reusing one renderer across scenarios is much cheaper, so instead
# the original set is snapshotted once at startup and always served from it.
# The uuid -> index mapping used by /update_actors is built at startup too, so
# it stays consistent with the snapshot.
def _install_actor_snapshot(server) -> int:
    snapshot = _get_actor_trajectories(server)

    def _frozen(self):
        return snapshot

    cl_server_mod.ClosedLoopServer.get_actor_trajectories = torch.no_grad()(_frozen)
    return len(snapshot)


# update_actor_trajectories has the same problem on the way in, and this one is
# not cosmetic: it is the call that places the staged collision vehicle. The
# product world_transform @ poses @ actor_transform mixes float32 and float64
# and raises, the endpoint 500s, and NeuroNCAP's HTTP client does not check
# status codes -- so the actor silently stays on its original logged path while
# the simulator collides the ego with a vehicle that was never rendered.
def _update_actor_trajectories(self, new_trajectories):
    device = self.model.device
    world = self.world_transform.to(torch.float64)
    actor = self.actor_transform.to(torch.float64)

    modified, actor_ids = [], []
    for traj in new_trajectories:
        poses = world @ traj["poses"].to(device=device, dtype=torch.float64) @ actor
        modified.append(
            {
                "poses": poses.to(torch.float32),
                "timestamps": (traj["timestamps"].to(torch.float64) / 1e6 - self.min_time).to(device),
                "dims": traj["dims"].to(device=device, dtype=torch.float32),
                "symmetric": True,
                "deformable": False,
                "uuid": traj["uuid"],
            }
        )
        actor_ids.append(self.actor_uuid_to_idx[traj["uuid"]])

    old_n_actors = self.model.dynamic_actors.n_actors
    self.model.dynamic_actors._populate_actors(modified)
    self.model.dynamic_actors.actor_to_id[:] = torch.tensor(actor_ids, device=device)
    self.model.dynamic_actors.n_actors = old_n_actors  # used in hash encoding
    self.model.dynamic_actors.to(device)
    self.actor_uuids = [traj["uuid"] for traj in new_trajectories]


cl_server_mod.ClosedLoopServer.update_actor_trajectories = torch.no_grad()(_update_actor_trajectories)


if __name__ == "__main__":
    import tyro
    import uvicorn

    from nerfstudio.scripts.closed_loop.main import app
    from nerfstudio.scripts.closed_loop.server import ClosedLoopServer
    import nerfstudio.scripts.closed_loop.main as cl_main

    tyro.extras.set_accent_color("bright_yellow")
    cl_server = tyro.cli(ClosedLoopServer)
    cl_server.main()
    cl_main.cl_server = cl_server  # the endpoints read this module-level name

    n_actors = _install_actor_snapshot(cl_server)
    print(f"[run_renderer] snapshotted {n_actors} original actors for /get_actors", flush=True)

    n_missing = len(_MISSING)
    print(f"[run_renderer] substituted {n_missing} missing lidar sweeps with empty clouds", flush=True)
    print(f"[run_renderer] listening on {cl_server.host}:{cl_server.port}", flush=True)
    uvicorn.run(app, host=cl_server.host, port=cl_server.port, log_level="warning")
