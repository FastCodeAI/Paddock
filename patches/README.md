# Patches

Edits to third-party source that could not be avoided by configuration.
`paddock setup` applies them, in filename order, after checking out the sha
`repos.yaml` pins.

**A patch is a maintenance cost paid on every upstream bump**, so it has to earn
its place. Before adding one, exhaust the alternatives in this order:
environment variable, CLI flag, symlink, then patch. See
[../docs/adding-a-model.md](../docs/adding-a-model.md).

Each patch carries a `Subject:` line and a rationale explaining *why* external
configuration was not possible, and whether it is upstreamable. Keep that habit:
the reader is deciding whether your patch still applies to a newer upstream, and
they cannot decide that from a diff alone.

## Current patches

| Repo | Patch | What and why |
|---|---|---|
| `neuro-ncap` | `0001-waypoint-dt-and-camera-selection` | Adds `--engine.waypoint-dt` and `--engine.cameras`. Needed to drive every model at one sim step while keeping the 0.5 s waypoint grid the scoring assumes, and to render only the cameras a model consumes. Additive; defaults to upstream behaviour. |
| `neuro-ncap` | `0002-fail-loudly-on-renderer-http-errors` | **Correctness.** `APIClient` ignored HTTP status, so a failed `/update_actors` left the actor on its logged path while the simulator staged a collision elsewhere — producing complete runs with believable, meaningless scores. Should go upstream. |
| `neuro-ncap` | `0003-optional-debug-views` | Adds the logger field `debug_views` (default True, upstream behaviour; `--engine.logger.no-debug-views` turns it off). False skips the two per-step debug views, FC_TRAJ and COMBINED_OUTPUTS, which only a person watches; drawing them took about 1 s of each 2.7 s step. Every record the scores and PDM terms read is still written. The regression gate turns it off (`[benchmarks.neuroncap] debug_views`). Additive. |
| `Drive-JEPA` | `0001-msda-compat-for-newer-gpus` | mmcv-full's compiled kernel has no SASS above sm_86 and no PTX, so it cannot run on newer cards. Falls back to the equivalent pure-PyTorch path. |
| `Drive-JEPA` | `0002-score-runs-with-failed-scenarios` | **Correctness.** One failed scenario destroyed a whole run: `compute_final_scores` stacked a column mixing arrays and scalar NaN, and the failed-token report used `not` on a Series. Scores valid rows, leaves failed rows NaN, negates with `~`. Upstreamable. |
| `Drive-JEPA` | `0003-navsim-v1-camera-path` | Adds `Camera.camera_path` to the **v1** devkit, the field navsim v2 already has, so paddock's proxy can ship image paths instead of pixels under `navsim_v1`. Additive; nothing in the devkit reads it. |
| `HUGSIM` | `0001-traj2control-heading-pr57` | **Correctness.** Upstream PR #57, verbatim (open since Nov 2025). `traj2control` read each waypoint as (right, forward) the wrong way round, so a straight plan got a reference heading of about +90 deg and the iLQR steered it right (4.3 m in 14 m on nuScenes scene-0013); with the fix, 0.00 m. Part of every model's lateral error was otherwise the controller's. Should go upstream. |
| `HUGSIM` | `0002-render-only-requested-cameras` | Adds `HUGSIM_RENDER_CAMS`: render only the named cameras and send the rest black, as `_get_obs` already does for the rear cameras of Waymo and KITTI-360. Rendering six views dominates a step and scoring never reads images, so scores are unchanged. Unset, upstream behaviour. Additive; upstreamable. |
| `Bench2Drive` | `0001-kill-only-own-carla-on-crash` | **Safety.** On "Simulation crashed" the evaluator killed every CARLA its user ran on the GPU (`grep -graphicsadapter=N | kill -9`), so one crashed route took parallel workers and other gate lanes down with it. Kills only its own server's process group. Should go upstream. |
| `Bench2Drive` | `0002-connect-timeout-for-a-booting-server` | Adds `--connect-timeout`. Each attempt to reach a just-launched CARLA waited the full client timeout (600 s), so a server slower than the 30 s sleep cost ten minutes before a retry that connected at once. `--timeout` itself also bounds the world load and the watchdogs, so it could not just be lowered. Without the flag, upstream behaviour. Upstreamable. |

## Verifying a patch

A patch must produce, from a clean checkout at the pinned sha, exactly the tree
you are running:

```bash
git clone --no-checkout --shared <repo> /tmp/verify && cd /tmp/verify
git checkout <pinned sha>
git apply /path/to/paddock/patches/<repo>/*.patch
diff -r /tmp/verify <your working repo>      # expect no differences in patched files
```

`paddock setup` does the apply half of this. Do the `diff` half yourself when
you add or update a patch — it is the only thing that proves the patch captures
what you actually ran.

## Not patches

Site-specific configuration is **not** a patch, even when it lives in a file
that upstream tracks. VisionPilot's `vision_pilot_test.conf` is the example: the
local version differs only by absolute paths to input clips, which is exactly
what `paddock.toml` and the dataset registry exist to supply. Paddock generates
that config at run time instead. Committing it as a patch would hardcode one
machine's layout into the shared repo.
