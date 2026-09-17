# Cosmos 3 Nano sim2real scripts

The converter and harness used for the sim2real work on Bench2Drive route
2802-Town12-InvadingTurn on 16–17 September 2026. What the pipeline is, what was
measured and what it means is in [docs/sim2real/cosmos3.md](../../../docs/sim2real/cosmos3.md);
this file says what each script is and how to run it.

**Diagnostic, not benchmark code.** Nothing here writes to `results/`. A model driving
on converted pixels is not a Bench2Drive score and must never appear in those tables.

## No path in here is a real path

These scripts ran across three machines: a workstation, a rented two-GPU pod, and a NAS.
Every location they used is now an argument or an environment variable, so the next person
can point them at their own layout instead of discovering someone else's. Each script's
header lists its variables; the ones that recur:

| Variable | What it names |
|---|---|
| `POD_ROOT` | the conversion pod's scratch root: `captures/`, `converted/`, `plans/`, `logs/`, and the paddock checkout |
| `POD_ROOT_REMOTE` | the same root as seen *on the pod*, for the local scripts that reach it over ssh |
| `POD_HOST`, `POD_SSH` | `user@host`, and the ssh command carrying the port and key |
| `COSMOS_PY`, `FLASHDREAMS_PY` | the two venv interpreters (cosmos-framework; flashdreams, which also provides `ninja` for the Ludus JIT) |
| `OPENLOOP_DIR` | the local working directory: `captures/`, `plans/`, `converted/`, `compare/` |
| `VIZ_PY` | a python with numpy, matplotlib and pillow, for the compare and page builders |
| `CLIPS_DIR`, `SITE_DIR` | where the mp4s and the HTML page go |
| `HF_TOKEN_FILE` | a **file** holding the Hugging Face token. No token is ever written into a script, and none is needed to read this directory |

`specs/*.json` name their prompt and control files relatively (`prompts/…`, `controls/…`),
so run the converters from this directory — `run_convert.sh` does. Create `controls/` with
`rgb.mp4`, `depth.mp4` and `wsm.mp4`: the sample loader resolves and validates those paths,
but both the resident service and the batch converter feed their own tensors, so the video
content is never read.

## services/ — resident per-tick converters (the closed-loop segment runs, on the pod)

- `cosmos_service.py` — HTTP service. `POST /frame {tick, rgb, depth, wsm}` returns one
  converted frame per call: Cosmos 3 Nano video-to-video, edge derived from the CARLA frame
  plus CARLA inverse depth plus the Ludus map as controls. Env: `COSMOS_STEPS`,
  `COSMOS_CHUNK` (5 or 9), `COSMOS_NCOND` (memory), `COSMOS_REANCHOR` (regenerate from the
  controls every N ticks), `COSMOS_RES`, `COSMOS_SPEC`, `COSMOS_OUTPUT_DIR`.
  Decided settings: `12 / 5 / 4 / 40`.
  Run: `cd <this dir> && COSMOS_STEPS=12 COSMOS_CHUNK=5 COSMOS_NCOND=4 COSMOS_REANCHOR=40 \
  "$COSMOS_PY" services/cosmos_service.py <port> specs/cosmos_live_spec.json`
- `wsm_service.py` — HTTP service. `POST /init {map, cams}` then
  `POST /wsm {timestamp_us, camera_to_world, actors, cam}` returns the Ludus map render.
  Cameras larger than 1280×720 render on a 1280×720 canvas (Ludus faults above that) and are
  cropped and resized back; the FTheta fit is always degree 5. Needs the flashdreams venv
  with its `bin` on `PATH`, for `ninja`.
- `replay_frames.py` — replays saved frames through `cosmos_service` tick by tick, for
  offline settings tests with no CARLA.
- `dreamer_service.py` — the earlier OmniDreams/FlashDreams service. Superseded by Cosmos 3
  for the live loop; kept because the 16 Sep closed-loop results came from it.

## batch/ — whole-sequence conversion (the open-loop study, on the pod)

- `wsm_batch.py` — renders the map control for every tick of a capture in-process
  (7 ms/frame): `python batch/wsm_batch.py <capture_dir> [cam ...]`
- `cosmos_batch.py` — converts one camera of a capture in 61-frame windows with 8 frames of
  overlap conditioned on the previous output (the framework's long-video mode); resumes from
  disk; `COSMOS_VIS_M=50` blanks the depth and map controls beyond that range.
  `COSMOS_STEPS=12 COSMOS_CHUNK=61 COSMOS_NCOND=8 COSMOS_VIS_M=50 python batch/cosmos_batch.py <capture> <cam> <out_dir> <spec.json> [t0 t1]`
- `run_convert.sh` — the runner: waits for each capture's `.upload_done`, renders wsm, then
  converts two cameras at a time, one per GPU.

Both batch scripts import their sibling service module rather than reimplementing it;
`COSMOS3_SERVICES` overrides where that module is found.

## specs/ — transfer sample specs

Only the prompt, negative prompt, guidance and control weights are read by the service and
the batch converter.

- `cosmos_live_spec.json` — **the settings every live run and the open-loop conversion used**:
  edge 0.5, depth 0.5, wsm 1.0, guidance 3.0, control guidance 1.5, seed 2026, `prompt_rain`.
- `cosmos_nohood_spec.json`, `cosmos_roof_spec.json`, `cosmos_d25_spec.json`,
  `cosmos_cg2_spec.json` — the bonnet experiments. Rejected: no effect.
- `cosmos_live_spec_ctrlhi.json` — control guidance 3.0, edge 0.8, depth 1.0. Rejected:
  oversaturated output.

## prompts/

`prompt_rain.json` is the one in use — route 2802 is rain and fog, so the prompt matches the
scene. `negative_prompt.json` is the framework's generic negative.

## openloop/ — capture → replay → compare → page

- `openloop_replay.py` — posts a captured tick sequence to a paddock model node, with the
  captured CARLA frames or `--frames CAM=<converted dir>`. Same wire body as the closed-loop
  agent, so the model sees identical states either way.
- `openloop_compare.py` — plan-versus-plan metrics (L2 at 1/2/3 s, planned speed, lateral
  offset), curves and BEV figures.
- `openloop_video.py` — the side-by-side clip, both plans projected through the capture's own
  calibration. `FFMPEG` names the binary.
- `build_openloop_page.py`, `finish_openloop.sh` — the results page and the one-shot finish.
- `pod_replay_queue.sh`, `local_replay_queue.sh` — the replay queues (models whose node needs
  more VRAM than the workstation has run on the pod).
- `build_model_envs.sh` — builds the paddock, alpamayo15 and qwendrive envs on a fresh pod.
- `paddock.toml.example` — the pod's site config, with every path as a placeholder.

`finish_openloop.sh` no longer restarts the site server: the original matched the process by
pattern, which on a shared box can kill a colleague's process
(see [docs/reviewing.md](../../../docs/reviewing.md) §8). Restart it yourself.

## The capture step lives in paddock

`src/paddock/adapters/benchmarks/bench2drive/segment.py`, spec `"mode": "capture"`, enabled
by `PADDOCK_SEGMENT=<spec.json>`. It replays a recorded run's poses and writes every tick's
RGB, inverse depth, camera pose, actor boxes and model inputs, so the whole recording can be
converted offline and replayed with no CARLA in the loop.

## Pod rebuild

About 25 minutes on a fresh two-GPU RTX PRO 6000 box; the recipe is in
[docs/sim2real/cosmos3.md](../../../docs/sim2real/cosmos3.md#pod-rebuild).
