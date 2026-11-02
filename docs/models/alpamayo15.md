# Alpamayo 1.5

NVIDIA's Alpamayo 1.5 (10B), a vision-language driving model: four cameras, four
frames each, 1.6 s of ego history and an optional navigation sentence in; a
reasoning trace and 64 poses 0.1 s apart (6.4 s) out. The adapter is
`src/paddock/adapters/models/alpamayo15/`; the model card and training data are
in [comparison.md](comparison.md), its KITScenes protocol in
[alpamayo15-eval-notes.md](alpamayo15-eval-notes.md), and what its reasoning
traces say in [alpamayo15-reasoning.md](alpamayo15-reasoning.md).

## On AlpaSim, through the bridge

`paddock run --model alpamayo15 --bench alpasim` drives the model on the settings
of its stored AlpaSim result (`results/alpasim/alpamayo15/legacy-alpamayo_proxy_referee_20hz`,
made by AlpaSim's own Alpamayo driver) and builds its input the way that driver
does. The settings are the `alpamayo15` entry in `alpasim/harness.py`'s
`DRIVER_PROFILES` and its driver config `paddock_bridge_alpamayo15.yaml`: four
f-theta cameras at 20 Hz (front wide, both cross cameras and the 30° telephoto),
a 3 s replayed warm-up with the driver called, 400 steps of 50 ms, and the 30
scenes pinned to their 26.01 artifacts. With `scene_set = "public_2601_30scenes"`
in `[benchmarks.alpasim]` it drives all 30.

The node switches to AlpaSim's inputs only on fields that profile's bridge
sends and no other client does, so every other benchmark gets exactly what it
got before (`tests/test_unchanged_inputs.py` pins seven of them, recorded from
the adapter before the change):

| Input | On AlpaSim | Switched on by |
|---|---|---|
| when the model is called | every 100 ms from the end of the warm-up, the model's own rate; the step between goes to `/observe`, and the bridge answers it from the last plan, re-timed and re-anchored to the car | `--replan-every-us 100000`, `--observe-between-replans` |
| cameras | all four, the telephoto as rendered at 30° where other benchmarks fill that slot with a crop of the front camera; f-theta frames go in as rendered, with no crop, resize or undistortion, as AlpaSim's driver does | a camera named `CAM_FRONT_TELE`; a camera with no pinhole entry |
| frames | the four newest per camera, 0.1 s apart | — |
| ego history | 16 poses 0.1 s apart from the 20 Hz pose stream | — |
| navigation text | AlpaSim's own rule (`route_nav.py`, ported from AlpaSim's `nav_text.py`): the route is moved into the newest pose's frame, and the first turn within 40 m becomes "Turn left in 20m" or similar; otherwise "Continue straight", and no text when the route is unusable | `route_xy` with `route_timestamp_us` (`--send-route`, `--send-route-time`) |
| output | all 64 points 0.1 s apart, with the model's own headings, and the first 400 characters of its reasoning in `extras` (so in the per-replan log). Any spacing other than 0.1 s is refused | `plan_dt` (`--send-plan-dt`) |

The model reads poses and pixels only, so no speed or acceleration switch is
on. Sampling is the model's own (`top_p 0.98`, temperature 0.6, one sample), so
a scene's drive is one draw.

**Inputs checked against AlpaSim's own driver.** Ten scenes' recorded simulator
messages were replayed offline through both: AlpaSim's Alpamayo driver code, and
the bridge with this profile into the paddock node, the model replaced by a
recorder on both sides. On 1,681 model calls after the warm-up, the call times,
camera indices, all 16 frame times, pixels, prompt text, navigation text and
sampling settings were identical, and the ego history agreed to 3.8e-6 m and
8.1e-6°. The replay was run at the author's site and is not committed.

The 30-scene run is `results/alpasim/alpamayo15/20261029T200113-9d900c`: valid,
one rollout per scene, `scene_score` **0.7952** (progress 0.843), one at-fault
collision and four off-road events. On the first ten scenes it scores 0.9000,
the same as the stored native row on every scene, including the at-fault
collision on `04749bb9`. The stored row ran on simulator image `8a69f005`, this
one on `6dad7728`.

On the seven scenes whose route turns, the node logged the navigation text it
gave the model on every call. It equals, call for call, the text AlpaSim's own
driver code computes from the same run's recorded messages.

What remains different from the native runs:

- **JPEG decoding.** AlpaSim's driver decoded frames on the GPU (nvJPEG); the
  bridge decodes them with PIL. The two can differ by rounding; not measured.
- **The plan's height and tilt.** The bridge returns every plan flat (x, y and
  heading); the native driver kept the model's full poses on model calls.
  AlpaSim's controller is planar, so this is not expected to matter; not
  measured.
- **Warm-up calls.** Natively the model was also called during the warm-up,
  whose plans AlpaSim does not drive; here those frames are observed only.
- **Speed.** A call takes 2.75 s at the median through paddock, about 3–4 times
  the native runs' 0.7 s, varying with other load on a shared GPU. AlpaSim waits
  for every plan, so this changes wall time, not the drive.

`make check-alpamayo ALPAMAYO_PY=<repos>/alpamayo1.5/a1_5_venv/bin/python` runs
the adapter's input tests on the CPU without weights; `make check` skips them,
because they need torch and `alpamayo1_5`.
