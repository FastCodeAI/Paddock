# KITScenes

Open-loop trajectory accuracy: ADE and FDE against the recorded future, over a
fixed set of 223 keyframes.

```bash
paddock run --bench kitscenes --model openpilot --limit 5   # smoke
paddock run --bench kitscenes --model openpilot             # full n=223
```

## The sample list is the benchmark

All models are scored on one shared keyframe list, configured per site:

```toml
[benchmarks.kitscenes]
sample_list = "/path/to/kitscenes_e2e_final.pkl"
```

Benchmark-level, not model-level, deliberately: two models scored on different
keyframes are not comparable however close the counts look. `paddock run
--dry-run` reports how many of the list this site's tiers can serve, and
`samples()` refuses to run a partial set rather than silently scoring fewer.

## Samples are sequences

Each sample is 48 preceding source frames plus the keyframe, not a single image.
The models need real history — openpilot fills a 24-deep feature buffer, Alpamayo
differences 16 ego poses to derive its speed — and handing them one frame would
measure a model working from a history it invented.

`Policy.infer_sequence` feeds the sequence and scores the last frame, disabling
self-priming for its duration. A model that primes *and* receives real history
sees a manufactured past followed by a real one.

## Camera adaptation

openpilot's ModelState compiles its warp for fixed geometries, so a source must
become 1928x1208 exactly. Which way depends on the source's aspect, and the
adapter derives it rather than hardcoding one:

| source | aspect | strategy | why |
|---|---:|---|---|
| KITScenes 3504x2272 | 1.542 | **crop** | taller than the target, so trimming height costs no horizontal field of view |
| nuScenes render 1600x900 | 1.778 | **letterbox** | wider, so cropping would trim width — and hFOV is scarce, the wide model view wants 58.7 deg of about 64 available |

Before paddock these were two separate classes, one per loop type, each
hardcoding one strategy — which is why the open-loop path could not run a
nuScenes frame and the closed-loop path could not run a KITScenes one.

## Reproduction status

Verified against the recorded `runs/full223` results, same keyframes:

| scene / keyframe | paddock ADE | recorded ADE |
|---|---:|---:|
| 00fc1b9b / 40 | 1.0783930800368695 | 1.0783930800368695 |
| 02be7d74 / 40 | 0.8611… | identical |
| 03ed2585 / 40 | 2.8353… | identical |

The **full n=223 run reproduces every metric exactly**:

| metric | exact / 223 | paddock | recorded |
|---|---:|---:|---:|
| ade | **223/223** | 1.599424 | 1.599424 |
| fde@1s | **223/223** | 0.836421 | 0.836421 |
| fde@2s | **223/223** | 1.808482 | 1.808482 |
| fde@3s | **223/223** | 2.893613 | 2.893613 |

892 of 892 metric comparisons identical at full float precision; ADE median
1.322747 against 1.322747. Roughly 85 minutes on this hardware.


## A bug worth knowing about

Reproducing the full n=223 found something a three-sample check could not: the
openpilot adapter cached its `CameraAdapter` across samples. It is built from
the first frame's calibration, and `reset()` did not clear it, so every scene
after the first was warped with the **first scene's mounting rotation**.

KITScenes rigs are recalibrated between sessions. Across 60 `val` scenes there
are three distinct front-camera rotations, up to 1 degree apart — so roughly a
quarter of samples were fed a subtly wrong view. Intrinsics are identical
throughout, which is why this survived the obvious checks.

What made it hard to see:

- **Nothing failed.** A degree of rotation shifts the warp; it does not raise.
- **The three samples used for the initial check all shared one rotation
  group**, so they matched bit-for-bit and looked like proof.
- The error only showed as a slowly growing gap in the running mean
  (+0.021 at 25 samples, +0.067 at 100), which is easy to attribute to noise.

The lesson generalises past this adapter: **anything derived from calibration is
episode state.** If a model caches it, `reset()` must clear it, and a test should
say so. `tests/test_policy.py::test_reset_clears_the_camera_adapter` does.

## Harness changes of 05 Sep 2026 (uncommitted, partially verified)

Made while trying to run all four models through this harness, then parked
because open-loop evaluation moved to nuScenes:

* **Route hint.** Every model used to get `command = STRAIGHT`. The reference
  Drive-JEPA run (`run_kitscenes_inference.py --command gt`) inferred
  left/right/straight from the recorded future (pose 8 steps × 5 frames ahead,
  lateral offset beyond 5 m). `route_hint()` now applies that rule for every
  model and also fills `nav_maneuver` / `nav_distance_m`.
* **Cameras.** Only `CAM_FRONT` was offered; Drive-JEPA (four slots) and
  Alpamayo (three) failed every sample. The four ring cameras are now offered
  under the nuScenes-style names, each located on its own tier, because the
  pruned NVMe tier lacks side or rear cameras for up to 58 of the 140 scenes.
  Verified on all 223 samples.
* **Duplicate warm-up frames.** Keyframe 40 with a 48-frame history repeats
  frame 0 eight times; Alpamayo's time-based pose resampler rejected the
  duplicate timestamps. Its policy now ignores repeated frames.
* **Known consequence, fixed 06 Sep:** openpilot's `model.yaml` declares the
  comma cameras for Bench2Drive, and `Observation.from_wire` keeps only
  declared cameras, so openpilot lost `CAM_FRONT` here. `CAM_FRONT` is now
  declared as well; the policy prefers the comma pair when present.

Only VisionPilot completed under this harness (n = 223, ADE 3.70 m, median
2.80). Drive-JEPA and Alpamayo were stopped mid-run and not rerun.
