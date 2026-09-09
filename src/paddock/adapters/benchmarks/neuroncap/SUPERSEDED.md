# Superseded files

Three rescued files were removed once their content had a proper home. They
remain in git history (commit `9a80414`, the rescue) if the original wording is
ever needed.

| Removed | Replaced by |
|---|---|
| `ladder.py` | `harness.py` for the driving logic, and `adapters/models/*/model.yaml` for the per-model launch matrix it hardcoded |
| `run_vp_variant.sh` | `VISIONPILOT_PROCESS_NOISE_DIST=0.3 paddock run ...` — see `adapters/models/visionpilot/model.yaml` |
| `../navsim/exports.sh.orig` | the `runtime.env` block in `adapters/models/drivejepa/model.yaml`, which carries the same five NAVSIM variables |
| `tools/benchmarks/neuroncap/make_videos.sh` | `NeuroNcapBenchmark.render_videos`, which runs automatically after every episode |

They were deleted rather than kept as reference copies because each contained a
full set of one machine's absolute paths. Keeping them would have meant either
an indefinite portability exemption, or a second definition of the launch matrix
that could silently disagree with `model.yaml`.

`make_videos.sh` was removed for a different reason than the three above: not
duplication of a launch matrix, but a layout it no longer matched. It looked for
frames at `<model>/<scenario>-0103/COMBINED_OUTPUTS`, which hardcodes one scene
of the fourteen and predates per-run subdirectories -- against a multi-run sweep
it found nothing and said `skip`. Its actual logic (10 fps, hold the last frame,
concat demuxer because the timestamps are not contiguous) moved into `_encode`
in `harness.py`, where it is covered by tests and applied without being asked.

`comparison_video.py` stays. It does something the harness does not -- all four
models side by side on one clock -- and it reads FC_TRAJ *frames*, so it needs
`video = "keep"` in `[benchmarks.neuroncap]`. It still hardcodes scene 0103 and
the old `alpamayo` model name; that is worth fixing when the full-set sweep gives
it more than one scene to compare.
