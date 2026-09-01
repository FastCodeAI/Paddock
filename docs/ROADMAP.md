# Roadmap

Milestone status for paddock. Kept in the repo rather than in anyone's notes,
because both organisations need to see it.

| | Milestone | Status | Evidence |
|---|---|---|---|
| **M1** | Foundation and contribution contract | **done** | `make check`; `paddock doctor`; `tests/test_alt_site.py` |
| **M2** | Rescue untracked work into adapters and patches | **done (models)** | patches verified byte-identical; 4 model adapters converted |
| **M3** | Registries, `paddock setup`, data projection | partial | registries and projection done; `paddock setup` not implemented |
| **M4** | Ingest existing results | **done** | 36 runs, 77 tests, golden values from the original reports |
| **M5** | Reproduce closed-loop (neuroNCAP) | **partial** | adapters proven identical to the original nodes; see `docs/benchmarks/neuroncap.md` |
| **M6** | Reproduce open-loop (KITScenes) | **not started** | — |

## What "done" means here, and what it does not

M1–M4 are verified in the sense that **the code does what it claims about data
that already exists**. No model has been run through paddock yet. The adapters
import cleanly, their contracts hold under test, and the ingest reproduces
published numbers — but inference has not happened.

That distinction matters, so it is spelled out:

| Claim | Verified how | Strength |
|---|---|---|
| Patches reproduce the working trees | clean checkout at pinned sha + apply → `diff` clean on every patched file | **proof** |
| Ingest matches published results | 25 golden assertions against the numbers in the original `report.md` files, not against paddock's own output | **strong** |
| `crop_box` unification is behaviour-preserving | KITScenes crop is byte-identical: `(1, 151, 3503, 2121)` | **strong** |
| Repo works at another site | `tests/test_alt_site.py` resolves a foreign layout; `test_portability.py` bans machine paths | **strong** |
| Adapters obey the Policy contract | per-model contract tests: frame convention, shapes, guards, reset semantics | **moderate** — no GPU, no weights |
| Adapters produce the same output as the original nodes | head-to-head on neuroNCAP: paddock and the original server both return `2.8269616956051293` | **proof**, for visionpilot + openpilot on one scenario |
| Adapters reproduce the **August** recorded numbers | they do not, and neither does the original code | **open** — see below |

M5 closed the first of those rows and opened a new question in doing so.
Running paddock's visionpilot adapter and the original `visionpilot_server.py`
head to head against the same renderer produced **the same score to full float
precision** — so the conversion is faithful. But both disagree with what the
same scenario recorded in August (2.827 today against 3.145 on 27 Aug).

Since the original code reproduces today's number and not August's, this is not
a paddock regression — but **the August neuroNCAP figures cannot currently be
reproduced**, and comparisons against them are unsafe until the cause is found.
Details and candidate causes in `docs/benchmarks/neuroncap.md`.

Two models on one scenario is not the whole of M5. The remaining models
(alpamayo15, drivejepa) and scenarios are still unrun.

This is also why **the original code has not been deleted**. It stays in place
until paddock reproduces its numbers.

## Remaining work

**M3 — `paddock setup`.** Clone or adopt vendor repos, build the `external/`
symlink farm, apply patches, run per-model `setup.sh`. Today `doctor` reports
what is missing; `setup` does not yet fix it.

**M5 — closed loop.** `paddock run --bench neuroncap --model all --runs 1
--scene 0103` must reproduce `ladder_current.log`:

| model | stationary | frontal | side |
|---|---:|---:|---:|
| openpilot | 5.0 | 1.862 | 5.0 |
| drivejepa | 5.0 | 2.072 | 5.0 |
| alpamayo | 5.0 | 0.000 | 1.176 |
| visionpilot | 3.145 | 1.684 | 1.074 |

Needs: the neuroNCAP benchmark adapter (rescued but not converted), the renderer
launcher, `runtime/proc.py` and `runtime/gpu.py`.

**M6 — open loop.** `paddock run --bench kitscenes --model openpilot` must
reproduce ADE 1.599 / median 1.323 / FDE@3s 2.894 at n=223, and
`--model alpamayo15 --arm nonav` must reproduce 0.921 / 2.003.

Needs: the KITScenes benchmark adapter converted, and the sample-list question
below settled.

## Open questions

**Which KITScenes tier serves the eval set.** The three copies may hold
different sequence sets rather than nested subsets. Until that is checked, a
"reproduced" number could be over different data. `paddock doctor` should
cross-check the sample list against each configured tier.

**The NAVSIM headline.** The CSV and its run log disagree (0.90673 vs 0.90488)
over the same 242 scenarios. Both are stored; which belongs in a comparison
depends on what the other side computed. See
`src/paddock/results/ingest/navsim.py`.

## Beyond reproduction

Not started, and larger than everything above:

- **Bench2Drive / CARLA** — no install, no checkout, no results anywhere. Greenfield.
- **nuScenes open-loop** — needs `nuscenes_infos*.pkl`, which do not exist, and the
  local copy has no `sweeps/` and empty radar. Camera-only is viable; fusion is not.
- **alpamayo2, vjepa2, sunnypilot** — never evaluated.
- **Training** — the framework's eventual purpose; out of scope until evaluation is trustworthy.
