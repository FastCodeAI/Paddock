# Open work at 22 October 2026, and what lands before v1.0.0

*Level 3, reference. A snapshot of open pull requests, issues and owed re-runs, with a
proposed disposition for each against the v1.0.0 feature freeze on 13 November.
Up: [8 · Release v1.0.0](../1-strategy/08-release-v1.0.0.md)*

**The rule proposed for the freeze:** before 13 November, land only what changes
certified numbers or the trial path. Everything else continues on `main` after the
release. Dispositions below are proposals for the release owner, not decisions.

## Paddock pull requests (14 open)

None of the fourteen is approved. Five conflict with their base branch, and one fails
CI.

| PR | What it adds | State on 22 Oct | Proposed |
|---|---|---|---|
| #100 | Regression gate: Bench2Drive crashes isolated, launches take turns per GPU, every verdict states its smallest detectable difference; run records carry the Paddock commit | CI green, clean | **land by 13 Nov**: certification and provenance need it |
| #102 | Gate: NAVSIM's safety and comfort terms beside every pair's score | stacked on #100 | **land by 13 Nov**, after #100 |
| #57 | VisionPilot synced to upstream; claims nuScenes L2 −26% to −32% | reviewer's blocker: the numbers predate the change | **land by 13 Nov after re-measuring**: un-stales VisionPilot's row |
| #95 | Drive-JEPA's momentum-aware selection on by default | gate runs pending; AlpaSim re-run waits on issue #94 | land by 13 Nov **if** its gate runs finish; otherwise after |
| #92 | AutoE2E on neuroNCAP: 3.93, 25% collisions | reviewer asked for a re-run with #89's fix; conflicts | land **if** the re-run is done by 13 Nov; otherwise after |
| #79 | openpilot on all 220 Bench2Drive routes: driving score 32.88 → 42.75 with two opt-in settings; failure analysis | conflicts; large (+31,544 lines) | land after a rebase if review allows; otherwise the evidence pack uses the merged 30.09 row |
| #96 | AutoE2E on Bench2Drive at 10 Hz; recorder, dashboard, early stop | no score yet (42 of 220 routes); conflicts | after v1.0.0 |
| #87 | A privileged ground-truth channel on the model interface | conflicts; nothing in Paddock uses it yet | after v1.0.0: an interface change is a poor fit for a release freeze |
| #86 | Alpamayo decision-analysis tools | stacked on #76 | after v1.0.0 |
| #76 | neuroNCAP reasoning-video tool made to fail loudly | based on the branch of #73, closed unmerged | after v1.0.0, rebased onto `main` |
| #83 | WA-JEPA adapter, one smoke run on scene 0103 | smoke only, `valid: false` | after v1.0.0 |
| #90 | VisionPilot scenario classifier for following distance | no measurements; a subset of #91 | after v1.0.0 |
| #91 | VisionPilot improvements and fine-tuning scaffolding | **fails CI** (machine-specific paths) | after v1.0.0 |
| #53 | Sim-to-real converter and study | no push for 31 days; review comments open | after v1.0.0 |

## Paddock issues (6 open)

| Issue | What | For v1.0.0 |
|---|---|---|
| #94 | One set of AlpaSim settings for every model | after, unless #95 needs it |
| #69 | AlpaSim results mix 10- and 30-scene rows, and a retrained checkpoint has no name of its own | **fix for the presentation leaderboard** |
| #48 | A run whose inputs were changed to fit a benchmark is still ranked; #82 and #101 may cover part | **check for the presentation leaderboard** |
| #50 | A column in the Qwen-Drive doc does not reproduce | fix, or keep it out of the evidence pack |
| #65, #67 | The re-ranker should check its inputs at load | after |

## Owed re-runs and fixes

From `docs/ROADMAP.md` and the marks in the support matrix:

| What | Why | For v1.0.0 |
|---|---|---|
| VisionPilot on every benchmark | a curvature-scale fix since its runs; every cell in its row is marked stale | **yes**, if VisionPilot stays in the trial matrix |
| openpilot on comma2k19, at its pinned commit | the recorded row came from a different commit | **yes** |
| comma2k19 ground truth in the car's frame | a ~2° frame error reorders the board | **yes**: re-score from the stored records |
| Drive-JEPA on nuScenes after its camera fix | the cell predates the fix | yes, if Drive-JEPA is in the matrix |
| Drive-JEPA and VisionPilot on NAVSIM | owed re-runs, ~4 h and ~3.5 h | yes, if in the matrix |
| nuPlan beyond 1 of 9 camera shards | every run is still invalid | no: nuPlan stays out of v1.0.0 |

## Gallops pull requests (4 open; Gallops has no CI)

| PR | What | State |
|---|---|---|
| #20 | Extra prediction heads on Drive-JEPA: held-out nuScenes L2 at 3 s 3.788 → 2.934 | "not a ship candidate": the gate has not run |
| #19 | nuScenes scene selection for teacher training | infrastructure, 5 tests |
| #18 | A privileged neuroNCAP teacher for Alpamayo | needs Paddock #87; no GPU runs |
| #17 | Expert imitation (brake, or brake then swerve) | learned on training scenes, did not transfer |

---

## Sources

**Verified:**
- Pull requests, issues, review and CI state: GitHub (`gh pr list`, `gh pr view`,
  `gh issue list`) on 22 October 2026. Each claim was checked against the PR's own
  description.
- Owed re-runs: `docs/ROADMAP.md` (Remaining work, M8) and the `README.md` support
  matrix.

**Argued:**
- Every disposition.
