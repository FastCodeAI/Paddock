# Reviewing a pull request

How to review a paddock PR so that the number it eventually produces can be
trusted by someone at the other site, on a machine you have never seen. This
complements the checklist in [CONTRIBUTING.md](../CONTRIBUTING.md); that list
says *what* a PR must have, this page says *how to check it* and what has
gone wrong before.

The guiding rule: **separate what you verified from what you argued.** A review
that says "looks right" is worth less than one that says "ran the golden test,
passed; the CARLA flags I checked against a 0.0.4 checkout; the PID import I
could not resolve anywhere." Write the review the way `docs/ROADMAP.md` records
status — claim, evidence, strength — and mark every finding 🔴 / 🟠 / 🟡 so
severity is visible before the argument (see §10).

---

## 1. Run it, do not read it

```bash
git fetch origin pull/<N>/head:pr-<N>
git worktree add /tmp/pr-<N> pr-<N>
cd /tmp/pr-<N> && make check; echo "exit=$?"
```

- CI (`.github/workflows/check.yml`) has already run `make check` on the PR
  head **without** a site config; a red check there is a finding in itself.
  Your run adds what CI cannot: the tests that skip without `paddock.toml`,
  and the site's own vendor checkouts. Say which of the two you ran.
- Check `$?` directly. `make check 2>&1 | tail` reports *tail's* exit code and
  has let a failing test through before.
- `make check` includes `tests/test_portability.py`, the gate on real machine
  paths. A green run is the portability check; you do not need to grep for it.
- Run the adapter's own tests too if they live under `adapters/**/tests/` —
  they are collected by `make check`, but say so in the review.
- A PR that touches `tools/models/drivejepa/alpasim_driver/`, **the scaffold it
  imports** (`src/paddock/adapters/benchmarks/alpasim/scaffold.py`) **or the
  AlpaSim bridge** (`bridge.py`, `replan_rhythm.py`, `drivejepa_rhythm.py` in
  the same folder) needs one more run that CI cannot make, because those tests
  need AlpaSim's interpreter: `make check-alpasim-driver ALPASIM_PY=<alpasim python>`.
  It includes the test that the existing bridge profiles still send the same
  bytes (`test_bridge_switches.py`), which `make check` skips. If your
  site has no AlpaSim checkout, say in the review that this was not run.
- A PR that touches the openpilot adapter (`src/paddock/adapters/models/openpilot/`)
  needs `make check-openpilot OPENPILOT_REPO=<repos>/openpilot` too. Its
  geometry tests import openpilot's transforms, so `make check` skips them
  everywhere; this runs them in paddock's interpreter with the checkout on
  `PYTHONPATH`. If your site has no openpilot checkout, say so in the review.
- A PR that touches the Alpamayo 1.5 adapter (`src/paddock/adapters/models/alpamayo15/`)
  needs `make check-alpamayo ALPAMAYO_PY=<repos>/alpamayo1.5/a1_5_venv/bin/python`
  too. Its input tests, including the pinned inputs of every other benchmark,
  need torch and `alpamayo1_5`, so `make check` skips them; they run on the CPU
  without weights. If your site has no Alpamayo environment, say so in the review.
- If the site has a checkout of the upstream tool the PR shells out to (CARLA,
  neuro-ncap, NAVSIM's devkit), open it and verify the flags, file layouts and
  output schema the PR assumes. "Matches the 0.0.4 checkout" is a verified
  claim; "the flags look right" is not.

## 2. Contracts that must be imported, not restated

Paddock has exactly one definition of each of these. A PR that re-types them is
correct today and silently wrong the day the definition moves.

| Contract | Lives in | What re-typing looks like |
|---|---|---|
| navigation commands, canbus offsets | `paddock.policy.protocol` (`Command`, `CANBUS_ACCEL`, `CANBUS_VELOCITY`) | `canbus[13] = speed`, `return 3` |
| trajectory validation | `paddock.policy.protocol.Trajectory` | `np.asarray(r.json()["trajectory"])` |
| image wire encoding | `decode_image` in `protocol.py`; the `.npy` encoder in `runner.py` | a private `_b64` helper |
| subprocess environment | `paddock.runtime.env.PASSTHROUGH` | a hand-built `{"PATH": ..., "HOME": ...}` |
| site settings | `BenchmarkOverride.get(key, default)` | a private `_setting()` copied from a sibling adapter |
| "no scene filter" | `scenes=None` in `ClosedLoopBenchmark.episodes` | a set of other benchmarks' default ids |

A client script that runs inside a foreign interpreter (a CARLA agent, a
renderer plugin) can still import these: every process paddock launches gets
`REPO_ROOT / "src"` on `PYTHONPATH` for exactly this reason. If a PR's client
cannot import `paddock.*`, the fix is the env, not a copy of the constants.

## 3. Every external dependency is preflighted and documented

`preflight()` exists so a missing asset is reported in seconds, not an hour into
a sweep. Trace every `import` in any script the adapter launches and every path
it constructs, and ask two questions:

- Does `preflight()` check it exists?
- Does `docs/benchmarks/<name>.md` (or `docs/models/<name>.md`) tell a stranger
  where to get it?

A module that lives in a *second* upstream checkout (a model zoo, a team-code
directory, a patched fork) is the common miss: it is on the author's `PYTHONPATH`
and on nobody else's. Ask the author where it came from; if the answer is a repo,
it belongs in `repos.yaml` with a pinned sha.

## 4. A crash must not become a score

Closed-loop scorers often write a numeric record for a route that did not really
run — the leaderboard marks it `Failed - Agent crashed` in a status field and
still fills in `score_composed`. Read the scorer's output schema and check:

- Every failure the scorer distinguishes (crash, invalid sensors, setup failure,
  simulation crash) raises `RuntimeError` from `run_episode`, so the runner marks
  the run `partial` and `valid=false`.
- Legitimate poor outcomes (blocked, deviated, timed out) *are* scored — those
  are the benchmark working.
- The distinction is tested, not just commented.

The runner records whatever `run_episode` returns as a valid measurement. A zero
that came from an HTTP 500 is indistinguishable from a model that drove into a
wall, and it averages into the leaderboard.

## 5. Golden tests pin the original system

- Fixture values must come from the *original* tool's output (a real
  `results.json`, a real `metrics.json`), never from a previous run of the code
  under test. A test that agrees with itself proves nothing.
- The fixture file should say where it came from (route id, date, tool version).
- The test's docstring states the claim being pinned in one sentence.
- A parse-only golden test does not exercise the launch path. Say so in the
  review, and record the end-to-end status in `docs/ROADMAP.md` as
  "unverified" rather than leaving it implicit.

## 6. Metrics and their direction

- New metric names are either canonical (already used by another benchmark) or
  documented in the benchmark doc's metrics table with a *direction*.
- `paddock.results.report.lower_is_better` decides how the leaderboard sorts.
  Check a new name against its markers: a name that contains `_err`, `ade`,
  `fde`, `seconds` etc. is rendered lower-is-better whether or not that is meant,
  and a lower-is-better metric that matches nothing (a rate, a penalty count)
  needs adding to the exact list.
- Per-episode metrics that the runner will average must be averageable: a 0/1
  flag becomes a rate, a percentage stays a percentage. Do not mix a
  per-route count with a per-route fraction under one name.

## 7. Configuration

- Every `[benchmarks.<name>]` / `[models.<name>]` key the code reads appears in
  `paddock.toml.example` with a placeholder and in the doc with its meaning.
- Keys the code reads but never documents are the ones that "silently do
  nothing" for the next site.
- Anything that varies by site (paths, ports, GPU index, timeouts) is a setting;
  anything that changes *what is measured* (route set, scenario list, waypoint
  dt) is either declared in tracked yaml or hashed into the run id. A site
  setting that changes what is measured, like a routes subset, needs to be
  recorded in `run.json` so two sites' numbers stay distinguishable.

## 8. Shared-machine hygiene

Paddock runs on boxes other people are using.

- No `pkill` by bare process name or pattern: it kills a colleague's simulator.
  Kill the process group you started (`start_new_session=True`, `os.killpg`).
- No unconditional per-tick recording of frames to the run directory. Estimate
  the volume for a full sweep (frames × routes × size) and gate it on a setting
  if it is more than a few GB.
- Long subprocesses stream their logs to files as they run, not into memory
  until exit, so a hung route is diagnosable while it hangs.
- Anything left running after a timeout is the adapter's problem, not the
  next user's.

## 9. Docs and the PR description agree with the diff

Check the diff against every prose claim, because prose is what the next person
reads first:

- Which models declare the benchmark (`model.yaml` edits vs the ROADMAP line).
- Whether the ROADMAP entry moved out of "not started" when the code landed,
  and whether its verification column says what was actually verified.
- README support matrix still says "planned" for something the PR ships.
- A docstring that says "replaces the previous X" where the repo never had X.

## 10. Writing the review

Every finding carries a severity marker, so the author can triage the review
at a glance and a reader at the other site can tell a merge-blocker from a
preference without reading the argument.

| Marker | Meaning | Merge rule |
|---|---|---|
| 🔴 **Blocker** | wrong numbers, a crash recorded as a result, harm to other users of the host, or the feature cannot run as documented | must be fixed before merge |
| 🟠 **Should fix** | a correctness or contract gap that will bite the next site or the next adapter: ignored flags, restated contracts, missing preflight, docs that do not match the diff | fix in this PR, or land a tracked follow-up (ROADMAP entry or issue) before merge |
| 🟡 **Nit** | quality, efficiency, wording | author's call; never blocks |
| ✅ **Verified** | you ran it, or read the upstream code it depends on | label the evidence, not the finding |

Rules of thumb for the scale:

- If the leaderboard would show a number that is not what was measured, it is
  🔴, however small the diff that fixes it.
- If it only fails on the author's machine's absence (a module on their
  `PYTHONPATH`, a path in their shell), it is 🔴: the whole point of paddock is
  the other site.
- If the code is correct but the *record* of what ran is wrong (`run.json`
  claims three repetitions, or a scene id from another benchmark), it is 🟠.
- Anything that could be a follow-up PR without changing a number is 🟡.

And the structure:

- Open with the verdict and a legend line. Then say what is *right* about the
  PR, specifically enough that the author knows what to keep doing. "The pure
  `parse_results` with a real fixture is the pattern every adapter should
  follow" is useful; "nice work" is not.
- A ✅ table of what you actually verified, with the result. "Nothing beyond
  the diff" is an honest row.
- Findings grouped 🔴, then 🟠, then 🟡, numbered so replies can cite them.
  Anchor every one to `file:line` in the PR's head commit. For each: what
  breaks, under what input, and the concrete fix. A finding without a failure
  scenario is a preference, and should be 🟡.
- Mirror the marker on the inline comment for the same finding, so the
  file view and the summary agree.
- Write as the most senior person in the room would: direct, specific, no
  hedging on things you verified, no certainty on things you only argued,
  and the principle stated once ("we do not kill processes we did not
  start") rather than re-derived per finding.

## Reviewer's quick list

- [ ] `make check` run on the PR branch, exit code checked
- [ ] `make check-alpasim-driver` run if the AlpaSim driver, its scaffold or the bridge changed, or marked not run
- [ ] `make check-openpilot` run if the openpilot adapter changed, or marked not run
- [ ] `make check-alpamayo` run if the Alpamayo 1.5 adapter changed, or marked not run
- [ ] upstream flags / schemas verified against a real checkout, or marked argued
- [ ] every external import and path is preflighted and documented
- [ ] scorer failure statuses raise `RuntimeError`; poor outcomes still score
- [ ] wire-format constants imported from `paddock.policy.protocol`
- [ ] subprocess env built from `runtime.env.PASSTHROUGH`, `REPO_ROOT/src` on path
- [ ] golden fixture values come from the original tool, provenance noted
- [ ] new metric names documented with direction; `lower_is_better` checked
- [ ] every config key read is in `paddock.toml.example` and the doc
- [ ] nothing kills processes by name; recording volume estimated and gated
- [ ] ROADMAP, README, docs and PR description agree with the diff
- [ ] review states what was verified vs argued
- [ ] every finding carries 🔴 / 🟠 / 🟡, summary and inline comments agree
