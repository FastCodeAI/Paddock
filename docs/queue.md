# The gate queue

One queue of regression-gate runs for the whole site. Anyone in the `paddock`
group submits a checkpoint; one worker, running as the `paddock` account, runs
the jobs one at a time with `tools.gate smoke` and writes each verdict to the
job's directory. `tools/gate_queue.py` is the code; [gate.md](gate.md) says what
the gate measures and how to read a verdict.

Why: when two people smoke checkpoints by hand on one shared GPU, both start
into the same free memory, both pin the same four cores, and each pairs against
baselines made in their own checkout under their own account. The queue runs
every job the same way.

## Using it

```bash
python -m tools.gate_queue submit --model drivejepa --env DRIVEJEPA_CHECKPOINT=/path/to/step_1000.ckpt
python -m tools.gate_queue submit ... --tier all --note "lateral loss, step 1000"
python -m tools.gate_queue submit ... --ref my-branch     # run a pushed branch's code, not main's
python -m tools.gate_queue ls                             # the queue and the last 10 finished jobs
python -m tools.gate_queue cancel <job>
python -m tools.gate_queue retry <job>                     # queue it again; finished runs are kept
```

Run these from a paddock checkout, with your site `paddock.toml` (it needs the
`[queue]` table below). `submit`:

- **checks what the job changes.** Each `--env` key must be one the model's
  `model.yaml` already sets (`runtime.env` or `runtime.tuning_env`), such as
  `DRIVEJEPA_CHECKPOINT`. A job cannot set `CUDA_VISIBLE_DEVICES` or
  `PYTHONPATH`: the runs would record the change as part of the model, or not at
  all.
- **copies the checkpoint.** A value that names a file or a directory is copied
  into the shared store, under a name made from its content (its sha256), and
  the job uses the copy. So you can overwrite or delete your file after
  submitting, and the job still measures what you submitted. A file brings the
  small files beside it (under 1 MB, such as gallops' `checkpoint.json`).
- **fixes the code.** The job runs the commit `origin/main` is at when you
  submit, or the branch or commit `--ref` names. That commit must be on GitHub,
  so whoever reads the result can check out exactly what it ran. Your local
  checkout's own state does not matter.

`--tier`, `--bench` and `--keep-going` mean what they do for `tools.gate smoke`.
The default tier is `quick` (about 40 minutes); `all` adds the closed-loop
benchmarks (hours).

**Where the result is.** `ls` prints each job's outcome: `done: SHIP`,
`done: HOLD`, `done: FAIL`, `cancelled` or `error: <why>`. The job's directory,
`<root>/jobs/<job>/`, holds:

| file | what it is |
|---|---|
| `job.json` | what was submitted: model, env, commit, tier, who, when |
| `gate.log` | everything the gate printed, including the comparison table |
| `result.json` | the outcome, written when the job ends |
| `verdict.json` | the verdict, each benchmark's run id, and every pair (`tools.gate smoke --json`) |
| `baseline.toml` | the baselines this job was compared with |
| `smoke/` | the gate's state directory: `manifest.toml`, one log per benchmark |

**Cancel and retry.** `cancel` drops a job that has not started, or stops a
running one within about 10 seconds (its running benchmark is stopped cleanly,
as `kill <gate pid>` does). Anyone in the group can cancel any job. `retry` puts
a finished, failed or cancelled job at the back of the queue; the gate resumes
it and keeps every benchmark that already finished. Use it when a run failed for
a reason that was not the checkpoint (a renderer that did not start in time).

## Baselines

The queue keeps its own baselines, made by the same account in the same way as
the jobs compared with them. Make them once per model, and again whenever the
model's own settings change on main (as PR #106 changed Drive-JEPA's):

```bash
python -m tools.gate_queue submit --model drivejepa --baseline --tier all
```

A `--baseline` job runs the released model (no `--env`). When it ends, its
finished runs replace those benchmarks' entries in
`<root>/baselines/<model>.toml`, and benchmarks it did not run keep theirs: a
`--tier quick` baseline job and a `--tier closed` one made later add up to both
tiers. The file as it was before moves to `<root>/baselines/replaced/`. A
benchmark the job ran and did not finish loses its entry (a comment says so);
`retry` the job to fill it in. `submit` refuses a candidate until its model has
baselines, or a baseline job for it is already queued.

Each candidate copies the baselines into its own directory when it starts, so
baselines replaced later do not change what it was compared with.

## How the worker runs a job

1. Takes the job that has waited longest. A job that was started and has no
   result (the worker was stopped, or the machine rebooted) comes first.
2. Checks the job's commit out under `<root>/checkouts/<commit>`, once per
   commit. The runs write their `run.json` into that checkout's `results/`.
3. Checks that git will read the model's own clone as the `paddock` account
   (see setup step 2): every run records that clone's commit, and a refused
   clone would be recorded as no commit, silently.
4. Runs `tools.gate smoke --execute` there, with the job's env, the baselines,
   the state directory inside the job's, and the worker's cores and ports. With
   `stage` set, the checkpoint is first copied to that local disk
   (`--stage-dir`), and the copy is deleted when the job ends.
5. Writes `result.json`. For a baseline job, also the model's baselines.

Stopping the worker (`systemctl --user stop paddock-queue`) stops the running
gate cleanly and leaves its job unfinished; the next start resumes it. A lock
file in `<root>` keeps a second worker from starting.

## Setting a site up (once)

You need the `paddock` account and group (members: everyone who submits), a
local directory and a large one, both owned by `paddock:paddock` with mode 2775
(group-writable, and new files stay in the group). Below, `<root>` is the local
directory and `<store>` the large one. Steps marked *as paddock* run under
`sudo -u paddock`.

1. **Site config.** Every submitter's `paddock.toml`, and the worker's own copy
   (say `<root>/paddock.toml`, a copy of the site's), gets:

   ```toml
   [queue]
   root  = "<root>/queue"          # jobs, the code checkouts, baselines, the worker's lock
   store = "<store>/checkpoints"   # submitted checkpoints, by content
   # The worker's own settings (submitters can leave them out):
   stage         = "<root>/stage"           # fast local copy of a running job's checkpoint
   python        = "<root>/venv/bin/python" # paddock's interpreter for the worker
   cores         = ["12-15"]                # one taskset lane per entry, away from people's own runs
   base_port     = 9800
   renderer_port = 8800
   ```

   In the worker's copy, point `[roots] runs` at a directory `paddock` can write
   (say `<store>/runs`), and copy the site runs folder's `geometry/` into it:
   neuroNCAP's PDM terms read the road-area files built there
   (`tools/benchmarks/nuscenes/build_pdm_geometry.py`), and without them its runs
   record no PDM terms.
2. **Git, as paddock.** `git config --global safe.directory '*'`. Git refuses a
   clone owned by another account ("dubious ownership"), and the vendor clones
   under `[roots] repos` belong to whoever made them.
3. **Read access.** `paddock` must read everything the site config names, and
   run every model's interpreter. Grant the group read access where it is
   missing, without changing owners: `setfacl -R -m g:paddock:rX <dir>`, plus
   `setfacl -d -m g:paddock:rX` on its directories so new files are readable
   too. On shakaal these were the NeuRAD checkpoints and the NAVSIM metric
   cache. Look also at where each model environment's `bin/python` link points
   (`readlink -f`): an environment built on a Python inside someone's home
   folder cannot be run by another account. CARLA needs no write access to its
   own folder: it writes its logs under the running account's home.
4. **The worker's Python, as paddock.** `uv venv <root>/venv --python 3.11`, then
   `uv pip install --python <root>/venv/bin/python '<a paddock checkout>[serve,viz,comma2k19]'`.
   (If `uv` is in your home folder, `paddock` cannot run it; give it a copy, such
   as `<root>/bin/uv`.)
5. **The queue, as paddock.** `PADDOCK_CONFIG=<root>/paddock.toml <root>/venv/bin/python -m tools.gate_queue init`,
   from a paddock checkout. It makes the directories and `<root>/queue/repo.git`,
   the repository submitters push each job's commit into.
6. **The first job.** Submit each model's `--baseline` job (above). It waits
   for the worker, and puts its commit in `repo.git` for the next step.
7. **The worker's own checkout, as paddock.**
   `git -C <root>/queue/repo.git worktree add --detach <root>/worker refs/jobs/<job>`,
   then `uv pip install --python <root>/venv/bin/python -e '<root>/worker[serve,viz,comma2k19]'`,
   so that the venv's own `paddock` is that checkout's: `paddock doctor` run as
   `paddock` then sees its models. Each job still runs its own checkout's code
   (the gate puts that checkout's `src/` first on the path). To update the
   worker later: `git -C <root>/worker checkout --detach <newer commit>`.
8. **The service, as paddock.** From `<root>/worker`:
   `PADDOCK_CONFIG=<root>/paddock.toml <root>/venv/bin/python -m tools.gate_queue service > ~/.config/systemd/user/paddock-queue.service`,
   then `systemctl --user daemon-reload && systemctl --user enable --now paddock-queue`.
   Under `sudo -u paddock`, `systemctl --user` also needs
   `XDG_RUNTIME_DIR=/run/user/$(id -u paddock)`, and the account needs lingering
   (`loginctl enable-linger paddock`) so the worker keeps running with nobody
   logged in. `journalctl --user -u paddock-queue` shows what it is doing.

## Limits

- **One job at a time.** The worker does not share the card between jobs; a
  job's own benchmarks share it as `tools.gate smoke` always does (lanes).
- **Simulators start cold for every benchmark run.** Keeping the neuroNCAP
  renderers and CARLA running between jobs is the next step, and changes how
  runs are made, so it comes with its own check that no score moves.
- **Nothing is cleaned up yet.** Checkouts, stored checkpoints and job
  directories stay until removed by hand. Do not remove a checkout that a
  baselines file names in `results =`.
- **Jobs run as a shared account.** A checkpoint is code (it is unpickled), and
  anyone in the group can submit one; the group is the trust boundary.
