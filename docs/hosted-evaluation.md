# Hosted evaluation

An author can have Paddock score a driving model without installing Paddock,
downloading a dataset or owning a GPU. They put a small folder in their own
repository and open an issue. Our GPU box builds the model, runs it on the
benchmarks and posts the scores on the issue.

This page has two halves: [for model authors](#for-model-authors) and
[for whoever runs the box](#for-whoever-runs-the-box).

## For model authors

### 1. Add a Paddock folder to your repository

Copy Paddock's model template,
[`src/paddock/adapters/models/_template/`](../src/paddock/adapters/models/_template/),
into a folder of your repository, `paddock/` unless you say otherwise. It holds
three files. [adding-a-model.md](adding-a-model.md) explains each in full; the
rules below are the ones a hosted run adds.

- **`model.yaml`.** `runtime.venv`, every `runtime.pythonpath` entry and
  `runtime.cwd` must start with `{repo}`, the root of your repository: the build
  can write nowhere else. `runtime.kind` is `uv`, `conda` or `shell`. List the
  benchmarks your model can run under `benchmarks:`; the others are skipped.
  Declare `cameras` and `vram_gb` truthfully, since the run renders and reserves
  what you declare.
- **`setup.sh`.** It builds your environment at `$PADDOCK_VENV` and downloads
  your released weights into `$PADDOCK_WEIGHTS`. It runs once, with internet
  access, for at most 60 minutes. `uv` is on the box's PATH; anything else your
  build needs, it installs itself.
- **`policy.py`.** `load()`, `reset()` and `infer(obs) -> Trajectory`, as in the
  template. Read images with `obs.image(camera)`: a hosted run sends frames as
  pixels, never as file paths. HD map files (`obs.map_files`) are not available
  to hosted runs yet.

The template already drives at constant velocity, so you can submit it unchanged
to check the plumbing, then put your model in.

### 2. Open an issue

Push the folder, copy the full 40-character commit hash, and open an
**Evaluate my model** issue. The form asks for the repository, the commit, the
folder, where your weights are and their licence, and the tier:

| Tier | What runs | GPU time |
| --- | --- | --- |
| quick | KITScenes, nuScenes, NAVSIM, a neuroNCAP subset, one Bench2Drive route, one AlpaSim scene | about 40 minutes |
| closed | the neuroNCAP, Bench2Drive and AlpaSim smoke subsets | hours |
| all | both | hours |

These are the regression gate's fixed subsets ([gate.md](gate.md)), so your
numbers line up with the in-repo models' own smoke runs.

### 3. Wait for the comment

A maintainer reads your folder before anything runs: it is code that will run
on our GPU. When they add the `approved` label, the issue is queued, and a
comment says how many jobs are ahead. When the job finishes, a second comment
gives each benchmark's headline metric, sample count and run id, or says why the
job stopped (a failed build or contract check quotes the end of its log). The
`closed` and `all` tiers need a second label, `approved-full`, which a
maintainer adds once a quick run has finished cleanly.

Smoke subsets are stored unranked, as every smoke run is. To join the ranked
[leaderboard](leaderboard.md), open a pull request that adds your folder to
Paddock ([adding-a-model.md](adding-a-model.md)).

### What your code can and cannot do on the box

While it is scored, your model runs with no network at all: it cannot download
anything, so `setup.sh` must fetch everything first. It sees the host's system
read-only, its own workspace (your repository, your environment, your weights)
read-write, and nothing of the datasets, other jobs or other models. One author
has one submission in the queue at a time.

## For whoever runs the box

### What runs where

Two long-running processes, both as the queue's `paddock` account:

- **The intake**, `python -m tools.submissions serve`. The only process with a
  GitHub token. Every 60 seconds it posts the results of finished jobs, then
  looks for open issues labelled `approved` and not yet `queued`. For each, it
  checks the form, fetches the author's repository at the named commit into a
  sandbox workspace, and queues a `submission` job pinned to Paddock's `main`.
- **The queue worker**, `python -m tools.gate_queue run` ([queue.md](queue.md)).
  For a submission job it makes a checkout of its own, copies the author's
  folder in as model `sub_<issue>`, runs `setup.sh` and `paddock verify` inside
  the sandbox, then runs the gate's smoke tier with nothing to compare against.

Every command that runs the author's code goes through bubblewrap
(`python -m paddock.runtime.sandbox`): the build, `paddock verify`'s driver and
the model's policy node. The job's own site config (`<root>/jobs/<job>/site.toml`)
names that launcher in `[models.sub_<issue>] launcher`, which is how `paddock run`
and `paddock verify` start the node inside it. The worker also sets
`PADDOCK_INLINE_IMAGES=1`, so open-loop frames go to the node as pixels and the
dataset folders can stay hidden.

### Setting it up (once)

The queue must already be set up ([queue.md](queue.md#setting-a-site-up-once)).

1. **bubblewrap.** `sudo apt install bubblewrap`, then, as paddock,
   `python -m paddock.runtime.sandbox check`. Both lines must say `ok`. On
   Ubuntu 24.04 the AppArmor profile that comes with the package allows the
   namespaces it needs; elsewhere, unprivileged user namespaces must be on.
2. **uv on the system PATH.** Homes are hidden from submitted code, so a `uv` in
   `~/.local/bin` is not there for its build. Install it to `/usr/local/bin`.
3. **Config.** Add to the worker's `paddock.toml`:

   ```toml
   [submissions]
   repo    = "FastCodeAI/Paddock"        # whose issues are submissions
   sandbox = "/srv/psbx"                 # one workspace per job; keep the path short
   token_env = "PADDOCK_GITHUB_TOKEN"    # the variable holding the token, never the token
   # poll_s = 60
   # build_timeout_s = 3600
   # hide = ["/path/to/other-teams-data"]  # more folders to hide, beyond the queue, roots and datasets
   ```

   Keep `sandbox` short: a node's Unix socket lives in it, and a socket path may
   be at most 107 bytes.
4. **Labels.** Create `evaluate`, `approved`, `approved-full`, `queued` and
   `evaluated` on the repository (`gh label create <name>`).
5. **Token.** A fine-grained token with read and write access to issues on that
   repository and nothing else. Put `PADDOCK_GITHUB_TOKEN=<token>` in
   `~/.config/paddock/intake.env`, mode 600.
6. **Start both.** `python -m tools.gate_queue service` and
   `python -m tools.submissions service` print their systemd user units. Install
   each in `~/.config/systemd/user/`, then
   `systemctl --user enable --now paddock-queue paddock-intake`.
7. **Try it.** Open an issue pointing at a repository that holds the unchanged
   model template, with `benchmarks: [nuscenes]` in its `model.yaml`. Add
   `approved` and wait for the scores comment.

### Day to day

- **Approving.** Read the folder: `setup.sh` runs with internet access. Add
  `approved` for a quick run and `approved-full` for a longer one.
  `python -m tools.submissions show <issue>` prints what the intake makes of an
  issue's form, without writing anything.
- **A refused form** gets a comment listing every problem and loses `approved`.
  Approve it again once the author has fixed it.
- **Watching and stopping.** Submissions are queue jobs:
  `python -m tools.gate_queue ls`, `cancel` and `retry` work on them as on any
  job. Each job's folder holds `build.log`, `verify.log` and `gate.log`.
- **Disk.** The intake removes a job's workspace once its result is posted. The
  job's checkout stays under `<root>/checkouts/<job>`, with the run records in
  its `results/`.

### What the sandbox protects against, and what it does not

| Risk | What stops it |
| --- | --- |
| Harmful code in `setup.sh` or `policy.py` | A maintainer reads it first; nothing runs before `approved`. |
| Code reading secrets, other jobs or the data | Homes, the queue, the site's roots, the datasets and the sandbox root are empty inside; only its own workspace is writable. `HF_TOKEN` and GitHub tokens are not passed in. |
| Code calling out while it is scored | The node has a network namespace of its own with only loopback. |
| Reading future frames or ground truth | Frames arrive as pixels; the dataset folders are hidden. Closed-loop simulators already send rendered frames. |
| A crafted image payload running code in the node | The torch-free decoder refuses every pickle global but the few a tensor needs. |

bubblewrap is a set of Linux namespaces, not a virtual machine: a kernel flaw
would cross it. The human read is the main guard. If submissions are ever run
without one, move the worker to a machine that holds nothing else.
