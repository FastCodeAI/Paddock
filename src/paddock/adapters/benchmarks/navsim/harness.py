"""NAVSIM as a paddock benchmark: PDMS/EPDMS over OpenScene scenarios.

Paddock does not score NAVSIM. NAVSIM's own PDM scorer does, in the vendor's
environment, and paddock supplies it a model. That division is deliberate --
`ExternallyScoredBenchmark` in paddock/benchmark.py says why -- and it is what
lets four models in four incompatible environments be scored by one scorer.

What happens on `paddock run --bench navsim --model X`:

1. paddock starts X's policy node and health-checks it (`proc.model_node`),
2. this harness runs `run_pdm_score_sequential.py` in the devkit's interpreter,
   pointed at `agent=paddock_proxy` with the node's URL,
3. the proxy agent (adapters/benchmarks/navsim/agent.py) translates each
   `AgentInput` into paddock's wire format and returns the node's plan,
4. NAVSIM writes its usual score CSV, which is parsed back through the *same*
   parser `paddock ingest` uses, so a run scored today is the same shape as the
   August run already in the store.

**A note on comparability.** Drive-JEPA's native NAVSIM agent receives
`metric_cache` and `past_ego_simulated_states` from the scoring loop and uses
them to re-rank its proposals against the scorer's own comfort metric. No model
behind an HTTP node can do that, and no other model in paddock has an
equivalent. Runs through `paddock_proxy` are therefore the comparable set;
the vendor's recorded 0.90673 is not a baseline the other three can be measured
against. Set `PADDOCK_NAVSIM_AGENT` to a hydra agent group name to run a
vendor's native agent instead -- used to reproduce a recorded run and to
quantify exactly that gap, not as a normal path.

Site settings, `[benchmarks.navsim]` in paddock.toml:

    python         interpreter that can import navsim and nuplan
    devkit         NAVSIM devkit root (the directory holding navsim/)
    metric_cache   prebuilt metric cache; `run_metric_caching.py` builds one
    split          train_test_split name, default navtest
    workers        scorer worker processes, default 4
"""

from __future__ import annotations

import csv
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from paddock.adapters.benchmarks.navsim.convert import NAVSIM_INTERVAL_S
from paddock.benchmark import Evaluation, ExternallyScoredBenchmark
from paddock.results.ingest import navsim as navsim_ingest
from paddock.runtime.proc import tail as _tail

#: Logs per scorer invocation. The vendor scorer keeps every result in worker
#: memory and writes one CSV at the very end, so anything that dies after the
#: last forward pass -- its own aggregation, the node, the box -- loses the
#: whole run. 12,146 alpamayo scenarios (11.5 h) were lost exactly that way.
#: Scoring in chunks bounds the loss to one chunk, writes a CSV per chunk as it
#: completes, and makes a run resumable from the chunks that finished.
#: 0 scores the split in one invocation, as the vendor does.
DEFAULT_CHUNK_LOGS = 4

#: A previous run's artifacts directory whose completed chunks are reused.
RESUME_ENV = "PADDOCK_NAVSIM_RESUME_FROM"


def chunk_logs(logs: list[str], size: int) -> list[list[str]]:
    """Split *logs* into consecutive chunks of *size*, in the given order."""
    if size <= 0:
        return [list(logs)]
    return [list(logs[i : i + size]) for i in range(0, len(logs), size)]


def hydra_list(items: list[str]) -> str:
    """A hydra list override literal. Quoted: log names carry dots."""
    return "[" + ",".join(f"'{item}'" for item in items) + "]"


def completed_chunk_csv(chunk_dir: Path) -> Path | None:
    """The scorer's CSV for a chunk that finished, else None."""
    csvs = sorted(chunk_dir.glob("*.csv")) if chunk_dir.is_dir() else []
    return csvs[-1] if csvs else None


def merge_chunk_csvs(csv_paths: list[Path], out_path: Path) -> Path:
    """Concatenate per-chunk score CSVs into one the parser already understands.

    Each chunk's CSV carries its own aggregate rows (average_all_frames and
    friends), which mean nothing across chunks and are dropped; paddock
    recomputes aggregates from per-sample records anyway.
    """
    header: list[str] | None = None
    rows: dict[str, dict[str, str]] = {}
    for path in csv_paths:
        with path.open() as handle:
            reader = csv.DictReader(handle)
            if header is None:
                header = list(reader.fieldnames or [])
            for row in reader:
                token = (row.get("token") or "").strip()
                if token in navsim_ingest.AGGREGATE_TOKENS:
                    continue
                # A token scored twice -- once failed, once by the retry pass --
                # keeps the scored row. A failed row never overwrites a scored one.
                valid = (row.get("valid") or "").strip().lower() != "false"
                if token in rows and not valid:
                    continue
                rows[token] = row
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header or [])
        writer.writeheader()
        writer.writerows(rows.values())
    return out_path


PROXY_AGENT = "paddock_proxy"

#: Where the devkit keeps each train_test_split's own config. `data_split:`
#: in there says which OpenScene split the scorer reads; preflight needs it
#: before hydra runs, so it is read from the same file rather than repeated.
SPLITS_DIR = "navsim/planning/script/config/common/train_test_split"

#: Environment switches that change what a NAVSIM run measures. Each is
#: recorded in run.json through `run_config()`, and the ones that make a run
#: unrankable set `valid: false` through `invalid_reasons()`.
ENV_AGENT = "PADDOCK_NAVSIM_AGENT"
ENV_ALLOW_PARTIAL = "PADDOCK_NAVSIM_ALLOW_PARTIAL"
ENV_ALLOW_SLOW = "PADDOCK_NAVSIM_ALLOW_SLOW"
ENV_OVERRIDES = "PADDOCK_NAVSIM_OVERRIDES"

#: Fewest scenarios NAVSIM can aggregate. `calculate_two_frame_extended_comfort`
#: pairs consecutive frames within a log -- `range(len(group) - 1)` -- so a
#: single scenario yields no pairs, the aggregate frame comes back with no
#: columns, and the run dies in pandas with a KeyError naming neither the cause
#: nor the fix. Two consecutive frames is the floor.
MIN_SCENARIOS = 2

#: OpenScene's observation rate. Not a configuration choice -- the logs carry one
#: keyframe every 0.500 s and exactly one image per keyframe, so there is no
#: denser history to build. A model needing more than this cannot be measured
#: here, only mis-measured.
INPUT_HZ = 1.0 / NAVSIM_INTERVAL_S
CONF_DIR = Path(__file__).resolve().parent / "conf"
SCORER = "navsim/planning/script/run_pdm_score_sequential.py"


def input_rate_problems(min_input_hz: float, model_name: str) -> list[str]:
    """Why a model needing denser input than OpenScene has cannot be scored.

    Pure, so the runner can ask before loading the model and `--dry-run` can
    report it. `PADDOCK_NAVSIM_ALLOW_SLOW` overrides -- for investigating the
    gap, not for producing a score, and the run is then stored `valid: false`.
    """
    if min_input_hz > INPUT_HZ and not os.environ.get(ENV_ALLOW_SLOW):
        return [
            f"{model_name} needs {min_input_hz:g} Hz observations and NAVSIM "
            f"supplies {INPUT_HZ:g} Hz. OpenScene stores one keyframe per 0.5 s and one image "
            f"per keyframe, so a denser history cannot be built without inventing frames.\n"
            f"A model that infers ego motion from frame-to-frame change reads this as a "
            f"stationary world and plans accordingly, which scores near the constant-velocity "
            f"baseline while measuring the rate gap rather than the model. See "
            f"docs/benchmarks/navsim.md.\n"
            f"Set {ENV_ALLOW_SLOW}=1 to run anyway -- for investigating the gap, "
            f"not for producing a score."
        ]
    return []


def limit_problems(limit: int | None) -> list[str]:
    """Why a `--limit` below NAVSIM's floor cannot be scored at all."""
    if limit is not None and limit < MIN_SCENARIOS:
        return [
            f"--limit {limit} is below NAVSIM's floor of {MIN_SCENARIOS}: two-frame extended "
            f"comfort compares consecutive frames, so one scenario has nothing to pair with "
            f"and the scorer fails in its own aggregation. Use --limit {MIN_SCENARIOS} or more."
        ]
    return []


class NavsimBenchmark(ExternallyScoredBenchmark):
    # -- site settings: `self.setting(key)` / `self.settings` from the base -----

    @property
    def split(self) -> str:
        return self.setting("split", "navtest") or "navtest"

    @property
    def devkit(self) -> Path | None:
        value = self.setting("devkit")
        return Path(value) if value else None

    def _split_config(self) -> Path | None:
        devkit = self.devkit
        return None if devkit is None else devkit / SPLITS_DIR / f"{self.split}.yaml"

    @property
    def data_split(self) -> str:
        """Which OpenScene split the scorer reads for `split`, from the devkit's own yaml.

        Read, not repeated: a table here would be correct today and stale the
        day the devkit renames one. Raises when it cannot be read; preflight
        reports the same thing as a problem before anything is started.
        """
        path = self._split_config()
        if path is not None and path.exists():
            for line in path.read_text().splitlines():
                if line.startswith("data_split:"):
                    return line.split(":", 1)[1].strip()
        raise RuntimeError(
            f"cannot read data_split for train_test_split {self.split!r}: "
            f"{path if path is not None else '[benchmarks.navsim] devkit is not set'}"
        )

    # -- preflight --------------------------------------------------------

    def preflight(self) -> list[str]:
        problems: list[str] = []

        if self.settings is None:
            return [
                "paddock.toml has no [benchmarks.navsim] section\n"
                "see docs/benchmarks/navsim.md for the four keys it needs"
            ]

        interpreter = self.setting("python")
        if not interpreter:
            problems.append("[benchmarks.navsim] python is not set")
        elif not Path(interpreter).exists():
            problems.append(f"[benchmarks.navsim] python does not exist: {interpreter}")

        devkit = self.devkit
        if devkit is None:
            problems.append("[benchmarks.navsim] devkit is not set")
        elif not (devkit / SCORER).exists():
            problems.append(f"no NAVSIM scorer at {devkit / SCORER}; is 'devkit' the devkit root?")
        else:
            try:
                _ = self.data_split
            except RuntimeError as exc:
                problems.append(f"split {self.split!r}: {exc}")
        if problems:
            return problems  # the checks below need the split

        dataset = self.dataset
        if dataset is None:
            problems.append("paddock.toml has no [datasets.navsim]; see docs/data-setup.md")
            return problems

        root = dataset.resolve()
        for required in (f"navsim_logs/{self.data_split}", f"sensor_blobs/{self.data_split}", "maps"):
            if not (root / required).is_dir():
                problems.append(f"navsim dataset is missing {required}/ under {root}")

        if "exp" not in dataset.extra:
            problems.append("[datasets.navsim] has no 'exp' key (the NAVSIM experiment root)")

        cache = self.setting("metric_cache")
        if not cache:
            problems.append("[benchmarks.navsim] metric_cache is not set")
        elif not Path(cache).is_dir():
            problems.append(
                f"metric cache not found at {cache}\n"
                f"build one with the devkit's run_metric_caching.py, or download the "
                f"vendor's; a scenario with no cache entry is silently skipped"
            )

        problems.extend(self._coverage_warnings(root, cache))
        return problems

    def _coverage(self) -> tuple[set[str], set[str]] | str | None:
        """`(scorable, cached)` log sets for this split; a string naming why they
        cannot be known; None when the inputs are not configured at all.

        `cached` is the cache's listing intersected with the split's own log
        list: the cache also holds a `metadata/` directory, so a raw directory
        count makes a complete 136-log download look like 136 of 137. There is
        no fallback to the raw count -- if the split's list cannot be read, that
        is reported rather than guessed around.
        """
        dataset = self.dataset
        cache = self.setting("metric_cache")
        if dataset is None or not cache or not Path(cache).is_dir():
            return None
        blobs = dataset.resolve() / "sensor_blobs" / self.data_split
        if not blobs.is_dir():
            return None
        declared = self._declared_logs()
        if declared is None:
            return (
                f"cannot read the log list for {self.split} from "
                f"{self.devkit / SPLITS_DIR / 'scene_filter' / (self.split + '.yaml')}"
            )
        have = {p.name for p in blobs.iterdir() if p.is_dir()}
        cached = {p.name for p in Path(cache).iterdir() if p.is_dir()} & declared
        return have & cached, cached

    def _coverage_warnings(self, root: Path, cache: str | None) -> list[str]:
        """Refuse a run whose sensor coverage is a small slice of the split.

        NAVSIM scores the intersection of the logs on disk and the metric cache,
        and says so only in a log line partway into the run. A site holding 5 of
        136 logs will otherwise get a headline number labelled `navtest` that is
        nothing of the sort -- which is exactly what the already-stored August
        run turned out to be. `PADDOCK_NAVSIM_ALLOW_PARTIAL` lets it run; the
        result is then stored `valid: false` with the coverage in `notes`.
        """
        coverage = self._coverage()
        if coverage is None:
            return []
        if isinstance(coverage, str):
            return [coverage]
        if os.environ.get(ENV_ALLOW_PARTIAL):
            return []
        scorable, cached = coverage
        if not scorable:
            return [
                f"no log has both sensor blobs and a metric cache entry "
                f"({len(cached)} cached); nothing can be scored"
            ]
        if len(scorable) < len(cached):
            return [
                f"partial coverage: {len(scorable)} of {len(cached)} {self.split} logs have sensor "
                f"blobs, so this run scores a subset, not {self.split}.\n"
                f"the missing blobs are the openscene_sensor_{self.data_split}_camera shards; "
                f"see docs/benchmarks/navsim.md. Set {ENV_ALLOW_PARTIAL}=1 to run anyway "
                f"-- the result is then stored valid=false, as a subset not comparable with a full run."
            ]
        return []

    # -- what a run under the current switches measures --------------------

    @staticmethod
    def _agent() -> str:
        return os.environ.get(ENV_AGENT, PROXY_AGENT)

    @staticmethod
    def _overrides() -> list[str]:
        return os.environ.get(ENV_OVERRIDES, "").split()

    def run_config(self) -> dict[str, Any]:
        """The switches that change what a NAVSIM run measures, for run.json."""
        return {
            "split": self.split,
            "agent": self._agent(),
            "allow_partial": bool(os.environ.get(ENV_ALLOW_PARTIAL)),
            "allow_slow": bool(os.environ.get(ENV_ALLOW_SLOW)),
            "overrides": self._overrides(),
        }

    def invalid_reasons(self) -> list[str]:
        """Why a run under the current switches cannot be ranked with the others."""
        reasons: list[str] = []
        if self._agent() != PROXY_AGENT:
            reasons.append(
                f"scored with the vendor's native agent {self._agent()!r}, not the paddock proxy; "
                f"not comparable with proxy runs (docs/benchmarks/navsim.md)"
            )
        if os.environ.get(ENV_ALLOW_SLOW):
            reasons.append(f"{ENV_ALLOW_SLOW}: a model below NAVSIM's input rate was run anyway")
        if os.environ.get(ENV_ALLOW_PARTIAL):
            coverage = self._coverage()
            if isinstance(coverage, tuple) and len(coverage[0]) < len(coverage[1]):
                scorable, cached = coverage
                reasons.append(
                    f"{ENV_ALLOW_PARTIAL}: {len(scorable)} of {len(cached)} {self.split} logs "
                    f"have sensor blobs; a subset, not {self.split}"
                )
        return reasons

    def model_problems(self, model, limit: int | None = None) -> list[str]:
        return input_rate_problems(float(model.min_input_hz or 0.0), model.name) + limit_problems(limit)

    # -- the run ----------------------------------------------------------

    @property
    def run_retries(self) -> int:
        """`run_retries` from [benchmarks.navsim], default 2. Meaningful only
        because chunks resume; each retry re-scores only what did not finish."""
        return max(0, int(self.setting("run_retries", "2") or 0))

    def node_count(self, model) -> int:
        """`nodes` from [benchmarks.navsim], applied to CPU-only models.

        The scorer runs `workers` processes and each is a serial client of one
        node, so a model that answers one request at a time caps throughput at
        one worker's worth regardless of how many the scorer has. VisionPilot
        is that model: onnxruntime on CPU, serving one request at a time. N
        nodes can make it N-fold faster -- but only with ORT's per-session
        thread pool capped (VISIONPILOT_ORT_THREADS). Left at its default every
        node claims every core, and four of them measured at ~16 cores for one
        node's throughput. OMP_NUM_THREADS=1 bounds MUMPS, not ORT.

        GPU models stay at one node. The card is shared, and a second 26 GB
        alpamayo would not start into the headroom left by the first; the
        `infer_sequence` override already took them off the critical path.
        """
        wanted = int(self.setting("nodes", "1") or 1)
        if model.vram_gb > 0:
            return 1
        return max(1, wanted)

    def _declared_logs(self) -> set[str] | None:
        """The logs this split declares, read from the devkit's own scene filter.

        None when it cannot be read -- no devkit, no file, no `log_names`, a
        yaml error -- and the caller reports that, rather than falling back to
        a directory count that is known to be wrong by one.
        """
        devkit = self.devkit
        if devkit is None:
            return None
        path = devkit / SPLITS_DIR / "scene_filter" / f"{self.split}.yaml"
        if not path.exists():
            return None
        try:
            import yaml

            names = yaml.safe_load(path.read_text()).get("log_names")
        except Exception:  # noqa: BLE001 - a preflight helper must not raise; None is the report
            return None
        return set(names) if names else None

    def _environment(self, root: Path, exp: Path) -> dict[str, str]:
        devkit = self.devkit
        env = dict(os.environ)
        env.update(
            {
                "NAVSIM_DEVKIT_ROOT": str(devkit),
                "NAVSIM_EXP_ROOT": str(exp),
                "OPENSCENE_DATA_ROOT": str(root),
                "NUPLAN_MAPS_ROOT": str(root / "maps"),
                "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
            }
        )
        # The proxy agent is imported inside the devkit's interpreter, where
        # paddock is not installed. It imports nothing from paddock, but hydra
        # still has to find the module by its dotted path.
        src = str(Path(__file__).resolve().parents[4])
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{src}:{existing}" if existing else src
        return env

    def _command(
        self,
        policy_url: str,
        out_dir: Path,
        config: dict[str, Any],
        limit: int | None,
        log_names: list[str] | None = None,
        tokens: list[str] | None = None,
    ) -> list[str]:
        root = self.dataset.resolve()
        agent = self._agent()
        cameras = list(config.get("cameras") or ["CAM_FRONT"])

        command = [
            str(self.setting("python")),
            "-u",
            str(self.devkit / SCORER),
            f"hydra.searchpath=[pkg://navsim.planning.script.config.common,file://{CONF_DIR}]",
            f"train_test_split={self.split}",
            f"agent={agent}",
            "worker=single_machine_thread_pool",
            f"worker.max_workers={self.setting('workers', '4')}",
            "worker.use_process_pool=true",
            f"metric_cache_path={self.setting('metric_cache')}",
            f"experiment_name=paddock_{config.get('model', 'model')}",
            # Keep the CSV, the hydra config and the scorer log with the rest of
            # this run's artifacts instead of in NAVSIM's own experiment tree.
            f"output_dir={out_dir}",
        ]
        if agent == PROXY_AGENT:
            urls = list(config.get("policy_urls") or [policy_url])
            command += [
                f"agent.node_url={policy_url}",
                "agent.node_urls=[" + ",".join(urls) + "]",
                "agent.cameras=[" + ",".join(cameras) + "]",
                f"agent.sensor_blobs_path={root / 'sensor_blobs' / self.data_split}",
                f"agent.dump_path={out_dir / 'trajectories.jsonl'}",
            ]
        if limit:
            command.append(f"train_test_split.scene_filter.max_scenes={limit}")
        if log_names:
            command.append(f"train_test_split.scene_filter.log_names={hydra_list(log_names)}")
        if tokens:
            command.append(f"train_test_split.scene_filter.tokens={hydra_list(tokens)}")
        return command + self._overrides()

    def _run_scorer(self, command: list[str], out_dir: Path, root: Path, exp: Path) -> Path:
        """One scorer invocation. Returns its CSV; raises with the log tail if not."""
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "navsim_scorer.log"
        with log_path.open("w") as handle:
            handle.write(" ".join(command) + "\n\n")
            handle.flush()
            # The scorer forks a worker pool. Give it its own process group so
            # that when this run is interrupted the pool goes with it, rather
            # than four orphans hammering a dead node on a machine other people
            # are using. The log streams to a file as it runs, so a hung
            # scenario is diagnosable while it hangs. See docs/reviewing.md §8.
            scorer = subprocess.Popen(
                command,
                cwd=str(self.devkit),
                env=self._environment(root, exp),
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = scorer.wait()
            except BaseException:
                _kill_group(scorer)
                raise
        if returncode != 0:
            raise RuntimeError(
                f"NAVSIM scorer exited {returncode}\n--- tail of {log_path} ---\n" + _tail(log_path)
            )
        csv_path = completed_chunk_csv(out_dir)
        if csv_path is None:
            raise RuntimeError(
                f"NAVSIM scorer exited 0 but wrote no CSV to {out_dir}\n"
                f"--- tail of {log_path} ---\n" + _tail(log_path)
            )
        return csv_path

    @staticmethod
    def _logs_of(merged_csv: Path, tokens: list[str]) -> list[str]:
        """The logs a set of tokens belong to, read from the scored CSV, so the
        retry pass loads only those logs' metadata."""
        wanted = set(tokens)
        logs: set[str] = set()
        with merged_csv.open() as handle:
            for row in csv.DictReader(handle):
                if (row.get("token") or "").strip() in wanted and row.get("log_name"):
                    logs.add(row["log_name"].strip())
        return sorted(logs)

    def _scorable_logs(self) -> list[str]:
        """The split's logs this site can score, in a fixed order."""
        root = self.dataset.resolve()
        blobs = root / "sensor_blobs" / self.data_split
        cache = self.setting("metric_cache")
        declared = self._declared_logs()
        have = {p.name for p in blobs.iterdir() if p.is_dir()} if blobs.is_dir() else set()
        cached = (
            {p.name for p in Path(cache).iterdir() if p.is_dir()} if cache and Path(cache).is_dir() else set()
        )
        return sorted(declared & have & cached)

    def evaluate(
        self,
        policy_url: str,
        out_dir: Path,
        config: dict[str, Any],
        limit: int | None = None,
        echo=print,
    ) -> Evaluation:
        # The runner has already asked `model_problems`; asked again here so a
        # caller that skips the runner is refused the same way.
        problems = input_rate_problems(
            float(config.get("min_input_hz") or 0.0), str(config.get("model", "this model"))
        ) + limit_problems(limit)
        if problems:
            raise RuntimeError("\n".join(problems))

        root = self.dataset.resolve()
        exp = self.dataset.extra["exp"]
        out_dir.mkdir(parents=True, exist_ok=True)
        began = time.time()

        # A smoke run (--limit) is one invocation: it exists to test the pipe,
        # and a per-invocation max_scenes would otherwise multiply by chunks.
        size = 0 if limit else int(self.setting("chunk_logs", str(DEFAULT_CHUNK_LOGS)) or 0)
        logs = self._scorable_logs() if size > 0 else []
        chunks = chunk_logs(logs, size) if logs else [[]]
        resume_from = config.get("resume_from") or os.environ.get(RESUME_ENV)
        summary = out_dir / "navsim_scorer.log"
        echo(
            f"  scoring {self.split} via {Path(self.setting('python') or 'python').name}"
            + (f" in {len(chunks)} chunk(s) of <= {size} log(s)" if size > 0 else "")
            + (f", resuming from {resume_from}" if resume_from else "")
        )

        csvs: list[Path] = []
        failed_chunks: list[list[str]] = []
        with summary.open("w") as log:
            for index, names in enumerate(chunks):
                chunk_dir = out_dir / "chunks" / f"{index:03d}" if size > 0 else out_dir
                # Resume: a chunk this run already completed, or one a previous
                # run completed, is reused rather than scored again.
                done = completed_chunk_csv(chunk_dir)
                if done is None and resume_from and size > 0:
                    previous = completed_chunk_csv(Path(resume_from) / "chunks" / f"{index:03d}")
                    if previous is not None:
                        chunk_dir.mkdir(parents=True, exist_ok=True)
                        done = Path(shutil.copy2(previous, chunk_dir / previous.name))
                if done is not None:
                    csvs.append(done)
                    log.write(f"chunk {index:03d}: reused {done}\n")
                    log.flush()
                    continue
                command = self._command(policy_url, chunk_dir, config, limit, names or None)
                try:
                    csvs.append(self._run_scorer(command, chunk_dir, root, exp))
                    log.write(f"chunk {index:03d}: ok ({len(names)} log(s))\n")
                    log.flush()
                except RuntimeError as exc:
                    # One chunk's failure must not take the others with it. It
                    # is recorded, the run is marked partial, and a later run
                    # can resume from what did complete.
                    failed_chunks.append(names)
                    log.write(f"chunk {index:03d}: FAILED -- {str(exc).splitlines()[0]}\n")
                    log.flush()
                    echo(f"  chunk {index:03d} FAILED ({', '.join(names) or 'whole split'}); continuing")

        if not csvs:
            raise RuntimeError(f"every chunk failed; nothing scored. See {summary} and {out_dir / 'chunks'}")
        merged = merge_chunk_csvs(csvs, out_dir / "scores.csv") if size > 0 else csvs[-1]
        samples, aggregates, failed = navsim_ingest.parse_scores(merged, split=self.split)

        # Scenarios the model failed on inside a chunk that otherwise completed
        # get one more pass of their own, by token. For an unseeded sampler that
        # is a fresh draw; for anything else it fails again and stays excluded.
        # Whole chunks that failed are the runner's to retry with fresh nodes.
        if failed and size > 0 and not completed_chunk_csv(out_dir / "chunks" / "retry"):
            echo(f"  re-scoring {len(failed)} failed scenario(s) by token")
            logs_of = self._logs_of(merged, failed)
            retry_dir = out_dir / "chunks" / "retry"
            try:
                csvs.append(
                    self._run_scorer(
                        self._command(policy_url, retry_dir, config, limit, logs_of or None, failed),
                        retry_dir,
                        root,
                        exp,
                    )
                )
                merged = merge_chunk_csvs(csvs, out_dir / "scores.csv")
                samples, aggregates, failed = navsim_ingest.parse_scores(merged, split=self.split)
            except RuntimeError as exc:
                echo(f"  retry pass FAILED ({str(exc).splitlines()[0]}); the {len(failed)} stay excluded")
        # The scorer's own headline is per invocation and means nothing across
        # chunks; only a single-invocation run keeps it.
        reported = navsim_ingest.log_reported_score(out_dir) if size == 0 else None
        for names in failed_chunks:
            failed.extend(f"{name}/*" for name in names)
        echo(
            f"  {len(samples)} scenario(s) in {time.time() - began:.0f}s"
            + (f", scorer reported {reported:.5f}" if reported is not None else "")
            + (f", {len(failed)} FAILED (excluded)" if failed else "")
        )
        # Kept on the run rather than folded into the score: the scorer's own
        # headline and paddock's recomputed mean answer slightly different
        # questions. Stored in the tracked metrics.json under `by_group`, the
        # same keys the ingest path writes. See paddock/results/ingest/navsim.py.
        return Evaluation(
            samples=samples,
            failed=failed,
            by_group={
                "source_aggregates": aggregates,
                "source_reported": {"final_average_valid": reported} if reported is not None else {},
            },
        )


def _kill_group(proc: subprocess.Popen) -> None:
    """Take the scorer and its whole worker pool down, TERM then KILL."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


BENCHMARK = NavsimBenchmark
