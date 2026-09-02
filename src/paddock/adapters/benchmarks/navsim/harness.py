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

import os
import subprocess
import time
from pathlib import Path
from typing import Any

from paddock.adapters.benchmarks.navsim.convert import NAVSIM_INTERVAL_S
from paddock.benchmark import ExternallyScoredBenchmark
from paddock.results.ingest import navsim as navsim_ingest

#: Which data_split each train_test_split reads. NAVSIM keeps this in its own
#: config (train_test_split/<name>.yaml: data_split), and preflight needs it
#: before hydra runs, so the mapping is repeated here for the splits paddock
#: supports and validated against the devkit's own yaml at preflight time.
SPLIT_DATA = {
    "navtest": "test",
    "navtest_two_stage": "test",
    "navmini": "mini",
    "navtrain": "trainval",
}

PROXY_AGENT = "paddock_proxy"

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


class NavsimBenchmark(ExternallyScoredBenchmark):
    # -- site settings ----------------------------------------------------

    @property
    def settings(self):
        return self.cfg.benchmarks.get("navsim")

    def _setting(self, key: str, default: str | None = None) -> str | None:
        settings = self.settings
        return settings.get(key, default) if settings is not None else default

    @property
    def split(self) -> str:
        return self._setting("split", "navtest") or "navtest"

    @property
    def data_split(self) -> str:
        return SPLIT_DATA.get(self.split, "test")

    @property
    def devkit(self) -> Path | None:
        value = self._setting("devkit")
        return Path(value) if value else None

    # -- preflight --------------------------------------------------------

    def preflight(self) -> list[str]:
        problems: list[str] = []

        if self.settings is None:
            return [
                "paddock.toml has no [benchmarks.navsim] section\n"
                "see docs/benchmarks/navsim.md for the four keys it needs"
            ]

        interpreter = self._setting("python")
        if not interpreter:
            problems.append("[benchmarks.navsim] python is not set")
        elif not Path(interpreter).exists():
            problems.append(f"[benchmarks.navsim] python does not exist: {interpreter}")

        devkit = self.devkit
        if devkit is None:
            problems.append("[benchmarks.navsim] devkit is not set")
        elif not (devkit / SCORER).exists():
            problems.append(f"no NAVSIM scorer at {devkit / SCORER}; is 'devkit' the devkit root?")
        elif self.split not in SPLIT_DATA:
            problems.append(
                f"split {self.split!r} is not one paddock knows; known: {', '.join(sorted(SPLIT_DATA))}"
            )
        else:
            # The mapping above must agree with the devkit's own config, or the
            # blob path preflight checks is not the one the scorer will read.
            splits = devkit / "navsim/planning/script/config/common/train_test_split"
            declared = splits / f"{self.split}.yaml"
            if declared.exists():
                for line in declared.read_text().splitlines():
                    if line.startswith("data_split:"):
                        actual = line.split(":", 1)[1].strip()
                        if actual != self.data_split:
                            problems.append(
                                f"{self.split} reads data_split {actual!r} but paddock expects "
                                f"{self.data_split!r}; SPLIT_DATA in harness.py is stale"
                            )

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

        cache = self._setting("metric_cache")
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

    def _coverage_warnings(self, root: Path, cache: str | None) -> list[str]:
        """Refuse a run whose sensor coverage is a small slice of the split.

        NAVSIM scores the intersection of the logs on disk and the metric cache,
        and says so only in a log line partway into the run. A site holding 5 of
        136 logs will otherwise get a headline number labelled `navtest` that is
        nothing of the sort -- which is exactly what the already-stored August
        run turned out to be.
        """
        blobs = root / "sensor_blobs" / self.data_split
        if not blobs.is_dir() or not cache or not Path(cache).is_dir():
            return []
        if os.environ.get("PADDOCK_NAVSIM_ALLOW_PARTIAL"):
            return []
        have = {p.name for p in blobs.iterdir() if p.is_dir()}
        cached = {p.name for p in Path(cache).iterdir() if p.is_dir()}
        # Against the split's own log list, not the raw directory count: the
        # cache also holds a `metadata/` directory, so counting directories
        # makes a complete 136-log download look like 136 of 137 and refuse.
        declared = self._declared_logs()
        cached = (cached & declared) if declared else cached
        scorable = have & cached
        if not scorable:
            return [
                f"no log has both sensor blobs and a metric cache entry "
                f"({len(have)} with blobs, {len(cached)} cached); nothing can be scored"
            ]
        if len(scorable) < len(cached):
            return [
                f"partial coverage: {len(scorable)} of {len(cached)} {self.split} logs have sensor "
                f"blobs, so this run scores a subset, not {self.split}.\n"
                f"the missing blobs are the openscene_sensor_{self.data_split}_camera shards; "
                f"see docs/benchmarks/navsim.md. Set PADDOCK_NAVSIM_ALLOW_PARTIAL=1 to run anyway "
                f"-- the result is recorded as a subset and is not comparable with a full run."
            ]
        return []

    # -- the run ----------------------------------------------------------

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
        wanted = int(self._setting("nodes", "1") or 1)
        if model.vram_gb > 0:
            return 1
        return max(1, wanted)

    def _declared_logs(self) -> set[str]:
        """The logs this split declares, read from the devkit's own scene filter.

        Empty when the devkit is not resolvable, in which case the caller falls
        back to the cache's directory listing.
        """
        devkit = self.devkit
        if devkit is None:
            return set()
        path = (
            devkit
            / "navsim/planning/script/config/common/train_test_split/scene_filter"
            / f"{self.split}.yaml"
        )
        if not path.exists():
            return set()
        try:
            import yaml

            return set(yaml.safe_load(path.read_text()).get("log_names") or [])
        except Exception:  # noqa: BLE001 - a preflight helper must not raise
            return set()

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
        self, policy_url: str, out_dir: Path, config: dict[str, Any], limit: int | None
    ) -> list[str]:
        root = self.dataset.resolve()
        agent = os.environ.get("PADDOCK_NAVSIM_AGENT", PROXY_AGENT)
        cameras = list(config.get("cameras") or ["CAM_FRONT"])

        command = [
            str(self._setting("python")),
            "-u",
            str(self.devkit / SCORER),
            f"hydra.searchpath=[pkg://navsim.planning.script.config.common,file://{CONF_DIR}]",
            f"train_test_split={self.split}",
            f"agent={agent}",
            "worker=single_machine_thread_pool",
            f"worker.max_workers={self._setting('workers', '4')}",
            "worker.use_process_pool=true",
            f"metric_cache_path={self._setting('metric_cache')}",
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
        extra = os.environ.get("PADDOCK_NAVSIM_OVERRIDES", "").split()
        return command + extra

    def evaluate(
        self,
        policy_url: str,
        out_dir: Path,
        config: dict[str, Any],
        limit: int | None = None,
        echo=print,
    ) -> list[dict[str, Any]]:
        needs = float(config.get("min_input_hz") or 0.0)
        if needs > INPUT_HZ and not os.environ.get("PADDOCK_NAVSIM_ALLOW_SLOW"):
            raise RuntimeError(
                f"{config.get('model', 'this model')} needs {needs:g} Hz observations and NAVSIM "
                f"supplies {INPUT_HZ:g} Hz. OpenScene stores one keyframe per 0.5 s and one image "
                f"per keyframe, so a denser history cannot be built without inventing frames.\n"
                f"A model that infers ego motion from frame-to-frame change reads this as a "
                f"stationary world and plans accordingly, which scores near the constant-velocity "
                f"baseline while measuring the rate gap rather than the model. See "
                f"docs/benchmarks/navsim.md.\n"
                f"Set PADDOCK_NAVSIM_ALLOW_SLOW=1 to run anyway -- for investigating the gap, "
                f"not for producing a score."
            )

        if limit is not None and limit < MIN_SCENARIOS:
            raise RuntimeError(
                f"--limit {limit} is below NAVSIM's floor of {MIN_SCENARIOS}: two-frame extended "
                f"comfort compares consecutive frames, so one scenario has nothing to pair with "
                f"and the scorer fails in its own aggregation. Use --limit {MIN_SCENARIOS} or more."
            )

        root = self.dataset.resolve()
        exp = self.dataset.extra["exp"]
        out_dir.mkdir(parents=True, exist_ok=True)

        command = self._command(policy_url, out_dir, config, limit)
        log_path = out_dir / "navsim_scorer.log"
        echo(f"  scoring {self.split} via {Path(command[0]).name} -> {log_path.name}")

        began = time.time()
        with log_path.open("w") as handle:
            handle.write(" ".join(command) + "\n\n")
            handle.flush()
            completed = subprocess.run(
                command,
                cwd=str(self.devkit),
                env=self._environment(root, exp),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )

        if completed.returncode != 0:
            raise RuntimeError(
                f"NAVSIM scorer exited {completed.returncode}\n--- tail of {log_path} ---\n" + _tail(log_path)
            )

        csvs = sorted(out_dir.glob("*.csv"))
        if not csvs:
            raise RuntimeError(
                f"NAVSIM scorer exited 0 but wrote no CSV to {out_dir}\n"
                f"--- tail of {log_path} ---\n" + _tail(log_path)
            )

        samples, aggregates = navsim_ingest.parse_scores(csvs[-1], split=self.split)
        reported = navsim_ingest.log_reported_score(out_dir)
        echo(
            f"  {len(samples)} scenario(s) in {time.time() - began:.0f}s"
            + (f", scorer reported {reported:.5f}" if reported is not None else "")
        )
        # Kept on the run rather than folded into the score: the scorer's own
        # headline and paddock's recomputed mean answer slightly different
        # questions. See paddock/results/ingest/navsim.py.
        for sample in samples:
            sample.setdefault("extra", {})
        if samples:
            samples[0]["extra"]["source_aggregates"] = aggregates
            samples[0]["extra"]["source_reported"] = reported
        return samples


def _tail(path: Path, lines: int = 30) -> str:
    if not path.exists():
        return "(no log)"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:]) or "(log empty)"


BENCHMARK = NavsimBenchmark
