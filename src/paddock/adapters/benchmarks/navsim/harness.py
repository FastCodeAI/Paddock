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
behind the proxy is handed them, and no other model in paddock has an
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

`adapters/benchmarks/navsim_v1/harness.py` subclasses `NavsimBenchmark` for the
v1 devkit and PDMS. The knobs that differ between devkit generations -- the
scorer script, the hydra conf directory, the score vocabulary, the smoke floor,
the version the devkit must declare -- are class attributes for that reason.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Collection
from pathlib import Path
from typing import Any

from paddock.adapters.benchmarks.navsim.convert import CAMERA_SLOTS, NAVSIM_INTERVAL_S
from paddock.adapters.benchmarks.navsim.dense import NUPLAN_HZ
from paddock.benchmark import Evaluation, ExternallyScoredBenchmark
from paddock.benchmark import input_rate_problems as shared_input_rate_problems
from paddock.results.ingest import navsim as navsim_ingest
from paddock.runtime import env as env_builder
from paddock.runtime import gpu
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

#: Another run whose *completed chunk CSVs* this run takes as its own, skipping
#: their logs. For continuing a run whose chunking changed -- `RESUME_ENV`
#: reuses chunks by index, which only works when the index means the same
#: logs. The other run must have been made under the same measurement
#: configuration; the record says where the rows came from (`merged_from`).
MERGE_ENV = "PADDOCK_NAVSIM_MERGE_FROM"


def node_alive(policy_url: str, timeout_s: float = 5.0) -> bool:
    """Whether the policy node still answers `/alive`. Standard library only:
    the harness must not pull an HTTP client into the runner's process."""
    try:
        with urllib.request.urlopen(f"{policy_url.rstrip('/')}/alive", timeout=timeout_s) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def chunk_logs(logs: list[str], size: int) -> list[list[str]]:
    """Split *logs* into consecutive chunks of *size*, in the given order."""
    if size <= 0:
        return [list(logs)]
    return [list(logs[i : i + size]) for i in range(0, len(logs), size)]


def hydra_list(items: list[str]) -> str:
    """A hydra list override literal. Quoted: log names carry dots."""
    return "[" + ",".join(f"'{item}'" for item in items) + "]"


def completed_chunk_csvs(run_dir: Path) -> list[Path]:
    """Every completed chunk CSV under another run's `chunks/`, retry pass included."""
    chunks = Path(run_dir) / "chunks"
    if not chunks.is_dir():
        return []
    found = [completed_chunk_csv(d) for d in sorted(chunks.iterdir()) if d.is_dir()]
    return [f for f in found if f is not None]


def tokens_in_csvs(csv_paths: list[Path]) -> set[str]:
    """Every `token` in these chunk CSVs. The scorer's CSV names no log -- its
    columns end at `score` -- so which log a token belongs to is the metric
    cache's to say (`_token_logs`)."""
    import csv

    tokens: set[str] = set()
    for path in csv_paths:
        with Path(path).open(newline="") as handle:
            for row in csv.DictReader(handle):
                token = (row.get("token") or "").strip()
                if token:
                    tokens.add(token)
    return tokens


def retry_tokens(failed: list[str], token_logs: dict[str, str], floor: int) -> list[str]:
    """The failed tokens, plus enough already-scored neighbours from the same
    logs for the scorer to aggregate them.

    The v2 scorer pairs consecutive frames within a log for two-frame comfort
    and dies in pandas below `floor` per log -- the benchmark's MIN_SCENARIOS,
    the same floor `--limit` guards (1 under v1, which pads nothing). One failed
    token per log is the common case, so alone it cannot
    be retried at all. The neighbours are re-scored and then discarded
    (`keep_tokens`): a stochastic model must not get a second draw on a
    scenario that did not fail.
    """
    wanted = set(failed)
    by_log: dict[str, list[str]] = {}
    for token in failed:
        log = token_logs.get(token)
        if log is not None:
            by_log.setdefault(log, []).append(token)
    padded = list(failed)
    for log, tokens in by_log.items():
        need = floor - len(tokens)
        if need > 0:
            others = sorted(t for t, owner in token_logs.items() if owner == log and t not in wanted)
            padded += others[:need]
    return padded


def keep_tokens(
    csv_path: Path, tokens: set[str], aggregate: frozenset[str] = frozenset(navsim_ingest.AGGREGATE_TOKENS)
) -> Path:
    """Rewrite a scorer CSV in place keeping only `tokens` and the aggregate rows."""
    with Path(csv_path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [
            row
            for row in reader
            if (row.get("token") or "").strip() in tokens or (row.get("token") or "").strip() in aggregate
        ]
    with Path(csv_path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return Path(csv_path)


def dense_history_summary(out_dir: Path) -> dict[str, int]:
    """What the plan dumps under `out_dir` say about the 10 Hz history each
    scenario was shown: how many scenarios had frames clamped at a log start
    or borrowed across a dropped frame (dense.py). Zero when no dump says so.

    A token planned more than once (the proxy's retries) counts once, by its
    last line. Summed into the tracked record because it is the one thing
    that changes what the model saw and would otherwise live only in the
    untracked dump.
    """
    import json

    last: dict[str, dict] = {}
    for dump in sorted(Path(out_dir).rglob("trajectories.jsonl")):
        with dump.open() as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                token = record.get("token")
                if token:
                    last[token] = record
    return {
        "scenarios": len(last),
        "clamped": sum(1 for r in last.values() if r.get("clamped")),
        "approximate": sum(1 for r in last.values() if r.get("approximate")),
    }


def completed_chunk_csv(chunk_dir: Path) -> Path | None:
    """The scorer's CSV for a chunk that finished, else None."""
    csvs = sorted(chunk_dir.glob("*.csv")) if chunk_dir.is_dir() else []
    return csvs[-1] if csvs else None


def merge_chunk_csvs(
    csv_paths: list[Path],
    out_path: Path,
    aggregate_tokens: Collection[str] = navsim_ingest.EPDMS.aggregate_tokens,
) -> Path:
    """Concatenate per-chunk score CSVs into one the parser already understands.

    Each chunk's CSV carries its own aggregate rows (average_all_frames and
    friends in v2, `average` in v1), which mean nothing across chunks and are
    dropped; paddock recomputes aggregates from per-sample records anyway.
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
                if token in aggregate_tokens:
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
#: The agents that need no policy node (nodeless.py). Each is a yaml under
#: every NAVSIM conf/agent/ directory; the names are the hydra group entries.
CONSTANT_VELOCITY_AGENT = "paddock_constant_velocity"
HUMAN_AGENT = "paddock_human"
REPLAY_AGENT = "paddock_replay"

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

#: Seconds of history NAVSIM's own AgentInput spans: four keyframes at 0.5 s.
NAVSIM_HISTORY_S = 1.5

#: [benchmarks.navsim] key naming a nuPlan root -- `splits/<split>/<log>.db` and
#: `sensor_blobs/<split>/<log>/<CHANNEL>/` at nuPlan's own 10 Hz. With it, a
#: model whose floor or native rate is above OpenScene's 2 Hz is fed a dense
#: history built from it (dense.py); without it such a model is refused, as
#: `input_rate_problems` explains. Which rate a run used is in `run.json`.
SETTING_NUPLAN = "nuplan"
#: [benchmarks.navsim] switch: on the dense path, a view the nuPlan pull does not hold (it has
#: F0/L0/R0) comes from OpenScene's blobs at its 2 Hz keyframes only. For a model whose camera
#: history is itself 2 Hz (AutoE2E: eight frames 0.5 s apart) that is every frame it reads;
#: the 10 Hz frames in between still carry the ego state. Off unless set.
SETTING_KEYFRAME_CAMERAS = "keyframe_cameras"
CONF_DIR = Path(__file__).resolve().parent / "conf"
SCORER = "navsim/planning/script/run_pdm_score_sequential.py"


def input_rate_problems(min_input_hz: float, model_name: str, supplied_hz: float = INPUT_HZ) -> list[str]:
    """Why a model needing denser input than this run supplies cannot be scored.

    `supplied_hz` is OpenScene's 2 Hz, or nuPlan's 10 Hz when the site holds the
    full-rate data (`SETTING_NUPLAN`). At nuPlan's rate there is no remedy left
    to offer -- nothing denser was ever recorded -- so that refusal is NAVSIM's
    own. Below it the refusal is `paddock.benchmark.input_rate_problems`, shared
    with comma2k19: both benchmarks refuse for the same reason and must say the
    same thing, and this wrapper supplies the facts only NAVSIM knows.
    """
    if supplied_hz > INPUT_HZ:
        if min_input_hz <= supplied_hz or os.environ.get(ENV_ALLOW_SLOW):
            return []
        return [
            f"{model_name} needs {min_input_hz:g} Hz observations and nuPlan's cameras run at "
            f"{supplied_hz:g} Hz; nothing denser was ever recorded."
        ]
    return shared_input_rate_problems(
        model_name=model_name,
        min_input_hz=min_input_hz,
        benchmark_name="NAVSIM",
        benchmark_hz=INPUT_HZ,
        allow_env=ENV_ALLOW_SLOW,
        why=(
            "OpenScene stores one keyframe per 0.5 s and one image per keyframe, "
            "so a denser history cannot be built without inventing frames.\n"
            f"The frames exist upstream: set [benchmarks.navsim] {SETTING_NUPLAN} to nuPlan's "
            f"full-rate sensor data and the model is fed {NUPLAN_HZ:g} Hz from it."
        ),
        doc="docs/benchmarks/navsim.md",
    )


def _has_jpg(directory: Path) -> bool:
    """Whether a directory holds any .jpg, stopping at the first.

    `Path.glob` reads the whole directory before yielding (Python 3.12), and a nuPlan camera
    directory on the NAS is ~4,000 entries: the dense-path check over 136 logs took ~45 min.
    """
    try:
        with os.scandir(directory) as entries:
            return any(entry.name.endswith(".jpg") for entry in entries)
    except OSError:
        return False


def supplied_cameras(cameras: list[str]) -> list[str]:
    """The declared cameras this rig has a view for, in declaration order.

    A model may declare cameras only one benchmark can render (openpilot's
    comma-three pair exists in CARLA alone) and fall back to CAM_FRONT elsewhere;
    the proxy asks NAVSIM for the ones that exist and the policy sees the rest
    as absent, as it does on KITScenes. None at all is a refusal.
    """
    return [c for c in cameras if c in CAMERA_SLOTS]


def camera_problems(cameras: list[str], model_name: str) -> list[str]:
    if supplied_cameras(cameras):
        return []
    return [
        f"{model_name} declares {', '.join(cameras) or 'no cameras'} and the nuPlan rig has none of "
        f"them; see CAMERA_SLOTS in convert.py"
    ]


def limit_problems(limit: int | None, floor: int = MIN_SCENARIOS) -> list[str]:
    """Why a `--limit` below the scorer's floor cannot be scored at all.

    The default floor is the v2 devkit's two-frame pairing; a devkit without
    that term (v1) passes its own floor of one.
    """
    if limit is not None and limit < floor:
        why = (
            "two-frame extended comfort compares consecutive frames, so one scenario has "
            "nothing to pair with and the scorer fails in its own aggregation"
            if floor >= 2
            else "the scorer needs a scenario to score"
        )
        return [f"--limit {limit} is below NAVSIM's floor of {floor}: {why}. Use --limit {floor} or more."]
    return []


class NavsimBenchmark(ExternallyScoredBenchmark):
    """NAVSIM v2 EPDMS. The class attributes are what a devkit generation
    changes; `navsim_v1/harness.py` overrides them and nothing else."""

    #: Scorer script, relative to the devkit root.
    SCORER = SCORER
    #: Hydra config directory holding paddock's agent group entries.
    CONF_DIR = CONF_DIR
    #: The agent group a comparable run -- a model behind a node -- uses.
    PROXY_AGENT = PROXY_AGENT
    #: Fewest scenarios the scorer can aggregate.
    MIN_SCENARIOS = MIN_SCENARIOS
    #: What the scorer's CSV calls its columns and aggregate rows.
    VOCABULARY = navsim_ingest.EPDMS
    #: Where refusal messages send people.
    DOC = "docs/benchmarks/navsim.md"
    #: Major version the devkit's setup.py must declare. Both generations ship a
    #: `run_pdm_score.py`, so the scorer path passing proves little, and a v2
    #: tree scored under a PDMS label would look exactly like a result.
    DEVKIT_MAJOR = "2"

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
        where = path if path is not None else f"[benchmarks.{self.spec.name}] devkit is not set"
        raise RuntimeError(f"cannot read data_split for train_test_split {self.split!r}: {where}")

    # -- preflight --------------------------------------------------------

    def preflight(self) -> list[str]:
        problems: list[str] = []
        section = f"[benchmarks.{self.spec.name}]"

        if self.settings is None:
            return [f"paddock.toml has no {section} section\nsee {self.DOC} for the keys it needs"]

        interpreter = self.setting("python")
        if not interpreter:
            problems.append(f"{section} python is not set")
        elif not Path(interpreter).exists():
            problems.append(f"{section} python does not exist: {interpreter}")

        devkit = self.devkit
        if devkit is None:
            problems.append(f"{section} devkit is not set")
        elif not (devkit / self.SCORER).exists():
            problems.append(f"no NAVSIM scorer at {devkit / self.SCORER}; is 'devkit' the devkit root?")
        else:
            problems.extend(self._devkit_problems(devkit))
            try:
                _ = self.data_split
            except RuntimeError as exc:
                problems.append(f"split {self.split!r}: {exc}")
        if problems:
            return problems  # the checks below need the split

        dataset = self.dataset
        if dataset is None:
            problems.append(f"paddock.toml has no [datasets.{self.spec.dataset}]; see docs/data-setup.md")
            return problems

        root = dataset.resolve()
        for required in (f"navsim_logs/{self.data_split}", f"sensor_blobs/{self.data_split}", "maps"):
            if not (root / required).is_dir():
                problems.append(f"navsim dataset is missing {required}/ under {root}")

        if "exp" not in dataset.extra:
            problems.append(f"[datasets.{self.spec.dataset}] has no 'exp' key (the NAVSIM experiment root)")

        nuplan = self.setting(SETTING_NUPLAN)
        if nuplan:
            section = f"[benchmarks.{self.spec.name}]"
            for required in (f"splits/{self.data_split}", f"sensor_blobs/{self.data_split}"):
                if not (Path(nuplan) / required).is_dir():
                    problems.append(
                        f"{section} {SETTING_NUPLAN} = {nuplan} has no {required}/; "
                        f"it must be a nuPlan root laid out as nuPlan's own (docs/benchmarks/navsim.md)"
                    )

        cache = self.setting("metric_cache")
        if not cache:
            problems.append(f"{section} metric_cache is not set")
        elif not Path(cache).is_dir():
            problems.append(
                f"metric cache not found at {cache}\n"
                f"build one with the devkit's run_metric_caching.py, or download the "
                f"vendor's; a scenario with no cache entry is silently skipped"
            )
        elif not self._cache_csvs(Path(cache)):
            problems.append(
                f"metric cache at {cache} has no metadata/*.csv: run_metric_caching.py writes it "
                f"last, so the build did not finish. Re-run it; it skips scenarios already cached"
            )

        problems.extend(self._coverage_warnings(root, cache))
        return problems

    def _devkit_problems(self, devkit: Path) -> list[str]:
        """Is this the devkit generation this benchmark scores with, patched as needed?

        `setup.py` declares the version; a tree without one (the tests' stand-ins)
        gets no opinion. The proxy also needs `Camera.camera_path`, which the v1
        devkit only has with paddock's patch applied -- without it every scenario
        fails at run time and the run ends as "no scenarios scored", hours later,
        rather than as a refusal here.
        """
        problems: list[str] = []
        setup = devkit / "setup.py"
        if setup.exists():
            match = re.search(r"""version\s*=\s*["'](\d+)(?:\.\d+)*["']""", setup.read_text(errors="replace"))
            if match and match.group(1) != self.DEVKIT_MAJOR:
                problems.append(
                    f"devkit at {devkit} declares navsim {match.group(1)}.x; [benchmarks.{self.spec.name}] "
                    f"needs a v{self.DEVKIT_MAJOR} devkit (see {self.DOC})"
                )
        dataclasses = devkit / "navsim" / "common" / "dataclasses.py"
        if dataclasses.exists() and "camera_path" not in dataclasses.read_text(errors="replace"):
            problems.append(
                f"{dataclasses} has no Camera.camera_path, which the proxy needs to ship image paths; "
                f"apply the patches repos.yaml lists for this devkit's repo (patches/README.md)"
            )
        return problems

    @staticmethod
    def _cache_csvs(cache: Path) -> list[Path]:
        """The metric cache's metadata csv(s). The scorer reads the cache through
        the first of these alone -- not the directory tree -- so a cache without
        one, or with a stale one, scores nothing or a subset."""
        metadata = cache / "metadata"
        return sorted(metadata.glob("*.csv")) if metadata.is_dir() else []

    def _cache_index(self) -> dict[str, str]:
        """token -> log name, from the metric cache's metadata csv; empty when absent.

        Rows are `<cache>/<log>/<scenario type>/<token>/metric_cache.pkl`, the
        same lines the devkit's MetricCacheLoader reads.
        """
        csvs = self._cache_csvs(Path(self.setting("metric_cache") or ""))
        if not csvs:
            return {}
        index: dict[str, str] = {}
        for line in csvs[0].read_text(errors="replace").splitlines()[1:]:
            parts = Path(line.strip()).parts
            if len(parts) >= 4:
                index[parts[-2]] = parts[-4]
        return index

    def _cache_coverage(self) -> tuple[int, int] | None:
        """`(listed, declared)`: how many of the split's declared scenarios the
        cache's metadata csv lists. None when either side is unknown -- a split
        whose scene filter names no tokens has no opinion, and a missing csv is
        preflight's to report."""
        declared = self._declared_tokens()
        if not declared:
            return None
        index = self._cache_index()
        if not index:
            return None
        return len(declared & set(index)), len(declared)

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
                f"see {self.DOC}. Set {ENV_ALLOW_PARTIAL}=1 to run anyway "
                f"-- the result is then stored valid=false, as a subset not comparable with a full run."
            ]
        listed = self._cache_coverage()
        if listed is not None and listed[0] < listed[1]:
            # The scorer scores what the cache's csv lists, not what the split
            # declares: an interrupted or smaller build gives a subset under the
            # split's name, and nothing later in the run would say so.
            return [
                f"partial metric cache: its metadata csv lists {listed[0]} of the {listed[1]} scenarios "
                f"{self.split} declares, so this run scores a subset, not {self.split}.\n"
                f"re-run run_metric_caching.py (it skips scenarios already cached) or set "
                f"{ENV_ALLOW_PARTIAL}=1 to run anyway -- the result is then stored valid=false."
            ]
        return []

    # -- what a run under the current switches measures --------------------

    def _agent(self) -> str:
        return os.environ.get(ENV_AGENT, self.PROXY_AGENT)

    @staticmethod
    def _overrides() -> list[str]:
        return os.environ.get(ENV_OVERRIDES, "").split()

    # -- which observation rate a model is fed ------------------------------

    @property
    def nuplan_root(self) -> Path | None:
        """The site's full-rate nuPlan data, when `SETTING_NUPLAN` names one."""
        raw = self.setting(SETTING_NUPLAN)
        return Path(raw) if raw else None

    def input_rate(self, min_input_hz: float, native_input_hz: float = 0.0) -> float:
        """Hz this model would be fed here: nuPlan's 10 when it needs or was
        built for more than OpenScene's 2 and the site holds nuPlan; 2 otherwise.

        A model that neither needs nor was built for a denser rate keeps
        OpenScene's frames even when nuPlan is present, so its recorded scores
        stay comparable with themselves.
        """
        wanted = max(float(min_input_hz or 0.0), float(native_input_hz or 0.0))
        if wanted > INPUT_HZ and self.nuplan_root is not None:
            return NUPLAN_HZ
        return INPUT_HZ

    def history_s(self, model_history_s: float, input_hz: float) -> float:
        """Seconds of history fed: the model's own ask on the dense path, or
        NAVSIM's 1.5 s where the scorer's AgentInput is what it gets."""
        if input_hz > INPUT_HZ and float(model_history_s or 0.0) > 0:
            return float(model_history_s)
        return NAVSIM_HISTORY_S

    def model_run_config(self, model) -> dict[str, Any]:
        hz = self.input_rate(model.min_input_hz, getattr(model, "native_input_hz", 0.0))
        config = {
            "input_hz": hz,
            "history_source": "nuplan" if hz > INPUT_HZ else "openscene",
            "history_s": self.history_s(getattr(model, "history_s", 0.0), hz),
        }
        if hz > INPUT_HZ and self.keyframe_cameras:
            config["keyframe_cameras"] = True
        return config

    @property
    def keyframe_cameras(self) -> bool:
        return str(self.setting(SETTING_KEYFRAME_CAMERAS, "") or "").lower() in ("1", "true", "yes")

    def _dense_problems(self, cameras: list[str]) -> list[str]:
        """Why nuPlan's data cannot feed this model's cameras for this split's logs."""
        root = self.nuplan_root
        if root is None:
            return []
        coverage = self._coverage()
        logs = sorted(coverage[0]) if isinstance(coverage, tuple) else sorted(self._declared_logs() or [])
        channels = [CAMERA_SLOTS[c].upper() for c in cameras if c in CAMERA_SLOTS]
        no_db = [log for log in logs if not (root / "splits" / self.data_split / f"{log}.db").is_file()]
        keyframes = (
            self.dataset.resolve() / "sensor_blobs" / self.data_split if self.keyframe_cameras else None
        )
        no_cam = [
            f"{log}/{channel}"
            for log in logs
            for channel in channels
            if not _has_jpg(root / "sensor_blobs" / self.data_split / log / channel)
            and not (keyframes is not None and _has_jpg(keyframes / log / channel))
        ]
        problems = []
        if no_db:
            problems.append(
                f"{len(no_db)} of {len(logs)} {self.split} logs have no nuPlan database under "
                f"{root / 'splits' / self.data_split} (first: {no_db[0]})"
            )
        if no_cam:
            problems.append(
                f"{len(no_cam)} log/camera directories have no nuPlan frames under "
                f"{root / 'sensor_blobs' / self.data_split} (first: {no_cam[0]}); "
                f"this model reads {', '.join(channels)}"
            )
        return problems

    def run_config(self) -> dict[str, Any]:
        """The switches that change what a NAVSIM run measures, for run.json."""
        return {
            "split": self.split,
            "agent": self._agent(),
            "allow_partial": bool(os.environ.get(ENV_ALLOW_PARTIAL)),
            "allow_slow": bool(os.environ.get(ENV_ALLOW_SLOW)),
            "overrides": self._overrides(),
            # Logs are chunked largest-first (see `_scorable_logs`). On the
            # record because `--resume` reuses chunks by index, and a run
            # chunked in another order has different logs behind each index.
            "chunk_order": "by_size",
            "merged_from": os.environ.get(MERGE_ENV) or None,
        }

    def invalid_reasons(self) -> list[str]:
        """Why a run under the current switches cannot be ranked with the others."""
        return self.agent_reasons() + self.switch_reasons()

    def agent_reasons(self) -> list[str]:
        """The agent switch alone: through the runner, any agent but the proxy
        means the plans did not come from the model whose name the run carries."""
        if self._agent() != self.PROXY_AGENT:
            return [
                f"scored with the vendor's native agent {self._agent()!r}, not the paddock proxy; "
                f"not comparable with proxy runs ({self.DOC})"
            ]
        return []

    def switch_reasons(self) -> list[str]:
        """The measurement switches other than the agent. Apart from the agent
        reason so that a run with no node by design -- the constant-velocity
        baseline, a replay of recorded plans (nodeless.py) -- is judged on these
        alone and states its own provenance."""
        reasons: list[str] = []
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
            listed = self._cache_coverage()
            if listed is not None and listed[0] < listed[1]:
                reasons.append(
                    f"{ENV_ALLOW_PARTIAL}: the metric cache lists {listed[0]} of the {listed[1]} "
                    f"scenarios {self.split} declares; a subset, not {self.split}"
                )
        return reasons

    def model_problems(self, model, limit: int | None = None) -> list[str]:
        hz = self.input_rate(model.min_input_hz, getattr(model, "native_input_hz", 0.0))
        problems = input_rate_problems(float(model.min_input_hz or 0.0), model.name, hz)
        # Cameras and the dense history are the proxy's concern; a node-less
        # agent (baseline, human, replay) loads no sensors at all.
        if self._agent() == self.PROXY_AGENT:
            problems += camera_problems(list(model.cameras), model.name)
            if hz > INPUT_HZ and not problems:
                problems += self._dense_problems(list(model.cameras))
        return problems + limit_problems(limit, self.MIN_SCENARIOS)

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

        GPU models get `gpu_nodes`, default 1: the card is shared, and the
        `infer_sequence` override already took most of them off the critical
        path. A site raises it for a small model bound by its per-request
        forward, not by memory (AutoE2E: ~3 GB a node). The setting reaches
        **every** GPU model the run scores, so it is capped by what the card
        holds, as `Comma2k19Benchmark.node_count` explains: `gpu.wait_for` gates
        each launch on declared VRAM, and four 26 GB alpamayo nodes into 30 GB
        free start one and park three in a two-hour poll. The card read is the
        one this model's node environment will land on; without nvidia-smi it is
        the site's `[gpu] budget_gb`.
        """
        wanted = int(self.setting("nodes", "1") or 1)
        if model.vram_gb > 0:
            nodes = max(1, int(self.setting("gpu_nodes", "1") or 1))
            if nodes == 1:
                return 1
            try:
                node_env = env_builder.build(model, self.cfg)
            except Exception:  # noqa: BLE001 - a node count is never worth failing a run over
                node_env = None
            available = gpu.free_gb(node_env)
            if available is None:
                available = float(self.cfg.gpu.budget_gb)
            return max(1, min(nodes, int(available // model.vram_gb)))
        return max(1, wanted)

    def _scene_filter_value(self, key: str) -> list | None:
        """One list from the devkit's own scene filter for this split.

        None when it cannot be read -- no devkit, no file, no such key, a yaml
        error -- and the caller reports that, rather than falling back to a
        directory count that is known to be wrong by one.
        """
        devkit = self.devkit
        if devkit is None:
            return None
        path = devkit / SPLITS_DIR / "scene_filter" / f"{self.split}.yaml"
        if not path.exists():
            return None
        try:
            import yaml

            value = yaml.safe_load(path.read_text()).get(key)
        except Exception:  # noqa: BLE001 - a preflight helper must not raise; None is the report
            return None
        return list(value) if value else None

    def _declared_logs(self) -> set[str] | None:
        """The logs this split declares."""
        names = self._scene_filter_value("log_names")
        return set(names) if names else None

    def _declared_tokens(self) -> set[str] | None:
        """The scenario tokens this split declares; navtest names all 12,146,
        a training split may name none."""
        tokens = self._scene_filter_value("tokens")
        return set(tokens) if tokens else None

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
        # The devkit root goes first: the interpreter's `navsim` may be an
        # editable install of the *other* generation (a .pth naming navsim_v2,
        # which site.py appends after site-packages), and PYTHONPATH decides
        # which tree `import navsim` -- and hydra's pkg://navsim searchpath --
        # resolves to. For the generation that is installed, nothing changes.
        # Then paddock's src: the proxy agent is imported inside the devkit's
        # interpreter, where paddock is not installed, and hydra has to find
        # the module by its dotted path.
        src = str(Path(__file__).resolve().parents[4])
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = ":".join([str(devkit), src] + ([existing] if existing else []))
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
        cameras = supplied_cameras(list(config.get("cameras") or ["CAM_FRONT"]))

        command = [
            str(self.setting("python")),
            "-u",
            str(self.devkit / self.SCORER),
            f"hydra.searchpath=[pkg://navsim.planning.script.config.common,file://{self.CONF_DIR}]",
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
        if agent == self.PROXY_AGENT:
            urls = list(config.get("policy_urls") or [policy_url])
            command += [
                f"agent.node_url={policy_url}",
                "agent.node_urls=[" + ",".join(urls) + "]",
                "agent.cameras=[" + ",".join(cameras) + "]",
                f"agent.sensor_blobs_path={root / 'sensor_blobs' / self.data_split}",
                f"agent.dump_path={out_dir / 'trajectories.jsonl'}",
            ]
            hz = self.input_rate(config.get("min_input_hz", 0.0), config.get("native_input_hz", 0.0))
            if hz > INPUT_HZ:
                command += [
                    f"agent.nuplan_root={self.nuplan_root}",
                    f"agent.nuplan_split={self.data_split}",
                    f"agent.dense_hz={hz:g}",
                    f"agent.history_s={self.history_s(config.get('history_s', 0.0), hz):g}",
                ]
                if self.keyframe_cameras:
                    command.append("agent.keyframe_cameras=true")
        if agent == REPLAY_AGENT and config.get("plans_dir"):
            # The run being replayed, by artifacts directory. Passed on the
            # command line rather than through PADDOCK_NAVSIM_OVERRIDES so the
            # tracked run.json records the run's id, not a machine path.
            command.append(f"agent.plans_dir={config['plans_dir']}")
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

    def _token_logs(self) -> dict[str, str]:
        """token -> log, once per harness: the metric cache's metadata csv
        (`_cache_index`) when it has one, else walked from the cache's own
        layout `<cache>/<log>/<type>/<token>/`. The scorer's CSV names no log,
        so this is the one place a token's log is known outside the scorer."""
        if getattr(self, "_token_log_map", None) is None:
            mapping = dict(self._cache_index())
            if not mapping:
                cache = Path(self.setting("metric_cache") or "")
                if cache.is_dir():
                    for log_dir in cache.iterdir():
                        if log_dir.is_dir():
                            for token_dir in log_dir.glob("*/*"):
                                if token_dir.is_dir():
                                    mapping[token_dir.name] = log_dir.name
            self._token_log_map = mapping
        return self._token_log_map

    def _logs_of(self, merged_csv: Path, tokens: list[str]) -> list[str]:
        """The logs a set of tokens belong to, so the retry pass loads only those
        logs' metadata. From the scored CSV when it has a `log_name` column,
        otherwise from the metric cache's metadata csv; the alternative is the
        scorer opening every log pickle in the split to find a handful of tokens."""
        wanted = set(tokens)
        logs: set[str] = set()
        with merged_csv.open() as handle:
            reader = csv.DictReader(handle)
            if "log_name" in (reader.fieldnames or []):
                for row in reader:
                    if (row.get("token") or "").strip() in wanted and row.get("log_name"):
                        logs.add(row["log_name"].strip())
                return sorted(logs)
        index = self._token_logs()
        return sorted({index[token] for token in wanted if token in index})

    def _scorable_logs(self) -> list[str]:
        """The split's logs this site can score, in a fixed order.

        Raises with `_coverage()`'s own message when the split's log list
        cannot be read: a caller that skips the runner gets the same refusal
        preflight gives, not a TypeError.
        """
        coverage = self._coverage()
        if coverage is None:
            return []
        if isinstance(coverage, str):
            raise RuntimeError(coverage)
        scorable, _ = coverage
        # Largest first. The scorer hands each of its workers one whole log,
        # so a chunk costs its longest log: chunked alphabetically, a 156-
        # scenario log next to 22-scenario ones left seven of eight nodes idle
        # for two hours at a time. Grouping logs of like size fills the
        # workers; name breaks ties so the order is fixed for `--resume`.
        return sorted(scorable, key=lambda log: (-self._cache_scenarios([log]), log))

    def _cache_scenarios(self, log_names: list[str]) -> int:
        """How many scenarios the metric cache holds for these logs.

        The cache is laid out `<cache>/<log>/<scenario type>/<token>/`, one
        directory per scenario, and the scorer scores exactly the cached
        scenarios of the logs it is given -- so this is what a chunk would have
        produced, and what a failed chunk lost.
        """
        cache = Path(self.setting("metric_cache") or "")
        total = 0
        for name in log_names:
            log_dir = cache / name
            if log_dir.is_dir():
                total += sum(1 for path in log_dir.glob("*/*") if path.is_dir())
        return total

    def _merge(self, csvs: list[Path], out_dir: Path) -> Path:
        return merge_chunk_csvs(csvs, out_dir / "scores.csv", self.VOCABULARY.aggregate_tokens)

    def _parse(self, merged: Path):
        """The scorer's CSV through the shared parser, in this devkit's vocabulary."""
        return navsim_ingest.parse_scores(
            merged, split=self.split, vocabulary=self.VOCABULARY, benchmark=self.spec.name
        )

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
        hz = self.input_rate(config.get("min_input_hz", 0.0), config.get("native_input_hz", 0.0))
        problems = input_rate_problems(
            float(config.get("min_input_hz") or 0.0), str(config.get("model", "this model")), hz
        ) + limit_problems(limit, self.MIN_SCENARIOS)
        if self._agent() == self.PROXY_AGENT:
            model_name = str(config.get("model", "this model"))
            problems += camera_problems(list(config.get("cameras") or []), model_name)
            if hz > INPUT_HZ and not problems:
                problems += self._dense_problems(list(config.get("cameras") or ["CAM_FRONT"]))
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
        # Rows another run already scored, taken as this run's own: their
        # logs are not chunked again. See MERGE_ENV.
        merged_csvs: list[Path] = []
        merged_logs: set[str] = set()
        merge_from = os.environ.get(MERGE_ENV)
        if merge_from and size > 0:
            merged_csvs = completed_chunk_csvs(Path(merge_from))
            token_logs = self._token_logs()
            merged_logs = {token_logs[t] for t in tokens_in_csvs(merged_csvs) if t in token_logs} & set(logs)
            if not merged_csvs:
                raise RuntimeError(f"{MERGE_ENV}={merge_from} has no completed chunk CSVs under chunks/")
        chunks = chunk_logs([log for log in logs if log not in merged_logs], size) if logs else [[]]
        # Only one way to reuse another run's chunks, and it is this variable;
        # `--resume` continues a run in its own directory and needs nothing here.
        resume_from = os.environ.get(RESUME_ENV)
        # What a complete run would score, so a partial one can say what it
        # lost. Unknown for a --limit smoke, which the scorer caps itself.
        n_expected = self._cache_scenarios(logs) if size > 0 else None
        # The chunk-by-chunk story of this run, across every attempt: its own
        # file, appended -- `navsim_scorer.log` is the scorer's, and for an
        # unchunked run lives at this same level.
        summary = out_dir / "chunks.log"
        echo(
            f"  scoring {self.split} via {Path(self.setting('python') or 'python').name}"
            + (f" in {len(chunks)} chunk(s) of <= {size} log(s)" if size > 0 else "")
            + (f", resuming from {resume_from}" if resume_from else "")
        )

        csvs: list[Path] = list(merged_csvs)
        failed_chunks: list[dict[str, Any]] = []
        with summary.open("a") as log:
            log.write(f"attempt {config.get('attempt', 1)}: {len(chunks)} chunk(s)\n")
            if merged_csvs:
                log.write(
                    f"merged {len(merged_csvs)} completed chunk CSV(s) from {merge_from}: "
                    f"{len(merged_logs)} log(s) not scored again\n"
                )
                echo(
                    f"  taking {len(merged_csvs)} completed chunk(s) from {merge_from}; "
                    f"{len(merged_logs)} log(s) skipped"
                )
            log.flush()
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
                    # is recorded -- with how many scenarios it would have
                    # scored, so the run says what it lost -- the run is marked
                    # partial, and a later run can resume from what completed.
                    failed_chunks.append(
                        {"index": index, "logs": list(names), "scenarios": self._cache_scenarios(names)}
                    )
                    log.write(f"chunk {index:03d}: FAILED -- {str(exc).splitlines()[0]}\n")
                    log.flush()
                    if self._agent() == self.PROXY_AGENT and not node_alive(policy_url):
                        # Every remaining chunk would fail the same way. Hand the
                        # run back to the runner, which restarts the nodes and
                        # re-invokes us; the chunks that completed are reused.
                        log.write(f"policy node {policy_url} is not answering; stopping for a restart\n")
                        raise RuntimeError(
                            f"policy node {policy_url} is not answering after chunk {index:03d} failed"
                        ) from exc
                    echo(f"  chunk {index:03d} FAILED ({', '.join(names) or 'whole split'}); continuing")

        if not csvs:
            raise RuntimeError(f"every chunk failed; nothing scored. See {summary} and {out_dir / 'chunks'}")
        merged = self._merge(csvs, out_dir) if size > 0 else csvs[-1]
        samples, aggregates, failed = self._parse(merged)

        # Scenarios the model failed on inside a chunk that otherwise completed
        # get one more pass of their own, by token. For an unseeded sampler that
        # is a fresh draw; for anything else it fails again and stays excluded.
        # Whole chunks that failed are the runner's to retry with fresh nodes.
        if failed and size > 0:
            retry_dir = out_dir / "chunks" / "retry"
            # A retry a previous attempt completed is merged, not run again.
            retried = completed_chunk_csv(retry_dir)
            if retried is None:
                tokens = retry_tokens(failed, self._token_logs(), self.MIN_SCENARIOS)
                neighbours = len(tokens) - len(failed)
                echo(
                    f"  re-scoring {len(failed)} failed scenario(s) by token"
                    + (f", with {neighbours} neighbour(s) so the scorer can aggregate" if neighbours else "")
                )
                logs_of = self._logs_of(merged, tokens)
                try:
                    retried = self._run_scorer(
                        self._command(policy_url, retry_dir, config, limit, logs_of or None, tokens),
                        retry_dir,
                        root,
                        exp,
                    )
                    keep_tokens(retried, set(failed), self.VOCABULARY.aggregate_tokens)
                except RuntimeError as exc:
                    echo(f"  retry pass FAILED ({str(exc).splitlines()[0]}); the {len(failed)} stay excluded")
                    retried = None
            if retried is not None:
                csvs.append(retried)
                merged = self._merge(csvs, out_dir)
                samples, aggregates, failed = self._parse(merged)
        # The scorer's own headline is per invocation and means nothing across
        # chunks; only a single-invocation run keeps it.
        reported = navsim_ingest.log_reported_score(out_dir) if size == 0 else None
        unscored = sum(chunk["scenarios"] for chunk in failed_chunks)
        lost = f", {len(failed_chunks)} chunk(s) never scored ({unscored} scenarios)" if failed_chunks else ""
        echo(
            f"  {len(samples)} scenario(s) in {time.time() - began:.0f}s"
            + (f", scorer reported {reported:.5f}" if reported is not None else "")
            + (f", {len(failed)} FAILED (excluded)" if failed else "")
            + lost
        )
        # Kept on the run rather than folded into the score: the scorer's own
        # headline and paddock's recomputed mean answer slightly different
        # questions. Stored in the tracked metrics.json under `by_group`, the
        # same keys the ingest path writes. See paddock/results/ingest/navsim.py.
        by_group: dict[str, Any] = {
            "source_aggregates": aggregates,
            "source_reported": {"final_average_valid": reported} if reported is not None else {},
        }
        if self._agent() == self.PROXY_AGENT and hz > INPUT_HZ:
            by_group["dense_history"] = dense_history_summary(out_dir)
        return Evaluation(
            samples=samples,
            failed=failed,
            by_group=by_group,
            failed_chunks=failed_chunks,
            n_expected=n_expected,
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
