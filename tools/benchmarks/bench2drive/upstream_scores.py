"""Bench2Drive's own per-route scores beside a pooled record: ``upstream_scores.json``.

A paddock record carries paddock's metrics (driving score, route completion, infraction
penalty, collision rate), the same for every benchmark. Bench2Drive's published tables also
report success, five abilities and their mean, efficiency and comfort, which only its own
scripts compute. This writes those, per route and pooled, into the record's directory, so a
220-route record can be set beside a published table:

    PADDOCK_CONFIG=<sweep>/paddock.toml python -m tools.benchmarks.bench2drive.upstream_scores \\
        results/bench2drive/<model>/<run_id> --runs <the ingest block's runs root> [...]

The routes are found as ``paddock ingest`` found them -- the latest scored attempt of each
route the record expects -- and refused unless they are exactly the record's: the same sample
ids (``dataset.sequences_sha256``), the same source runs, and for each route the very
``results.json`` the record pooled (its run and sha256, from the record's ``samples.jsonl``,
which ``paddock ingest`` writes under the runs root of the config it ran with -- so run this
with that config). The scores are computed by
Bench2Drive's own code at the pinned checkout, in its CARLA-client interpreter
(``upstream_scores_client.py``), and the per-route ability credits are checked against
``tools/ability_benchmark.py``'s own pooled numbers on the same routes.

Pooled values are over the record's routes, as its paddock metrics are: the routes set aside
for every model are in none of them. Bench2Drive's ``merge_route_json.py`` divides driving
score and success by 220 instead, a missing route counting as 0; each route's row is here, so
either convention can be recomputed. Regenerate the file, never edit it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from paddock.adapters.benchmarks.bench2drive.harness import BENCHMARK, ended_early, parse_routes
from paddock.config import SiteConfig
from paddock.registry import Registry
from paddock.results.ingest.bench2drive import discover, route_parts
from paddock.results.store import run_dir, sequence_hash

CLIENT = Path(__file__).with_name("upstream_scores_client.py")
OUT = "upstream_scores.json"


class Mismatch(ValueError):
    """The runs tree no longer holds exactly the attempts the record was pooled from."""


def record_attempts(record: dict[str, Any], roots: list[Path], samples: Path) -> dict[str, Any]:
    """The record's route folder -> its ``Attempt``, found as the ingest found them, or ``Mismatch``.

    ``samples`` is the record's ``samples.jsonl``, where ``paddock ingest`` wrote the run and the
    sha256 of each route's ``results.json``. Route names and source runs agreeing is not enough: a
    run continued after pooling (``--resume``) can hold a later drive of a route the record took
    from another run, so each attempt found must be the very file the record pooled.
    """
    config, dataset = record.get("config") or {}, record.get("dataset") or {}
    wanted = [str(r) for r in config.get("routes_subset") or []]
    if not wanted:
        raise Mismatch("the record names no routes (config.routes_subset)")
    survey = discover(roots, wanted)
    scored = {f: a for f, a in survey.scored.items() if route_parts(f) and route_parts(f)[0] in set(wanted)}
    ids = [f"bench2drive/{f}" for f in scored]
    if sequence_hash(ids) != dataset.get("sequences_sha256"):
        raise Mismatch(
            f"the trees give {len(ids)} scored route(s) whose hash is not the record's "
            f"({dataset.get('n_samples')} sample(s)): pool the record again, or point --runs at its trees"
        )
    runs = sorted({a.run_id for a in scored.values()})
    if runs != sorted(config.get("source_runs") or []):
        raise Mismatch("the trees' latest scored attempts come from other runs than the record's source_runs")
    pooled = _pooled_from(samples)
    for folder, attempt in sorted(scored.items()):
        found = (attempt.run_id, hashlib.sha256(attempt.results.read_bytes()).hexdigest())
        if pooled.get(f"bench2drive/{folder}") != found:
            raise Mismatch(f"{folder}: the trees' latest scored attempt is not the drive the record pooled")
    return scored


def _pooled_from(samples: Path) -> dict[str, tuple[str, str]]:
    """Sample id -> the run and the sha256 of the ``results.json`` the record pooled for it."""
    if not samples.is_file():
        raise Mismatch(
            f"no {samples}: pool the record with `paddock ingest` under this config, which writes it"
        )
    rows = [json.loads(line) for line in samples.read_text().splitlines() if line.strip()]
    return {r["sample_id"]: (r["native_id"]["run_id"], r["native_id"]["sha256"]) for r in rows}


def request_for(scored: dict[str, Any], routes_xml: Path) -> dict[str, Any]:
    """What the client is asked: each route's results.json and metric_info.json, and its early stop."""
    routes = []
    for folder, attempt in sorted(scored.items(), key=lambda kv: int(route_parts(kv[0])[0])):
        rid, town, scenario = route_parts(folder)
        route_dir = attempt.results.parent
        record = json.loads(attempt.results.read_text())["_checkpoint"]["records"][0]
        metric_info = route_dir.parent / "metric_info" / record.get("save_name", "") / "metric_info.json"
        ended = [k[len("ended_") :] for k, v in ended_early(route_dir).items() if v]
        routes.append(
            {
                "route_id": rid,
                "town": town,
                "scenario": scenario,
                "results": str(attempt.results),
                "metric_info": str(metric_info) if metric_info.is_file() else None,
                "ended_early": ended[0] if ended else None,
            }
        )
    return {"routes_xml": str(routes_xml), "routes": routes}


def scores(
    record_dir: Path, roots: list[Path], cfg: SiteConfig, samples: Path | None = None
) -> dict[str, Any]:
    """Run the client on the record's routes; the file's content. ``samples`` defaults to where
    ``paddock ingest`` wrote the record's ``samples.jsonl``: under this config's runs root."""
    record = json.loads((record_dir / "run.json").read_text())
    samples = (
        samples or run_dir(record["benchmark"], record["model"], record["run_id"], cfg.runs) / "samples.jsonl"
    )
    harness = BENCHMARK(Registry.load().benchmarks["bench2drive"], cfg)
    python, repo, routes_xml = harness.setting("python"), harness.repo, harness.routes_file
    if not python or repo is None or routes_xml is None or not routes_xml.exists():
        raise SystemExit("[benchmarks.bench2drive] python, repo and the routes file are needed")
    scored = record_attempts(record, roots, samples)
    request = request_for(scored, routes_xml)
    with tempfile.TemporaryDirectory() as tmp:
        asked, answer = Path(tmp) / "request.json", Path(tmp) / "answer.json"
        asked.write_text(json.dumps(request))
        client = subprocess.run(
            [python, "-u", str(CLIENT), str(repo), str(asked), str(answer)],
            env=harness._environment({}),
            cwd=tmp,
        )
        if client.returncode != 0:  # it has said why on stderr, above
            raise SystemExit(f"{CLIENT.name} exited {client.returncode}; nothing written")
        result = json.loads(answer.read_text())
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    known = {rid for rid, _, _ in parse_routes(routes_xml)}
    return {
        "record": record["run_id"],
        "model": record["model"],
        "bench2drive": sha or None,
        "routes_file": routes_xml.name,
        "routes_set_aside": (record.get("config") or {}).get("routes_set_aside") or [],
        "routes_in_file": len(known),
        "pooled": result["pooled"],
        "routes": result["routes"],
        "how": (
            "Bench2Drive's own scorers at the checkout named, over the record's routes: success as "
            "merge_route_json.py counts it, abilities as tools/ability_benchmark.py does (its pooled "
            "numbers checked against the per-route credits), efficiency and comfort as "
            "tools/efficiency_smoothness_benchmark.py does. Pooled values are over these routes, not "
            "divided by 220."
        ),
    }


def dumps(content: dict[str, Any]) -> str:
    """The file's text: indented, but each route's row on one line, so a diff shows which routes changed."""
    head = {k: v for k, v in content.items() if k != "routes"}
    text = json.dumps(head, indent=1)[:-2]
    rows = ",\n".join("  " + json.dumps(row, sort_keys=True) for row in content["routes"])
    return f'{text},\n "routes": [\n{rows}\n ]\n}}\n'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("record", type=Path, help="results/bench2drive/<model>/<run_id>")
    ap.add_argument("--runs", type=Path, nargs="+", required=True, help="the ingest block's runs root(s)")
    args = ap.parse_args(argv)
    try:
        content = scores(args.record, args.runs, SiteConfig.load())
    except Mismatch as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    (args.record / OUT).write_text(dumps(content))
    pooled = content["pooled"]
    print(
        f"{args.record / OUT}: {pooled['routes']} routes, success {pooled['success']} "
        f"({pooled['success_rate']:.4f}), multi-ability {pooled['multi_ability_mean']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
