"""Bench2Drive's own per-route scores, computed by Bench2Drive's own code: the client half.

Runs inside the CARLA-client interpreter (Python 3.8, carla 0.9.15), because the upstream
scorers import ``carla`` and its route planner. ``upstream_scores.py`` (paddock's side) writes
the request and reads the answer:

    python upstream_scores_client.py <bench2drive repo> <request.json> <answer.json>

The request lists the record's routes: id, town, scenario, ``results.json`` and the route's
``metric_info.json`` (or null). The answer holds, per route, upstream's success, the abilities
its scenario counts towards and its credit in each, efficiency and comfort; and, pooled, what
``tools/ability_benchmark.py`` itself prints for these routes.

Upstream's code is imported from the checkout, never restated -- ``Ability``,
``get_infraction_status``, ``load_offline_map``, ``get_position``, ``get_waypoint_route``,
``read_from_json``, ``seg_compute_comfort_metric`` -- with one exception: the traffic-sign test
for a route that did not succeed is inline in ``ability_benchmark.main``, so it is restated
below, and the per-route credits are checked against ``main``'s own pooled numbers on the same
routes before anything is written. A disagreement is an error, not an answer.

Kept 3.8-compatible, like ``agent.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("request")
    ap.add_argument("answer")
    args = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(args.repo, "tools"))
    try:
        import ability_benchmark as ab
        import efficiency_smoothness_benchmark as es
    except ImportError as exc:  # scipy and networkx: the scorers need them, driving does not
        sys.exit(
            f"Bench2Drive's scorers cannot be imported in {sys.executable}: {exc}. "
            "They need scipy and networkx there (docs/benchmarks/bench2drive.md, Efficiency and comfort)."
        )

    request = json.load(open(args.request))
    routes_xml = request["routes_xml"]
    map_root = os.path.join(os.environ["CARLA_ROOT"], "CarlaUE4/Content/Carla/Maps")

    # One map and one route planner per town, shared with upstream's own pass below: Town12's
    # and Town13's OpenDRIVE files are 71 and 89 MB.
    maps, planners = {}, {}
    load_map = ab.load_offline_map

    def cached_map(town, root):
        if town not in maps:
            maps[town] = load_map(town, root)
        return maps[town]

    ab.load_offline_map = cached_map

    def planner(town):
        if town not in planners:
            planners[town] = ab.GlobalRoutePlanner(cached_map(town, map_root), 1.0)
        return planners[town]

    import xml.etree.ElementTree as ET

    xml_routes = {r.get("id"): r for r in ET.parse(routes_xml).getroot().findall("route")}

    work = tempfile.mkdtemp(prefix="upstream_scores_")
    rows, records = [], []
    for route in request["routes"]:
        record = json.load(open(route["results"]))["_checkpoint"]["records"][0]
        record.pop("index", None)
        records.append(record)
        scenario = route["scenario"]
        success = record["status"] in ("Completed", "Perfect") and not ab.get_infraction_status(record)
        abilities = [name for name, scenarios in ab.Ability.items() if scenario in scenarios]
        credit = {name: 1.0 if success else 0.0 for name in abilities}
        traffic_sign_test = None
        if "Traffic_Signs" in abilities and not success:
            # ability_benchmark.main at 7ec25d1, its Traffic_Signs branch, word for word but for
            # the cached map and planner: credit when the car got past the route's first junction
            # (plus 8 waypoints) with no stop-sign or red-light infraction.
            locations = ab.get_position(xml_routes[route["route_id"]])
            waypoints = ab.get_waypoint_route(locations, planner(route["town"]))
            count = 0
            for wp in waypoints:
                count += 1
                if wp.is_junction:
                    break
            if not wp.is_junction:
                raise RuntimeError(f"No junction found in route {route['route_id']}")
            junction_completion = float(count + 8) / len(waypoints)
            record_completion = record["scores"]["score_route"] / 100.0
            passed = (
                record_completion > junction_completion
                and not record["infractions"]["stop_infraction"]
                and not record["infractions"]["red_light"]
            )
            traffic_sign_test = {"junction_completion": round(junction_completion, 6), "passed": bool(passed)}
            credit["Traffic_Signs"] = 1.0 if passed else 0.0
        efficiency, comfort, comfort_error = None, None, None
        metric_info = route.get("metric_info")
        ticks = bool(metric_info) and os.path.getsize(metric_info) > 0
        # read_from_json joins its metric_dir with each record's save_name: one record, one link.
        # Efficiency needs only the record, so a route without its ticks gets an empty file there,
        # and only its comfort is missing, with the reason.
        one = os.path.join(work, route["route_id"])
        os.makedirs(one, exist_ok=True)
        link = os.path.join(one, record["save_name"])
        if ticks:
            os.symlink(os.path.dirname(metric_info), link)
        else:
            os.makedirs(link)
            json.dump({}, open(os.path.join(link, "metric_info.json"), "w"))
            comfort_error = "metric_info.json is empty" if metric_info else "no metric_info.json"
        merged_one = os.path.join(one, "merged.json")
        json.dump({"_checkpoint": {"records": [record]}}, open(merged_one, "w"))
        try:
            data, driving_efficiency = es.read_from_json(merged_one, one)
            efficiency = driving_efficiency[0] if driving_efficiency else None
            if ticks:
                comfort = float(es.seg_compute_comfort_metric(**data[0]))
        except Exception as exc:  # upstream raises on a route too short to segment; say so
            comfort_error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "route_id": route["route_id"],
                "town": route["town"],
                "scenario": scenario,
                "status": record["status"],
                "success": bool(success),
                "driving_score": record["scores"]["score_composed"],
                "route_completion": record["scores"]["score_route"],
                "infraction_penalty": record["scores"]["score_penalty"],
                "infractions": {k: len(v) for k, v in record["infractions"].items()},
                "abilities": credit,
                "traffic_sign_test": traffic_sign_test,
                "efficiency": efficiency,
                "comfort": comfort,
                "comfort_error": comfort_error,
                "duration_game_s": (record.get("meta") or {}).get("duration_game"),
                "ended_early": route.get("ended_early"),
            }
        )

    # Upstream's own pass over the same routes, as `python tools/ability_benchmark.py -r merged.json`.
    merged = os.path.join(work, "merged.json")
    json.dump({"_checkpoint": {"records": records}}, open(merged, "w"))
    missing = [name for name in ab.Ability if not any(name in r["abilities"] for r in rows)]
    if missing:
        # ability_benchmark.main divides by each ability's route count and would raise ZeroDivisionError.
        raise RuntimeError(
            f"no route here counts towards {', '.join(missing)}, and tools/ability_benchmark.py cannot "
            "score a set without one"
        )
    ab.main(argparse.Namespace(file=routes_xml, result_file=merged))
    upstream = json.load(open(merged[: -len(".json")] + "_ability.json"))

    per_ability = {}
    for name in ab.Ability:
        credits = [r["abilities"][name] for r in rows if name in r["abilities"]]
        per_ability[name] = {
            "routes": len(credits),
            "score": sum(credits) / len(credits) if credits else None,
        }
        if credits and not math.isclose(per_ability[name]["score"], upstream[name], abs_tol=1e-9):
            raise RuntimeError(
                f"{name}: the per-route credits give {per_ability[name]['score']!r}, "
                f"ability_benchmark.main gives {upstream[name]!r}"
            )
    efficiencies = [r["efficiency"] for r in rows if r["efficiency"] is not None]
    comforts = [r["comfort"] for r in rows if r["comfort"] is not None]
    pooled = {
        "routes": len(rows),
        "success": sum(1 for r in rows if r["success"]),
        "success_rate": sum(1 for r in rows if r["success"]) / len(rows) if rows else None,
        "driving_score": sum(r["driving_score"] for r in rows) / len(rows) if rows else None,
        "abilities": per_ability,
        "multi_ability_mean": upstream["mean"],
        "efficiency": sum(efficiencies) / len(efficiencies) if efficiencies else None,
        "efficiency_routes": len(efficiencies),
        "comfort": sum(comforts) / len(comforts) if comforts else None,
        "comfort_routes": len(comforts),
    }
    json.dump({"routes": rows, "pooled": pooled}, open(args.answer, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
