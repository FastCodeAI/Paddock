# Fixtures

| File | Provenance |
|---|---|
| `results_collisions.json` | A real `results.json` written by Bench2Drive 0.0.4 for `RouteScenario_1773` (ParkedObstacle, Town12): route completed with two vehicle collisions, so penalty 0.6 × 0.6 = 0.36 and driving score 36.0. Recorded before this adapter existed, by the checkout's own `run_evaluation.sh`. |
| `results_agent_crashed.json` | **Synthesised**, see its `_provenance` field: the record shape and the `Failed - Agent crashed` status string are copied from `statistics_manager.py` at 7ec25d1; the numbers are what that code writes for a route stopped mid-way. No crashed run has been kept at any site yet -- replace it with a real one when there is one. |
| `evaluator_carla_crash.log` | The last 11 lines, verbatim, of a real `evaluator.log` for route 11715 (Town12 EnterActorFlow), driven by openpilot on 7 Oct 2026: the agent's last tick at 53.65 s of game time, then CARLA's `CommonUnixCrashHandler: Signal=11`. The log's first line, the evaluator's command, is left out: it names the machine's paths. |
| `routes_sample.xml` | The id, town and scenario type of two routes in Bench2Drive's `leaderboard/data/bench2drive220.xml` (1711 and 1773), the names Paddock's results already use for them. The rest of each entry -- its waypoints, weather and scenario parameters, 97% of its text -- is left out. Bench2Drive's route list is under CC BY-NC-ND 4.0. |
