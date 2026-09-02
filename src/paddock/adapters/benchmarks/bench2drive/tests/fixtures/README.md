# Fixtures

| File | Provenance |
|---|---|
| `results_collisions.json` | A real `results.json` written by Bench2Drive 0.0.4 for `RouteScenario_1773` (ParkedObstacle, Town12): route completed with two vehicle collisions, so penalty 0.6 × 0.6 = 0.36 and driving score 36.0. Recorded before this adapter existed, by the checkout's own `run_evaluation.sh`. |
| `results_agent_crashed.json` | **Synthesised**, see its `_provenance` field: the record shape and the `Failed - Agent crashed` status string are copied from `statistics_manager.py` at 7ec25d1; the numbers are what that code writes for a route stopped mid-way. No crashed run has been kept at any site yet -- replace it with a real one when there is one. |
| `routes_sample.xml` | Two routes cut verbatim from `leaderboard/data/bench2drive220.xml` (ids 1711 and 1773). |
