# Fixtures

| File | Provenance |
|---|---|
| `results_summary_two_scenes.json` | **Made-up numbers in a real summary's shape.** The keys, their order and the `score_criteria` are those of a `results-summary.json` written by AlpaSim `affc2ea`; every measured value and run id is invented, so the repository holds no results measured on licensed scenes. The scene ids are two of AlpaSim's own published ones (`data/scenes/`). One scene passes cleanly (score 1.0); the other makes 96% progress and is then zeroed by an offroad event, which is the rule worth pinning. Trimmed to the `rollouts`, `scene_score_enabled` and `score_criteria` keys — the `metrics_results` block is deliberately absent, because a reader that trusts it drops the score. |
| `fake_wizard.py` | **Synthesised stand-in** for `alpasim_wizard.main`, so the launch path (exit codes, timeouts, process-group cleanup, a missing summary) is tested rather than argued. It writes a canned summary and never starts a container. Not a recording of anything. |
