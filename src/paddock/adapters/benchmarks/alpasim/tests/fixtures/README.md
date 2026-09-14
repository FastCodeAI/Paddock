# Fixtures

| File | Provenance |
|---|---|
| `results_summary_two_scenes.json` | Two rollouts cut verbatim from a real `results-summary.json` written by AlpaSim `affc2ea` for the `alpamayo15` / `public_2601-first10` run of 2026-09-07. One scene passed cleanly (score 1.0); the other made 98.8% progress and was then zeroed by an offroad event, which is the rule worth pinning. Trimmed to the `rollouts`, `scene_score_enabled` and `score_criteria` keys — the `metrics_results` block is deliberately absent, because a reader that trusts it drops the score. |
| `fake_wizard.py` | **Synthesised stand-in** for `alpasim_wizard.main`, so the launch path (exit codes, timeouts, process-group cleanup, a missing summary) is tested rather than argued. It writes a canned summary and never starts a container. Not a recording of anything. |
