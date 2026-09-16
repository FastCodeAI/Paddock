# Fixtures

| Directory | Provenance |
|---|---|
| `alpamayo_two_scenes/` | Cut from the real AlpaSim run `alpamayo_pilot10` (`run_metadata.yaml` `run_time: 2026-09-07 08:13:51`, run uuid `fbd6cf24…`): Alpamayo 1.5 driver `0.91.0`, `public_2601` first ten, 4 cameras at 10 Hz, scored by AlpaSim `affc2ea`. `aggregate/results-summary.json` keeps two of the ten rollouts verbatim — `clipgt-01d503d4…` (clean pass, score 1.0) and `clipgt-02e075b9…` (real progress, then zeroed by an offroad event) — and the run's `metrics_results` block unchanged, so that block still describes all ten scenes, which is the trap the reader must not fall into. The YAML files are trimmed to the keys the reader reads, and the console log is three lines taken verbatim from `wizard_console_alpamayo_pilot10.log`, renamed for this fixture. |
