# Fixtures

| Directory | Provenance |
|---|---|
| `alpamayo_two_scenes/` | Laid out like the AlpaSim run `alpamayo_pilot10` (`run_metadata.yaml` `run_time: 2026-09-07 08:13:51`): Alpamayo 1.5 driver `0.91.0`, `public_2601` first ten, 4 cameras at 10 Hz, scored by AlpaSim `affc2ea`. The YAML files are that run's, trimmed to the keys the reader reads, and the console log is three lines taken from `wizard_console_alpamayo_pilot10.log`, renamed for this fixture; none of them holds a score. **`aggregate/results-summary.json` holds made-up numbers** in that run's shape: every measured value and run id is invented, so the repository holds no results measured on licensed scenes. It has two rollouts — `clipgt-02e075b9…` (clean pass, score 1.0) and `clipgt-01d503d4…` (96% progress, then zeroed by an offroad event); which scene passes is made up as well — and a `metrics_results` block, invented too, that still describes ten scenes, which is the trap the reader must not fall into. |
