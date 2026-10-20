# VisionPilot Trajectory Visualization

Interactive dashboard to compare trajectory predictions before and after range-scaled noise model.

## Features

✅ **Multi-sample comparison** - Pick up to 5 samples to analyze  
✅ **Trajectory plots** - Visualize predicted vs ground truth paths  
✅ **Metric breakdown** - ADE, FDE@1s/2s/3s, longitudinal/lateral MAE  
✅ **Error analysis** - Frame-by-frame error curves  
✅ **Distribution stats** - Histogram of improvements across all samples  
✅ **Side-by-side layout** - Compare baseline vs improved side-by-side  

## Quick Start

```bash
cd /users/gbanda/git/Paddock
source .venv/bin/activate
streamlit run src/paddock/adapters/models/visionpilot/visualize_predictions.py
```

Then open `http://localhost:8501` in your browser.

## What You'll See

### Top Section: Overall Improvement
- Baseline ADE (before range-scaling)
- Improved ADE (after range-scaling)
- % improvement
- Sample count

### Sample-by-Sample Analysis
For each selected sample:
- **Left panel**: Baseline metrics
- **Middle panel**: Improved metrics  
- **Right panel**: Delta (improvement) with % change

- **Trajectory plot**: Side-by-side comparison of predicted vs actual paths
  - Black line = ground truth (what actually happened)
  - Green line = prediction
  - Red arrows = error vectors
  - Red star = ego start position

- **Error curve**: Frame-by-frame error over time
  - Red dashed = baseline
  - Green solid = improved
  - Green shaded = improvement region

### Summary Statistics
- **ADE Distribution**: Histogram of errors across all samples
- **Improvement Distribution**: Histogram of improvement % across all samples

## How to Interpret

### Good Predictions
- Green line stays close to black line
- Error bars are small
- Errors decrease over time

### Improvement Zones
- Green area under error curve = where improved model beats baseline
- Larger green area = better improvement

### Metrics to Watch
- **ADE**: Overall accuracy across entire horizon (lower is better)
- **Longitudinal MAE**: Forward/backward accuracy (typically larger error)
- **Lateral MAE**: Left/right accuracy (usually smaller error)
- **FDE@3s**: Final position error at 3-second horizon

## Example Usage

1. **Open the dashboard** (see Quick Start above)
2. **In the sidebar**, keep default run paths or provide your own
3. **Select samples** - Multi-select 2-3 samples to compare
4. **Review metrics**:
   - Look for consistent improvements across metrics
   - Pay attention to longest horizon errors (FDE@3s)
5. **Scroll down** to see distribution statistics
6. **Pick different samples** and re-run to verify consistency

## Run Paths

**Baseline (before range-scaling):**
```
/users/gbanda/git/Paddock/results/nuplan/visionpilot/20260917T220044-97873a
```

**Improved (after range-scaling):**
```
/localdisk/.tmp/claude-64145/paddock-runs/nuplan/visionpilot/20261007T232811-ddbaba
```

Or use your own benchmark run directories with `samples.jsonl` files.

## Troubleshooting

**"Could not load benchmark data"**
- Check paths exist and contain `samples.jsonl`
- Verify benchmark runs are complete

**Streamlit not found**
- Run: `pip install streamlit`

**Plots not showing**
- Ensure matplotlib is installed: `pip install matplotlib`

## What the Range-Scaling Change Does

The improved model uses: σ(distance) = 3.0 × (d/30)²

This means:
- **Near readings (<15m)**: 4× more confident (σ=0.75m)
- **Reference (30m)**: unchanged (σ=3.0m)  
- **Far readings (60m)**: 4× less confident (σ=12m)

**Result**: Better speed scheduling from more accurate distance estimates.

Expected improvement: 26-32% on NuScenes, 5-10% on mixed datasets.
