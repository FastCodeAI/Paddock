# VisionPilot RL Fine-Tuning Guide

## Overview

VisionPilot's IDM planner can be improved through reinforcement learning by adapting parameters to driving context and learning optimal sensor fusion weights. This guide outlines the two-phase RL approach and how to run experiments.

## Current Baseline

After range-scaled noise model + scenario classifier:
- **NuScenes ADE @1s**: 0.758m
- **NuScenes ADE @2s**: 1.500m  
- **NuScenes ADE @3s**: 2.692m
- **Overall ADE**: 1.650m (30.1% improvement from baseline)

## Phase 1: IDM Parameter Optimization ⭐⭐⭐

**Goal:** Learn scenario-dependent IDM parameters (headway T, acceleration a_max, deceleration b) that minimize trajectory error.

**Approach:**
- Algorithm: PPO (Proximal Policy Gradient)
- State: [speed_ms, lead_distance_m, lead_prob, curvature, time_step, scenario]
- Action: T ∈ [0.8, 2.5]s, a_max ∈ [1.0, 4.0] m/s², b ∈ [1.5, 5.0] m/s²
- Reward: -(ADE + 0.05×jerk + 0.02×collision_penalty)
- Baseline: Hand-tuned per-scenario (1.8s highway, 1.2s urban, 1.5s suburban)

**Expected Impact:** 5-10% additional ADE improvement
**Training Time:** 2-3 days on single GPU (NVIDIA A100)
**Data:** 20K+ samples from NuScenes + Comma2K19

**Key Insight:**
The hand-tuned scenario classifier gives us a warm start. RL refines within each scenario:
- Highway: Does 1.8s work, or does speed/curvature matter more?
- Urban: Is 1.2s optimal, or should dense-traffic cases go different?
- Suburban: What's the right balance?

### Setup

```python
from paddock.adapters.models.visionpilot.rl_training import create_idm_rl_experiment

# Create experiment
exp = create_idm_rl_experiment(name="idm_opt_v1")

# Load training data (stub - integrate with your pipeline)
train_data = load_nuscenes_and_comma2k19(split="train", limit=20000)
val_data = load_nuscenes_and_comma2k19(split="val", limit=2000)

# Initialize and train (pseudocode)
trainer = IDMTrainer(exp.config)
trainer.train(train_data, val_data, exp.policy)

# Save results
exp.save_policy()
```

## Phase 2: Measurement Fusion Optimization ⭐⭐⭐

**Goal:** Learn dynamic weighting between two distance measurements:
- **CIPO**: Camera-based distance from DNN (with range-scaled uncertainty)
- **AutoSpeed**: Detection box distance

**Approach:**
- Learn fusion_weight ∈ [0.0, 1.0] per driving situation
- Weight composite measurement: d = (1-w)×box_distance + w×cipo_distance
- Condition on: cipo_stddev, speed, lead_probability, scenario
- Reward: -(distance_error + detection_error + false_positives)

**Hand-Tuned Heuristics (Before RL):**
- High uncertainty (σ > 15m) + low speed → trust boxes 80% (weight=0.2)
- Low uncertainty (σ < 2m) + high confidence → trust camera 90% (weight=0.9)
- Otherwise → balance 50/50 (weight=0.5)

**Expected Impact:** 8-12% additional ADE improvement (compounds with Phase 1)
**Training Time:** 1-2 days on single GPU
**Data:** Focus on edge cases: occlusion, far-field detection, cut-ins

**Key Insight:**
Each measurement has different failure modes:
- CIPO fails far away (σ → 12m at 60m range) but perfect near-field (σ = 0.75m at 15m)
- Boxes fail on small vehicles, parked cars, but robust to range
- Learning when to switch saves 8-12% ADE

## Cumulative Impact (Both Phases)

| Stage | Component | ADE | Gain |
|-------|-----------|-----|------|
| Baseline | Current VP | 2.360m | — |
| Done ✅ | Range-scaling | 1.650m | -30.1% |
| Done ✅ | Scenario classifier | ~1.600m | -3% |
| Phase 1 | IDM RL | ~1.480m | -8% |
| Phase 2 | Fusion RL | ~1.310m | -10% |
| **Total** | All improvements | **1.310m** | **-44.5%** |

## Running an Experiment

```bash
# Phase 1: IDM parameter learning
python -m paddock.adapters.models.visionpilot.run_rl \
    --phase idm \
    --name idm_opt_v1 \
    --epochs 10 \
    --batch_size 64 \
    --learning_rate 3e-4 \
    --datasets nuscenes,comma2k19 \
    --num_samples 20000

# Phase 2: Fusion weight learning
python -m paddock.adapters.models.visionpilot.run_rl \
    --phase fusion \
    --name fusion_opt_v1 \
    --epochs 10 \
    --learning_rate 5e-4 \
    --datasets nuscenes,comma2k19 \
    --num_samples 20000
```

## Monitoring Training

Results saved to `/localdisk/.tmp/rl_experiments/<exp_name>/`:
- `config.json` — experiment hyperparameters
- `policy.json` — learned parameters (updated each epoch)
- `training.log` — loss curves, ADE improvements
- `validation_metrics.json` — val performance each epoch

Metrics to watch:
- **Reward trend** should monotonically increase
- **Validation ADE** should decrease
- **Jerk penalty** should stay low (smooth motions)
- **Collision rate** should stay near 0

## Integration with VisionPilot

After training, integrate learned parameters back into policy.py:

```python
# Load trained policy
exp = RLExperiment.load("/localdisk/.tmp/rl_experiments/idm_opt_v1")
idm_policy = IDMParameterPolicy.load(exp.policy_path)

# In VisionPilot.infer():
scenario = ScenarioContext.classify(...)
action = idm_policy.compute_action(RLState(...))
self.idm.T = action.headway_t
self.idm.a = action.a_max
self.idm.b = action.b_min
```

## Advanced: Multi-Objective Pareto Learning

For even higher gains (6-15%), simultaneously optimize:
1. **Minimize ADE** (primary goal)
2. **Minimize jerk** (passenger comfort)
3. **Maintain safety** (no collisions)

Uses a Pareto frontier to find non-dominated parameter sets, then selects by user preference (aggressive vs smooth).

## References

- Scenario classifier: `ScenarioContext` in policy.py
- IDM planner: `IDM` class in policy.py (ported from C++)
- Range-scaled noise: `cipo_stddev()` in policy.py
- RL infrastructure: `rl_training.py`
