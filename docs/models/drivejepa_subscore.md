# drivejepa_subscore — analysis only

Drive-JEPA with 32 extra swerves and a learned sub-score scorer that may override its pick. It
measures, closed loop on neuroNCAP, a scorer trained in gallops (`gallops.train.scorer_train`) on
cached features of five reconstructed nuScenes train scenes with an inserted car (0053, 0058, 0064,
0070, 0075). It is not a model to rank: it has seen neuroNCAP-style scenarios in training, and its
effect on normal driving is unmeasured.

The base model has the camera-frame fix (#62). An earlier, camera-blind version of this adapter and
its runs are kept on branch `record/drivejepa-camera-blind`
(`docs/models/drivejepa-camera-blind.md` there).

## What it does per tick

1. Pass 1 is the unchanged `drivejepa` forward; a hook keeps the refiner's last stage (queries,
   image features, the 32 native plans and their tokens).
2. A vocabulary of 32 drivable swerves plus 32 far-ahead probe paths is built from the released
   pick and the ego speed (`vocab.py`, `probe.py`, the files the training cache was built with).
3. Pass 2 reruns only the refiner's last stage, 32 paths at a time, each in the slot of the native
   plan nearest it, so the refiner reads the image along each swerve and probe.
4. Five scorer seeds (weights from `DRIVEJEPA_SUBSCORE_DIR`) give each of the 64 plans six answers:
   no collision, on road, no TTC violation, in lane, progress, final. The weights' layout is
   `gallops.train.subscorer.layout()`.
5. The released pick drives unless the rule overrides it, and never before the history spans 0.45 s:
   - `DRIVEJEPA_SUBSCORE_RULE=relative` (default): when the best plan by
     `nc^4 * on_road * (5 progress + 5 no_ttc + 2 in_lane) / 12` beats the released pick's
     no-collision answer by `DRIVEJEPA_SUBSCORE_MARGIN` (0.2).
   - `absolute`: when the released pick's no-collision answer is below 0.5.

`DRIVEJEPA_SUBSCORE_GATE=off` runs every pass but always drives the released pick;
`DRIVEJEPA_SUBSCORE_LOG` writes one JSON line per tick (released pick, pick, override, answers).

## What it was trained on, and what that showed open loop

Leave-one-scene-out on the camera-fixed cache (gallops `scorer_folds --variant probe-geo`, five seeds
averaged): on the held-out scene, frontal ticks more than 2 s before contact were rescued 36–84% of
the time in four of five scenes (8% on 0058), against 6–35% when the features were camera-blind;
collision AUC 0.74–0.92. Plans that were already safe stayed safe on 95–99.7% of ticks in the three
Boston scenes but only 73% and 89% in the two Singapore ones. Closed-loop results are in
`results/neuroncap/drivejepa_subscore/`.
