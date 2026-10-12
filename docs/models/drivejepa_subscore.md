# drivejepa_subscore — analysis only, camera-blind era

Drive-JEPA with 32 extra swerves and a learned sub-score scorer that may override
its pick. Built to test, closed loop on neuroNCAP, a scorer trained in gallops on
cached features (`experiments/2026-10-blind-scorer` there). **It ran before #62,
so the base model was camera-blind** — see
[drivejepa-camera-blind.md](drivejepa-camera-blind.md). It is kept as a record,
not as a model to rank.

## What it does per tick

1. Pass 1 is the unchanged `drivejepa` forward; a hook keeps the refiner's last
   stage (queries, image features, the 32 native plans and their tokens).
2. A vocabulary of 32 drivable swerves plus 32 far-ahead probe paths is built
   from the released pick and the ego speed (`vocab.py`, `probe.py`).
3. Pass 2 reruns only the refiner's last stage, 32 paths at a time, each in the
   slot of the native plan nearest it.
4. Five scorer seeds (weights from `DRIVEJEPA_SUBSCORE_DIR`, written by gallops)
   give each of the 64 plans six answers: no collision, on road, no TTC, in lane,
   progress, final.
5. The released pick drives unless the rule overrides it, and never before the
   history spans 0.45 s:
   - `DRIVEJEPA_SUBSCORE_RULE=absolute`: when the pick's no-collision answer is
     below 0.5.
   - `relative`: when the best plan by `nc^4 * on_road * (5 progress + 5 no_ttc +
     2 in_lane) / 12` beats the pick's no-collision answer by
     `DRIVEJEPA_SUBSCORE_MARGIN`.

`DRIVEJEPA_SUBSCORE_GATE=off` runs every pass but always drives the released
pick; `DRIVEJEPA_SUBSCORE_LOG` writes one JSON line per tick.

## What it measured (scene 0103, 3 runs each)

| rule | frontal | stationary | side |
|---|---|---|---|
| absolute | 2.20 / 2.26 / 2.21, all hit | 5.0 ×3 | 5.0 ×3 |
| relative, margin 0.2 | 2.13 hit, 5.0, 5.0 | 5.0 ×3 | 5.0 ×3 |

Neither reacted to the hazard. Every override happened in the first 3 s with the
same times, offsets and scores in frontal, stationary and side runs; the relative
rule's drift of about 2 m left is what let two frontal runs pass. The camera-blind
base model explains it: the scorer's inputs held no image.
