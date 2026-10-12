# drivejepa_ncapdump — analysis only, camera-blind era

`drivejepa`, driving exactly as before, with all 32 candidates logged per tick
(proposals, the scorer's logits and `pdm_score`, the refined trajectory) to
`DRIVEJEPA_DUMP_DIR`, one JSONL per episode. It answered whether any of Drive-JEPA's
candidates stops short of or steers clear of the neuroNCAP target actor; the
findings are in [drivejepa-neuroncap.md](drivejepa-neuroncap.md).

**Every run of it was camera-blind.** It ran before #62, when the `drivejepa`
adapter placed its cameras relative to nuScenes' LIDAR_TOP (yawed −90°) and no
image sample landed in the front image. The candidate sets it recorded are those
of a model planning from ego status alone; "no swerve among the 32" has to be
re-measured on the seeing model.

Switches (all default to `drivejepa`'s exact input): `DRIVEJEPA_ACCEL=zero`,
`DRIVEJEPA_COMMAND=left|right|straight`, `DRIVEJEPA_AUX_HEADS=1` (runs
`pred_col_agent` and `pred_area` beside the pick, never feeding it),
`DRIVEJEPA_DRIVE=gt` with `DRIVEJEPA_GT_DIR` (the car follows the logged human
drive while the model is recorded in shadow), and `DRIVEJEPA_FEATURES=1` (saves
the scorer heads' input features as float16 `.npy`). See the policy docstring.
