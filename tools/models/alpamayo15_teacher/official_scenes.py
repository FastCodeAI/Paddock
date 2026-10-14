"""The one definition of which nuScenes scenes the official neuroNCAP sweep
drives. Split out of `select_scenes.py` so anything that just needs this set
(e.g. `expert/scene_manifest.py`) doesn't also pull in nuscenes-devkit, which
only the `ncap` conda env carries (see `select_scenes.py`'s own docstring on
that env split)."""

from __future__ import annotations

# Mirrors harness.py's OFFICIAL_SET in Paddock
# (adapters/benchmarks/neuroncap/harness.py) -- the 14 scenes the official
# neuroNCAP sweep drives, excluded from both the training and validation pools
# so nothing trained on ever overlaps the eval set.
NEUROCAP_SCENES = {
    "scene-0099",
    "scene-0101",
    "scene-0103",
    "scene-0106",
    "scene-0108",
    "scene-0110",
    "scene-0278",
    "scene-0331",
    "scene-0346",
    "scene-0783",
    "scene-0796",
    "scene-0921",
    "scene-0923",
    "scene-0966",
}
