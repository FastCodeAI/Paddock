"""32 probe paths per tick: features of the road far ahead, for the scorer to read as context.

They are never driven. Each is a straight line along the ego heading with 8 waypoints at fixed
distances, so its tokens sample the image where an oncoming car is 2-5 s before contact (median
35-45 m on our frontal ticks), which no candidate plan reaches (released pick 8-13 m).
8 lateral offsets (+-1, +-3, +-5, +-7 m: own lane and up to two lanes either side) x 4 reaches.
"""

import numpy as np

LATERAL = (-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0)
SPACING = (4.0, 8.0, 12.0, 16.0)  # waypoint k at (k+1)*spacing m: reaches 32, 64, 96, 128 m


def vocabulary(natives, pick_index, speed, accel=0.0):
    paths, meta = [], []
    for spacing in SPACING:
        x = np.arange(1, 9) * spacing
        for lat in LATERAL:
            paths.append(np.column_stack([x, np.full(8, lat), np.zeros(8)]))
            meta.append({"probe_spacing_m": spacing, "lateral_m": lat})
    return np.stack(paths), meta
