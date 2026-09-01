#!/usr/bin/env python3
"""Run dmonitoringmodeld over the cabin camera and cache driverStateV2 outputs."""

import os
import pickle

import numpy as np
from openpilot.selfdrive.test.process_replay.model_replay import trim_logs
from openpilot.selfdrive.test.process_replay.process_replay import get_process_config, replay_process
from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.openpilotci import get_url

from tools import paths

ROUTE, SEG = "8494c69d3c710e81|000001d4--2648a9a404", 4
START, END = 0, int(os.getenv("NFRAMES", "600"))
OUT = str(paths.repo("openpilot") / "openpilot/model_output")
PKL = f"{OUT}/driverstate_{START}_{END}.pkl"

lr = list(LogReader(get_url(ROUTE, SEG, "rlog.zst")))
frs = {
    "cabinCameraState": FrameReader(
        get_url(ROUTE, SEG, "dcamera.hevc"), pix_fmt="nv12", cache_size=END - START
    )
}
logs = trim_logs(lr, START, END, {"cabinCameraState"}, {"cabinEncodeIdx", "carParams", "can"})
for s in ("extrinsicsCalibration", "deviceState"):
    m = next(x for x in lr if x.which() == s).as_builder()
    m.logMonoTime = lr[0].logMonoTime
    logs.insert(1, m.as_reader())

msgs = replay_process(get_process_config("dmonitoringmodeld"), logs, frs)

out = []
for m in msgs:
    if m.which() != "driverStateV2":
        continue
    ds = m.driverStateV2
    rhd = ds.wheelOnRightProb > 0.5
    dd = ds.rightDriverData if rhd else ds.leftDriverData
    out.append(
        {
            "frameId": ds.frameId,
            "rhd": bool(rhd),
            "wheelOnRightProb": ds.wheelOnRightProb,
            "facePosition": np.array(list(dd.facePosition)),
            "faceOrientation": np.array(list(dd.faceOrientation)),
            "faceOrientationStd": np.array(list(dd.faceOrientationStd)),
            "faceProb": dd.faceProb,
            "leftEyeProb": dd.leftEyeProb,
            "rightEyeProb": dd.rightEyeProb,
            "leftBlinkProb": dd.leftBlinkProb,
            "rightBlinkProb": dd.rightBlinkProb,
            "sunglassesProb": dd.sunglassesProb,
            "phoneProb": dd.phoneProb,
            "sleepProb": dd.sleepProb,
            "execTime": ds.modelExecutionTime,
        }
    )
pickle.dump(out, open(PKL, "wb"))
print(f"\n{len(out)} driverStateV2 messages -> {PKL}")

fp = np.array([x["facePosition"] for x in out])
print(
    f"facePosition  x: [{fp[:, 0].min():+.3f}, {fp[:, 0].max():+.3f}]   y: [{fp[:, 1].min():+.3f}, {fp[:, 1].max():+.3f}]"
)
for k in (
    "faceProb",
    "leftEyeProb",
    "rightEyeProb",
    "leftBlinkProb",
    "sunglassesProb",
    "phoneProb",
    "sleepProb",
    "wheelOnRightProb",
):
    a = np.array([x[k] for x in out])
    print(f"  {k:<18} min {a.min():.3f}  mean {a.mean():.3f}  max {a.max():.3f}")
print(f"  RHD frames: {sum(x['rhd'] for x in out)}/{len(out)}")
ex = np.array([x["execTime"] for x in out]) * 1e3
print(f"  inference: mean {ex.mean():.0f} ms  ({1000 / ex.mean():.1f} Hz)")
