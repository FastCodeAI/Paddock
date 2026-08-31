#!/usr/bin/env python3
"""Run modeld over a public segment and print real predictions."""
import os, pickle, numpy as np
from openpilot.selfdrive.test.process_replay.process_replay import get_process_config, replay_process
from openpilot.selfdrive.test.process_replay.model_replay import TEST_ROUTE, SEGMENT, START_FRAME, END_FRAME, trim_logs
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.openpilotci import get_url

CACHE = f'/tmp/model_replay_cache/{TEST_ROUTE}_{SEGMENT}_{START_FRAME}_{END_FRAME}.pkl'
frs = pickle.load(open(CACHE, 'rb'))
print(f"loaded cached frames: {list(frs)}")

lr = list(LogReader(get_url(TEST_ROUTE, SEGMENT, "rlog.zst")))
cams = {"narrowRoadCameraState", "wideRoadCameraState"}
logs = trim_logs(lr, START_FRAME, END_FRAME, cams,
                 {"narrowRoadEncodeIdx", "wideRoadEncodeIdx", "carParams", "carState", "carControl", "can"})
for s in ('extrinsicsCalibration', 'deviceState'):
    m = next(x for x in lr if x.which() == s).as_builder()
    m.logMonoTime = lr[0].logMonoTime
    logs.insert(1, m.as_reader())

msgs = replay_process(get_process_config("modeld"), logs, frs)
mv2 = [m.modelV2 for m in msgs if m.which() == 'modelV2']
print(f"\n=== got {len(mv2)} modelV2 messages ===\n")

print(f"{'frame':>6} {'curv':>9} {'accel':>8} {'stop':>5} {'v@1s':>7} {'x@5s':>7} "
      f"{'lead_d':>7} {'lead_p':>6} {'llProb':>6} {'conf':>7}")
for md in mv2[::6]:
    lead = md.leadsV3[0] if len(md.leadsV3) else None
    print(f"{md.frameId:>6} {md.action.desiredCurvature:>9.5f} {md.action.desiredAcceleration:>8.3f} "
          f"{str(md.action.shouldStop):>5} {md.velocity.x[10]:>7.2f} {md.position.x[20]:>7.2f} "
          f"{(lead.x[0] if lead else float('nan')):>7.2f} {(lead.prob if lead else float('nan')):>6.2f} "
          f"{np.mean(md.laneLineProbs):>6.2f} {str(md.confidence):>7}")

md = mv2[len(mv2) // 2]
print(f"\n--- detail @ frame {md.frameId} ---")
print(f"position.x[:8]   {np.array(list(md.position.x))[:8].round(2)}")
print(f"position.y[:8]   {np.array(list(md.position.y))[:8].round(2)}")
print(f"velocity.x[:8]   {np.array(list(md.velocity.x))[:8].round(2)}")
print(f"orientation.z[:8]{np.array(list(md.orientation.z))[:8].round(4)}")
print(f"laneLineProbs    {np.array(list(md.laneLineProbs)).round(3)}")
print(f"leads: " + ", ".join(f"[d={l.x[0]:.1f}m v={l.v[0]:.1f}m/s p={l.prob:.2f}]" for l in md.leadsV3))
print(f"desireState      {np.array(list(md.meta.desireState)).round(3)}")
print(f"engagedProb      {md.meta.engagedProb:.3f}   hardBrakePredicted={md.meta.hardBrakePredicted}")
dp = md.meta.disengagePredictions
print(f"gasPressProbs    {np.array(list(dp.gasPressProbs)).round(3)}")
print(f"brakePressProbs  {np.array(list(dp.brakePressProbs)).round(3)}")
print(f"brake3ms2Probs   {np.array(list(dp.brake3MetersPerSecondSquaredProbs)).round(3)}")

co = [m.cameraOdometry for m in msgs if m.which() == 'cameraOdometry']
print(f"\n--- cameraOdometry (visual odometry), {len(co)} msgs ---")
for c in co[::12]:
    print(f"  trans(m/s)={np.array(list(c.trans)).round(2)}  rot(rad/s)={np.array(list(c.rot)).round(4)}")
