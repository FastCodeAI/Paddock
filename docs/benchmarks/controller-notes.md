# VisionPilot, OpenPilot and the Bench2Drive controller

Checked against the local adapter code on 9 September 2026. These descriptions
refer to our evaluated integrations, not every version of the upstream products.

## What VisionPilot produces and what we plot

One front camera feeds three perception networks: AutoSteer produces 64 image-space
path positions and masks; AutoDrive estimates road curvature and the closest
in-path object's distance/existence; AutoSpeed produces boxes with threat classes.
The adapter projects AutoSteer pixels onto a flat road using camera calibration,
fits a near-field quadratic path, and fuses object observations with a particle
filter to estimate gap and relative velocity. An IDM speed schedule supplies
forward distances; the adapter returns `(distance, fitted_lateral(distance))`.
It approximates forward x by integrated distance; it is not a full bicycle-model
rollout along arc length, particularly on curves.

In `tools/models/visionpilot/video.py`, yellow dots are raw AutoSteer image points,
orange dashes are the fitted ground path projected back into the image, and green
points are the adapter's timed trajectory. The Bench2Drive video builder plots
saved predicted waypoints, not future ground-truth ego positions. These are plans;
the actual driven path depends on tracking and subsequent replanning.

The timed plan is **not constant velocity**. IDM is integrated at 0.05 s (20 Hz;
some source comments incorrectly say 50 Hz), then sampled every 0.5 s. The current
adapter returns eight points through 4 s; Bench2Drive uses six through 3 s.
VisionPilot's original planning horizon is 1 s, so later points extrapolate the
adapter's fitted path and speed model. Its native lateral CTE/yaw fusion and
controller are not reproduced by this fitted-path/PID integration.

## OpenPilot in this adapter

The front image is warped into narrow and wide model views. These are two views
of the same supplied camera, not two independent recorded cameras. Supercombo
uses temporal image/features history and predicts a metric motion plan and other
outputs. `plan_to_grid` interpolates the native 33 timestamps onto 0.5 s intervals
and flips lateral right-positive to Paddock's left-positive convention.

Its model clock is 20 Hz; actual camera capture rate, repeated input frames,
model updates, and waypoint spacing are different quantities. nuScenes evaluation
uses approximately 12 Hz real front-camera history at 2 Hz scoring anchors.
Repeating frames does not create new visual observations.

For Bench2Drive, the adapter can send native longitudinal outputs or send only
waypoints to the shared tracker. Historical runs used different modes, including
resume assistance. The newly agreed OpenPilot experiment is waypoint PID with
0.05 m/s low-speed braking and the existing overspeed rule. It must be labelled
separately from those historical runs. VisionPilot keeps 0.4 m/s.

## What Bench2Drive controls

Bench2Drive evaluates an agent's steering, throttle and brake; it does not mandate
one PID for all models. Our `WaypointTracker` is TCP-derived, not an unchanged
upstream Bench2Drive controller. Steering uses a heading-error PID aimed about
4 m ahead. Speed PID targets mean consecutive waypoint spacing divided by 0.5 s
(the origin-to-first-point segment is excluded). Throttle is capped at 0.75.
It applies full brake and zero throttle if desired speed is below 0.4 m/s or
actual speed exceeds 1.1 times desired speed. Steering Kp is 1.25 here versus
0.75 in the referenced UniAD/VAD tracker; route-target steering blending is omitted.

For example, 0.15 m average spacing implies 0.3 m/s and triggers full brake.
That can prevent following a small but nonzero plan. Correct execution of this
rule is not proof of a good model/controller match, nor proof that a stall is
solely a model failure. Lowering the stop threshold does not fix wrong geometry,
unsafe selection, or an inadequate acceleration response.

## IDM and LQR

**IDM (Intelligent Driver Model)** decides longitudinal acceleration from ego
speed, desired cruise speed, lead gap and closing speed. It accelerates toward
cruise speed on a clear road and brakes when the gap is too small or closing too
fast. Our defaults include 1.5 m/s² acceleration, 3 m/s² comfortable deceleration,
1.5 s headway and 2 m standstill gap; curvature also caps cruise speed.
The comfortable deceleration parameter is not a hard acceleration clamp.
The current horizon loop holds the estimated relative velocity fixed while
updating gap and planned ego speed; it does not simulate a reacting lead vehicle.
See the [IDM authors' explanation](https://traffic-simulation.de/info/info_IDM.html).

**LQR (Linear Quadratic Regulator)** uses a linear vehicle model and a weighted
cost for tracking error and control effort. In lateral tracking, it can choose
steering to reduce lateral/heading errors while avoiding excessive steering.
Its feedback has the form `u = -K e`; weights decide the trade-off. This is a
controller, not an image detector or trajectory predictor. Our VisionPilot
Bench2Drive path uses the PID above, not LQR. See
[MIT's LQR notes](https://underactuated.mit.edu/lqr.html).

## What the VisionPilot nuScenes L2/ADE measures

It scores the **adapter-synthesized trajectory**, not a native learned metric plan.
The harness supplies recorded CAN-bus speed and camera calibration; no road speed
limit is supplied. `infer_sequence` resets state per sample and processes its
history. The cruise target is `max(first history observation's speed, 1.0)` m/s;
IDM integration at the scoring frame starts from that frame's recorded speed.
There is **no single fixed CV value** such as 3 or 6 m/s. For a history starting
at 8 m/s, cruise is 8 m/s, but predicted speed can change with curvature and lead
conditions. The standalone CV baseline instead predicts straight motion at the
current recorded speed.

The score depends on flat-road projection/calibration, fitted path shape, recorded
speed, cruise latching, filter history and IDM assumptions, plus extrapolation
beyond 1 s. It measures the complete adaptation. The reported point-averaged L2
and cumulative/per-step L2 are different summaries; do not call them interchangeable
ADE values or mix their numbers. Open-loop closeness to the logged driver does
not establish collision avoidance or closed-loop driving quality.
