# Constant velocity (baseline)

The floor. `constant_velocity` holds the ego's current speed and heading and
projects a straight line: waypoint *k* is at `(speed · k · dt, 0)` in the ego
frame. It reads only ego speed from the CAN bus, so it has no perception, no
weights and no environment of its own -- `runtime.kind: builtin` runs it in
paddock's own interpreter.

```bash
paddock run --bench nuscenes    --model constant_velocity
paddock run --bench bench2drive --model constant_velocity
```

Purpose: a reference every learned model must beat. On nuScenes its L2 grows
with the horizon exactly as the road curves away from a straight line; on
Bench2Drive it drives straight at its entry speed until the route turns or an
obstacle stops it, so it is a low floor there. NAVSIM uses its own native
constant-velocity agent (`paddock_constant_velocity`), not this adapter.

The same `ConstantVelocityPolicy` serves open- and closed-loop: in closed loop
the shared Bench2Drive tracker turns the straight plan into throttle and steer.
