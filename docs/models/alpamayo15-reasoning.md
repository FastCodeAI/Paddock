# Chain-of-Causation records — Alpamayo-1.5 on KITScenes

All 223 scored samples. Each row is the reasoning trace emitted alongside the
trajectory, for each arm. `nav` received a ground-truth-derived route hint; `nonav`
did not; `letterbox` is the alternative camera warp. ADE is the medoid over K=6 draws.

## c365b427 k=40  ·  v0=0.0 m/s  ·  nav ADE 0.04 (Drive-JEPA 0.27)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.04): Stop to keep distance to the lead vehicle since it is stationary ahead
- **nav** (ADE 0.04): Stop to keep distance to the lead vehicle since it is stationary ahead
- **letterbox** (ADE 0.04): Stop to keep distance to the lead vehicle since it is stationary ahead

## 39ed6bdc k=310  ·  v0=0.0 m/s  ·  nav ADE 0.07 (Drive-JEPA 0.63)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.07): Nudge to the right to create clearance for the oncoming vehicle
- **nav** (ADE 0.07): Yield to the cut-in vehicle since it is entering our lane ahead
- **letterbox** (ADE 0.07): Nudge to the right to create clearance for the vehicle pulling out from the left

## b4d2fb1b k=40  ·  v0=8.9 m/s  ·  nav ADE 0.08 (Drive-JEPA 0.60)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.10): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.08): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.28): Keep distance to the lead vehicle since it is directly ahead in our lane

## f6bb38bc k=40  ·  v0=9.9 m/s  ·  nav ADE 0.09 (Drive-JEPA 0.83)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.31): Slow down to keep distance to the lead vehicle since it is stopped ahead
- **nav** (ADE 0.09): Slow down to maintain a safe distance to the lead vehicle since it is stopped ahead
- **letterbox** (ADE 1.99): Nudge left due to the stopped van blocking the right side of our lane

## 5aaa3555 k=40  ·  v0=0.6 m/s  ·  nav ADE 0.14 (Drive-JEPA 0.13)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.26): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.14): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.43): Keep distance to the lead vehicle since it is directly ahead in our lane

## 766aceef k=40  ·  v0=5.0 m/s  ·  nav ADE 0.14 (Drive-JEPA 2.83)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.14): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.14): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.25): Keep distance to the lead vehicle since it is directly ahead in our lane

## d427b10f k=310  ·  v0=6.2 m/s  ·  nav ADE 0.15 (Drive-JEPA 1.27)

- *route hint:* `Turn right in 36m`
- **nonav** (ADE 1.45): Resume speed since the lane is clear after passing the construction barricade
- **nav** (ADE 0.15): Turn right at the intersection since the right-turn traffic light is green
- **letterbox** (ADE 0.79): Yield to the cyclists since they are crossing our lane ahead

## 367b80cc k=400  ·  v0=9.4 m/s  ·  nav ADE 0.16 (Drive-JEPA 6.18)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.13): Keep lane since the lane is clear ahead
- **nav** (ADE 0.16): Nudge to the left to clear the obstacle blocking the right side of our lane
- **letterbox** (ADE 0.28): Keep distance to the lead vehicle since it is directly ahead in our lane

## 61a2563e k=40  ·  v0=3.7 m/s  ·  nav ADE 0.17 (Drive-JEPA 1.36)

- *route hint:* `Turn left in 26m`
- **nonav** (ADE 0.32): Turn right at the intersection since the right-turn traffic light is green
- **nav** (ADE 0.17): Turn left at the intersection since the left-turn traffic light is green
- **letterbox** (ADE 0.54): Change to the left lane after turning left due to the right curb lane being constrained/turn-only and to align with the through lane once the signal permits movement and cross traffic clears.

## d96782f8 k=310  ·  v0=7.0 m/s  ·  nav ADE 0.19 (Drive-JEPA 0.91)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.43): Keep lane since the lane is clear
- **nav** (ADE 0.19): Keep lane since the lane is clear ahead
- **letterbox** (ADE 0.36): Keep lane since the lane is clear and there is no lead vehicle ahead

## 1db9c8a0 k=40  ·  v0=4.3 m/s  ·  nav ADE 0.19 (Drive-JEPA 1.33)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.31): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.19): Keep distance to the lead vehicle since it is slowing ahead
- **letterbox** (ADE 0.42): Keep distance to the lead vehicle since it is directly ahead in our lane

## c1bf5a93 k=40  ·  v0=6.4 m/s  ·  nav ADE 0.19 (Drive-JEPA 1.67)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.27): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.19): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.39): Keep distance to the lead vehicle since it is directly ahead in our lane

## 4bcb0526 k=40  ·  v0=3.2 m/s  ·  nav ADE 0.19 (Drive-JEPA 3.21)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.28): Keep distance to the lead vehicle since it is stopped ahead
- **nav** (ADE 0.19): Keep distance to the lead vehicle since it is stopped ahead
- **letterbox** (ADE 0.30): Keep distance to the lead vehicle since it is stopped ahead

## b2f44b75 k=40  ·  v0=4.5 m/s  ·  nav ADE 0.20 (Drive-JEPA 0.53)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.57): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.20): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.34): Keep distance to the lead vehicle since it is moving slowly ahead in our lane

## 236144ec k=40  ·  v0=3.2 m/s  ·  nav ADE 0.21 (Drive-JEPA 0.19)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.22): Keep distance to the lead vehicle since it is slowing ahead
- **nav** (ADE 0.21): Keep distance to the lead vehicle since it is moving slowly ahead
- **letterbox** (ADE 0.28): Keep distance to the lead vehicle since it is slowing ahead

## 7080aa7f k=40  ·  v0=12.1 m/s  ·  nav ADE 0.21 (Drive-JEPA 0.67)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.17): Lane change to the left due to the right lane becoming a right-turn-only lane ahead and the presence of a safe gap behind/around adjacent vehicles, then match speed with traffic after merging.
- **nav** (ADE 0.21): Wait for a gap because a car is coming from behind in the left lane blocking an immediate merge to the left
- **letterbox** (ADE 0.20): Keep distance to the lead vehicle since it is directly ahead in our lane

## de354ef1 k=40  ·  v0=8.2 m/s  ·  nav ADE 0.22 (Drive-JEPA 0.63)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.45): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.22): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.35): Keep distance to the lead vehicle since it is directly ahead in our lane

## d398a7a5 k=40  ·  v0=4.9 m/s  ·  nav ADE 0.22 (Drive-JEPA 1.18)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.24): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.22): Accelerate to maintain following distance to the lead vehicle ahead
- **letterbox** (ADE 0.28): Keep distance to the lead vehicle since it is directly ahead in our lane

## af665a7a k=220  ·  v0=4.3 m/s  ·  nav ADE 0.23 (Drive-JEPA 2.44)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.40): Nudge to the right to increase clearance from the cyclist ahead
- **nav** (ADE 0.23): Nudge to the right to increase clearance from the cyclist ahead
- **letterbox** (ADE 0.22): Nudge to the right to clear the stopped vehicle blocking the left side of our lane

## 950dda14 k=40  ·  v0=0.4 m/s  ·  nav ADE 0.23 (Drive-JEPA 0.05)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.19): Stop to keep distance to the lead vehicle since it is stationary ahead
- **nav** (ADE 0.23): Stop to keep distance to the lead vehicle since it is stationary ahead
- **letterbox** (ADE 0.13): Keep distance to the lead vehicle since it is stopped ahead

## 546bc7e5 k=40  ·  v0=3.4 m/s  ·  nav ADE 0.24 (Drive-JEPA 0.27)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.17): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.24): Keep distance to the lead vehicle since it is slowing ahead
- **letterbox** (ADE 0.37): Keep distance to the lead vehicle since it is slowing ahead

## 38fd42fe k=40  ·  v0=1.1 m/s  ·  nav ADE 0.24 (Drive-JEPA 0.88)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.19): Accelerate to proceed through the intersection since the straight traffic light turns green
- **nav** (ADE 0.24): Accelerate to proceed through the intersection since the straight traffic light turns green
- **letterbox** (ADE 0.17): Accelerate to proceed through the intersection since the straight traffic light turns green

## c78f7cc6 k=130  ·  v0=14.6 m/s  ·  nav ADE 0.24 (Drive-JEPA 4.93)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.19): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.24): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.72): Keep distance to the lead vehicle since it is directly ahead in our lane

## b574fe64 k=40  ·  v0=2.3 m/s  ·  nav ADE 0.24 (Drive-JEPA 0.66)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.61): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.24): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.58): Keep distance to the lead vehicle since it is directly ahead in our lane

## da084a65 k=40  ·  v0=7.2 m/s  ·  nav ADE 0.25 (Drive-JEPA 0.44)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.22): Stop due to the red traffic light
- **nav** (ADE 0.25): Stop due to the red traffic light ahead
- **letterbox** (ADE 0.18): Stop due to the red traffic light ahead

## eca7bf3a k=220  ·  v0=5.4 m/s  ·  nav ADE 0.25 (Drive-JEPA 0.96)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.36): Stop for the red traffic light since the signal is red
- **nav** (ADE 0.25): Stop for the red traffic light ahead
- **letterbox** (ADE 0.37): Stop for the red traffic light since the signal is red

## 142f1419 k=40  ·  v0=7.5 m/s  ·  nav ADE 0.25 (Drive-JEPA 1.69)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.29): Keep distance to the lead vehicle since it is slowing ahead
- **nav** (ADE 0.25): Keep distance to the lead vehicle since it is slowing ahead
- **letterbox** (ADE 0.23): Keep distance to the lead vehicle since it is slowing ahead

## 00fc1b9b k=40  ·  v0=2.1 m/s  ·  nav ADE 0.25 (Drive-JEPA 0.57)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.27): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.25): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.90): Keep distance to the lead vehicle since it is directly ahead in our lane

## a724baf7 k=130  ·  v0=12.8 m/s  ·  nav ADE 0.26 (Drive-JEPA 11.41)

- *route hint:* `Turn right in 64m`
- **nonav** (ADE 0.45): Nudge to the right to clear the roadwork barricade blocking the left side of our lane
- **nav** (ADE 0.26): Lane change to the right due to slow traffic ahead in our lane near the signalized junction and a clear gap to merge into, then reduce speed approaching the junction/crosswalk for safety.
- **letterbox** (ADE 3.75): Keep distance to the cut-in vehicle since it merges into our lane ahead

## f10a3467 k=130  ·  v0=1.0 m/s  ·  nav ADE 0.27 (Drive-JEPA 1.72)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.33): Keep distance to the lead vehicle since it is moving slowly ahead
- **nav** (ADE 0.27): Keep distance to the lead vehicle since it is moving slowly ahead
- **letterbox** (ADE 0.21): Keep distance to the lead vehicle since it is moving slowly ahead

## 6848358c k=40  ·  v0=3.8 m/s  ·  nav ADE 0.27 (Drive-JEPA 0.84)

- *route hint:* `Turn left in 22m`
- **nonav** (ADE 0.39): Yield to the cyclist since it is directly ahead in our path during our right turn
- **nav** (ADE 0.27): Keep distance to the scooter ahead since it is directly in our path after our left turn
- **letterbox** (ADE 0.71): Keep distance to the cyclist since it is directly ahead in our lane

## 5c20d8ff k=130  ·  v0=9.7 m/s  ·  nav ADE 0.27 (Drive-JEPA 3.87)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.78): Maintain speed due to green traffic light ahead
- **nav** (ADE 0.27): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.81): Nudge right due to the vehicle encroaching from the left

## 3cd1a6eb k=40  ·  v0=9.0 m/s  ·  nav ADE 0.29 (Drive-JEPA 0.54)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.22): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.29): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.35): Keep distance to the lead vehicle since it is directly ahead in our lane

## d96782f8 k=130  ·  v0=8.7 m/s  ·  nav ADE 0.29 (Drive-JEPA 0.45)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.40): Keep lane since the lane is clear ahead
- **nav** (ADE 0.29): Keep lane since the lane is clear ahead
- **letterbox** (ADE 0.37): Keep lane since the lane is clear ahead

## 6487419d k=40  ·  v0=7.0 m/s  ·  nav ADE 0.30 (Drive-JEPA 3.42)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.40): Nudge left due to the roadwork barricades blocking the right side of our lane
- **nav** (ADE 0.30): Decelerate to maintain a safe distance to the lead vehicle since it is slowing ahead
- **letterbox** (ADE 0.95): Nudge left due to the stopped car blocking the right side of our lane

## 2673dca8 k=130  ·  v0=4.4 m/s  ·  nav ADE 0.30 (Drive-JEPA 0.30)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.20): Yield to the van merging from the left into our lane ahead
- **nav** (ADE 0.30): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.33): Nudge right due to the van encroaching from the left

## f88f4a8e k=40  ·  v0=8.9 m/s  ·  nav ADE 0.31 (Drive-JEPA 1.62)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.29): Nudge left due to the stopped vehicle blocking the right side of our lane
- **nav** (ADE 0.31): Nudge to the left to clear the stopped trailer blocking the right side of our lane
- **letterbox** (ADE 0.50): Nudge left due to the stopped van blocking the right side of our lane

## 35b08325 k=40  ·  v0=0.0 m/s  ·  nav ADE 0.31 (Drive-JEPA 0.70)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.63): Accelerate to proceed through the intersection since the traffic light turns green
- **nav** (ADE 0.31): Accelerate to proceed through the intersection since the traffic light turns green
- **letterbox** (ADE 0.53): Accelerate to proceed through the intersection since the traffic light is green

## 3919596d k=40  ·  v0=7.5 m/s  ·  nav ADE 0.31 (Drive-JEPA 0.62)

- *route hint:* `Turn left in 40m`
- **nonav** (ADE 0.37): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.31): Change to the right lane to prepare for the upcoming right turn at the signalized intersection, following the lead van and ensuring clearance with vehicles in adjacent lanes; then modulate speed for the signal ahead after merging into the right‑turn lane.
- **letterbox** (ADE 1.08): Keep distance to the lead vehicle since it is directly ahead in our lane

## 3a616875 k=130  ·  v0=1.5 m/s  ·  nav ADE 0.33 (Drive-JEPA 0.25)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.52): Stop to yield to the cross-traffic since vehicles are crossing ahead
- **nav** (ADE 0.33): Yield to the cross-traffic since a vehicle is crossing ahead
- **letterbox** (ADE 0.54): Stop to yield to cross traffic blocking our path

## fdcc7efa k=40  ·  v0=8.0 m/s  ·  nav ADE 0.33 (Drive-JEPA 11.68)

- *route hint:* `Turn left in 40m`
- **nonav** (ADE 0.87): Keep distance to the cut-in vehicle since it merges into our lane ahead
- **nav** (ADE 0.33): Split to the left to take the side road since the left split traffic light is green
- **letterbox** (ADE 1.71): Split to the right to take the off-ramp since the lane markings indicate a right exit

## 324fb34d k=40  ·  v0=7.4 m/s  ·  nav ADE 0.33 (Drive-JEPA 2.19)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.39): Maintain speed due to green traffic light ahead
- **nav** (ADE 0.33): Maintain speed due to green traffic light ahead
- **letterbox** (ADE 0.35): Maintain speed due to a green traffic light ahead

## c73fd01c k=40  ·  v0=6.9 m/s  ·  nav ADE 0.33 (Drive-JEPA 3.19)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.46): Nudge left due to the stopped vehicle blocking the right side of our lane
- **nav** (ADE 0.33): Nudge left due to the stopped vehicle blocking the right side of our lane
- **letterbox** (ADE 0.32): Nudge left due to the stopped car blocking the right side of our lane

## c73fd01c k=130  ·  v0=6.9 m/s  ·  nav ADE 0.33 (Drive-JEPA 1.79)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.35): Nudge left due to the stopped vehicle blocking the right side of our lane
- **nav** (ADE 0.33): Nudge left due to the stopped vehicle blocking the right side of our lane
- **letterbox** (ADE 0.25): Nudge left due to the stopped vehicle blocking the right side of our lane

## ced33ece k=220  ·  v0=10.4 m/s  ·  nav ADE 0.33 (Drive-JEPA 2.67)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.38): Keep lane since the lane is clear ahead
- **nav** (ADE 0.33): Keep lane since the lane is clear ahead
- **letterbox** (ADE 0.26): Keep lane since the lane is clear ahead

## 0cfdeb83 k=40  ·  v0=4.0 m/s  ·  nav ADE 0.33 (Drive-JEPA 0.97)

- *route hint:* `Turn right in 24m`
- **nonav** (ADE 0.48): Accelerate to proceed through the right turn since the right-turn traffic light is green
- **nav** (ADE 0.33): Turn right at the intersection since the right-turn traffic light is green
- **letterbox** (ADE 1.17): Turn right at the intersection since the right-turn traffic light is green

## 1e20c4a2 k=40  ·  v0=5.8 m/s  ·  nav ADE 0.34 (Drive-JEPA 0.22)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.33): Keep lane since the lane is clear ahead
- **nav** (ADE 0.34): Keep lane since the lane is clear ahead
- **letterbox** (ADE 0.37): Keep lane since the lane is clear ahead

## 17719e25 k=40  ·  v0=8.5 m/s  ·  nav ADE 0.34 (Drive-JEPA 2.31)

- *route hint:* `Turn left in 39m`
- **nonav** (ADE 0.51): Change to the left lane due to the right curb lane narrowing with construction cones, then match speed with traffic after merging because cyclists appear ahead after the merge and flow is slow approaching the signalized intersection.
- **nav** (ADE 0.34): Keep distance to the scooter ahead since it is in our lane
- **letterbox** (ADE 0.28): Keep distance to the scooter ahead since it is in our lane

## c78f7cc6 k=40  ·  v0=14.6 m/s  ·  nav ADE 0.34 (Drive-JEPA 5.63)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.34): Keep lane since the lane is clear ahead
- **nav** (ADE 0.34): Keep lane since the lane is clear ahead
- **letterbox** (ADE 0.38): Keep lane since the lane is clear ahead

## 85c30383 k=130  ·  v0=7.2 m/s  ·  nav ADE 0.35 (Drive-JEPA 2.53)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.40): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.35): Adapt speed to maintain a safe distance from the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.34): Keep distance to the lead vehicle since it is directly ahead in our lane

## acbbe9df k=40  ·  v0=0.7 m/s  ·  nav ADE 0.35 (Drive-JEPA 1.81)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.56): Keep distance to the lead vehicle since it is moving slowly ahead
- **nav** (ADE 0.35): Keep distance to the lead vehicle since it is moving slowly ahead
- **letterbox** (ADE 0.68): Maintain lane and search for a gap to merge left because a large truck is blocking our current right lane while cars are alongside in the left lane, preventing an immediate merge; wait for space as our lane is slowed by the truck ahead and right‑lane vehicles are merging left due to cones/keep‑left island.

## 367b80cc k=220  ·  v0=8.5 m/s  ·  nav ADE 0.35 (Drive-JEPA 0.69)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.50): Accelerate gently due to a green traffic light
- **nav** (ADE 0.35): Accelerate through the intersection because of green traffic light
- **letterbox** (ADE 0.41): Keep distance to the lead vehicle since it is directly ahead in our lane

## 6f47eb66 k=40  ·  v0=3.4 m/s  ·  nav ADE 0.36 (Drive-JEPA 0.42)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.53): Nudge left due to the stopped car blocking the right side of our lane
- **nav** (ADE 0.36): Nudge left due to the stopped vehicle blocking the right side of our lane
- **letterbox** (ADE 0.53): Nudge left due to the stopped vehicle blocking the right side of our lane

## cf8cfcba k=40  ·  v0=2.8 m/s  ·  nav ADE 0.36 (Drive-JEPA 6.12)

- *route hint:* `Turn left in 20m`
- **nonav** (ADE 0.25): Turn left at the intersection since the left-turn traffic light turns green
- **nav** (ADE 0.36): Turn right at the intersection since the right-turn traffic light is green
- **letterbox** (ADE 1.34): Turn left at the intersection since the left-turn traffic light is green

## 45ac20f8 k=40  ·  v0=1.6 m/s  ·  nav ADE 0.36 (Drive-JEPA 2.86)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.42): Yield to the cross-traffic since a vehicle is crossing ahead in our path
- **nav** (ADE 0.36): Nudge left due to construction barricades blocking the right side
- **letterbox** (ADE 0.33): Turn right since the lane ahead is blocked by construction barricades

## 02be7d74 k=40  ·  v0=5.5 m/s  ·  nav ADE 0.38 (Drive-JEPA 1.08)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.28): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.38): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 1.09): Keep distance to the lead vehicle since it is directly ahead in our lane

## 367b80cc k=130  ·  v0=8.4 m/s  ·  nav ADE 0.38 (Drive-JEPA 0.76)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.42): Slow down due to the lead vehicle ahead
- **nav** (ADE 0.38): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.48): Keep distance to the lead vehicle since it is directly ahead in our lane

## bc171ed9 k=40  ·  v0=5.9 m/s  ·  nav ADE 0.38 (Drive-JEPA 1.74)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.40): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.38): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.37): Keep distance to the lead vehicle since it is directly ahead in our lane

## f745d301 k=40  ·  v0=7.6 m/s  ·  nav ADE 0.38 (Drive-JEPA 0.71)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.46): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.38): Nudge to the left to clear the construction barricade blocking the right side of our lane
- **letterbox** (ADE 0.44): Nudge left due to the construction barricade blocking the right side of our lane

## 2381584f k=130  ·  v0=13.8 m/s  ·  nav ADE 0.38 (Drive-JEPA 4.87)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.49): Keep lane since the lane is clear ahead
- **nav** (ADE 0.38): Nudge to the right to avoid the road debris blocking the lane
- **letterbox** (ADE 0.48): Nudge to the right to avoid the road obstacle blocking the lane

## c542e25e k=40  ·  v0=0.8 m/s  ·  nav ADE 0.39 (Drive-JEPA 0.20)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.20): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.39): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.59): Keep distance to the lead vehicle since it is directly ahead in our lane

## 95bac4b1 k=40  ·  v0=11.1 m/s  ·  nav ADE 0.39 (Drive-JEPA 1.14)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.25): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.39): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.19): Keep distance to the lead vehicle since it is directly ahead in our lane

## fd274f08 k=40  ·  v0=12.6 m/s  ·  nav ADE 0.40 (Drive-JEPA 3.80)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.32): Keep distance to the cyclist since they are traveling ahead in our path
- **nav** (ADE 0.40): Keep distance to the cyclist since they are traveling ahead in our lane
- **letterbox** (ADE 0.40): Nudge right due to the vehicle encroaching from the left side of our lane

## eca7bf3a k=130  ·  v0=11.2 m/s  ·  nav ADE 0.40 (Drive-JEPA 7.41)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.58): Lane change to the left due to construction blocking the right lane ahead and the right lane transitioning to a turn/curbside zone; then continue with mild slowing approaching the signalized intersection and pedestrians nearby.
- **nav** (ADE 0.40): Change to the left lane due to the right lane becoming right-turn-only ahead and to maintain a straight-through path, then slow for the signalized intersection and crossing area with nearby pedestrians.
- **letterbox** (ADE 0.43): Keep distance to the lead vehicle since it is directly ahead in our lane

## a5ef9e38 k=40  ·  v0=10.7 m/s  ·  nav ADE 0.40 (Drive-JEPA 0.88)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.39): Slow down for the red traffic light since it is red
- **nav** (ADE 0.40): Slow down for the red traffic light since it is red
- **letterbox** (ADE 0.66): Keep distance to the lead vehicle since it is directly ahead in our lane

## ed483615 k=40  ·  v0=3.8 m/s  ·  nav ADE 0.41 (Drive-JEPA 1.32)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.45): Change to the left lane due to the right lane turning into a right‑turn slip road and a slow/turning lead car ahead, with a green signal allowing progression through the intersection after merging into a safe gap behind the box truck.
- **nav** (ADE 0.41): Change to the left lane due to the rightmost lane turning into a right‑turn‑only slip lane, then proceed through the green signal while following the lead vehicles and monitoring potential cyclists/pedestrians at the crossing, which also necessitates slowing as needed after merging.
- **letterbox** (ADE 0.74): Change to the left lane and proceed through the green signal because the right side narrows into a right‑turn/side‑street and the left lane is clear, following the lead vehicle into the through lanes with safe spacing.

## c71eb5a2 k=40  ·  v0=10.1 m/s  ·  nav ADE 0.41 (Drive-JEPA 0.47)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.62): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.41): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.26): Keep distance to the lead vehicle since it is directly ahead in our lane

## d45fc68a k=40  ·  v0=3.0 m/s  ·  nav ADE 0.41 (Drive-JEPA 0.36)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.47): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.41): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.35): Keep distance to the lead vehicle since it is directly ahead in our lane

## 85c30383 k=40  ·  v0=8.9 m/s  ·  nav ADE 0.41 (Drive-JEPA 0.96)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.33): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.41): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.24): Keep distance to the lead vehicle since it is directly ahead in our lane

## 3ba62d26 k=40  ·  v0=10.3 m/s  ·  nav ADE 0.41 (Drive-JEPA 1.29)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.43): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.41): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.41): Keep distance to the lead vehicle since it is directly ahead in our lane

## d96782f8 k=40  ·  v0=3.0 m/s  ·  nav ADE 0.43 (Drive-JEPA 1.42)

- *route hint:* `Turn right in 18m`
- **nonav** (ADE 0.51): Maintain lane and gently accelerate to proceed through the intersection
- **nav** (ADE 0.43): Accelerate to proceed through the right turn because of clear path ahead
- **letterbox** (ADE 1.16): Accelerate to proceed through the right turn since the right-turn traffic light is green

## 04a39028 k=220  ·  v0=11.0 m/s  ·  nav ADE 0.43 (Drive-JEPA 1.32)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.94): Create a usable gap for a right-lane merge because a car is alongside in the right lane blocking an immediate merge
- **nav** (ADE 0.43): Create a usable gap to merge right because a car is alongside in the right lane blocking an immediate merge
- **letterbox** (ADE 0.63): Keep distance to the lead vehicle since it is directly ahead in our lane

## 51d6f5e5 k=40  ·  v0=4.3 m/s  ·  nav ADE 0.44 (Drive-JEPA 0.30)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.54): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.44): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.20): Accelerate to proceed through the intersection since the straight traffic light turns green

## ba2f9a7d k=130  ·  v0=13.5 m/s  ·  nav ADE 0.44 (Drive-JEPA 2.03)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.27): Keep lane since the lane is clear ahead
- **nav** (ADE 0.44): Keep lane since the lane is clear and no lead vehicle is present
- **letterbox** (ADE 0.38): Keep lane since the lane is clear ahead

## cf8cfcba k=130  ·  v0=9.5 m/s  ·  nav ADE 0.44 (Drive-JEPA 1.14)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.44): Nudge to the left to clear the small obstacle blocking the right side of our lane
- **nav** (ADE 0.44): Keep lane since the lane is clear ahead
- **letterbox** (ADE 0.40): Keep lane since the lane is clear ahead

## f0e0c753 k=130  ·  v0=10.2 m/s  ·  nav ADE 0.45 (Drive-JEPA 0.71)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.68): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.45): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.68): Lane change to the right due to a slower lead vehicle in the current lane and a safe gap behind the right-lane car, then match its speed after merging.

## 97a94a07 k=130  ·  v0=11.3 m/s  ·  nav ADE 0.45 (Drive-JEPA 1.32)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.40): Keep distance to the lead vehicle since it is moving slowly ahead
- **nav** (ADE 0.45): Keep distance to the lead vehicle since it is slowing ahead
- **letterbox** (ADE 0.57): Keep distance to the lead vehicle since it is slowing ahead

## d7b1c9c6 k=40  ·  v0=10.9 m/s  ·  nav ADE 0.45 (Drive-JEPA 1.39)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.75): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.45): Nudge right due to the cones blocking the left side of our lane
- **letterbox** (ADE 0.83): Nudge left due to the stopped vehicle blocking the right side of our lane

## 0b795c4c k=40  ·  v0=6.3 m/s  ·  nav ADE 0.45 (Drive-JEPA 9.58)

- *route hint:* `Turn left in 33m`
- **nonav** (ADE 0.63): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.45): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 1.14): Keep distance to the lead vehicle since it is directly ahead in our lane

## e8dae25a k=40  ·  v0=7.2 m/s  ·  nav ADE 0.46 (Drive-JEPA 2.05)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.36): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.46): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.67): Keep distance to the lead vehicle since it is directly ahead in our lane

## 0f8ffa88 k=40  ·  v0=8.7 m/s  ·  nav ADE 0.46 (Drive-JEPA 2.48)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.33): Nudge left due to the stopped van blocking the right side of our lane
- **nav** (ADE 0.46): Keep lane since the lane is clear ahead
- **letterbox** (ADE 0.26): Nudge left due to the stopped van blocking the right side of our lane

## 56e5816f k=40  ·  v0=3.4 m/s  ·  nav ADE 0.46 (Drive-JEPA 0.38)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.25): Yield to the cyclists since they have priority on our right-turn slip road
- **nav** (ADE 0.46): Turn right at the intersection since the right-turn traffic light is green
- **letterbox** (ADE 0.35): Adjust speed and keep safe distance due to the lead vehicle ahead

## 1df9975b k=40  ·  v0=12.0 m/s  ·  nav ADE 0.47 (Drive-JEPA 0.99)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.57): Nudge right due to the vehicle encroaching from the left
- **nav** (ADE 0.47): Nudge right due to the stopped truck blocking the left side of our lane
- **letterbox** (ADE 0.61): Nudge left due to the stopped van blocking the right side of our lane

## 39ed6bdc k=400  ·  v0=3.6 m/s  ·  nav ADE 0.47 (Drive-JEPA 6.94)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.50): Nudge left due to the stopped vehicle blocking the right side of our lane
- **nav** (ADE 0.47): Nudge left due to the stopped vehicle blocking the right side of our lane
- **letterbox** (ADE 0.58): Nudge left due to the stopped vehicle blocking the right side of our lane

## d427b10f k=40  ·  v0=7.3 m/s  ·  nav ADE 0.48 (Drive-JEPA 2.01)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.64): Stop to yield to the pedestrians walking across the crosswalk ahead
- **nav** (ADE 0.48): Stop to yield to the pedestrians walking across the crosswalk ahead
- **letterbox** (ADE 0.56): Keep distance to the lead vehicle since it is slowing ahead

## 18f9f9c8 k=40  ·  v0=10.0 m/s  ·  nav ADE 0.48 (Drive-JEPA 1.16)

- *route hint:* `Turn left in 52m`
- **nonav** (ADE 0.59): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.48): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.74): Keep distance to the lead vehicle since it is directly ahead in our lane

## f0b1b61b k=40  ·  v0=3.8 m/s  ·  nav ADE 0.48 (Drive-JEPA 0.55)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.58): Yield to the traffic in the roundabout since vehicles are circulating ahead
- **nav** (ADE 0.48): Yield to the traffic in the roundabout since a vehicle is circulating ahead of our entry
- **letterbox** (ADE 0.12): Yield to the traffic in the roundabout since vehicles are circulating ahead

## ba826f49 k=130  ·  v0=11.6 m/s  ·  nav ADE 0.49 (Drive-JEPA 2.82)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.89): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.49): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.51): Keep distance to the lead vehicle since it is directly ahead in our lane

## af665a7a k=40  ·  v0=1.4 m/s  ·  nav ADE 0.49 (Drive-JEPA 2.45)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.56): Nudge to the right to create clearance for the oncoming vehicle
- **nav** (ADE 0.49): Nudge to the right to create clearance for the oncoming vehicle
- **letterbox** (ADE 1.01): Nudge left due to the stopped car blocking the right side of our lane

## 8fb5bd8e k=40  ·  v0=5.0 m/s  ·  nav ADE 0.50 (Drive-JEPA 1.16)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.30): Maintain a steady following speed due to the lead vehicle ahead
- **nav** (ADE 0.50): Accelerate to maintain following distance because of the lead vehicle ahead
- **letterbox** (ADE 0.44): Accelerate to maintain following distance behind the lead vehicle

## 04a39028 k=130  ·  v0=11.8 m/s  ·  nav ADE 0.51 (Drive-JEPA 2.88)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.27): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.51): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.94): Keep distance to the lead vehicle since it is directly ahead in our lane

## 5c20d8ff k=220  ·  v0=8.9 m/s  ·  nav ADE 0.51 (Drive-JEPA 0.45)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.53): Nudge to the left to clear the obstacle blocking the right side of our lane
- **nav** (ADE 0.51): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.66): Keep distance to the lead vehicle since it is directly ahead in our lane

## e6e18423 k=40  ·  v0=13.7 m/s  ·  nav ADE 0.51 (Drive-JEPA 3.85)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.55): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.51): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.93): Nudge left to increase clearance to the car encroaching from the right

## c16f7986 k=40  ·  v0=6.4 m/s  ·  nav ADE 0.52 (Drive-JEPA 1.76)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.66): Keep distance to the cyclists since they are riding alongside in our path
- **nav** (ADE 0.52): Yield due to cyclists ahead in our lane
- **letterbox** (ADE 0.37): Keep distance to the cyclists ahead since they are directly in our path

## 93b7a937 k=40  ·  v0=9.1 m/s  ·  nav ADE 0.52 (Drive-JEPA 1.16)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.56): Accelerate to maintain following distance to the lead vehicle ahead
- **nav** (ADE 0.52): Accelerate to maintain following distance to the lead vehicle ahead
- **letterbox** (ADE 0.65): Accelerate to maintain following distance to the lead vehicle ahead

## 39ed6bdc k=220  ·  v0=1.3 m/s  ·  nav ADE 0.52 (Drive-JEPA 1.14)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.47): Yield to the cut-in vehicle from the right since it is entering our lane ahead
- **nav** (ADE 0.52): Yield to the cut-in vehicle from the right since it is entering our lane ahead
- **letterbox** (ADE 0.93): Turn right at the intersection since the cross-traffic van has cleared

## c1d9c491 k=40  ·  v0=7.8 m/s  ·  nav ADE 0.52 (Drive-JEPA 3.86)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.34): Nudge to the left to clear the cones blocking the right side of our lane
- **nav** (ADE 0.52): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.67): Keep distance to the lead vehicle since it is directly ahead in our lane

## b3dd3c95 k=40  ·  v0=3.2 m/s  ·  nav ADE 0.53 (Drive-JEPA 0.17)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.58): Stop to yield to the cyclist crossing ahead
- **nav** (ADE 0.53): Slow down to yield to the cyclist crossing ahead
- **letterbox** (ADE 0.39): Yield to the cross-traffic since a car is crossing ahead at the yield-controlled junction

## 6f47eb66 k=220  ·  v0=7.3 m/s  ·  nav ADE 0.53 (Drive-JEPA 0.51)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.20): Keep distance to the lead vehicle since it is moving slowly ahead in our lane
- **nav** (ADE 0.53): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 1.21): Nudge right due to the stopped van blocking the lane ahead

## af665a7a k=310  ·  v0=0.7 m/s  ·  nav ADE 0.54 (Drive-JEPA 0.82)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.39): Yield to the pedestrian since they are walking across our lane ahead
- **nav** (ADE 0.54): Keep distance to the pedestrian since they are walking close to our path ahead
- **letterbox** (ADE 0.64): Nudge left due to the bollards blocking the lane ahead

## 18f9f9c8 k=130  ·  v0=7.7 m/s  ·  nav ADE 0.54 (Drive-JEPA 0.52)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.51): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.54): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.83): Nudge right due to the barrier blocking the left side of our lane

## bc3300c2 k=40  ·  v0=4.2 m/s  ·  nav ADE 0.54 (Drive-JEPA 0.22)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.39): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.54): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.27): Keep distance to the lead vehicle since it is directly ahead in our lane

## 9d111fa2 k=40  ·  v0=8.0 m/s  ·  nav ADE 0.56 (Drive-JEPA 0.78)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.46): Keep distance to the cut-in vehicle since it merges into our lane ahead
- **nav** (ADE 0.56): Decelerate to create a gap for merging left because our current right lane tapers into parked cars/bus stop ahead, with vehicles occupying the left through-lane blocking an immediate merge
- **letterbox** (ADE 0.62): Keep distance to the cut-in vehicle since it merges into our lane ahead

## ba826f49 k=40  ·  v0=14.1 m/s  ·  nav ADE 0.56 (Drive-JEPA 4.96)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.30): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.56): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.61): Keep distance to the lead vehicle since it is directly ahead in our lane

## 5c20d8ff k=40  ·  v0=11.0 m/s  ·  nav ADE 0.56 (Drive-JEPA 1.71)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.52): Yield due to slow lead vehicle ahead
- **nav** (ADE 0.56): Slow down due to the lead vehicle queue ahead
- **letterbox** (ADE 0.71): Yield to the slow lead vehicle ahead

## fbb55f13 k=40  ·  v0=9.4 m/s  ·  nav ADE 0.57 (Drive-JEPA 0.65)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.64): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.57): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.35): Keep distance to the lead vehicle since it is directly ahead in our lane

## f0e0c753 k=40  ·  v0=3.5 m/s  ·  nav ADE 0.58 (Drive-JEPA 0.50)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.75): Change lanes to the right to bypass slower lead vehicles and use a clear right through lane under a green signal, then match speed with right-lane traffic after merging.
- **nav** (ADE 0.58): Lane change to the right to prepare for the upcoming right turn
- **letterbox** (ADE 0.34): Keep distance to the lead vehicle since it is directly ahead in our lane

## 2673dca8 k=40  ·  v0=5.5 m/s  ·  nav ADE 0.58 (Drive-JEPA 0.30)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.85): Nudge left due to the parked van protruding into our lane
- **nav** (ADE 0.58): Nudge left due to the stopped van blocking the right side of our lane
- **letterbox** (ADE 1.69): Nudge left due to the stopped van blocking the right side of our lane

## 39ed6bdc k=130  ·  v0=5.1 m/s  ·  nav ADE 0.59 (Drive-JEPA 0.84)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.58): Nudge to the left to clear the stopped van blocking the right side of our lane
- **nav** (ADE 0.59): Nudge left due to the stopped van blocking the right side of our lane
- **letterbox** (ADE 1.59): Nudge left due to the stopped vehicle blocking the right side of our lane

## 97dabcaa k=40  ·  v0=3.6 m/s  ·  nav ADE 0.59 (Drive-JEPA 0.54)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.66): Keep distance to the cut-in vehicle since it merges into our lane ahead
- **nav** (ADE 0.59): Yield to the cut-in vehicle since it merges into our lane ahead
- **letterbox** (ADE 0.63): Keep distance to the cut-in vehicle since it merges into our lane ahead

## a724baf7 k=40  ·  v0=9.4 m/s  ·  nav ADE 0.59 (Drive-JEPA 1.15)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.76): Nudge left due to the stopped vehicle blocking the right side of our lane
- **nav** (ADE 0.59): Nudge left due to the car merging from the right
- **letterbox** (ADE 0.42): Nudge left due to the car encroaching from the right side of our lane

## 93bcdd59 k=130  ·  v0=10.8 m/s  ·  nav ADE 0.59 (Drive-JEPA 2.57)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.04): Slow down since the straight traffic light is red
- **nav** (ADE 0.59): Slow down since the traffic light is red
- **letterbox** (ADE 1.26): Decelerate to stop for the red traffic light ahead

## ced33ece k=40  ·  v0=4.0 m/s  ·  nav ADE 0.59 (Drive-JEPA 0.16)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.70): Accelerate to proceed through the intersection since the straight traffic light is green
- **nav** (ADE 0.59): Accelerate to proceed through the intersection since the straight traffic light is green
- **letterbox** (ADE 0.23): Accelerate to proceed through the intersection since the traffic light is green

## 54cd7acd k=40  ·  v0=15.9 m/s  ·  nav ADE 0.59 (Drive-JEPA 8.69)

- *route hint:* `Continue straight`
- **nonav** (ADE 2.28): Lane change to the right due to a slow bus ahead in our lane and a clear right lane, then match speed with right-lane traffic after merging.
- **nav** (ADE 0.59): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 1.01): Keep distance to the lead vehicle since it is directly ahead in our lane

## ba2f9a7d k=40  ·  v0=7.1 m/s  ·  nav ADE 0.62 (Drive-JEPA 0.91)

- *route hint:* `Turn right in 54m`
- **nonav** (ADE 0.45): Turn right at the intersection since the right-turn traffic light is green
- **nav** (ADE 0.62): Accelerate to proceed through the intersection since the right-turn traffic light is green
- **letterbox** (ADE 0.75): Accelerate to proceed through the intersection since the straight traffic light is green

## e983ab19 k=40  ·  v0=7.1 m/s  ·  nav ADE 0.62 (Drive-JEPA 3.39)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.62): Nudge left due to the stopped vehicle blocking the right side of our lane
- **nav** (ADE 0.62): Nudge left to avoid the obstruction in our lane ahead
- **letterbox** (ADE 0.62): Nudge left due to the stopped vehicle blocking the right side of our lane

## d0d0d336 k=40  ·  v0=2.4 m/s  ·  nav ADE 0.62 (Drive-JEPA 0.54)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.34): Stop to keep distance to the lead vehicle since it is stopped ahead
- **nav** (ADE 0.62): Keep distance to the lead vehicle since it is stopped ahead
- **letterbox** (ADE 0.46): Keep distance to the lead vehicle since it is slowing ahead

## 04a39028 k=40  ·  v0=13.1 m/s  ·  nav ADE 0.63 (Drive-JEPA 3.88)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.49): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.63): Change lanes to the right due to a clear gap in the right lane and slower traffic ahead, then match speed with right‑lane traffic after merging.
- **letterbox** (ADE 0.71): Keep distance to the lead vehicle since it is directly ahead in our lane

## dda6802f k=40  ·  v0=2.9 m/s  ·  nav ADE 0.64 (Drive-JEPA 4.13)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.73): Nudge to the left to pass the stopped van blocking the lane ahead
- **nav** (ADE 0.64): Nudge to the left to clear the stopped van blocking the right side of our lane
- **letterbox** (ADE 0.27): Nudge left due to the stopped van blocking the right side of our lane

## 3689ea5c k=40  ·  v0=3.0 m/s  ·  nav ADE 0.64 (Drive-JEPA 1.56)

- *route hint:* `Turn right in 21m`
- **nonav** (ADE 0.70): Lane change to the left due to slow traffic in the current right lane and a safe gap behind the box truck, then match speed with left-lane traffic after merging.
- **nav** (ADE 0.64): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 1.76): Lane change to the left due to slow traffic ahead in our lane, with a clear gap behind the lead truck and a green signal allowing continued movement; then match speed with following distance after merging.

## 85dbe6d6 k=40  ·  v0=1.6 m/s  ·  nav ADE 0.64 (Drive-JEPA 0.70)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.66): Turn left at the intersection since the left-turn traffic light turns green
- **nav** (ADE 0.64): Turn left through the intersection since the traffic light turns green
- **letterbox** (ADE 0.95): Accelerate to proceed through the intersection since the straight traffic light turns green

## 50df6381 k=40  ·  v0=10.0 m/s  ·  nav ADE 0.65 (Drive-JEPA 1.08)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.84): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.65): Nudge to the left to clear the cone blocking the right side of our lane
- **letterbox** (ADE 0.56): Nudge to the left to clear the stopped motorcycle blocking the right side of our lane

## 2381584f k=40  ·  v0=14.6 m/s  ·  nav ADE 0.66 (Drive-JEPA 5.19)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.47): Keep lane since the lane is clear ahead
- **nav** (ADE 0.66): Nudge to the left to clear the road debris blocking the right side of our lane
- **letterbox** (ADE 0.68): Keep distance to the lead motorcycle since it is directly ahead in our lane

## 2a28c841 k=40  ·  v0=7.4 m/s  ·  nav ADE 0.67 (Drive-JEPA 0.48)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.75): Accelerate to maintain following distance to the lead vehicle
- **nav** (ADE 0.67): Accelerate to match the lead vehicle ahead
- **letterbox** (ADE 0.78): Accelerate to clear the intersection because of the green traffic light

## a83fd3b2 k=40  ·  v0=10.6 m/s  ·  nav ADE 0.67 (Drive-JEPA 1.15)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.63): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.67): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.31): Keep distance to the lead vehicle since it is directly ahead in our lane

## e6826f0a k=130  ·  v0=4.4 m/s  ·  nav ADE 0.68 (Drive-JEPA 3.63)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.96): Create a usable gap because cars alongside in the right lane block an immediate merge to the right
- **nav** (ADE 0.68): Keep distance to the lead vehicle since it is slowing ahead
- **letterbox** (ADE 0.68): Keep distance to the lead vehicle since it is moving slowly ahead

## d427b10f k=130  ·  v0=5.3 m/s  ·  nav ADE 0.68 (Drive-JEPA 2.62)

- *route hint:* `Turn right in 28m`
- **nonav** (ADE 1.89): Stop for the red traffic light since it is red
- **nav** (ADE 0.68): Stop for the red traffic light ahead since it is red
- **letterbox** (ADE 0.49): Adapt speed for the right curve since the lane bends right ahead

## 03ed2585 k=40  ·  v0=4.9 m/s  ·  nav ADE 0.69 (Drive-JEPA 3.69)

- *route hint:* `Turn right in 28m`
- **nonav** (ADE 0.89): Maintain lane and gently accelerate to proceed through the intersection
- **nav** (ADE 0.69): Slow down due to the tram crossing our path
- **letterbox** (ADE 1.10): Adjust speed and keep safe distance due to the vehicle ahead

## 2673dca8 k=220  ·  v0=2.5 m/s  ·  nav ADE 0.69 (Drive-JEPA 1.47)

- *route hint:* `Turn right in 12m`
- **nonav** (ADE 0.60): Stop to yield to the pedestrians walking across the crosswalk ahead
- **nav** (ADE 0.69): Slow down to turn right because of pedestrians at the curb ahead
- **letterbox** (ADE 0.99): Stop to yield to the pedestrian in the crosswalk ahead

## 9e40e0fc k=40  ·  v0=10.2 m/s  ·  nav ADE 0.70 (Drive-JEPA 1.95)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.59): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.70): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.75): Keep distance to the lead vehicle since it is directly ahead in our lane

## 6f47eb66 k=310  ·  v0=6.7 m/s  ·  nav ADE 0.71 (Drive-JEPA 1.90)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.66): Nudge right due to the stopped van blocking the left side of our lane
- **nav** (ADE 0.71): Nudge right due to the stopped van blocking the left side of our lane
- **letterbox** (ADE 2.43): Nudge right due to the stopped van blocking the left side of our lane

## 3a616875 k=40  ·  v0=4.7 m/s  ·  nav ADE 0.72 (Drive-JEPA 6.93)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.55): Nudge left due to the stopped van blocking the right side of our lane
- **nav** (ADE 0.72): Nudge left due to the stopped van blocking the right side of our lane
- **letterbox** (ADE 0.74): Nudge left due to the stopped vehicle blocking the right side of our lane

## 35eb1fee k=40  ·  v0=14.0 m/s  ·  nav ADE 0.73 (Drive-JEPA 5.09)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.45): Keep lane since the lane is clear ahead
- **nav** (ADE 0.73): Keep lane since the lane is clear ahead
- **letterbox** (ADE 0.70): Keep distance to the cut-in vehicle since it merges into our lane ahead

## 367b80cc k=40  ·  v0=8.6 m/s  ·  nav ADE 0.74 (Drive-JEPA 1.87)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.55): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.74): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.80): Keep distance to the lead vehicle since it is directly ahead in our lane

## ba826f49 k=310  ·  v0=15.7 m/s  ·  nav ADE 0.76 (Drive-JEPA 6.83)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.68): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.76): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.69): Keep distance to the lead vehicle since it is directly ahead in our lane

## 367b80cc k=310  ·  v0=9.7 m/s  ·  nav ADE 0.77 (Drive-JEPA 2.49)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.75): Nudge right due to the van encroaching from the left
- **nav** (ADE 0.77): Nudge right to overtake the slow van ahead
- **letterbox** (ADE 0.52): Keep distance to the lead vehicle since it is directly ahead in our lane

## b0b4b2b5 k=40  ·  v0=11.4 m/s  ·  nav ADE 0.78 (Drive-JEPA 2.49)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.64): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.78): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 1.36): Keep distance to the lead vehicle since it is directly ahead in our lane

## 50df6381 k=130  ·  v0=7.0 m/s  ·  nav ADE 0.80 (Drive-JEPA 1.26)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.98): Keep distance to the lead vehicle since it is slowing ahead
- **nav** (ADE 0.80): Keep distance to the lead vehicle since it is slowing ahead
- **letterbox** (ADE 1.56): Decelerate to maintain a safe distance to the lead vehicle since it is slowing ahead

## fd274f08 k=130  ·  v0=11.7 m/s  ·  nav ADE 0.80 (Drive-JEPA 1.13)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.86): Keep distance to the scooter ahead since it is directly in our lane
- **nav** (ADE 0.80): Keep distance to the scooter ahead since it is moving slowly in our lane
- **letterbox** (ADE 0.36): Nudge to the left to increase clearance to the scooter ahead

## b9c27408 k=40  ·  v0=8.1 m/s  ·  nav ADE 0.81 (Drive-JEPA 10.44)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.35): Nudge to the right to clear the parked trailer blocking the left side of our lane
- **nav** (ADE 0.81): Nudge to the right to clear the parked vehicle blocking the left side of our lane
- **letterbox** (ADE 0.55): Nudge to the right to clear the stopped trailer blocking the left side of our lane

## 5d443f77 k=40  ·  v0=4.8 m/s  ·  nav ADE 0.82 (Drive-JEPA 0.68)

- *route hint:* `Turn right in 28m`
- **nonav** (ADE 0.38): Keep distance to the lead vehicle since it is directly ahead in our lane after the merge
- **nav** (ADE 0.82): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 1.16): Keep distance to the lead vehicle since it is directly ahead in our lane

## 9ffae20c k=130  ·  v0=8.1 m/s  ·  nav ADE 0.82 (Drive-JEPA 1.19)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.80): Nudge right to increase clearance to the truck encroaching from the left
- **nav** (ADE 0.82): Nudge left due to cones blocking the right side of our lane
- **letterbox** (ADE 0.30): Nudge left to avoid the construction cones blocking the right side of our lane

## 138e9096 k=40  ·  v0=8.9 m/s  ·  nav ADE 0.82 (Drive-JEPA 5.81)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.33): Lane change to the right due to a slower lead vehicle in our lane and a clear right-lane gap, then match speed with right-lane traffic after merging.
- **nav** (ADE 0.82): Lane change to the right due to a slower lead car in our lane and a clear right lane, then match speed with right-lane traffic after merging.
- **letterbox** (ADE 0.79): Lane change to the right to stay in the through lane as the left lane bends away around a median ahead, merging smoothly behind traffic already in that right lane; slow to create a safe gap, then accelerate after completing the merge since no lead vehicle blocks our path but lane boundary and curbside obstacles constrain our movement.

## 840e435a k=40  ·  v0=10.6 m/s  ·  nav ADE 0.83 (Drive-JEPA 0.81)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.57): Nudge to the left to clear the construction barricade blocking the right side of our lane
- **nav** (ADE 0.83): Nudge to the left to clear the roadwork barricade blocking the right side of our lane
- **letterbox** (ADE 0.51): Nudge left due to the construction barricade blocking the right side of our lane

## 77c7c33c k=130  ·  v0=6.6 m/s  ·  nav ADE 0.83 (Drive-JEPA 1.77)

- *route hint:* `Turn right in 37m`
- **nonav** (ADE 0.35): Change to the right lane due to entering a two‑lane street after the right turn, aligning with the right travel lane and then matching speed with lead vehicles ahead, as indicated by the trajectory crossing the right lane boundary and subsequent acceleration once the merge is completed.
- **nav** (ADE 0.83): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 1.62): Resume speed since the lane ahead is clear after turning right

## f10a3467 k=40  ·  v0=3.7 m/s  ·  nav ADE 0.83 (Drive-JEPA 1.90)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.51): Keep distance to the lead vehicle since it is slowing ahead
- **nav** (ADE 0.83): Keep distance to the lead vehicle since it is stopped ahead
- **letterbox** (ADE 0.65): Keep distance to the lead vehicle since it is slowing ahead

## c34c778f k=130  ·  v0=7.4 m/s  ·  nav ADE 0.84 (Drive-JEPA 0.65)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.48): Adapt speed to maintain a safe distance from the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.84): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.59): Keep distance to the lead vehicle since it is directly ahead in our lane

## 03ed2585 k=130  ·  v0=8.1 m/s  ·  nav ADE 0.84 (Drive-JEPA 1.78)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.96): Stop for the red traffic light since our lane is stopped ahead
- **nav** (ADE 0.84): Stop for the red traffic light ahead
- **letterbox** (ADE 0.63): Stop for the red traffic light since it is red ahead

## ab1cafcc k=40  ·  v0=7.8 m/s  ·  nav ADE 0.85 (Drive-JEPA 0.44)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.64): Adapt speed for the right curve since the lane bends right ahead
- **nav** (ADE 0.85): Keep distance to the lead truck since it is stopped ahead in our lane
- **letterbox** (ADE 0.43): Keep distance to the lead vehicle since it is directly ahead in our lane

## ced33ece k=130  ·  v0=10.9 m/s  ·  nav ADE 0.85 (Drive-JEPA 1.50)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.65): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.85): Keep lane since the lane is clear ahead
- **letterbox** (ADE 0.85): Keep lane since the lane is clear ahead

## 873eebf4 k=220  ·  v0=9.8 m/s  ·  nav ADE 0.85 (Drive-JEPA 1.22)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.36): Nudge left due to the cones blocking the right side of our lane
- **nav** (ADE 0.85): Nudge left due to the roadwork barrier blocking the right side of our lane
- **letterbox** (ADE 1.06): Nudge left due to construction cones blocking the right side of our lane

## 0f8ffa88 k=130  ·  v0=7.9 m/s  ·  nav ADE 0.87 (Drive-JEPA 0.51)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.30): Nudge left due to the stopped van blocking the right side of our lane
- **nav** (ADE 0.87): Nudge left due to the vehicle encroaching from the right side of the lane
- **letterbox** (ADE 0.82): Nudge left due to the stopped van blocking the right side of our lane

## 6370e4b6 k=40  ·  v0=3.4 m/s  ·  nav ADE 0.87 (Drive-JEPA 0.66)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.43): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.87): Keep distance to the lead vehicle since it is turning right ahead
- **letterbox** (ADE 1.28): Keep distance to the lead vehicle since it is directly ahead in our lane after the right turn

## d96782f8 k=220  ·  v0=4.4 m/s  ·  nav ADE 0.89 (Drive-JEPA 0.54)

- *route hint:* `Turn left in 15m`
- **nonav** (ADE 0.44): Turn left at the intersection since cross traffic has cleared
- **nav** (ADE 0.89): Yield to the cross-traffic since vehicles are crossing ahead on our path
- **letterbox** (ADE 1.43): Yield to the cross-traffic since vehicles are crossing ahead on our merge path

## 85dbe6d6 k=130  ·  v0=5.4 m/s  ·  nav ADE 0.90 (Drive-JEPA 10.39)

- *route hint:* `Turn left in 39m`
- **nonav** (ADE 0.48): Yield to the cyclist merging into our lane ahead
- **nav** (ADE 0.90): Keep distance to the cyclists since they merge into our lane ahead
- **letterbox** (ADE 0.75): Keep distance to the cyclist since it is directly ahead in our lane

## d7b1c9c6 k=130  ·  v0=12.6 m/s  ·  nav ADE 0.91 (Drive-JEPA 2.34)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.64): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.91): Keep lane since the lane is clear ahead
- **letterbox** (ADE 0.90): Keep lane since the lane is clear ahead

## dbce1c0b k=40  ·  v0=7.4 m/s  ·  nav ADE 0.92 (Drive-JEPA 1.33)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.42): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.92): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 1.40): Nudge left due to pedestrians standing at the right edge of our lane

## 97a94a07 k=40  ·  v0=13.6 m/s  ·  nav ADE 0.92 (Drive-JEPA 4.71)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.43): Slow down due to the lead vehicle ahead
- **nav** (ADE 0.92): Yield to the lead vehicle ahead
- **letterbox** (ADE 0.68): Maintain speed due to clear lane ahead

## be0dbfb9 k=40  ·  v0=4.5 m/s  ·  nav ADE 0.92 (Drive-JEPA 0.42)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.75): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 0.92): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.92): Keep distance to the lead vehicle since it is directly ahead in our lane

## 9ffae20c k=40  ·  v0=5.7 m/s  ·  nav ADE 0.93 (Drive-JEPA 0.88)

- *route hint:* `Turn left in 24m`
- **nonav** (ADE 0.71): Accelerate to proceed through the intersection since the straight traffic light is green
- **nav** (ADE 0.93): Turn left at the intersection since the left-turn traffic light is green
- **letterbox** (ADE 1.14): Keep distance to the lead vehicle since it is directly ahead in our lane

## 4c944492 k=40  ·  v0=9.7 m/s  ·  nav ADE 0.93 (Drive-JEPA 6.52)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.91): Nudge to the right to clear the road debris blocking the lane
- **nav** (ADE 0.93): Keep lane since the lane is clear ahead
- **letterbox** (ADE 0.86): Keep lane since the lane is clear ahead

## 873eebf4 k=40  ·  v0=9.9 m/s  ·  nav ADE 0.99 (Drive-JEPA 0.94)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.08): Nudge to the left to clear the stopped car with an open door blocking the right side of our lane
- **nav** (ADE 0.99): Nudge left due to the stopped vehicle with open door blocking the right side of our lane
- **letterbox** (ADE 1.42): Keep distance to the scooter ahead since it is in our lane

## 39ed6bdc k=40  ·  v0=5.5 m/s  ·  nav ADE 0.99 (Drive-JEPA 1.43)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.69): Nudge left due to the stopped van blocking the right side of our lane
- **nav** (ADE 0.99): Nudge right due to the vehicle protruding from the left side of the lane
- **letterbox** (ADE 0.35): Nudge to the left to clear the stopped van blocking the right side of our lane

## 8c501afb k=40  ·  v0=4.2 m/s  ·  nav ADE 0.99 (Drive-JEPA 2.84)

- *route hint:* `Turn left in 17m`
- **nonav** (ADE 0.96): Turn left at the intersection since the left-turn traffic light is green
- **nav** (ADE 0.99): Keep distance to the lead vehicle since it is turning left ahead
- **letterbox** (ADE 1.81): Keep distance to the lead vehicle since it is directly ahead in our lane

## 5bf1c78c k=40  ·  v0=4.2 m/s  ·  nav ADE 1.00 (Drive-JEPA 1.69)

- *route hint:* `Turn left in 18m`
- **nonav** (ADE 0.96): Nudge to the left to clear the stopped vehicle blocking the lane ahead
- **nav** (ADE 1.00): Nudge left due to the stopped vehicle blocking the right side of our lane
- **letterbox** (ADE 1.09): Nudge left due to stopped vehicles blocking the right side of our lane

## 33a55e0a k=40  ·  v0=7.1 m/s  ·  nav ADE 1.00 (Drive-JEPA 0.79)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.02): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 1.00): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.95): Keep distance to the lead vehicle since it is directly ahead in our lane

## bb2f22e1 k=130  ·  v0=5.3 m/s  ·  nav ADE 1.00 (Drive-JEPA 9.14)

- *route hint:* `Turn left in 31m`
- **nonav** (ADE 0.97): Change to the left lane due to the upcoming lane split where the right lane turns right; moving left keeps the intended through route while monitoring nearby cross-traffic, cyclists, and pedestrian crossings; then accelerate after clearing them with no lead vehicle blocking.
- **nav** (ADE 1.00): Change to the left lane due to needing to leave the right-turn/edge lane after the left turn and align with the main through lane, then accelerate once the merge is completed since there is no blocking traffic ahead and cross traffic has cleared at the yield-controlled junction.
- **letterbox** (ADE 1.84): Turn right at the intersection since the junction is clear and controlled by a yield sign

## 67abb4e2 k=40  ·  v0=2.5 m/s  ·  nav ADE 1.01 (Drive-JEPA 0.74)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.85): Adapt speed to maintain a safe distance from the lead vehicle since it is slowing ahead
- **nav** (ADE 1.01): Adapt speed to maintain a safe distance from the lead vehicle since it is slowing ahead
- **letterbox** (ADE 0.94): Adapt speed to maintain a safe distance from the lead vehicle since it is slowing ahead

## 59dc3fbb k=40  ·  v0=2.2 m/s  ·  nav ADE 1.01 (Drive-JEPA 1.14)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.63): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 1.01): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.75): Keep distance to the lead vehicle since it is directly ahead in our lane

## 3cd1a6eb k=130  ·  v0=11.1 m/s  ·  nav ADE 1.05 (Drive-JEPA 1.59)

- *route hint:* `Turn right in 56m`
- **nonav** (ADE 1.45): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 1.05): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 1.11): Keep distance to the lead vehicle since it is directly ahead in our lane

## d427b10f k=220  ·  v0=6.6 m/s  ·  nav ADE 1.07 (Drive-JEPA 0.65)

- *route hint:* `Turn left in 32m`
- **nonav** (ADE 1.05): Change to the left lane and proceed through the intersection because the signal is green, lane markings indicate a clear left through lane, and there is no blocking traffic ahead, so positioning left is appropriate for continuing straight after crossing the tracks.
- **nav** (ADE 1.07): Change to the left through lane and proceed because the right side becomes a right‑turn/slip lane controlled by signals, and the left lane is open with a green light, with no immediate blocking vehicles but with attention to cyclists/pedestrians near the crossing island after merging left.
- **letterbox** (ADE 1.08): Adapt speed for the left curve since the lane bends left ahead

## 8d3308e6 k=130  ·  v0=4.0 m/s  ·  nav ADE 1.07 (Drive-JEPA 0.11)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.21): Nudge left due to the stopped van blocking the right side of our lane
- **nav** (ADE 1.07): Nudge left due to the stopped van blocking the right side of our lane
- **letterbox** (ADE 0.85): Nudge left due to the stopped van blocking the right side of our lane

## 3acd59be k=40  ·  v0=8.7 m/s  ·  nav ADE 1.09 (Drive-JEPA 2.93)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.78): Nudge to the left in the same lane to clear the construction barricades blocking the right side of our lane
- **nav** (ADE 1.09): Nudge to the left to clear the construction barricades blocking the right side of our lane
- **letterbox** (ADE 0.78): Nudge to the left to clear the construction barricades blocking the right side of our lane

## d5388adc k=40  ·  v0=8.7 m/s  ·  nav ADE 1.13 (Drive-JEPA 0.60)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.09): Yield to the cross-traffic since a yield sign controls the merge ahead
- **nav** (ADE 1.13): Stop for the red right-turn traffic light ahead
- **letterbox** (ADE 0.62): Adapt speed for the right curve since the lane bends right ahead

## e983ab19 k=130  ·  v0=6.0 m/s  ·  nav ADE 1.14 (Drive-JEPA 0.32)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.98): Nudge left due to the parked vehicles blocking the right side of our lane
- **nav** (ADE 1.14): Nudge left due to the stopped vehicle blocking the right side of our lane
- **letterbox** (ADE 1.20): Nudge left due to the stopped vehicle blocking the right side of our lane

## 3a616875 k=400  ·  v0=6.2 m/s  ·  nav ADE 1.14 (Drive-JEPA 0.64)

- *route hint:* `Continue straight`
- **nonav** (ADE 2.93): Nudge to the right to clear the stopped van blocking the lane ahead
- **nav** (ADE 1.14): Nudge to the right to clear the parked van blocking the left side of our lane
- **letterbox** (ADE 0.86): Nudge right due to the stopped van blocking the left side of our lane

## 0dd003b2 k=40  ·  v0=12.8 m/s  ·  nav ADE 1.14 (Drive-JEPA 3.65)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.18): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 1.14): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.97): Keep distance to the lead vehicle since it is directly ahead in our lane

## ba826f49 k=220  ·  v0=12.7 m/s  ·  nav ADE 1.15 (Drive-JEPA 2.78)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.96): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 1.15): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 1.28): Keep distance to the lead vehicle since it is directly ahead in our lane

## 6f47eb66 k=130  ·  v0=2.2 m/s  ·  nav ADE 1.18 (Drive-JEPA 4.01)

- *route hint:* `Turn left in 16m`
- **nonav** (ADE 1.37): Turn right at the intersection since the crossing vehicle has cleared
- **nav** (ADE 1.18): Turn left at the intersection since the crossing vehicle has cleared
- **letterbox** (ADE 1.41): Turn left at the intersection since cross-traffic has cleared

## c34c778f k=40  ·  v0=8.1 m/s  ·  nav ADE 1.21 (Drive-JEPA 1.26)

- *route hint:* `Turn right in 41m`
- **nonav** (ADE 1.62): Change to the right lane and merge behind the lead vehicle due to the lane split and keep-right signage guiding traffic to the right, then match speed with the lead car after merging.
- **nav** (ADE 1.21): Wait for a gap because a car is coming from behind in the right lane blocking an immediate merge to the right
- **letterbox** (ADE 1.84): Keep distance to the lead vehicle since it is directly ahead in our lane

## be5138b4 k=40  ·  v0=4.0 m/s  ·  nav ADE 1.22 (Drive-JEPA 2.33)

- *route hint:* `Turn right in 24m`
- **nonav** (ADE 1.21): Slow down due to a cyclist crossing our right-turn path
- **nav** (ADE 1.22): Slow down to turn right because of cyclists crossing ahead
- **letterbox** (ADE 2.35): Slow down due to a cyclist crossing our right turn path

## c365b427 k=130  ·  v0=6.1 m/s  ·  nav ADE 1.23 (Drive-JEPA 0.42)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.01): Accelerate to maintain following distance behind the lead vehicle
- **nav** (ADE 1.23): Accelerate to maintain following distance behind the lead bus
- **letterbox** (ADE 1.04): Accelerate to maintain following distance behind the lead vehicle

## f7a2812c k=40  ·  v0=2.4 m/s  ·  nav ADE 1.24 (Drive-JEPA 0.18)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.08): Keep distance to the lead vehicle since it is moving slowly ahead
- **nav** (ADE 1.24): Keep distance to the lead vehicle since it is moving slowly ahead
- **letterbox** (ADE 0.62): Keep distance to the lead vehicle since it is moving slowly ahead

## 3a616875 k=220  ·  v0=2.8 m/s  ·  nav ADE 1.26 (Drive-JEPA 6.22)

- *route hint:* `Turn left in 21m`
- **nonav** (ADE 0.81): Lane change to the left due to stopped vehicles blocking the right lane ahead, then match speed with left-lane traffic after merging.
- **nav** (ADE 1.26): Keep lane since the lane is clear ahead
- **letterbox** (ADE 1.51): Turn left to enter the main road since cross traffic has cleared

## e6826f0a k=40  ·  v0=2.1 m/s  ·  nav ADE 1.34 (Drive-JEPA 2.97)

- *route hint:* `Turn left in 17m`
- **nonav** (ADE 1.18): Turn left at the intersection since the left turn traffic light turns green
- **nav** (ADE 1.34): Turn left at the intersection since the left-turn traffic light turns green
- **letterbox** (ADE 1.37): Turn left at the intersection since cross traffic is clear after stopping at the stop sign

## 45ac20f8 k=130  ·  v0=7.1 m/s  ·  nav ADE 1.34 (Drive-JEPA 10.34)

- *route hint:* `Continue straight`
- **nonav** (ADE 0.38): Keep lane since the lane is clear ahead
- **nav** (ADE 1.34): Keep lane since the lane is clear
- **letterbox** (ADE 3.60): Nudge to the left to clear the vehicle encroaching from the right

## d7b1c9c6 k=220  ·  v0=2.7 m/s  ·  nav ADE 1.37 (Drive-JEPA 2.80)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.47): Keep distance to the lead vehicle since it is slowing ahead
- **nav** (ADE 1.37): Slow down for the leading vehicle ahead to keep a safe distance
- **letterbox** (ADE 1.16): Keep distance to the lead vehicle since it is directly ahead in our lane

## 93bcdd59 k=40  ·  v0=13.7 m/s  ·  nav ADE 1.38 (Drive-JEPA 7.86)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.69): Keep lane since the lane is clear ahead
- **nav** (ADE 1.38): Nudge to the left to clear the road debris blocking the right side of our lane
- **letterbox** (ADE 0.86): Nudge left due to the parked cars blocking the right side of our lane

## 6f47eb66 k=490  ·  v0=3.3 m/s  ·  nav ADE 1.41 (Drive-JEPA 5.70)

- *route hint:* `Turn right in 18m`
- **nonav** (ADE 1.14): Nudge left due to the stopped car blocking the right side of our lane
- **nav** (ADE 1.41): Nudge left due to a stopped vehicle blocking the right side of our lane
- **letterbox** (ADE 2.21): Nudge left due to the stopped car blocking the right side of our lane

## bb2f22e1 k=40  ·  v0=5.9 m/s  ·  nav ADE 1.45 (Drive-JEPA 2.14)

- *route hint:* `Turn right in 30m`
- **nonav** (ADE 1.10): Yield to the traffic in the roundabout since a vehicle is circulating ahead
- **nav** (ADE 1.45): Keep distance to the lead vehicle since it is slowing for the roundabout
- **letterbox** (ADE 0.27): Adapt speed for the roundabout since the entry is yield controlled

## 766aceef k=130  ·  v0=8.0 m/s  ·  nav ADE 1.49 (Drive-JEPA 2.60)

- *route hint:* `Continue straight`
- **nonav** (ADE 2.11): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 1.49): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 2.46): Keep distance to the cyclist since it is directly ahead in our lane

## 0638dbd2 k=40  ·  v0=5.9 m/s  ·  nav ADE 1.58 (Drive-JEPA 4.86)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.65): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 1.58): Adjust speed and keep safe distance due to the lead vehicle ahead
- **letterbox** (ADE 0.55): Keep distance to the lead vehicle since it is directly ahead in our lane

## 6f47eb66 k=400  ·  v0=5.4 m/s  ·  nav ADE 1.61 (Drive-JEPA 0.68)

- *route hint:* `Turn right in 17m`
- **nonav** (ADE 2.16): Nudge right due to the stopped van blocking the right-turn entry
- **nav** (ADE 1.61): Yield to the traffic in the roundabout since a van is circulating ahead of our entry
- **letterbox** (ADE 3.85): Nudge right due to the car encroaching from the left

## 8c501afb k=130  ·  v0=10.2 m/s  ·  nav ADE 1.62 (Drive-JEPA 1.53)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.60): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 1.62): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 1.46): Keep distance to the lead vehicle since it is directly ahead in our lane

## 59dc3fbb k=130  ·  v0=5.9 m/s  ·  nav ADE 1.70 (Drive-JEPA 1.66)

- *route hint:* `Turn left in 35m`
- **nonav** (ADE 1.85): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 1.70): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 3.35): Nudge left due to the stopped car blocking the right side of our lane

## 3b1a16a1 k=40  ·  v0=11.8 m/s  ·  nav ADE 1.75 (Drive-JEPA 3.18)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.10): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 1.75): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 2.22): Nudge right due to the stopped van blocking the left side of our lane

## 77c7c33c k=40  ·  v0=0.2 m/s  ·  nav ADE 1.76 (Drive-JEPA 1.25)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.36): Nudge left due to the van encroaching from the right side of our lane
- **nav** (ADE 1.76): Accelerate to proceed through the intersection since the straight traffic light turns green
- **letterbox** (ADE 1.90): Keep distance to the lead vehicle since it is directly ahead in our lane

## eca7bf3a k=40  ·  v0=11.4 m/s  ·  nav ADE 1.77 (Drive-JEPA 7.79)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.55): Nudge left due to the cyclist ahead in our lane
- **nav** (ADE 1.77): Nudge left due to the cyclist ahead in our lane
- **letterbox** (ADE 1.28): Nudge left due to the cyclist ahead in our lane

## cc31f13d k=40  ·  v0=11.6 m/s  ·  nav ADE 1.80 (Drive-JEPA 1.89)

- *route hint:* `Continue straight`
- **nonav** (ADE 2.23): Keep lane since the lane is clear ahead
- **nav** (ADE 1.80): Nudge to the left to clear the road debris blocking the right side of our lane
- **letterbox** (ADE 1.75): Keep lane since the lane is clear and no immediate lead vehicle is present

## 789908e0 k=40  ·  v0=7.1 m/s  ·  nav ADE 1.82 (Drive-JEPA 2.34)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.59): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 1.82): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.85): Adapt speed for the right curve since the lane bends right ahead

## ae85d240 k=40  ·  v0=11.6 m/s  ·  nav ADE 1.84 (Drive-JEPA 2.11)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.56): Keep lane since the lane is clear ahead
- **nav** (ADE 1.84): Keep lane since the lane is clear ahead
- **letterbox** (ADE 0.93): Nudge left due to the parked van blocking the right side of our lane

## 9ffae20c k=220  ·  v0=10.1 m/s  ·  nav ADE 1.85 (Drive-JEPA 0.76)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.71): Nudge left due to the stopped vehicle blocking the right side of our lane
- **nav** (ADE 1.85): Nudge left due to the stopped vehicle blocking the right side of our lane
- **letterbox** (ADE 2.05): Nudge left due to the stopped vehicle blocking the right side of our lane

## af665a7a k=400  ·  v0=0.7 m/s  ·  nav ADE 1.86 (Drive-JEPA 0.48)

- *route hint:* `Continue straight`
- **nonav** (ADE 2.38): Turn left at the intersection since the corner is narrowed by bollards
- **nav** (ADE 1.86): Reverse slowly because forward is blocked by bollards/obstacles and there is a pedestrian walking along the left side, so backing out while yielding is required for safe clearance.
- **letterbox** (ADE 2.02): Keep distance to the scooter since it is directly ahead in our path

## b8002a79 k=40  ·  v0=12.8 m/s  ·  nav ADE 1.90 (Drive-JEPA 5.83)

- *route hint:* `Turn right in 64m`
- **nonav** (ADE 1.44): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 1.90): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 0.55): Keep distance to the lead vehicle since it is directly ahead in our lane

## 1df9975b k=130  ·  v0=13.1 m/s  ·  nav ADE 1.97 (Drive-JEPA 11.88)

- *route hint:* `Continue straight`
- **nonav** (ADE 2.19): Nudge left due to the stopped van blocking the right side of our lane
- **nav** (ADE 1.97): Nudge left due to the stopped van blocking the right side of our lane
- **letterbox** (ADE 2.29): Nudge left due to the van encroaching from the right side of our lane

## 873eebf4 k=130  ·  v0=2.9 m/s  ·  nav ADE 2.05 (Drive-JEPA 2.23)

- *route hint:* `Turn left in 9m`
- **nonav** (ADE 3.48): Turn left at the intersection since the left-turn traffic light turns green
- **nav** (ADE 2.05): Slow down to turn left at the intersection
- **letterbox** (ADE 3.05): Turn left at the intersection since the left-turn traffic light is green

## d398a7a5 k=130  ·  v0=6.2 m/s  ·  nav ADE 2.08 (Drive-JEPA 0.67)

- *route hint:* `Turn right in 28m`
- **nonav** (ADE 2.89): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 2.08): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 2.97): Accelerate to proceed through the intersection since the pedestrian has cleared ahead

## dbba76d9 k=40  ·  v0=8.0 m/s  ·  nav ADE 2.20 (Drive-JEPA 3.05)

- *route hint:* `Continue straight`
- **nonav** (ADE 2.53): Nudge to the right to clear the stopped van blocking the lane ahead
- **nav** (ADE 2.20): Nudge to the right to clear the stopped van blocking the lane ahead
- **letterbox** (ADE 3.59): Keep distance to the cut-in van since it merges into our lane ahead

## dbba76d9 k=130  ·  v0=3.9 m/s  ·  nav ADE 2.20 (Drive-JEPA 6.24)

- *route hint:* `Turn left in 18m`
- **nonav** (ADE 2.03): Nudge left due to pedestrians walking at the right edge ahead
- **nav** (ADE 2.20): Nudge left due to the stopped vehicle blocking the right side of our lane
- **letterbox** (ADE 2.08): Nudge to the left to clear the bollard blocking the center of our lane

## 703ee0c4 k=40  ·  v0=13.0 m/s  ·  nav ADE 2.30 (Drive-JEPA 0.70)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.46): Keep lane since the lane is clear ahead
- **nav** (ADE 2.30): Keep distance to the cut-in vehicle since it merges into our lane ahead
- **letterbox** (ADE 1.56): Keep lane since the lane is clear ahead

## 18775d16 k=40  ·  v0=11.7 m/s  ·  nav ADE 2.46 (Drive-JEPA 3.09)

- *route hint:* `Turn left in 66m`
- **nonav** (ADE 2.76): Accelerate to match the lead vehicle’s speed for a smooth merge into our lane, since cyclists are alongside in the adjacent bike lane and pedestrians are at the curb, leaving our lane clear ahead but not for an immediate lane change.
- **nav** (ADE 2.46): Lane change to the right due to the left lane becoming left-turn-only ahead and to continue straight while maintaining safe clearance from cyclists and following the lead vehicle; then match speed with traffic after merging.
- **letterbox** (ADE 5.00): Create a usable gap for a right lane merge because a car is behind in our right lane blocking an immediate merge

## 8d3308e6 k=40  ·  v0=6.7 m/s  ·  nav ADE 2.53 (Drive-JEPA 1.11)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.00): Stop for the red traffic light ahead
- **nav** (ADE 2.53): Stop for the red traffic light ahead
- **letterbox** (ADE 2.08): Keep distance to the lead vehicle since it is directly ahead in our lane

## e57f9b38 k=40  ·  v0=2.3 m/s  ·  nav ADE 2.66 (Drive-JEPA 0.82)

- *route hint:* `Continue straight`
- **nonav** (ADE 1.25): Create a usable gap to merge left because a truck is alongside in the left lane blocking an immediate merge
- **nav** (ADE 2.66): Create a usable gap for a left lane change because a truck is alongside blocking an immediate merge and our lane bends left ahead
- **letterbox** (ADE 0.62): Nudge left due to cones blocking the right side of our lane

## 67abb4e2 k=130  ·  v0=2.9 m/s  ·  nav ADE 2.66 (Drive-JEPA 0.91)

- *route hint:* `Turn right in 22m`
- **nonav** (ADE 1.20): Yield to the truck in the roundabout since it has priority
- **nav** (ADE 2.66): Yield to the truck circulating in our roundabout since it has priority
- **letterbox** (ADE 1.42): Yield to the truck in the roundabout since it has priority

## 916b4c97 k=40  ·  v0=0.0 m/s  ·  nav ADE 2.77 (Drive-JEPA 1.47)

- *route hint:* `Continue straight`
- **nonav** (ADE 3.08): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 2.77): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 2.08): Keep distance to the lead vehicle since it is directly ahead in our lane

## 8dcfc2ba k=40  ·  v0=6.8 m/s  ·  nav ADE 2.94 (Drive-JEPA 2.06)

- *route hint:* `Turn right in 34m`
- **nonav** (ADE 3.50): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 2.94): Keep distance to the lead vehicle since it is directly ahead in our lane
- **letterbox** (ADE 5.33): Keep distance to the lead vehicle since it is directly ahead in our lane

## 3a616875 k=310  ·  v0=4.6 m/s  ·  nav ADE 3.04 (Drive-JEPA 8.31)

- *route hint:* `Turn left in 25m`
- **nonav** (ADE 2.71): Turn left at the intersection since the intersection is clear
- **nav** (ADE 3.04): Nudge to the left to clear the stopped van blocking the right side of our lane
- **letterbox** (ADE 3.25): Turn left at the intersection since the intersection is clear

## 9ffae20c k=310  ·  v0=8.9 m/s  ·  nav ADE 3.09 (Drive-JEPA 2.28)

- *route hint:* `Continue straight`
- **nonav** (ADE 3.08): Nudge left to avoid the construction barrier blocking the right side of our lane
- **nav** (ADE 3.09): Nudge left due to construction barricades blocking the right side of our lane
- **letterbox** (ADE 3.16): Nudge left due to the construction barrier blocking the right side of our lane

## af665a7a k=130  ·  v0=5.2 m/s  ·  nav ADE 3.17 (Drive-JEPA 1.52)

- *route hint:* `Continue straight`
- **nonav** (ADE 3.26): Nudge right due to the stopped car blocking the left side of our lane
- **nav** (ADE 3.17): Nudge to the right to create clearance for the oncoming cyclist
- **letterbox** (ADE 3.30): Nudge right due to the stopped car blocking the left side of our lane

## 10edd8d1 k=40  ·  v0=11.7 m/s  ·  nav ADE 3.88 (Drive-JEPA 4.01)

- *route hint:* `Continue straight`
- **nonav** (ADE 2.81): Keep distance to the lead vehicle since it is directly ahead in our lane
- **nav** (ADE 3.88): Lane change to the left to stay in a through lane as the right lane becomes right-turn-only ahead, merging into a safe gap behind vehicles already in that lane, then adjusting speed for traffic after the merge.
- **letterbox** (ADE 4.78): Keep distance to the lead vehicle since it is directly ahead in our lane

## b8002a79 k=130  ·  v0=14.6 m/s  ·  nav ADE 4.30 (Drive-JEPA 9.36)

- *route hint:* `Turn right in 79m`
- **nonav** (ADE 4.29): Accelerate to merge into traffic because the lane ahead is clear
- **nav** (ADE 4.30): Adapt speed for the right curve since the lane bends right ahead
- **letterbox** (ADE 5.04): Create a usable gap for a right lane merge because a car is behind in our right lane blocking an immediate merge

## f0b1b61b k=130  ·  v0=5.6 m/s  ·  nav ADE 5.80 (Drive-JEPA 11.46)

- *route hint:* `Turn right in 38m`
- **nonav** (ADE 6.50): Adapt speed for the right curve since the lane bends right ahead
- **nav** (ADE 5.80): Stop to yield to the pedestrian walking across the crosswalk ahead
- **letterbox** (ADE 2.12): Stop to yield to the pedestrian walking across the crosswalk ahead

## 67abb4e2 k=220  ·  v0=6.2 m/s  ·  nav ADE 6.37 (Drive-JEPA 0.43)

- *route hint:* `Turn right in 41m`
- **nonav** (ADE 3.87): Keep distance to the cut-in vehicle since it merges into our lane ahead
- **nav** (ADE 6.37): Keep distance to the lead vehicle since it is directly ahead in our lane after our right turn
- **letterbox** (ADE 2.31): Keep distance to the lead truck since it is directly ahead in our lane
