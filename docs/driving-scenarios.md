# Hard driving scenarios, and what solves them

Two examples started this doc. Tesla FSD slows down for speed bumps. Waymo reads the
turn signals of the cars around it. The question was how many scenarios like
these exist, and whether each one is solved by better **sensors**, by **maps**,
by **models**, or by some mix of the three.

## How many are there?

Nobody publishes one complete list. Each company keeps its own catalogue, and
these catalogues are not public. Two public lists give a sense of the size:

- NHTSA's *Pre-Crash Scenario Typology* (Najm et al., 2007) sorts US crashes into
  **37 pre-crash scenarios**.
- The CARLA Leaderboard 2.0 scenario set, which our Bench2Drive benchmark uses,
  has **44 scenario types**.

Both lists describe crash geometry: who hit whom, from which direction. Neither
covers perception problems like "is that light blinking" or "is that bump real".
The tables below list about 50 scenarios that keep coming up in public
reporting, in seven groups. The real long tail is much longer than this list.

## How to read the tables

**Mainly solved by** names the approach that carries most of the load. When
more than one is listed, the first one matters most.

| Code | Approach | What it means |
|---|---|---|
| **S** | Sensors | Hardware that measures something a camera cannot, e.g. lidar for depth, radar for speed through fog, microphones for sirens, thermal cameras for warm bodies at night |
| **M** | Maps | Knowledge stored before the drive, e.g. HD maps, crowd-sourced maps, mapped traffic-light positions |
| **L** | Learned model | A network that reads the sensor data and decides what it sees, what others will do, and how to drive |
| **F** | Fleet | Humans or the wider fleet help: remote assistance, hazard reports shared across cars, service limits in bad weather |
| **V** | V2X | The road or other cars send the information directly, e.g. traffic-light timing over a radio link |

**State**

- **Shipping**: in a public product and widely seen working.
- **Partial**: works much of the time, but public failures are on record or the
  hard version of the case is unsolved.
- **Open**: nobody has publicly shown it working reliably.

**Evidence.** This is a survey of company statements, press reports and owner
videos up to mid-2026. The rows were **not** checked one by one against primary
sources. When a row names a company, that company has talked about the case or
been seen handling or failing it. That does not mean other companies cannot do
it. Rows marked *(inferred)* are our reading of a sensor stack, not a public
claim.

## 1. Reading signals from other road users

| Scenario | Why it is hard | Mainly solved by | Public example | State |
|---|---|---|---|---|
| Turn signal of a nearby car | Small light, flashes at 1–2 Hz. One frame cannot tell blinking from steady, so the model needs several frames, and it has to attach the light to the right car | L | Waymo has said its perception classifies other vehicles' turn signals | Shipping (robotaxi) |
| Brake lights and hazard lights | Hazards usually mean the car is stopping or parked, a different prediction from braking | L | Standard in camera stacks *(inferred)* | Shipping |
| Reverse lights: car backing out of a parking space | The car is barely moving; the light is the earliest warning | L | *(inferred)* | Partial |
| Cyclist hand signals | Small gesture, often half-made | L | Google's self-driving project described reading cyclist hand signals (2016) | Partial |
| Police officer or flagger directing traffic, stop/slow paddle | The gesture overrides the traffic light; each person signals differently | L + F | Waymo says it responds to traffic officers; remote assistance covers the hard cases | Partial |
| Pedestrian intent: about to cross, waving the car on, looking at a phone | Intent is not visible, only guessed from posture and gaze | L | All robotaxi operators; active research area | Partial |
| Siren heard before the vehicle is visible | The siren is audible around corners where no camera or lidar can see | S (microphones) + L | Waymo described external microphones for sirens (2017) | Shipping (robotaxi) |
| Emergency vehicle: yield, pull over, do not block it | Needs a manoeuvre that is normally illegal, e.g. entering an intersection on red | L + F | Cruise vehicles blocked emergency responders in San Francisco (2023) | Partial |
| School bus with its stop arm out | Legally you must stop even in the opposite lane on many roads | L + M | Waymo recalled software in late 2025 after its cars passed stopped school buses in Austin | Partial |
| Parked car's door opening into the lane | Very little warning; the obstacle is thin | L | CARLA `VehicleOpensDoorTwoWays` scenario | Partial |

## 2. Road surface and geometry

| Scenario | Why it is hard | Mainly solved by | Public example | State |
|---|---|---|---|---|
| **Speed bumps and speed tables** | A bump is a few centimetres of height, judged from a flat image. Painted bumps and real bumps look alike | L, or M, or S | Tesla FSD slows for them on cameras alone (widely reported by owners). Mercedes Magic Body Control (2013 S-Class) scans the road ahead with a stereo camera, but for the suspension, not for driving decisions. Maps can tag them, e.g. OpenStreetMap `traffic_calming=bump` | Shipping |
| Potholes and broken road | Same height problem as bumps, and they change week to week | L + F | Fleet hazard reporting *(inferred)* | Partial |
| Ice, black ice, oil | Mostly invisible; detected after a wheel slips | F + V | Volvo and Audi share slippery-road warnings between cars through the cloud | Partial |
| Flooded road or deep water | Depth cannot be judged from the surface | L + F | Robotaxi operators limit service in severe weather | Open |
| Faded, missing or snow-covered lane lines | The lanes still exist; the paint does not | M + L | Waymo and Mobileye (REM crowd-sourced map) fall back on the map; Tesla infers lanes from the scene | Partial |
| Unmarked or dirt roads | No lane structure at all | L | | Partial |
| Blind crests and sharp hills | The road beyond is not visible | M + L | *(inferred)* | Partial |
| Low clearance: parking barriers, gates, low branches, low bridges | Cameras judge height poorly; obstacle is above the usual detection zone | S (lidar) + M | *(inferred)* | Partial |
| Wet concrete, fresh paint | Looks like normal road | L + M | A Cruise car drove into wet concrete in San Francisco (2023) | Open |

## 3. Traffic control

| Scenario | Why it is hard | Mainly solved by | Public example | State |
|---|---|---|---|---|
| Traffic light state, and which light belongs to your lane | Big intersections have many lights facing different lanes | M + L | Waymo uses mapped light positions; Tesla reads lights from cameras alone | Shipping |
| Dark traffic lights in a power cut (treat as all-way stop) | The car must notice the light is off, and switch rules | L + F | Waymo cars stalled at dark intersections during a San Francisco power cut (December 2025) | Partial |
| Flashing red or yellow, arrows, bus-only signals | Same light, different rule | L + M | | Partial |
| Temporary signs, message boards, handwritten signs | The car has to read and understand free text | L | | Open |
| Construction zones: cones, lane shifts, closed lanes | The map is now wrong, and the new layout is only implied by cones | L + M + F | CARLA `ConstructionObstacle` scenario | Partial |
| Railroad crossings | Rare, and the cost of a mistake is total | M + L | Press reports in 2025 of FSD mishandling level crossings | Partial |
| School zones and time-of-day speed limits | The rule depends on the clock and the day | M + L | | Partial |
| Conditional signs ("except buses", "7–9 am") | Has to read the sign and apply the condition | L + M | | Open |

## 4. Unusual objects and hazards

| Scenario | Why it is hard | Mainly solved by | Public example | State |
|---|---|---|---|---|
| Debris, lost cargo, tyre treads on a highway | Small and low, seen late at highway speed | S (lidar, radar) + L | | Partial |
| Animals: deer, dogs, kangaroos | Unusual shapes and motion | L + S (thermal at night) | Volvo reported its detection struggled with kangaroos' hopping (2017) | Partial |
| Plastic bag or rock: safe to drive over? | Same size; the answer depends on what it is made of | L | | Open |
| Ball rolls into the road, child may follow | The hazard is the one you cannot see yet | L | | Open |
| Stopped or crashed vehicle in the lane at speed | Radar stacks used to ignore stationary returns to avoid false braking | S + L | NHTSA investigated Tesla Autopilot crashes into stopped emergency vehicles (2021) | Partial |
| Pedestrian stepping out from behind a parked bus or van | Hidden until the last second | L + S | CARLA `PedestrianCrossing` and `DynamicObjectCrossing` scenarios | Partial |
| Odd shapes: overhanging loads, trailers, wheelchairs, scooters, people in costume | Outside what the detector was trained on | L + S | | Partial |
| Wrong-way driver | Breaks every prior about traffic direction | L | | Partial |
| Cones placed on the car's bonnet | Not a driving problem; the car must stop and wait for help | F | Activists "coned" Waymo and Cruise cars in San Francisco (2023) | n/a (operations) |

## 5. Weather and visibility

| Scenario | Why it is hard | Mainly solved by | Public example | State |
|---|---|---|---|---|
| Low sun glare | Blinds cameras for seconds at a time | S (radar, lidar) + L | NHTSA opened a probe into FSD in reduced visibility: sun glare, fog, dust (October 2024) | Partial |
| Fog, heavy rain, dust | Cameras and lidar both degrade; radar does not | S (radar, thermal) | Zoox fits long-wave infrared cameras | Partial |
| Snow | Lanes hidden, sensors coated, traction low | S + M | Most robotaxi service runs in warm cities | Open |
| Night, unlit pedestrian | Low contrast for cameras | S (lidar, thermal) + L | Uber ATG test-car fatality, Tempe, 2018 | Partial |
| Dirty or blocked sensor, rain on the lens | The car has to know its own view is bad | S (cleaning) + L (self-check) | Robotaxis carry sensor-cleaning systems | Shipping |
| Tunnels and urban canyons (no GPS) | Position has to come from somewhere else | M (lidar matched to the map) + S (IMU) | | Shipping |

## 6. Negotiation and social driving

| Scenario | Why it is hard | Mainly solved by | Public example | State |
|---|---|---|---|---|
| Unprotected left turn across traffic | Must judge gaps in oncoming traffic and commit | L | CARLA `SignalizedJunctionLeftTurn`, `NonSignalizedJunctionLeftTurn` | Partial |
| Four-way stop: who goes first | Rules plus local habits and eye contact | L | | Partial |
| Merging into dense or slow traffic, zipper merge | Too polite and you never merge; too pushy and you collide | L | CARLA `MergerIntoSlowTraffic`, `EnterActorFlow` | Partial |
| Cut-ins | The other car decides late | L | CARLA `HighwayCutIn`, `ParkingCutIn`; common in ADAS | Shipping |
| Going round a double-parked car using the oncoming lane | Briefly breaking a rule on purpose | L | CARLA `Accident`, `ParkedObstacle`, `InvadingTurn` | Partial |
| Another driver waves you through | Gesture through a windscreen | L | | Open |
| Pick-up and drop-off: where to stop without blocking | Depends on the kerb, the rider and traffic | M + L + F | | Partial |
| Roundabouts | Yield rules plus reading intent | L + M | | Partial |

## 7. When the car cannot decide

| Scenario | Why it is hard | Mainly solved by | Public example | State |
|---|---|---|---|---|
| Stuck: no rule-following path exists | Some situations need a human judgement call | F | Waymo "fleet response" and Zoox "teleguidance": a remote human gives a hint, the car still drives | Shipping (robotaxi) |
| Safe stop when the system is unsure | Stopping in a live lane can itself be dangerous | L | | Partial |

## The two examples, in detail

**Speed bumps.** There are three ways to know a bump is there, and companies use
different ones.

- *Model on cameras* (Tesla). The network learns that a certain pattern of paint
  and shading means "slow down". It finds new bumps no one has mapped. It has to
  judge a few centimetres of height from a flat image, and a painted fake bump
  looks the same as a real one.
- *Map*. A bump tagged in the map is known before it comes into view, even past a
  crest. A new bump that is not yet mapped is missed.
- *Lidar*. Measures the height directly. Cost is the sensor.

Mercedes has scanned the road ahead with a stereo camera since 2013. That
system tunes the suspension, so the car is comfortable over the bump; it does
not slow the car down. What is new in FSD is that the driving decision comes
from the camera.

**Turn signals.** This is a model problem, and it is mainly a problem of
**time**. A blinker flashes at roughly 1–2 Hz. A single frame cannot tell a
blinking light from a light that is on, so the model must look at several
frames in a row, at a frame rate well above the flash rate. A system that
samples the scene at 2 Hz cannot see a blinker at all. Some of our own
benchmarks feed the model frames at 2 Hz (NAVSIM's default), which is one
reason paddock records each model's `min_input_hz`.

## The pattern

- **Sensors** cover physics a camera cannot measure: depth in the dark, speed
  through fog, sound around a corner, heat at night. The cost is hardware on
  every car.
- **Maps** cover things that are fixed but hard to see: which light belongs to
  which lane, a bump past a crest, a lane under snow. They break when the world
  changes, e.g. construction.
- **Models** cover anything that needs reading the scene or guessing intent:
  blinkers, gestures, signs, negotiation. A model is cheap per car, but it only
  learns a rare case if the training data contains that case.
- **Fleet support** catches what all three miss. It is too slow for
  split-second events.

The two companies in the conversation sit at opposite ends. Tesla is all model:
cameras only (radar removed in 2021, ultrasonic sensors in 2022), light map
use. Waymo uses all four: lidar, radar, cameras, microphones, HD maps and
remote assistance. Mobileye is in between: cameras plus a crowd-sourced map,
with radar and lidar added at higher autonomy levels.

For a chip vendor the split matters because each approach loads the compute
differently. A sensor-heavy stack spends its compute on fusing point clouds and
radar returns. A model-heavy stack spends it on large camera networks, and
cases like blinkers push those networks to run on more frames per second and
to keep several frames in memory.

## What paddock tests today

| Benchmark | Which of the groups above it scores |
|---|---|
| [neuroNCAP](benchmarks/neuroncap.md) | Group 4: a vehicle inserted on a collision course, in frontal, side and stationary variants |
| [Bench2Drive](benchmarks/bench2drive.md) | Groups 1, 3, 4 and 6 through its CARLA scenarios: `YieldToEmergencyVehicle`, `ConstructionObstacle`, `PedestrianCrossing`, `DynamicObjectCrossing`, junction turns, cut-ins, merges, obstacles in lane |
| [NAVSIM](benchmarks/navsim.md) (v2) | Group 3, partly: its score includes a traffic-light compliance term. [NAVSIM v1](benchmarks/navsim_v1.md) does not |
| Open-loop: [nuScenes](benchmarks/nuscenes.md), [KITScenes](benchmarks/kitscenes.md), [nuPlan](benchmarks/nuplan.md), [comma2k19](benchmarks/comma2k19.md) | None directly. Real drives contain many of these scenes, but ADE/FDE averages over all frames and does not score them one by one |

Going by what each benchmark's score is made of, none of them scores reading
blinkers, hand signals, sirens, speed bumps or bad weather.
