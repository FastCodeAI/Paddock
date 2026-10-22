# Glossary

*Level 3, reference. The terms used across the release documents, in plain words.*

*Up: [executive plan](../plan-v1.0.0.md)*

## The programme

| Term | Meaning |
|---|---|
| **RAIL** | Renesas AI Labs, the umbrella for all of this work |
| **Paddock** | the benchmark and analysis framework: every model through one interface, scored the same way, with the failure behind each score |
| **Gallops** | the fine-tuning toolkit: trains models against gaps Paddock measured; Paddock scores the result |
| **Jockey** | the proposed name for RAIL's own driving model; its first design is the one Gallops' docs call `gallops-v0` |
| **Hosted twin** | Paddock run by us, for demos, hands-on trials and proofs of value |
| **On-prem kit** | the same Paddock images, run inside a customer's network |
| **Proof of value** | a customer's own model scored and diagnosed, hosted under NDA or on their site |
| **Customer zero** | the first team outside the core contributors to install a release unaided |

## How Paddock tests a model

| Term | Meaning |
|---|---|
| **Open loop** | replay a recording, ask the model for a plan, and compare it with what the human driver did. Cheap, but it measures prediction, not driving |
| **Closed loop** | a simulator drives the car with the model's plan, so the model lives with its own mistakes |
| **NAVSIM** | in between: the model's plan is simulated for four seconds against recorded traffic, and scored for safety, progress and comfort |
| **Adapter** | the small piece of code that connects one model, or one benchmark, to Paddock's single interface |
| **Policy** | that interface: `load`, `reset`, and `infer`, from an observation to a trajectory |
| **Trajectory** | the model's plan: waypoints ahead of the car, in the car's own frame |
| **Run** | one model on one benchmark, recorded with how it was made |
| **Valid / invalid** | whether a run is comparable with the rest of its table. A subset, changed inputs or a failure make it invalid, with the reason |
| **Marks A, P, S, R** | in the support matrix: **A**dapted (trained or tuned for this benchmark), **P**ipeline built around the model, **S**tale (owed a re-run), **R**un differently |
| **Constant velocity** | the floor: "keep going as you are". A model that cannot beat it has learned little |
| **Regression gate** | Paddock's tool for asking "did this change help?": it pairs two runs sample by sample and says ship, hold or fail |
| **Smallest detectable difference** | how large a change must be before a given benchmark can see it; a smaller change reads as "no evidence", not "the same" |
| **Certification run** | reference models re-run on a new install, to prove the site reproduces reference numbers |
| **Provenance** | what a run records about itself: code commit, image, weights and data, so it can be traced and repeated |

## The scores

| Score | Benchmark | Reads as |
|---|---|---|
| **ADE / FDE** | KITScenes, comma2k19, nuPlan | average and final distance between the plan and the human's path, in metres; lower is better |
| **L2** | nuScenes | the same kind of distance, averaged UniAD-style; lower is better |
| **EPDMS / PDMS** | NAVSIM v2 / v1 | 0 to 1, combining no collisions, staying on the road, progress and comfort; higher is better |
| **Driving score** | Bench2Drive | 0 to 100: route completion times an infraction penalty; higher is better |
| **NCAP score** | neuroNCAP | 0 to 5, from a safety rating applied to staged collision scenarios; higher is better |
| **Scene score** | AlpaSim | 0 to 1 per scene; an at-fault collision zeroes it; higher is better |

## Models and techniques

| Term | Meaning |
|---|---|
| **End-to-end model** | one network from camera pixels to a plan, with no hand-written planner in between |
| **Self-supervised** | learned from unlabelled video; how Drive-JEPA's encoder was trained |
| **Vision-language-action (VLA)** | a large language-and-vision model that also outputs driving actions, and can explain them; Alpamayo is one |
| **Bird's-eye view (BEV)** | a top-down map of the scene built from the cameras; AutoE2E uses one |
| **Re-ranker** | a small model that picks among a frozen model's own candidate plans |
| **Fine-tuning** | further training of part of an existing model on new data |
| **Distillation** | a large "teacher" model trains a small "student" to reproduce its features; the student is what ships |
| **DAgger** | drive, relabel what went wrong, retrain, repeat |
| **Camera-blind** | a model whose plan does not change when its camera image does; several early runs were, and preflight now refuses it |
| **Checkpoint sidecar** | the record written beside a trained model's weights: what it is, from which commit, trained on which data |

## The chip

| Term | Meaning |
|---|---|
| **R-Car** | Renesas' automotive system-on-chip family; Jockey's design names R-Car X5 as its target |
| **fp32 / int8** | 32-bit floating point, the training precision, and 8-bit integers, the precision embedded accelerators run fastest |
| **Quantisation** | converting a model from fp32 to int8 |
| **int8 delta** | how much quantisation costs, measured as fp32 against int8 on identical samples |
| **Operator allowlist** | the operations an accelerator runs natively; anything else falls back to slow paths |
| **ONNX** | a standard file format for neural networks, the usual hand-off to a chip's toolchain |
| **Hardware in the loop (HIL)** | the real chip runs the model while a simulator plays the world around it |
| **Shadow mode** | the model runs in a real car and is logged, but does not control it |
| **MCP** | Model Context Protocol, the open protocol AI agents use to call tools; the planned agent gateway speaks it |
