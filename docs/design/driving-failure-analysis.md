# Design: why a driving model fails, and an agent to help find out

**Status:** §1 describes what was built and run on the openpilot × Bench2Drive-220 campaign of October 2026.
§2 is a proposal; none of it is built. §3 is a procedure, carried out once, in §3.3.

The leaderboard says how well a model drives. This note is about *why*, and answers three questions:

1. How were the campaign's results analysed and catalogued? (§1)
2. What would it take to build an agent that does part of that work, including reading the camera frames? (§2)
3. When the catalogue, or such an agent, misses a failure mode, how is it added so it is not missed again? (§3)

The findings themselves are in
[openpilot_drv_supcombo_bench2drive_failure_analysis.md](../../tools/models/openpilot/analysis/openpilot_drv_supcombo_bench2drive_failure_analysis.md).
This note is about the method. Claims carry the labels of [reviewing.md](../reviewing.md):
- **verified**: checked in code, a log or a frame;
- **measured**: computed from the record;
- **argued**: a reading, or a design not yet tested.

---

## 1. How the results were analysed and catalogued

### 1.1 The aim

A driving score of 33 is an average over 218 different stories. The aim was to tell each story in one sentence a
reviewer can check:
- one **primary cause** per route, so the points lost across routes add up to the total lost;
- every other true statement about the route as a **secondary label**;
- for each, **evidence**: a time, and what to look at.

Two rules followed from that.
- **Record once, analyse many times.** Driving the routes took about a day of wall clock on five CARLAs.
  Re-deriving every table from the record takes 81 s and reproduces it byte for byte (measured). A new question
  never needs a new drive, unless the record lacks the signal (§3.2).
- **Every label is a function of the record.** Nothing is typed by hand. Fixing the classifier relabels every route
  at once, and a disputed label is settled by re-running it, not by argument.

### 1.2 The record

With `record = "true"` for bench2drive in the site config, the agent
([agent.py](../../src/paddock/adapters/benchmarks/bench2drive/agent.py),
[recorder.py](../../src/paddock/adapters/benchmarks/bench2drive/recorder.py)) writes per route:

| File | Content |
|---|---|
| `ticks.jsonl` | One line per 20 Hz tick, holding three things. The node's whole reply: plan, lead, lanes, pose, sigma, desire. What the agent sent and actuated, and the inference time. Privileged state the model never sees: ego pose and speed, every actor within 60 m in the ego frame, the lane and junction, progress along the route, and the next light with its state. |
| `actors.jsonl` | Each actor once: type, role, size |
| `route_plan.json` | The dense route: positions, distance along it, the manoeuvre at each point, the lights on it |
| `model_road/`, `model_wide/` | The two images the model was given, 482×302, every other tick |
| `front/`, `bev/` | A front camera (800×450) and a top-down camera (512×512), every tick |
| `results.json` | The leaderboard's record: status, scores, every infraction with its location |

That is ~100 MB for a 67 s route; the campaign's attempts total 47 GB (measured). Beside it is the **expert**:
Think2Drive's drive of the same route, from Bench2Drive's public dataset
([expert.py](../../tools/benchmarks/bench2drive/expert.py)), available for 209 of the 218 scored routes.

Recording the model's outputs and the privileged world side by side, per tick, is what makes most findings
mechanical: they compare the two. For example, the lead head reported the obstacle, yet the car stopped 33 m short
of it. Or: the model's own turn desire averaged 0.01 inside left turns it was never told about.

The analysis forced two fixes on the record (verified, commits `87ababd` and `0214a69`):
- The base agent kept only a 50 m downsample of the route, too coarse to place a turn or a light. The agent now
  keeps the dense plan.
- The dataset's top-level x/y for the expert are GNSS: ~1.5 m off, and far noisier than the ground-truth pose of
  the ego's bounding box. `expert.py` now reads the bounding box.

### 1.3 From ticks to events

[openpilot_drv_supcombo_bench2drive_lib.py](../../tools/models/openpilot/analysis/openpilot_drv_supcombo_bench2drive_lib.py)
reads each route's record for the attempt that counts: the latest one that scored
([runs.py](../../tools/benchmarks/bench2drive/runs.py)). It turns the record into per-tick arrays (`Signals`). Two
derived signals carry most of the analysis:
- **standstill**: speed below 0.3 m/s for at least 1 s.
- **ahead**: what blocked the lane at a tick. The first match wins:
  1. a red or yellow light on the route within 30 m;
  2. the nearest actor in the ego's lane (|y| < 2.5 m in the ego frame), typed: a stationary vehicle, prop or
     pedestrian within 40 m, or a moving vehicle within 15 m;
  3. the model's own lead (probability > 0.5) within 40 m, when the privileged state puts nothing in the lane;
  4. otherwise, *nothing*.

From these come four event tables, one row per event, each row carrying what is needed to explain it:

| Table | One row per | The columns that explain it |
|---|---|---|
| `standstills.csv` | standstill | what was ahead when it began and when it ended (or the route did), whether it ended, the plan's reach while standing, the wait after green, how many actors moved in view |
| `lights.csv` | light on the route that the car reached | its state as the car crossed, speed, whether the car saw red and stopped |
| `turns.csv` | junction manoeuvre | reached, completed, where the car left the route, the model's own turn desire inside it |
| `collisions.csv` | collision | speed, the other actor, the lead head and braking in the 2 s before |

There is also one row per route, `route_features.csv`. It holds the leaderboard's numbers, the launch time, pace
against the expert, time stopped, the final standstill, and summaries of the four tables.

Events are facts; causes are interpretations. Keeping them in separate tables means a dispute about a cause never
has to re-open the facts.

### 1.4 From events to a cause: the pattern catalogue

`PATTERNS` is a list of `Pattern(name, label, family, primary, summary, test)`. `test` takes the route's
`RouteContext`: its features, status, leaderboard result and event lists. It returns a short evidence string when
the pattern holds, and `None` otherwise:

```python
Pattern("blocked_by_prop", "stuck behind cones / a parked mesh", "blocked, no overtake", 12,
        "construction cones, warning signs or a parked car in its lane; no lane change to pass",
        lambda c: _final_ahead(c, "prop")),
```

On route 2513 it holds with the evidence "stood 194 s from t=6 s to the end; prop 37 m ahead".

- Every pattern that holds goes to `patterns.csv` with its evidence.
- `primary` is a number or `None`. The route's cause is the holding pattern with the **lowest number**. The order
  is causal precedence: whatever happened first and made the rest moot.

  | Number | Patterns | Why there |
  |---|---|---|
  | 0–4 | not scored; success; completed with a collision, a red light or another infraction | the outcome itself |
  | 10 | never pulled away | nothing else could happen |
  | 11–11.8 | missed a junction turn; left the route elsewhere; collided, then stood | each ends or freezes the route where it happens |
  | 12–16 | what it stood behind at the end: a stationary vehicle, prop or pedestrian (12); queued traffic or a red light (13); nothing, after a red light (14), after a lead left (15), or on a clear road (16) | the final standstill, by what was in the lane |
  | 20, 30 | a collision without a final standstill; ran out of time while moving | |
  | 99 | `other_failure` | the fallback |

- **Secondary labels** (`primary=None`) are true of the route, but never why it scored what it did. Examples: ran
  a red light, stopped for red, slow restart after green, a lane change, a turn taken without route input, the
  kind of each collision.
- The **fallback** holds only when no other primary does: it is the catalogue admitting it has no name for a
  route. On the campaign it holds for 0 routes. That shows coverage, not correctness: a named cause can still be
  wrong (§3.3).
- Each route's lost points (100 − DS) go to its primary cause and are summed per family. That sum is the findings
  note's "where the score went" table.

Adding a mode means adding one `Pattern` (§3).

### 1.5 How the catalogue was checked

The classifier was wrong twice on the finished campaign. The checks below are what found it.

1. **Each label against the scenarios it landed on** (verified, `331271a`). The first classifier was one if-chain,
   and two of its labels did not fit their scenarios' stories.
   - Its "left the route (not at a turn)" held 46 routes that had all missed their one junction turn. It compared
     where the deviation was flagged with where the turn was, but the leaderboard flags a deviation only once the
     car is 30 m off the route.
   - Its "ran out of time" held 43 routes that had stood still for ~194 of their 200 s. The tick cap ended them,
     because those scenarios switch the blocked test off.

   The if-chain became the catalogue.
2. **Against the leaderboard's own counts.** The light table found 25 red-light crossings; the leaderboard records
   22. The report now uses the leaderboard's count.
3. **Against the videos.** Sample each pattern's hits on the dashboard, at the time given in the evidence. Each
   route's video ([route_video.py](../../tools/benchmarks/bench2drive/route_video.py)) puts five things in one frame:
   the BEV, the front camera, the plan against the expert's path, the speed, and the score components.
4. **Against the prose.** Every sentence of the findings note carries a label, and every number was re-derived from
   the tables before commit. That pass corrected several claims. Two examples: how many overtaking routes changed
   lanes, and the wait after green, which is bimodal rather than uniformly slow.

### 1.6 What to keep for another model or benchmark

- Record privileged state beside the model's outputs, per tick.
- Write the event tables first and the classifier second.
- Give each route one primary cause with an explicit precedence; make everything else secondary; count the
  fallback.
- Put evidence with a time on every label.
- Name every threshold with its reason, and look at the distribution around it (§3.3 shows why).
- Look in the dashboard; count in the tables. Never count from what you looked at.

---

## 2. What it would take to build an analysis agent

Everything in this section is proposed (argued). None of it is built or measured.

### 2.1 What it would add, and what it must not replace

The catalogue names only what someone encoded, and it cannot see pixels. Both limits show up on this campaign
(verified, §3.3). Of the 41 routes it labels "stopped with a clear road", 10 have a stopped vehicle or cones in the
lane 43–60 m ahead. That is past the 40 m the catalogue looks. Their frames differ: one shows an empty junction on
a green light, two show stopped traffic, and one shows an accident in fog. The record says "an actor 55 m ahead";
only the frame says whether it matters.

An agent would add three things:
1. **Reading frames** at the moments the record flags, and saying what a person would see there.
2. **Answering open questions** over the tables, without someone writing a query for each.
3. **Proposing patterns** the catalogue lacks, with the routes that motivate them.

The opposite limit matters just as much: a frame has no time in it. One frame cannot tell a car that has stood for
190 s from one about to go. Nor can it tell a car that never moved from one that drove 80 m first, which is the
mistake in §3.3's first cut. So the agent reads the tables as text and the frames as images, never the frames alone.

The agent must not produce any number in a report. Its output is a proposal with evidence. Counts come from the
catalogue, and a mode the agent finds is counted only once it is a pattern (§3).

A version with nothing built already exists. This campaign's analysis was done interactively with Claude Code: it
read the tables, wrote the queries and opened frames. The horizon finding in §3.3 came from looking at four frames.
That version lacks two things:
- **coverage**: it looked at the routes someone asked about, not all 218;
- **measurement**: nobody scored how often it was right.

The two stages below make both systematic.

### 2.2 Shape: a batch workflow first, an agent second

Most of the work needs no autonomy, so there are two stages.

**Stage A, per-route triage: one model call per route, no loop.** Code assembles a packet for each route:
- its `route_features.csv` row, its patterns with evidence, and its event rows;
- the dashboard's `summary.png`: BEV, the two paths, and the speed;
- up to ~12 `model_road` frames, picked at event times: each standstill's first and last tick, 2 s before each
  collision, each deviation, each light crossing.

The model answers in a fixed schema (structured output):

```text
diagnosis              what happened, and why the route scored what it did
primary_failure        a catalogue pattern name, or "none of these" plus a short name of its own
evidence               [{t_s, source: frame | table, observation}]
agrees_with_catalogue  true / false, and why not
proposed_pattern       optional: {name, description, the signal in the record that would detect it}
confidence             low / medium / high
```

The 218 requests go through the Message Batches API, keyed by route id: at half price, with nothing to throttle.

**Stage B, the analysis agent: a tool loop** over the triage output and the tables. It clusters disagreements,
answers questions, and drafts pattern proposals (§3) for a person to implement. Its tools are read-only and
deterministic, and they return little:

| Tool | Returns |
|---|---|
| `routes(where)` | route ids with cause, family, DS and scenario, filtered |
| `route(route_id)` | the features row, the patterns with evidence, and the Stage A triage |
| `events(route_id, kind)` | standstill, light, turn or collision rows |
| `ticks(route_id, t0, t1, fields, hz)` | a downsampled slice of `ticks.jsonl` |
| `frames(route_id, t_s, camera)` | the `model_road`, `model_wide`, `front` or `bev` image at a time |
| `expert(route_id, s0, s1)` | Think2Drive's speed and position over the same stretch of route |
| `sql(query)` | DuckDB over the CSVs |
| `propose_pattern(...)` | appends to a review file; this is its only write |

It never edits the catalogue or the analysis outputs.

### 2.3 Models and API, as of October 2026

- **Stage A:** `claude-sonnet-5-5`, with vision, at $2 / $10 per million input / output tokens. Try
  `claude-haiku-4-5` only if the labelled set (§2.5) says it is good enough.
- **Stage B:** `claude-opus-5-5`, at $4 / $20. Its thinking is always on; the effort setting controls depth.
- **Output shape.** Stage A uses structured outputs: `client.messages.parse(..., output_format=<pydantic model>)`.
  Stage B uses strict tool schemas, with `tool_choice` left on auto, because forcing a tool choice returns a 400
  on both models.
- **Discounts.** Prompt caching covers the shared prefix: the instructions plus `pattern_catalogue.csv`. Stage A
  also goes through Batches, and the two discounts stack.
- **Harness.** Use the Python SDK's tool runner (`client.beta.messages.tool_runner`, with `@beta_tool` functions),
  so the tools run here, next to the 47 GB record. There are two alternatives:
  - The Claude Agent SDK, if the agent should write and run its own pandas. It brings Claude Code's file and shell
    tools.
  - Managed Agents, which run the loop on Anthropic's side. The tools run in a hosted sandbox, or in a self-hosted
    one on this machine, so the frames need not move. This fits a scheduled pass after every sweep, at the cost
    of a larger platform to learn.

### 2.4 Cost (arithmetic, not measured)

Image tokens scale with pixels, roughly width × height / 750: ~200 tokens for a 482×302 model frame, and ~1,400
for the 1920×540 summary. A Stage A packet then holds:
- ~3k tokens of instructions and catalogue, cached;
- ~4k tokens of tables;
- ~2.4k tokens for 12 frames, and ~1.4k for the summary.

That is ~11k tokens in and ~3k out per route, thinking included. Over 218 routes it comes to ~2.4M in and ~0.7M
out: about $12 on Sonnet 5.5 at list price, and half that through Batches. Measure one real packet with
`count_tokens()` before trusting these figures.

With caching, a Stage B question that reads twenty routes' tables and fifty frames costs a few dollars, not tens.
Money is not the constraint; the labelled set is.

### 2.5 Evaluation

- **A labelled set.** About 40 routes, stratified over the catalogue's causes, plus every route where a reviewer
  disagreed with the catalogue. Each one gets three things from a person: the primary cause, the time the outcome
  became inevitable, and the frame that shows it. On the dashboard that takes a few minutes per route, two to
  three hours in all.
- **Scores.**
  - Agreement with the person's cause.
  - **Evidence validity**: does the cited tick or frame show what the agent says? Code checks the table
    citations; the frame citations are spot-checked.
  - The **precision of its disagreements** with the catalogue: how many are real.
  - **Calibration**: is "high" right more often than "low"?
- **Anchoring.** Run Stage A twice on the labelled set:
  - **blind**, with no catalogue labels in the packet. Its agreement with the catalogue measures whether the agent
    sees what the code sees.
  - **informed**, with the labels. Its disagreements are where the agent adds anything at all.
- **Regression.** Every miss found later (§3) joins the set, with its label.

### 2.6 Guardrails

- A claim that cites no tick range or frame is dropped when the triage is ingested.
- Each triage record carries provenance, as `run.json` does: the model id, the prompt hash, the catalogue's git sha
  and the packet's hash.
- The dashboard already shows any per-route CSV beside each route (`--features`). The triage goes there, marked as
  model output. The report never counts it.

### 2.7 Effort (estimate)

| Piece | Estimate |
|---|---|
| Packet builder: frames at event times, summary, tables | 1 day |
| Stage A prompt, schema and batch runner; ingest to a CSV for the dashboard | 1–2 days |
| Labelled set of ~40 routes, and the scorer | 1 day, 2–3 hours of it a person watching routes |
| Stage B tools and loop | 1–2 days |
| Tuning the prompts against the labelled set | 1–2 days |

That is about a week and a half for one person who knows the record. Most of it is not model work. The packet
builder and the labelled set would pay for themselves even without an agent. These are usable as they stand:
`runs.py`, the lib's `Signals` and event tables, the videos and summaries, and the dashboard's `--features`.

---

## 3. Adding a failure mode the catalogue, or an agent, missed

### 3.1 Where missed modes show up

- **The fallback.** Routes whose cause is `other_failure`. There are none today, which shows coverage, not
  correctness.
- **Disagreement.** A Stage A triage that disagrees with the catalogue's cause, or a reviewer whose reading of the
  video differs from the label.
- **Labels true for the wrong reason.** In §3.3's first cut, 18 routes "stopped short" because they never moved.
- **Pile-ups at a threshold or a horizon.** A distribution that crowds a cut-off is often the visible part of
  something the cut-off hides. In §3.3, the distances crowd just under the 40 m look-ahead.
- **A new configuration.** Turning on driver-resume or route desire will produce behaviour the catalogue never
  needed to name.

### 3.2 The procedure

1. **State the mode as a fact about the record**, not about the model's mind. "Pulled away, stopped for an obstacle
   more than 15 m ahead, and stood there to the end" can be tested. "Was afraid of the obstacle" cannot.
2. **Find the signal.**
   - If the record has it, write the test over `RouteContext`. Add a field to `Standstill` or `RouteFeatures` if
     you need one.
   - If it does not, extend the recorder, and accept that only routes driven afterwards will carry it. A full
     sweep here is about a day on five workers.
3. **Decide: primary or secondary.**
   - Secondary, if it says how the route went but not why it scored what it did. It moves no cause and needs no
     argument.
   - Primary, only if it should take routes from an existing cause. Its number is then a claim about precedence,
     and the commit must name the routes that move, and why.
4. **Name the threshold, with its reason**, and look at the distribution around it. If there is no gap, say so,
   and give the counts at neighbouring values.
5. **Run it over every route** and check, in this order:
   - the hit count;
   - the cause diff: a secondary moves nothing, and a primary moves only routes you can name;
   - the hits: read their evidence, and watch a sample on the dashboard at the evidence's time;
   - the near misses, on both sides of every condition;
   - hits that are true for the wrong reason.
6. **Commit** the pattern with the counts and the cause diff in the message. Regenerate the report, and update the
   findings note.
7. **Close the loop.**
   - `pattern_catalogue.csv` is the agent's vocabulary. Stage A's prompt carries it, so the next pass names the
     mode instead of rediscovering it.
   - The routes that revealed the mode join the labelled set.

### 3.3 Worked example: `stopped_short_of_obstacle`

This was carried out on the 218 scored routes, and the pattern is now in the catalogue as a secondary label. All
numbers are measured unless marked otherwise.

**Where it came from.** An ad-hoc query for the findings note found that the blocked cars stopped a median 31 m
short of their obstacles. A finding that lives only in prose is never re-checked when the catalogue changes, so it
became a pattern.

**First cut.** The test was a final standstill with a stationary vehicle or prop more than 15 m ahead at the end,
as a secondary label. It held for 40 routes and moved no cause. But the hits were not what the label said:
- 18 of the 40 never moved. Those routes start 28–35 m behind their scenario's obstacle, so the distance described
  the spawn point, not an approach.
- One had stopped for a red light, with a car 22 m ahead.

**Second cut.** The test now holds only after the car has pulled away, and only for a stop that began with the
obstacle, or the model's lead, ahead. It holds for 21 routes:
- 14 whose cause is "stuck behind a stationary vehicle", and 7 "stuck behind cones / a parked mesh". That is 21
  of the 30 routes with those two causes.
- They drove 10–90 m first, reaching 5.8–12.6 m/s.
- They stopped a median 33 m short of the obstacle (18–38 m), at a median 7.6 s into the route.
- They stood a median 172 s, planning a median 0.12 m of travel.
- On 19 of the 21, the model's lead head reported a lead (probability > 0.5) when the car stopped.

**The threshold.** There is no gap at 15 m. Of the 36 cars that pulled away and ended behind a stationary vehicle
or prop, 14 stood within 15 m and 22 beyond. But the count barely depends on the threshold:

| Threshold | Routes |
|---|---:|
| 10 m | 23 |
| 15 m | 21 |
| 20 m | 20 |
| 25 m | 19 |
| 30 m | 17 |

15 m was kept. The distance is measured centre to centre, and a car queued normally behind another reads 8–10 m.

**The horizon.** The distances crowd 29–38 m, just under the 40 m that `Signals.ahead` looks. So the next question
is whether they continue past it, and they do.
- Of the 41 routes labelled "stopped with a clear road", 10 have a stationary vehicle or prop in the lane 43–60 m
  ahead at the end.
- On those 10, the model's lead head puts something 16–31 m ahead, with probability 0.3–0.8.

The last tick's `model_road` frame tells four of them apart (verified):

| Route | What the frame shows | Reading |
|---|---|---|
| 2847 | an empty junction on a green light | a clear-road trap, labelled correctly |
| 2539 | the scenario's accident, in fog, ~44 m ahead in its lane | stopped short of the obstacle |
| 2709, 28154 | stopped traffic ahead | stopped well behind stopped traffic |

**Decision.**
- `AHEAD_M` stays at 40 m. Raising it would mislabel 2847 and routes like it, which is worse than undercounting.
- The 10 routes are recorded as a known limit in the findings note.
- The record already holds the signal that would separate them: whether a junction on the route lies between the
  car and the actor. `route_plan.json` marks every junction manoeuvre by distance along the route. Using it is a
  primary-cause change, so it goes through step 3. At most these 10 routes would move, from "stopped with a clear
  road" to a blocked or queued cause.

**What it says about the model** (argued). On these routes the model sees the obstacle and plans to stop a long way
before it. The standstill trap (the findings note's §1) then holds it there. A driver would roll up to the obstacle
before deciding how to pass it.

**What a frame-reading pass would and would not have caught.**
- It would have caught the horizon. "An accident ahead in fog" against "clear road" is exactly a disagreement that
  Stage A reports.
- It would not have caught the first cut's 18 hits that were true for the wrong reason. A frame from a car that
  never moved looks like a frame from a car that drove up and stopped.

So the agent gets the tables as well as the frames, and a mode it finds still goes through steps 4 and 5 before it
is counted.

### 3.4 Checklist for a pattern PR

- [ ] The mode is stated as a fact about the record.
- [ ] The test returns evidence with a time.
- [ ] Primary or secondary is decided. For a primary, the commit lists the routes that move.
- [ ] The threshold is named with its reason, and the counts at neighbouring values are given.
- [ ] The hits are read and a sample watched; near misses are checked on both sides; hits true for the wrong
      reason are excluded.
- [ ] The report is regenerated and the findings note updated.
- [ ] The routes that revealed the mode are added to the agent's labelled set, once there is one.
