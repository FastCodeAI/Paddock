# Driving failure videos

The MP4s are **not tracked**. `.gitignore` and CONTRIBUTING both say run
artifacts — predictions, frames, videos — live outside the repo, and this repo
has no LFS; 7.4 MiB of video would sit in every clone and every CI checkout at
both sites forever. What is tracked is [`manifest.json`](manifest.json), which
identifies the files wherever they are.

| Video | Path | Contents |
|---|---|---|
| VisionPilot | `{runs}/bench2drive/evidence/driving-videos/visionpilot-cipo-collision.mp4` | Collision replay: CIPO boxes and recorded trajectory overlay. |
| OpenPilot | `{runs}/bench2drive/evidence/driving-videos/openpilot-converted-comparison.mp4` | Original versus Cosmos Transfer 2.5 base frames, with predicted paths. |

`{runs}` is `[roots.runs]` from your `paddock.toml`; `paddock config` prints the
expansion for this site. `manifest.json` carries the `video_sha256` of both
files, so check a copy before trusting it:

```bash
cd "$(paddock config | awk '$1=="runs" {print $2}')/bench2drive/evidence/driving-videos"
sha256sum -c <(python3 -c "import json;[print(v,' ',k) for k,v in json.load(open('manifest.json'))['video_sha256'].items()]")
```

VisionPilot's detection boxes come from replay; the cyan trajectory is recorded.
OpenPilot shows prediction replay, not a closed-loop rollout or proof of sim2real
causation.
