# Security policy

## Reporting a vulnerability

Please do not open a public issue. Use the private form instead: on this repository's
**Security** tab, choose **Report a vulnerability**. Tell us what you found and how to
reproduce it, and we will reply there.

Fixes go into the latest release only.

## What to know before you run Paddock

- **Model nodes have no authentication.** Each model runs as an HTTP server that listens
  on `127.0.0.1` by default. Run Paddock on a machine you trust, and do not expose those
  ports.
- **Paddock runs other people's code.** `paddock setup` clones upstream repositories and
  runs each model's `setup.sh`, and a run loads that model's code and weights. Read what
  you are about to run, as you would for any research code.
