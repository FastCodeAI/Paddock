"""Fetch the OpenScene test sensor blobs so NAVSIM can score the full navtest split.

Today this site holds 5 of navtest's 136 logs, so every NAVSIM run scores 242 of
12,146 scenarios and is a subset wearing the split's name. This closes that.

Storage strategy, measured on this machine rather than assumed:

    HF download        ~50 MB/s   <- the bottleneck
    NAS write          ~53 MB/s   (21 T, 13 T free) -> where the blobs live
    NAS small files  ~1035 file/s (~363k JPEGs, so ~6 min of the total)

Two things that are not obvious and cost a wasted download if missed:

* **The metadata tarball is required, not optional.** NAVSIM builds its token
  list from the per-log ``.pkl`` files in ``navsim_logs/``, not from the images.
  With 5 of those present, downloading every image still scores 242 scenarios.
  It is 476 MB against the camera set's 106 GiB, and it is the half that decides
  what gets evaluated.

* **Lidar is not needed.** No model here requests ``lidar_pc`` -- the proxy agent
  builds its SensorConfig from the model's declared cameras -- and the metric
  cache is already built, so the 32 lidar shards are skipped entirely. That is
  half the published download.

Resumable at two levels: ``curl -C -`` continues a partial shard, and a shard
already extracted is skipped on the next run. Peak local disk is one shard
(~3.4 GiB); the tarball is deleted once its contents are on the NAS.

    python -m tools.benchmarks.navsim.fetch_test_blobs --dest <nas>/navsim --dry-run
    python -m tools.benchmarks.navsim.fetch_test_blobs --dest <nas>/navsim

`--dest` is the only site-specific input and is deliberately not defaulted: where
the bulk copy lives differs at every site, and guessing would put 106 GiB
somewhere nobody asked for.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from tools import paths

BASE = "https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main/openscene-v1.1"
METADATA = "openscene_metadata_test.tgz"
CAMERA_SHARDS = 32
#: Inside the tarballs: openscene-v1.1/sensor_blobs/test/<log>/<CAM>/*.jpg
#: Extraction targets <dest>/sensor_blobs/test/, so all three leading components
#: come off and a log directory lands directly under it. Stripping two instead
#: leaves the `test/` component and nests a second one inside.
STRIP = 3


def navtest_logs(split: str = "navtest") -> set[str]:
    """The logs the split declares, from the devkit's own scene filter.

    Filtering to these is what keeps the extraction honest: the test split holds
    more logs than navtest scores, and pulling the rest would inflate the NAS
    copy with data no benchmark reads.
    """
    settings = paths.config().benchmarks.get("navsim")
    devkit = Path(settings.get("devkit")) if settings and settings.get("devkit") else None
    if devkit is None:
        raise SystemExit("[benchmarks.navsim] devkit is not set in paddock.toml")
    path = devkit / "navsim/planning/script/config/common/train_test_split/scene_filter" / f"{split}.yaml"
    if not path.exists():
        raise SystemExit(f"no scene filter at {path}")
    return set(yaml.safe_load(path.read_text()).get("log_names") or [])


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1024**3


def download(url: str, target: Path, echo=print) -> None:
    """Resumable fetch. Leaves the partial file in place if interrupted."""
    target.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = subprocess.run(
        # --no-progress-meter: curl's bar writes thousands of carriage-returned
        # lines per shard, which makes the log unreadable when following a run
        # that lasts hours. The per-shard size and duration below is the signal.
        [
            "curl",
            "-fL",
            "-C",
            "-",
            "--no-progress-meter",
            "--retry",
            "5",
            "--retry-delay",
            "10",
            "-o",
            str(target),
            url,
        ]
    )
    if result.returncode != 0:
        raise SystemExit(f"download failed ({result.returncode}): {url}")
    size = target.stat().st_size / 1024**3
    echo(f"    {size:.2f} GiB in {time.time() - started:.0f}s")


def extract(tarball: Path, dest: Path, wanted: set[str], echo=print) -> int:
    """Extract only *wanted* logs. Returns how many log directories landed.

    Selective by log rather than by camera: NAS space is not scarce here, and a
    camera filter would bake in today's model set -- adding a model that reads
    CAM_L1 would mean re-downloading tarballs this script has deleted.
    """
    listing = subprocess.run(
        ["tar", "-tzf", str(tarball)], capture_output=True, text=True, check=True
    ).stdout.splitlines()

    # Files only -- never the directory entries. Naming a directory makes GNU tar
    # extract that subtree by prefix match, after which every explicitly named
    # file inside it reports "Not found in archive" and tar exits 2, despite
    # having extracted all of them. Directories are created implicitly by the
    # file extraction anyway, so dropping them is both the fix and correct.
    members = [
        line for line in listing if not line.endswith("/") and any(f"/test/{log}/" in line for log in wanted)
    ]
    if not members:
        echo("    no navtest logs in this shard")
        return 0

    logs = {line.split("/test/", 1)[1].split("/", 1)[0] for line in members}
    dest.mkdir(parents=True, exist_ok=True)
    started = time.time()
    # Names go in on stdin, not argv. A shard holds up to ~20k members and the
    # names average ~100 bytes, so passing them as arguments sits right at
    # ARG_MAX (2 MB here): shard 0 fit, shard 3 did not, and the failure is an
    # OSError from exec rather than anything tar can report. --null because the
    # list is machine-generated and a newline in a name would silently split it.
    subprocess.run(
        [
            "tar",
            "-xzf",
            str(tarball),
            "-C",
            str(dest),
            f"--strip-components={STRIP}",
            "--null",
            "--files-from",
            "-",
        ],
        input="\0".join(members).encode() + b"\0",
        check=True,
    )
    # Verify the files landed where the scorer will look, rather than trusting
    # the strip depth. A wrong --strip-components silently nests everything one
    # level deeper and nothing complains until a run scores zero scenarios; this
    # catches it on the first shard instead of the thirty-second.
    landed = [log for log in logs if (dest / log).is_dir()]
    if len(landed) != len(logs):
        stray = sorted({p.name for p in dest.iterdir() if p.is_dir()} - logs)[:3]
        raise SystemExit(
            f"extracted {len(logs)} log(s) but only {len(landed)} are at {dest}/<log>/ -- "
            f"STRIP={STRIP} is wrong for this archive layout" + (f"; found {stray} instead" if stray else "")
        )
    echo(f"    {len(logs)} log(s), {len(members)} entries in {time.time() - started:.0f}s")
    return len(logs)


def verify(blobs: Path, logs_dir: Path, wanted: set[str], echo=print) -> int:
    """Cross-check each log's frames on disk against its metadata. Returns the fault count.

    A shard that extracts partially leaves a log directory that looks fine --
    the right name, the right cameras, just fewer JPEGs -- and NAVSIM only
    notices when a scenario fails deep inside a scoring run hours later. The
    metadata says exactly how many keyframes each log has, so the check is cheap
    and exact.
    """
    import pickle

    faults = 0
    checked = 0
    for log in sorted(wanted):
        directory = blobs / log
        if not directory.is_dir():
            continue
        meta = logs_dir / f"{log}.pkl"
        if not meta.exists():
            echo(f"  {log}: no metadata")
            faults += 1
            continue
        with meta.open("rb") as handle:
            frames = pickle.load(handle)
        expected = len(frames)
        cameras = sorted(p.name for p in directory.iterdir() if p.is_dir())
        checked += 1
        for camera in cameras:
            found = sum(1 for _ in (directory / camera).glob("*.jpg"))
            if found != expected:
                echo(f"  {log}/{camera}: {found} frames, metadata says {expected}")
                faults += 1
    echo(f"verified {checked} log(s), {faults} fault(s)")
    return faults


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, help="NAS root; sensor_blobs/ is created under it")
    ap.add_argument(
        "--stage",
        default=None,
        help="staging dir for tarballs; keep it local. Default is beside the navsim dataset, "
        "because staging on the NAS would write each 3.4 GiB tarball over the network and "
        "then read it straight back to extract -- doubling the slowest link in the chain.",
    )
    ap.add_argument("--split", default="navtest")
    ap.add_argument("--shards", type=int, default=CAMERA_SHARDS)
    ap.add_argument("--skip-metadata", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=200.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true", help="check what is on disk, fetch nothing")
    args = ap.parse_args()

    wanted = navtest_logs(args.split)
    dest = Path(args.dest)
    blobs = dest / "sensor_blobs" / "test"
    stage = Path(args.stage) if args.stage else paths.dataset("navsim").parent / ".openscene-stage"
    logs_dir = paths.dataset("navsim") / "navsim_logs" / "test"

    have = {p.name for p in blobs.iterdir() if p.is_dir()} if blobs.is_dir() else set()
    print(f"{args.split}: {len(wanted)} logs declared, {len(have)} already on the NAS")
    print(f"blobs   -> {blobs}")
    print(f"staging -> {stage}")
    print(
        f"metadata-> {logs_dir}  ({len(list(logs_dir.glob('*.pkl'))) if logs_dir.is_dir() else 0} pkl present)"
    )

    if args.verify:
        raise SystemExit(1 if verify(blobs, logs_dir, wanted) else 0)

    if args.dry_run:
        print(f"\nwould fetch {args.shards} camera shards (~3.4 GiB each) plus metadata")
        print("dry run: nothing downloaded")
        return

    dest.mkdir(parents=True, exist_ok=True)
    if free_gb(dest) < args.min_free_gb:
        raise SystemExit(f"only {free_gb(dest):.0f} GiB free at {dest}; need {args.min_free_gb:.0f}")

    if not args.skip_metadata:
        marker = stage / ".metadata.done"
        if marker.exists():
            print("\nmetadata: already extracted")
        else:
            print(f"\nmetadata: {METADATA}")
            tarball = stage / METADATA
            download(f"{BASE}/{METADATA}", tarball)
            logs_dir.mkdir(parents=True, exist_ok=True)
            listing = subprocess.run(
                ["tar", "-tzf", str(tarball)], capture_output=True, text=True, check=True
            ).stdout.splitlines()
            members = [
                line for line in listing if line.endswith(".pkl") and any(log in line for log in wanted)
            ]
            subprocess.run(
                [
                    "tar",
                    "-xzf",
                    str(tarball),
                    "-C",
                    str(logs_dir),
                    "--strip-components=3",
                    "--null",
                    "--files-from",
                    "-",
                ],
                input="\0".join(members).encode() + b"\0",
                check=True,
            )
            print(f"    {len(members)} log metadata file(s) -> {logs_dir}")
            tarball.unlink()
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()

    total_logs = 0
    for index in range(args.shards):
        name = f"openscene_sensor_test_camera_{index}.tgz"
        marker = stage / f".camera_{index}.done"
        if marker.exists():
            print(f"\n[{index + 1}/{args.shards}] {name}: already extracted")
            continue
        print(f"\n[{index + 1}/{args.shards}] {name}")
        tarball = stage / name
        download(f"{BASE}/openscene_sensor_test_camera/{name}", tarball)
        total_logs += extract(tarball, blobs, wanted)
        tarball.unlink()
        marker.touch()

    have = {p.name for p in blobs.iterdir() if p.is_dir()} if blobs.is_dir() else set()
    missing = wanted - have
    print(f"\n{len(have)}/{len(wanted)} {args.split} logs on the NAS")
    if missing:
        print(f"still missing {len(missing)}: {', '.join(sorted(missing)[:3])}...")
        sys.exit(1)
    print("complete -- point [datasets.navsim] sensor_blobs at this NAS copy, then")
    print("`paddock run --bench navsim --model all` scores the full split.")


if __name__ == "__main__":
    main()
