"""Fetch KITScenes: full scenes archived to NAS, a fast working subset kept on NVMe.

Storage strategy, measured rather than assumed:

    HF download   ~41 MB/s   <- the bottleneck
    NAS write    ~110 MB/s   (21 T, 16 T free)  -> archive, full scenes, never re-download
    p4 NVMe     ~1.5 GB/s    (384 G, 353 G free) -> working set for the eval

Because the network is 2.7x slower than the NAS, archiving costs no wall-clock time.
The eval reads only 6.4% of a scene, and I/O is under 2% of its runtime, so the p4
copy exists for convenience and future work rather than speed.

Scenes are fetched in priority order: the 127 needed for the 223-keyframe eval first,
so scoring is never blocked behind train-split data it does not use.
"""
from __future__ import annotations

import argparse, os, queue, shutil, subprocess, sys, threading, time
from pathlib import Path

KITSCENES_REPO = Path("/shakaal_hdd_1/workspace/naren/renesas/kitscenes")
DL_PYTHON = Path("/shakaal_hdd_1/workspace/naren/renesas/alpamayo1.5/a1_5_venv/bin/python")

# kept on p4, in priority order; walked until the scene's share of --budget-gb is spent.
# calibration + poses are free and always required.
ESSENTIAL = ["calibration", "poses.txt", "camera_ring_front"]
OPTIONAL  = ["camera_ring_front_left", "camera_ring_front_right", "lidar_top",
             "camera_ring_rear", "camera_ring_rear_left", "camera_ring_rear_right"]
SPLITS = ("val", "overlap_train_val", "train", "test")


def du(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def find_scene(root: Path, sid: str) -> Path | None:
    for sp in SPLITS:
        p = root / "data" / sp / sid
        if p.is_dir():
            return p
    return None


def archive_to_nas(scene: Path, nas_root: Path, split: str, sid: str) -> tuple[bool, float]:
    dest = nas_root / "data" / split / sid
    if dest.is_dir() and du(dest) >= du(scene) * 0.98:
        return True, 0.0            # already archived
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    t0 = time.time()
    try:
        shutil.copytree(scene, tmp)
        shutil.rmtree(dest, ignore_errors=True)
        tmp.rename(dest)            # rename only after a complete copy
        return True, time.time() - t0
    except Exception as e:
        print(f"    NAS archive failed: {type(e).__name__}: {e}", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        return False, time.time() - t0


def prune_to_budget(scene: Path, allowance: int) -> int:
    """Keep ESSENTIAL, then OPTIONAL in order while under `allowance` bytes."""
    keep = set(ESSENTIAL)
    used = sum(du(scene / n) for n in ESSENTIAL if (scene / n).exists())
    for name in OPTIONAL:
        p = scene / name
        if not p.exists():
            continue
        sz = du(p)
        if used + sz <= allowance:
            keep.add(name); used += sz
    for child in scene.iterdir():
        if child.name not in keep:
            shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)
    return used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/p4/naren/kitscenes")
    ap.add_argument("--nas", default="/mnt/nas/naren/kitscenes")
    ap.add_argument("--scenes-file", default="/tmp/todo_scenes.txt")
    ap.add_argument("--budget-gb", type=float, default=200.0, help="p4 working-set cap")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-nas", action="store_true")
    ap.add_argument("--min-free-gb", type=float, default=40.0)
    args = ap.parse_args()

    root, nas = Path(args.root), Path(args.nas)
    root.mkdir(parents=True, exist_ok=True)
    todo = [s.strip() for s in open(args.scenes_file) if s.strip()]
    if args.limit:
        todo = todo[: args.limit]

    log = root / "fetched.txt"
    done = set(log.read_text().split()) if log.exists() else set()
    todo = [s for s in todo if s not in done]
    # per-scene allowance: equal share of the budget (scene sizes vary 1.8-13.6 G,
    # but the working set we keep is dominated by fixed-count sensors, not scene length)
    allowance = int(args.budget_gb * 1e9 / max(len(todo) + len(done), 1))
    print(f"{len(todo)} scenes -> {root}\n  archive: {'off' if args.no_nas else nas}\n"
          f"  p4 budget {args.budget_gb:.0f} G  ({allowance/1e6:.0f} MB/scene)", flush=True)

    # Archive+prune runs on a worker thread so the NAS copy of scene N overlaps the
    # download of N+1. Measured: download ~85 s and archive ~66 s for a median scene,
    # so overlapping cuts ~35% off a multi-hour run. Depth 2 bounds unpruned scenes on
    # p4 to ~27 G worst case, well inside the free-space floor.
    work: queue.Queue = queue.Queue(maxsize=2)
    totals = {"kept": 0, "nas": 0}
    lock = threading.Lock()

    def archiver():
        while True:
            item = work.get()
            if item is None:
                work.task_done(); return
            scene, split, sid, full = item
            nas_s = 0.0
            try:
                if not args.no_nas:
                    okn, nas_s = archive_to_nas(scene, nas, split, sid)
                    if okn:
                        with lock: totals["nas"] += full
                used = prune_to_budget(scene, allowance)   # only after the archive lands
                with lock:
                    totals["kept"] += used
                    done.add(sid); log.write_text("\n".join(sorted(done)))
                print(f"    archived {sid[:8]} {full/1e9:5.1f}G -> p4 {used/1e6:5.0f}M"
                      f"{'' if args.no_nas else f' + nas {nas_s:4.0f}s'} | "
                      f"p4 {totals['kept']/1e9:5.1f}G nas {totals['nas']/1e9:6.1f}G", flush=True)
            except Exception as e:
                print(f"    archiver error on {sid[:8]}: {type(e).__name__}: {e}", flush=True)
            finally:
                work.task_done()

    th = threading.Thread(target=archiver, daemon=True); th.start()
    t_all = time.time(); kept = 0; nas_bytes = 0
    for i, sid in enumerate(todo, 1):
        free = shutil.disk_usage(root).free / 1e9
        if free < args.min_free_gb:
            print(f"STOP: only {free:.0f} G free on p4 (floor {args.min_free_gb:.0f} G)", flush=True)
            break
        scene = find_scene(root, sid)
        if scene is None:
            ok = False
            for attempt in range(3):
                staging = root / f".staging_{sid[:8]}"
                shutil.rmtree(staging, ignore_errors=True)
                env = {**os.environ, "PYTHONPATH": str(KITSCENES_REPO),
                       "KITSCENES_DOWNLOAD_STAGING": str(staging)}
                r = subprocess.run([str(DL_PYTHON), "-m", "kitscenes.download", str(root),
                                    "--scenes", sid], env=env, capture_output=True, text=True)
                shutil.rmtree(staging, ignore_errors=True)
                scene = find_scene(root, sid)      # exit 0 is not proof it arrived
                if scene:
                    ok = True; break
                tail = (r.stderr or r.stdout).strip().splitlines()[-1:] or ["no output"]
                print(f"    attempt {attempt+1} failed: {tail[0][:120]}", flush=True)
            if not ok:
                print(f"[{i}/{len(todo)}] {sid[:8]} FAILED", flush=True); continue

        split = scene.parent.name
        full = du(scene)
        el = time.time() - t_all
        print(f"[{i}/{len(todo)}] {sid[:8]} {split:17} {full/1e9:5.1f}G downloaded | "
              f"free {shutil.disk_usage(root).free/1e9:.0f}G | "
              f"{el/60:.0f}m, eta {(el/i)*(len(todo)-i)/60:.0f}m", flush=True)
        work.put((scene, split, sid, full))     # blocks if the archiver is 2 behind

    work.put(None); work.join(); th.join(timeout=30)
    print(f"\ndone in {(time.time()-t_all)/60:.0f} min — p4 {totals['kept']/1e9:.1f} G, "
          f"nas {totals['nas']/1e9:.1f} G", flush=True)


if __name__ == "__main__":
    main()
