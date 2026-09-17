"""Serve a built viewer bundle: play, pause and step through ten scenes.

    python -m tools.benchmarks.comma2k19.viewer.server --scene-set viewer10 --port 8850

Four routes, and deliberately nothing else. The bundle on disk is the whole
state of this thing; the server only hands it over:

    GET  /                    the page
    GET  /api/index           index.json  -- generation, models present
    GET  /api/scene/{slug}    that scene's scene.json
    GET  /media/...           the frames, the BEV panels, the mp4s

Every request **re-reads** the file rather than caching it, because `build.py`
rewrites the bundle while this is up -- once per model, six times over a couple
of hours. A cache here would mean restarting the server to see a model land,
which is exactly the thing the incremental build exists to avoid. The files are
small (a `scene.json` is ~56 KB) and rewritten atomically, so a read racing a
rebuild sees the old bundle or the new one, never half of either.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools import paths

from .build import bundle_root

STATIC = Path(__file__).parent / "static"


def build_app(root: Path):
    """The app serving the bundle at `root`. Takes a path, knows nothing else."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="comma2k19 side-by-side viewer")

    def read(path: Path) -> dict:
        try:
            return json.loads(path.read_text())
        except FileNotFoundError:
            raise HTTPException(404, f"{path.name} is not in the bundle yet; run build.py") from None
        except ValueError as exc:
            raise HTTPException(503, f"{path.name} is being rewritten: {exc}") from None

    @app.get("/")
    async def page() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/index")
    async def index() -> JSONResponse:
        return JSONResponse(read(root / "index.json"))

    @app.get("/api/scene/{slug}")
    async def scene(slug: str) -> JSONResponse:
        # `slug` names a directory in the bundle, so it must not be able to name
        # one outside it: a request for `../../etc` would otherwise be served.
        if slug != Path(slug).name:
            raise HTTPException(400, "slug must name one scene directory")
        return JSONResponse(read(root / slug / "scene.json"))

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    app.mount("/media", StaticFiles(directory=root), name="media")
    return app


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene-set", default="viewer10")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8850)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    import uvicorn

    args = parse_args(argv)
    root = bundle_root(paths.config(), args.scene_set)
    if not (root / "index.json").exists():
        raise SystemExit(
            f"no bundle at {root}\n"
            f"build one first: python -m tools.benchmarks.comma2k19.viewer.build "
            f"--scene-set {args.scene_set}"
        )
    print(f"serving {root}\n  http://{args.host}:{args.port}/")
    uvicorn.run(build_app(root), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
