"""The comma2k19 side-by-side viewer: build a bundle, serve it, drive the runs.

Three modules, in the order they are used:

    pipeline.py   run the models one at a time, rebuilding after each
    build.py      turn the runs on disk into a bundle of frames, BEV panels and JSON
    server.py     serve that bundle with play/pause/step

The split is the same one `paddock render` makes: building knows about models
and benchmarks, serving knows only about a directory of files. So the browser
can be open, and useful, before the first model has finished -- which is the
point of building incrementally rather than once at the end.
"""
