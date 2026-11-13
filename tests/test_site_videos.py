"""tools/site/videos.py: which runs it hands a benchmark's video hooks."""

from __future__ import annotations

from tools.site import videos


def test_the_floor_row_is_marked_so_a_benchmark_can_leave_it_out_of_the_picking() -> None:
    rows = videos.table_runs("nuscenes")
    floors = [model for model, _, _, floor in rows if floor]
    assert floors == ["constant_velocity"]
    assert all(len(row) == 4 for row in rows)


def test_a_drawn_run_keeps_the_tables_name_and_its_floor_flag() -> None:
    run = videos.Named("constant_velocity", "nuscenes", "r", {}, {}, [], display="Baseline", floor=True)
    assert (run.label, run.floor) == ("Baseline", True)
