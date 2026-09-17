"""Tests for the dataset survey and the rows `paddock doctor` prints from it.

The fixture is a six-segment dataset whose counts are known by hand, which is
the only way to test a counter: against the real 2035-segment copy every
assertion would either restate the implementation or depend on which chunks
this machine happens to have downloaded.

What is worth testing here is not arithmetic but *what the walk decides to
ignore*. A partial download leaves half-written directories behind, the harness
is right to skip them silently, and a person setting the dataset up needs to be
told about every one -- so the anomaly list, not the segment count, is the part
these tests lean on.
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from paddock.adapters.benchmarks.comma2k19 import inventory as INV
from paddock.adapters.benchmarks.comma2k19 import loader as L
from paddock.adapters.benchmarks.comma2k19.tests.synthetic import straight_pose, write_segment
from paddock.config import Dataset

RAV4_ROUTE = f"{L.RAV4_DONGLE}|2018-07-27--06-03-57"
CIVIC_ROUTE = f"{L.CIVIC_DONGLE}|2018-08-01--09-00-00"
CIVIC_ROUTE_2 = f"{L.CIVIC_DONGLE}|2018-09-02--10-11-12"


def _segment(root, chunk: int, route: str, n: int):
    """One segment complete enough for `loader._is_segment` and the spot check."""
    seg = root / f"Chunk_{chunk}" / route / str(n)
    write_segment(seg, straight_pose(n=120))
    for name in ("video.hevc", "preview.png", "raw_log.bz2"):
        (seg / name).write_bytes(b"")
    return seg


@pytest.fixture
def dataset(tmp_path) -> Dataset:
    """Two chunks, three routes, six segments: 3 rav4 and 3 civic."""
    root = tmp_path / "raw_data"
    for n in (0, 1, 2):
        _segment(root, 1, RAV4_ROUTE, n)
    for n in (0, 1):
        _segment(root, 1, CIVIC_ROUTE, n)
    _segment(root, 2, CIVIC_ROUTE_2, 0)
    return Dataset(name="comma2k19", path=root)


# --------------------------------------------------------------------------
# The walk
# --------------------------------------------------------------------------


def test_survey_counts_segments_routes_chunks_and_cars(dataset):
    found = INV.survey(dataset)
    assert found.n_segments == 6
    assert found.n_routes == 3
    assert found.chunks == {1: 5, 2: 1}
    assert found.vehicles == {"rav4": 3, "civic": 3}
    assert found.hours == pytest.approx(0.1)  # one minute per segment
    assert found.vehicle_summary() == "civic 3, rav4 3"
    assert found.anomalies == []
    assert found.deep == {}


def test_survey_names_every_directory_the_walk_passed_over(dataset, tmp_path):
    """Four ways a partial copy goes wrong, each reported and none fatal."""
    root = tmp_path / "raw_data"
    (root / "Chunk_beta").mkdir()
    (root / "Chunk_1" / "README.txt").write_text("notes")
    (root / "Chunk_1" / "deadbeef-not-a-route").mkdir()
    # A segment whose video never finished downloading: present, unusable.
    (root / "Chunk_1" / RAV4_ROUTE / "2" / "video.hevc").unlink()

    found = INV.survey(dataset)
    assert found.n_segments == 5  # the truncated one is gone from the count
    notes = "\n".join(found.anomalies)
    assert "Chunk_beta: not Chunk_<n>" in notes
    assert "README.txt: file where a route directory goes" in notes
    assert "deadbeef-not-a-route: not a dongle_id|start_time" in notes
    assert f"Chunk_1/{RAV4_ROUTE}: 1 segment(s) without a video.hevc" in notes


def test_survey_keeps_counting_when_a_root_is_not_mounted(tmp_path):
    """The normal state before the NAS is mounted, and not a crash."""
    absent = tmp_path / "not-here"
    found = INV.survey(Dataset(name="comma2k19", path=absent))
    assert found.n_segments == 0
    assert found.anomalies == [f"{absent} does not exist"]


def test_survey_reuses_a_walk_the_caller_already_did(dataset, monkeypatch):
    """`harness.status` passes its own list; a second walk is seconds on a NAS."""
    segments = L.list_segments(dataset)
    monkeypatch.setattr(L, "list_segments", lambda _: pytest.fail("walked twice"))
    assert INV.survey(dataset, segments=segments).n_segments == 6


def test_survey_spot_checks_the_files_scoring_does_not_need(dataset, tmp_path):
    """`preview.png` is optional to the harness, so its absence is a row."""
    victim = tmp_path / "raw_data" / "Chunk_2" / CIVIC_ROUTE_2 / "0"
    (victim / "preview.png").unlink()
    (victim / "raw_log.bz2").unlink()

    found = INV.survey(dataset)
    assert found.checked == 6  # fewer segments than SAMPLE_SEGMENTS: all of them
    assert found.incomplete == [f"2/{CIVIC_ROUTE_2}/0: missing preview.png, raw_log.bz2"]
    assert found.n_segments == 6  # still scoreable, so still counted


def test_an_unrecognised_dongle_is_named_rather_than_dropped(dataset, tmp_path):
    root = tmp_path / "raw_data"
    _segment(root, 2, "0123456789abcdef|2018-10-01--00-00-00", 0)
    found = INV.survey(dataset)
    assert found.vehicles["unknown"] == 1
    assert found.vehicle_summary().endswith("unknown dongle 1")


def test_a_route_renamed_by_a_filesystem_that_rejects_a_pipe_still_parses():
    """Copying through exFAT or S3 turns `|` into `_`; that copy is still fine."""
    stamp = "2018-07-27--06-03-57"
    for sep in ("|", "_"):
        match = INV.ROUTE_RE.match(f"{L.RAV4_DONGLE}{sep}{stamp}")
        assert match is not None, sep
        assert match.group("dongle") == L.RAV4_DONGLE
        assert match.group("start") == stamp
    assert INV.ROUTE_RE.match(f"{L.RAV4_DONGLE}-{stamp}") is None
    assert INV.ROUTE_RE.match(f"nothex0123456789|{stamp}") is None


# --------------------------------------------------------------------------
# Small measurements
# --------------------------------------------------------------------------


def test_rate_hz_measures_delivered_samples_not_the_nominal_rate():
    t = np.arange(0, 10, 0.05)  # 20 Hz for 10 s
    assert INV.rate_hz(t) == pytest.approx(20.0)
    # A gap must move the number: that is the point of the mean over the median.
    dropped = np.delete(t, slice(40, 100))
    assert INV.rate_hz(dropped) < 19.0
    assert np.isnan(INV.rate_hz(np.array([1.0])))
    assert np.isnan(INV.rate_hz(None))
    assert np.isnan(INV.rate_hz(np.zeros(8)))  # no span, no rate


def test_human_reaches_for_the_unit_a_person_would_use():
    assert INV.human(512) == "512 B"
    assert INV.human(1024) == "1.0 KiB"
    assert INV.human(37_400_000) == "35.7 MiB"
    assert INV.human(107_800_000_000) == "100.4 GiB"


def test_png_header_reads_geometry_without_decoding(tmp_path):
    png = pytest.importorskip("PIL.Image")
    path = tmp_path / "preview.png"
    png.new("RGB", (1164, 874)).save(path)
    assert INV.png_header(path) == {"width": 1164, "height": 874, "bit_depth": 8, "color_type": 2}
    (tmp_path / "not.png").write_bytes(b"\x00" * 64)
    assert INV.png_header(tmp_path / "not.png") is None


# --------------------------------------------------------------------------
# Probing video without PyAV
# --------------------------------------------------------------------------


def test_ffprobe_is_looked_for_beside_the_configured_ffmpeg(tmp_path, monkeypatch):
    """A site that set PADDOCK_FFMPEG into another project's venv has it there."""
    binaries = tmp_path / "bin"
    binaries.mkdir()
    for name in ("ffmpeg", "ffprobe"):
        (binaries / name).write_text("")
    monkeypatch.delenv("PADDOCK_FFPROBE", raising=False)
    monkeypatch.setenv("PADDOCK_FFMPEG", str(binaries / "ffmpeg"))
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/ffprobe")
    assert INV._ffprobe() == str(binaries / "ffprobe")

    # Nothing beside it and nothing on PATH: say so rather than guess.
    (binaries / "ffprobe").unlink()
    monkeypatch.setattr(shutil, "which", lambda _: None)
    assert INV._ffprobe() is None


def test_a_missing_probe_is_a_row_in_the_report_not_an_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(INV, "_ffprobe", lambda: None)
    out = INV._probe_with_ffprobe(tmp_path / "video.hevc")
    assert "PADDOCK_FFMPEG" in out["error"] and "pip install av" in out["error"]


def test_a_video_that_cannot_be_probed_is_a_row_too(tmp_path, monkeypatch):
    """A truncated download is the finding, so it must not raise past the report."""
    monkeypatch.setattr(INV, "_ffprobe", lambda: shutil.which("false") or "/bin/false")
    out = INV._probe_with_ffprobe(tmp_path / "missing.hevc")
    assert "error" in out


# --------------------------------------------------------------------------
# The rows doctor prints
# --------------------------------------------------------------------------


def _bench(dataset, tmp_path, **settings):
    from paddock.adapters.benchmarks.comma2k19.harness import Comma2k19Benchmark

    class _Settings:
        def get(self, key, default=None):
            return settings.get(key, default)

    bench = Comma2k19Benchmark.__new__(Comma2k19Benchmark)
    bench.spec = type("S", (), {"name": "comma2k19", "dataset": "comma2k19"})()
    bench.cfg = type(
        "C",
        (),
        {
            "benchmarks": {"comma2k19": _Settings()},
            "datasets": {"comma2k19": dataset},
            # No clone: the toolkit row should say so, not raise.
            "repo": staticmethod(lambda name: tmp_path / "repos" / name),
        },
    )()
    return bench


def test_status_describes_the_copy_on_this_machine(dataset, tmp_path, monkeypatch):
    monkeypatch.delenv("PADDOCK_COMMA2K19_INPUT_HZ", raising=False)
    rows = dict(_bench(dataset, tmp_path).status())
    assert rows["segments"] == "6 (0.1 h) in 3 routes, civic 3, rav4 3"
    assert rows["chunks"] == "2 of 10 present -- 1:5 2:1"
    assert rows["input rate"].startswith("20 Hz native, evaluating at 20 Hz (native)")
    assert rows["integrity"].startswith("6/6 sampled segments hold every optional file")
    # Every row is a fact about the subject; none is a verdict on it.
    assert not any("[" in value for value in rows.values())


def test_status_keeps_its_other_rows_when_the_toolkit_is_missing(dataset, tmp_path):
    """A description must survive the machine it is describing.

    `_bench` points `repo` at an empty directory, which is the state of any site
    that has the dataset but has not cloned the toolkit. The calibration row
    needs the toolkit's pose maths and so cannot be computed -- but it used to
    *raise*, and `doctor` catches that by replacing the whole list with the
    exception, so the reader lost the six rows that were perfectly answerable.
    """
    rows = dict(_bench(dataset, tmp_path).status())
    assert rows["calibration"].startswith("not estimable without the toolkit")
    assert "comma2k19 toolkit not found" in rows["calibration"]
    # The point of the fix: the answerable rows are still there.
    assert rows["segments"] == "6 (0.1 h) in 3 routes, civic 3, rav4 3"
    assert rows["chunks"] == "2 of 10 present -- 1:5 2:1"
    assert rows["integrity"].startswith("6/6 sampled segments")


def test_status_estimates_the_mounting_angle_when_it_can(dataset, tmp_path, rotations):
    """And with the pose maths available, the row is the estimate itself."""
    rows = dict(_bench(dataset, tmp_path).status())
    assert rows["calibration"].startswith("3 routes sampled")


def test_status_reports_a_rate_that_cannot_be_reached_without_raising(dataset, tmp_path):
    """`preflight` owns the refusal; `status` still has to describe the request."""
    rows = dict(_bench(dataset, tmp_path, input_hz="7").status())
    assert rows["input rate"] == "20 Hz native, 7 Hz requested and unreachable -- see preflight"


def test_status_counts_anomalies_it_skipped(dataset, tmp_path):
    (tmp_path / "raw_data" / "Chunk_1" / "stray-directory").mkdir()
    assert "1 directory anomaly skipped" in dict(_bench(dataset, tmp_path).status())["integrity"]


def test_status_is_empty_when_the_site_has_not_configured_the_dataset(tmp_path):
    assert _bench(None, tmp_path).status() == []
