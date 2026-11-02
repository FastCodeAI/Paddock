"""Where the Qwen3-VL processor files are read from. Needs the model's environment
(torch, transformers, alpamayo1_5), so it runs in `make check-alpamayo`."""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="the model's environment supplies torch")
pytest.importorskip("alpamayo1_5", reason="the model's environment supplies alpamayo1_5")

import transformers  # noqa: E402
from alpamayo1_5 import helper  # noqa: E402

from paddock.adapters.models.alpamayo15 import model_loader  # noqa: E402


def test_a_processor_under_weights_is_read_from_there_with_upstreams_resize_limits(
    tmp_path, monkeypatch
) -> None:
    calls = []

    class _Processor:
        tokenizer = None

    def from_pretrained(name, **kwargs):
        calls.append((name, kwargs))
        return _Processor()

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", from_pretrained)
    monkeypatch.setattr(helper, "get_processor", lambda tokenizer: pytest.fail("went to the hub name"))
    processor = model_loader.get_processor("the tokenizer", local=tmp_path)
    assert calls == [(str(tmp_path), {"min_pixels": helper.MIN_PIXELS, "max_pixels": helper.MAX_PIXELS})]
    assert processor.tokenizer == "the tokenizer"


def test_without_one_the_upstream_hub_name_load_is_unchanged(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(helper, "get_processor", lambda tokenizer: ("upstream", tokenizer))
    assert model_loader.get_processor("tok", local=tmp_path / "absent") == ("upstream", "tok")
