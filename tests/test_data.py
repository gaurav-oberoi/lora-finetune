"""Pure tests for data loading/formatting (no torch). The training loop itself
is covered by test_train.py, which self-skips when ML deps aren't installed."""

import pytest

from app.data import build_texts, format_example, format_prompt, load_jsonl


def test_format_prompt_shape():
    p = format_prompt("  What is X?  ")
    assert "### Question:" in p
    assert "### Answer:" in p
    assert "What is X?" in p
    assert p.endswith("### Answer:\n")  # ends ready for the model to continue


def test_format_example_appends_completion_and_eos():
    ex = format_example("Q?", "A.", eos="<|end|>")
    assert ex.startswith("### Question:")
    assert "A." in ex
    assert ex.endswith("<|end|>")


def test_load_jsonl_and_build_texts():
    records = load_jsonl("data/faq.jsonl")
    assert len(records) == 16
    assert all("prompt" in r and "completion" in r for r in records)
    texts = build_texts(records, eos="</s>")
    assert len(texts) == 16
    assert all(t.endswith("</s>") for t in texts)


def test_load_jsonl_rejects_empty(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("\n\n")
    with pytest.raises(ValueError):
        load_jsonl(p)


def test_load_jsonl_skips_malformed_rows(tmp_path):
    p = tmp_path / "mixed.jsonl"
    p.write_text('{"prompt":"a","completion":"b"}\n{"nope":1}\n\n{"prompt":"c","completion":"d"}\n')
    records = load_jsonl(p)
    assert len(records) == 2  # the row missing prompt/completion is skipped
