"""Dataset loading and prompt formatting — pure Python, no ML deps, so it can be
unit-tested on its own.

Training data is JSONL with {"prompt": ..., "completion": ...}. We render each
pair into a single instruction-style string the causal LM learns to continue.
"""

from __future__ import annotations

import json
from pathlib import Path

PROMPT_TEMPLATE = "### Question:\n{prompt}\n\n### Answer:\n"


def format_prompt(prompt: str) -> str:
    """The exact string shown to the model at inference time (no answer yet)."""
    return PROMPT_TEMPLATE.format(prompt=prompt.strip())


def format_example(prompt: str, completion: str, eos: str = "") -> str:
    """A full training example: prompt template + the target completion (+ EOS)."""
    return format_prompt(prompt) + completion.strip() + eos


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" in obj and "completion" in obj:
                records.append({"prompt": str(obj["prompt"]), "completion": str(obj["completion"])})
    if not records:
        raise ValueError(f"no usable {{prompt, completion}} rows in {path}")
    return records


def build_texts(records: list[dict], eos: str = "") -> list[str]:
    return [format_example(r["prompt"], r["completion"], eos) for r in records]
