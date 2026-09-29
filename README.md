# lora-finetune

Fine-tune a small language model on your own data with **LoRA** — on a **CPU**, no GPU required.
LoRA (Low-Rank Adaptation) freezes the base model and trains tiny adapter matrices on top, so you
update **~0.4% of the parameters** (a few MB) instead of retraining the whole network. This repo
fine-tunes `distilgpt2` on a small FAQ so the adapter reproduces the answers in a question→answer format.

![lora](https://img.shields.io/badge/PEFT-LoRA-1c3c3c) ![python](https://img.shields.io/badge/python-3.10%2B-blue)

```
$ python -m app.train
Done. Trained 294,912 LoRA params (0.359% of 82,207,488) in 400 steps; final loss 1.11
       Adapter saved to adapter/

$ python -m app.infer "How much does Nimbus Pro cost?"
Nimbus Pro is $9 per member per month and includes unlimited projects.   # learned from the adapter
```

## How it works

- **Data** ([`app/data.py`](app/data.py)) — JSONL `{prompt, completion}` pairs are rendered into a
  consistent instruction template the causal LM learns to continue. Pure, unit-tested.
- **Train** ([`app/train.py`](app/train.py)) — loads the base model, wraps it with a PEFT
  `LoraConfig` (rank-16 adapters on GPT-2's attention projection), tokenises the data, and runs a
  `Trainer` loop on CPU. Saves **only the adapter** (`adapter_config.json` + weights).
- **Infer** ([`app/infer.py`](app/infer.py)) — loads the base model, applies the adapter with
  `PeftModel`, and generates.

> **Note:** `distilgpt2` is a tiny base model chosen so the whole thing runs in seconds on a laptop
> CPU — great for demonstrating the *pipeline*, modest in output quality. Point `BASE_MODEL` at a
> larger model (e.g. a 1–3B instruct model) and add data/steps for real results.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m app.train                              # fine-tune on data/faq.jsonl -> adapter/
python -m app.infer "Does Nimbus offer refunds?"
```

Configure with `BASE_MODEL`, `DATA_PATH`, `OUTPUT_DIR`. Bring your own `{prompt, completion}` JSONL.

## Tests

```bash
pip install -r requirements-dev.txt   # just pytest — for the fast data tests
pytest -q
```

- **Data tests** always run (formatting, JSONL loading, malformed-row handling) — no ML deps.
- **Training integration test** (`tests/test_train.py`) trains one LoRA step on a tiny model, saves,
  reloads the adapter, and generates. It **self-skips** unless `torch`/`peft`/`transformers` are
  installed (so the lightweight CI stays fast); run it locally after `pip install -r requirements.txt`.

## License

MIT
