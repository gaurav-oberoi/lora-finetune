"""LoRA fine-tuning of a small causal LM, on CPU.

LoRA (Low-Rank Adaptation) freezes the base model and trains tiny adapter
matrices on top — a few MB instead of retraining hundreds of millions of
parameters — which is what makes fine-tuning feasible without a GPU. Heavy ML
imports live inside the function so importing this module (or the data tests)
doesn't require torch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class TrainConfig:
    base_model: str = "distilgpt2"
    data_path: str = "data/faq.jsonl"
    output_dir: str = "adapter"
    # This is a memorisation demo on a tiny FAQ, so we deliberately train long
    # enough for the adapter to reproduce the answers (that's the visible effect).
    epochs: float = 50.0
    max_steps: int = -1            # >0 overrides epochs (used by the fast test)
    batch_size: int = 2
    learning_rate: float = 1e-3
    lora_r: int = 16
    lora_alpha: int = 32
    max_length: int = 128


def train(cfg: TrainConfig) -> dict:
    # Imported here so `app.data` and its tests don't pull in torch.
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    from .data import build_texts, load_jsonl

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    records = load_jsonl(cfg.data_path)
    texts = build_texts(records, eos=tokenizer.eos_token)
    ds = Dataset.from_dict({"text": texts}).map(
        lambda b: tokenizer(b["text"], truncation=True, max_length=cfg.max_length),
        batched=True,
        remove_columns=["text"],
    )

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=0.05,
        target_modules=["c_attn"],  # GPT-2 attention projection
    )
    model = get_peft_model(model, lora)
    trainable, total = model.get_nb_trainable_parameters()

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        use_cpu=True,
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    result = trainer.train()

    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save_pretrained(cfg.output_dir)      # saves ONLY the LoRA adapter
    tokenizer.save_pretrained(cfg.output_dir)

    return {
        "adapter_dir": cfg.output_dir,
        "train_loss": round(float(result.training_loss), 4),
        "steps": int(result.global_step),
        "trainable_params": int(trainable),
        "total_params": int(total),
        "trainable_pct": round(100 * trainable / total, 3),
    }


def main() -> None:
    cfg = TrainConfig(
        base_model=os.getenv("BASE_MODEL", "distilgpt2"),
        data_path=os.getenv("DATA_PATH", "data/faq.jsonl"),
        output_dir=os.getenv("OUTPUT_DIR", "adapter"),
    )
    info = train(cfg)
    print(
        f"Done. Trained {info['trainable_params']:,} LoRA params "
        f"({info['trainable_pct']}% of {info['total_params']:,}) "
        f"in {info['steps']} steps; final loss {info['train_loss']}. Adapter saved to {info['adapter_dir']}/"
    )


if __name__ == "__main__":
    main()
