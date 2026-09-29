"""Generate text from the base model + a trained LoRA adapter."""

from __future__ import annotations

import os
import sys

from .data import format_prompt


def generate(prompt: str, base_model: str = "distilgpt2", adapter_dir: str = "adapter", max_new_tokens: int = 60) -> str:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir if os.path.isdir(adapter_dir) else base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(base_model)
    if os.path.isdir(adapter_dir):
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    text = format_prompt(prompt)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    # Return just the answer portion.
    return full[len(text):].strip() if full.startswith(text) else full.strip()


def main() -> None:
    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        sys.exit('usage: python -m app.infer "your question"')
    print(generate(prompt, base_model=os.getenv("BASE_MODEL", "distilgpt2"), adapter_dir=os.getenv("OUTPUT_DIR", "adapter")))


if __name__ == "__main__":
    main()
