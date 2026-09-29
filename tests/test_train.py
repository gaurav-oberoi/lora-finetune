"""Integration test: actually run one LoRA training step on a tiny model, save
the adapter, reload it, and generate. Self-skips if the ML stack isn't
installed, so it doesn't burden the lightweight CI — run it locally with the
full requirements installed."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("peft")
pytest.importorskip("transformers")


def test_lora_train_save_and_generate(tmp_path):
    from app.infer import generate
    from app.train import TrainConfig, train

    out = str(tmp_path / "adapter")
    info = train(
        TrainConfig(
            base_model="sshleifer/tiny-gpt2",  # tiny test model
            data_path="data/faq.jsonl",
            output_dir=out,
            max_steps=1,
            batch_size=2,
        )
    )

    # LoRA trains only a sliver of the parameters.
    assert info["trainable_params"] > 0
    assert info["trainable_pct"] < 50
    assert info["steps"] == 1

    # The adapter was saved and is loadable for generation.
    import os

    assert os.path.exists(os.path.join(out, "adapter_config.json"))
    text = generate("What is Nimbus?", base_model="sshleifer/tiny-gpt2", adapter_dir=out, max_new_tokens=5)
    assert isinstance(text, str)
