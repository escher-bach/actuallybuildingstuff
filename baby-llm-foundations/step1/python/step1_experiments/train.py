"""Transformers Trainer entry point for teacher-conditioned causal training."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import partial
from pathlib import Path

import torch

from .artifacts import atomic_json
from .data import BinaryShard, collate
from .standard_stack import (
    EXPECTED_PARAMETER_COUNT,
    assert_model_contract,
    create_model,
    load_tokenizer,
)


def _training_arguments(config: dict, checkpoint_dir: Path, max_steps: int):
    from transformers import TrainingArguments

    training = config["training"]
    return TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=training["microbatch_sequences"],
        gradient_accumulation_steps=max(1, training["global_tokens_per_update"] // (training["microbatch_sequences"] * config["world"]["context_length"])),
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        lr_scheduler_type="cosine",
        warmup_ratio=training["warmup_fraction"],
        max_steps=max_steps,
        fp16=torch.cuda.is_available(),
        bf16=False,
        save_strategy="steps",
        save_steps=max(1, min(max_steps, training["save_tokens"][0] // max(1, training["global_tokens_per_update"]))),
        save_total_limit=2,
        logging_strategy="steps",
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=config["run"]["root_seed"],
        data_seed=config["run"]["root_seed"],
        average_tokens_across_devices=True,
    )


def _trainer(config: dict, run_dir: Path, max_steps: int):
    from transformers import Trainer

    dataset = BinaryShard(run_dir / "datasets" / "train.bin")
    model = create_model()
    assert_model_contract(model)
    return Trainer(
        model=model,
        args=_training_arguments(config, run_dir / "checkpoints", max_steps),
        train_dataset=dataset,
        data_collator=partial(collate, context=config["world"]["context_length"]),
        processing_class=load_tokenizer(),
    ), dataset


def _model_artifact(run_dir: Path) -> Path:
    return run_dir / "model"


def _save_model_artifact(trainer, run_dir: Path, config: dict) -> Path:
    artifact = _model_artifact(run_dir)
    trainer.save_model(str(artifact))
    load_tokenizer().save_pretrained(artifact)
    atomic_json(artifact / "experiment.json", {
        "initialization": "random_from_local_gpt_neox_config",
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "config_hash": config["_meta"]["hash"],
        "model_config_sha256": hashlib.sha256((artifact / "config.json").read_bytes()).hexdigest(),
        "root_seed": config["run"]["root_seed"],
    })
    return artifact


def _real_batch_preflight(trainer, dataset, config: dict, run_dir: Path) -> None:
    """Exercise a real Rust batch, then prove save/reload logits are stable."""
    batch = collate([dataset[0]], config["world"]["context_length"])
    if not (batch["labels"] == -100).any() or not (batch["labels"] != -100).any():
        raise AssertionError("real batch must contain both context and supervised labels")
    trainer.train()
    artifact = _save_model_artifact(trainer, run_dir, config)
    from transformers import AutoModelForCausalLM
    trainer.model.eval()
    reloaded = AutoModelForCausalLM.from_pretrained(artifact, local_files_only=True).to(trainer.args.device).eval()
    inputs = {name: value.to(trainer.args.device) for name, value in batch.items() if name != "labels"}
    with torch.no_grad():
        expected = trainer.model(**inputs).logits.float().cpu()
        actual = reloaded(**inputs).logits.float().cpu()
    if not torch.allclose(expected, actual, rtol=2e-3 if trainer.args.fp16 else 1e-5, atol=2e-3 if trainer.args.fp16 else 1e-6):
        raise AssertionError("save_pretrained reload changed fixed-batch logits")


def run(resolved_config: Path, run_dir: Path, preflight_only: bool = False, resume: bool = False) -> None:
    config = json.loads(resolved_config.read_text())
    # A one-update preflight is a plumbing diagnostic, not a capability gate.
    target_tokens = config["training"]["global_tokens_per_update"] if preflight_only else config["training"]["token_budget"]
    max_steps = max(1, math.ceil(target_tokens / config["training"]["global_tokens_per_update"]))
    trainer, dataset = _trainer(config, run_dir, max_steps)
    try:
        if preflight_only:
            _real_batch_preflight(trainer, dataset, config, run_dir)
        else:
            from transformers.trainer_utils import get_last_checkpoint
            checkpoint = get_last_checkpoint(str(run_dir / "checkpoints")) if resume else None
            trainer.train(resume_from_checkpoint=checkpoint)
            _save_model_artifact(trainer, run_dir, config)
    finally:
        dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.resolved_config, args.run_dir, args.preflight_only, args.resume)
