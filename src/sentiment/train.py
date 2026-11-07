"""Fine-tune a transformer on TweetEval sentiment from an :class:`ExperimentConfig`.

Model selection uses **validation macro-F1 only**. This module never loads
test labels; test evaluation lives in ``sentiment.final_eval`` and is run once,
after the final model has been chosen.

Outputs (``<output_root>/<name>/``):
    result.json      config, environment, timings, validation metrics, log history
    val_probs.npy    softmax probabilities on the validation set (for ensembling)
    best/            best checkpoint (model + tokenizer), unless keep_checkpoint=False
"""

from __future__ import annotations

import gc
import json
import math
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from sentiment.config import ExperimentConfig
from sentiment.data import load_tweeteval_split
from sentiment.labels import ID2LABEL, LABEL2ID
from sentiment.metrics import classification_metrics, expected_calibration_error
from sentiment.preprocess import normalize_tweet


def _environment() -> dict:
    import torch
    import transformers

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        env["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    return env


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    if "bertweet" in model_name.lower():
        # BERTweet ships its own tweet normaliser; needs the slow tokenizer.
        return AutoTokenizer.from_pretrained(model_name, normalization=True, use_fast=False)
    return AutoTokenizer.from_pretrained(model_name)


def load_model_for_training(model_name: str, model_cfg):
    """Load a classification model with **fp32 master weights**.

    transformers>=5 loads weights in the dtype they were saved in. Checkpoints
    stored in fp16 (e.g. microsoft/deberta-v3-base) then make AMP's GradScaler
    fail with "Attempting to unscale FP16 gradients". Mixed precision must keep
    fp32 parameters and only autocast activations, so the dtype is forced here.
    """
    import torch
    from transformers import AutoModelForSequenceClassification

    return AutoModelForSequenceClassification.from_pretrained(
        model_name, config=model_cfg, ignore_mismatched_sizes=True, dtype=torch.float32
    )


def run_experiment(
    cfg: ExperimentConfig,
    data_root: str | Path,
    output_root: str | Path,
    keep_checkpoint: bool = True,
    overwrite: bool = False,
) -> dict:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoConfig,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    out_dir = Path(output_root) / cfg.name
    result_path = out_dir / "result.json"
    if result_path.exists() and not overwrite:
        print(f"[skip] {cfg.name}: result exists")
        return json.loads(result_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    # Intermediate Trainer checkpoints go to fast *local* disk, never to the
    # (Drive-mounted) results folder: on Colab every byte written to Drive is
    # also cached locally and deleted files land in Drive's trash, which filled
    # a 112 GB disk during the large-model stage. Only the final best weights
    # are copied to ``out_dir``.
    ckpt_dir = Path(os.environ.get("SENTIMENT_CKPT_DIR", out_dir.parent / "_local_ckpt")) / cfg.name
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    stale = out_dir / "checkpoints"  # left by older versions / interrupted runs
    shutil.rmtree(stale, ignore_errors=True)
    set_seed(cfg.seed)

    train_df = load_tweeteval_split(data_root, "train")
    val_df = load_tweeteval_split(data_root, "validation")
    if cfg.max_train_samples:
        train_df = train_df.sample(cfg.max_train_samples, random_state=cfg.seed)
    if cfg.normalize_text:
        train_df = train_df.assign(text=train_df["text"].map(normalize_tweet))
        val_df = val_df.assign(text=val_df["text"].map(normalize_tweet))

    tokenizer = _load_tokenizer(cfg.model_name)
    model_cfg = AutoConfig.from_pretrained(
        cfg.model_name, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )
    if cfg.dropout is not None:
        for attr in ("hidden_dropout_prob", "attention_probs_dropout_prob", "dropout"):
            if hasattr(model_cfg, attr):
                setattr(model_cfg, attr, cfg.dropout)
    model = load_model_for_training(cfg.model_name, model_cfg)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

    train_ds = Dataset.from_pandas(train_df, preserve_index=False).map(tok, batched=True)
    val_ds = Dataset.from_pandas(val_df, preserve_index=False).map(tok, batched=True)
    train_ds = train_ds.rename_column("label", "labels").remove_columns(["text"])
    val_ds = val_ds.rename_column("label", "labels").remove_columns(["text"])

    steps_per_epoch = max(1, len(train_ds) // cfg.effective_batch_size)
    eval_steps = max(1, steps_per_epoch // cfg.evals_per_epoch)
    total_steps = int(math.ceil(steps_per_epoch * cfg.num_train_epochs))
    # Explicit integer warmup works on transformers 4.x and 5.x (5.x removed warmup_ratio).
    warmup_steps = int(math.ceil(cfg.warmup_ratio * total_steps))

    class_weights = None
    if cfg.class_weighting:
        counts = np.bincount(train_df["label"], minlength=3).astype(float)
        class_weights = torch.tensor(counts.sum() / (3 * counts), dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            w = class_weights.to(outputs.logits.device) if class_weights is not None else None
            loss = torch.nn.functional.cross_entropy(
                outputs.logits.float(), labels, weight=w, label_smoothing=cfg.label_smoothing
            )
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        m = classification_metrics(labels, np.argmax(logits, axis=1))
        return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "macro_recall": m["macro_recall"]}

    use_fp16 = cfg.fp16 and torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=128,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        weight_decay=cfg.weight_decay,
        warmup_steps=warmup_steps,
        lr_scheduler_type=cfg.lr_scheduler_type,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=1,
        save_only_model=True,  # no optimizer state: ~3x smaller checkpoints
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=max(1, eval_steps // 4),
        fp16=use_fp16,
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_num_workers=2,
        report_to="none",
    )
    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)],
    )

    t0 = time.time()
    trainer.train()
    train_seconds = time.time() - t0

    pred = trainer.predict(val_ds)
    probs = _softmax(pred.predictions.astype(np.float64))
    val_metrics = classification_metrics(val_df["label"].values, probs.argmax(axis=1))
    val_metrics["ece"] = expected_calibration_error(probs, val_df["label"].values)
    np.save(out_dir / "val_probs.npy", probs.astype(np.float32))

    state = trainer.state
    result = {
        "name": cfg.name,
        "config": cfg.to_dict(),
        "environment": _environment(),
        "train_seconds": round(train_seconds, 1),
        "peak_gpu_mem_gb": (
            round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None
        ),
        "best_step": state.best_global_step if hasattr(state, "best_global_step") else None,
        "global_step": state.global_step,
        "epochs_completed": round(state.epoch or 0, 2),
        "best_val_macro_f1_during_training": state.best_metric,
        "validation": val_metrics,
        "log_history": state.log_history,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }

    if keep_checkpoint:
        trainer.save_model(str(out_dir / "best"))
        tokenizer.save_pretrained(str(out_dir / "best"))
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    result_path.write_text(json.dumps(result, indent=2, default=str))

    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    print(
        f"[done] {cfg.name}: val macro-F1={val_metrics['macro_f1']:.4f} "
        f"acc={val_metrics['accuracy']:.4f} ({train_seconds/60:.1f} min)"
    )
    return result
