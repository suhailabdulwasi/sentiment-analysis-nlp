"""Typed experiment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    name: str
    model_name: str
    learning_rate: float = 2e-5
    num_train_epochs: float = 4
    per_device_train_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    lr_scheduler_type: str = "linear"
    max_length: int = 128
    seed: int = 42
    fp16: bool = True
    early_stopping_patience: int = 2
    evals_per_epoch: int = 2
    class_weighting: bool = False
    label_smoothing: float = 0.0
    dropout: float | None = None
    normalize_text: bool = True
    max_train_samples: int | None = None  # smoke tests only
    stage: str = "screen"
    hypothesis: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError("name must be non-empty and contain no '/'")
        if not 0 < self.learning_rate < 1e-2:
            raise ValueError(f"learning_rate out of range: {self.learning_rate}")
        if self.num_train_epochs <= 0 or self.per_device_train_batch_size <= 0:
            raise ValueError("epochs and batch size must be positive")
        if not 8 <= self.max_length <= 512:
            raise ValueError("max_length must be in [8, 512]")
        if not 0 <= self.warmup_ratio < 1 or not 0 <= self.label_smoothing < 1:
            raise ValueError("warmup_ratio and label_smoothing must be in [0, 1)")
        if self.dropout is not None and not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiments(path: str | Path) -> list[ExperimentConfig]:
    """Load ``defaults`` + ``experiments`` from a YAML file.

    Unknown keys raise immediately so a typo cannot silently fall back to a default.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or "experiments" not in raw:
        raise ValueError(f"{path}: expected a mapping with an 'experiments' list")
    defaults = raw.get("defaults", {}) or {}
    allowed = {f.name for f in fields(ExperimentConfig)}
    out, seen = [], set()
    for exp in raw["experiments"]:
        merged = {**defaults, **exp}
        unknown = set(merged) - allowed
        if unknown:
            raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
        cfg = ExperimentConfig(**merged)
        if cfg.name in seen:
            raise ValueError(f"{path}: duplicate experiment name {cfg.name}")
        seen.add(cfg.name)
        out.append(cfg)
    return out
