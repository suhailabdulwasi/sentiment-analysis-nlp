"""Generate later experiment stages from earlier *validation* results."""

from __future__ import annotations

from sentiment.config import ExperimentConfig

LARGE_MODELS = {
    # exact base-model id -> large counterpart from the same pre-training family.
    # Exact keys (not substrings): "roberta-base" is a substring of
    # "cardiffnlp/twitter-roberta-base-sentiment-latest", which previously
    # mapped the Twitter model to general-domain FacebookAI/roberta-large.
    "cardiffnlp/twitter-roberta-base-sentiment-latest": "cardiffnlp/twitter-roberta-large-2022-154m",
    "cardiffnlp/twitter-roberta-base-2022-154m": "cardiffnlp/twitter-roberta-large-2022-154m",
    "vinai/bertweet-base": "vinai/bertweet-large",
    "FacebookAI/roberta-base": "FacebookAI/roberta-large",
    "microsoft/deberta-v3-base": "microsoft/deberta-v3-large",
}


def _short(model_name: str) -> str:
    return model_name.split("/")[-1].replace("-", "_").replace(".", "_")


def _ranked(results: list[dict], stage: str) -> list[dict]:
    rs = [r for r in results if r["config"].get("stage") == stage]
    return sorted(rs, key=lambda r: r["validation"]["macro_f1"], reverse=True)


def lr_sweep(results: list[dict], top_k: int = 2, lrs=(1e-5, 3e-5)) -> list[ExperimentConfig]:
    """Stage 2: alternative learning rates (+ class weighting) for the top-k screened models."""
    out, picked, seen = [], [], set()
    for r in _ranked(results, "screen"):  # top-k *distinct* architectures
        if r["config"]["model_name"] not in seen:
            seen.add(r["config"]["model_name"])
            picked.append(r)
    for r in picked[:top_k]:
        base = {**r["config"], "stage": "lr_sweep", "tags": []}
        for lr in lrs:
            if abs(lr - base["learning_rate"]) < 1e-12:
                continue
            out.append(ExperimentConfig(**{
                **base, "learning_rate": lr,
                "name": f"s2_{_short(base['model_name'])}_lr{lr:.0e}".replace("-0", "-"),
                "hypothesis": f"LR {lr:g} beats {base['learning_rate']:g} for this architecture.",
            }))
        out.append(ExperimentConfig(**{
            **base, "class_weighting": True,
            "name": f"s2_{_short(base['model_name'])}_classweight",
            "hypothesis": "Inverse-frequency class weights help the minority negative class "
                          "(15.5% of train) and raise macro-F1.",
        }))
    return out


def large_models(results: list[dict], top_k: int = 2) -> list[ExperimentConfig]:
    """Stage 3: large counterparts of the best base architectures that have one."""
    pool = _ranked(results, "screen") + _ranked(results, "lr_sweep")
    pool.sort(key=lambda r: r["validation"]["macro_f1"], reverse=True)
    out, seen = [], set()
    for r in pool:
        name = r["config"]["model_name"]
        large = LARGE_MODELS.get(name)
        if not large or large in seen:
            continue
        seen.add(large)
        out.append(ExperimentConfig(
            name=f"s3_{_short(large)}",
            model_name=large,
            learning_rate=1e-5,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=2,
            warmup_ratio=0.1,
            max_length=128,
            class_weighting=r["config"].get("class_weighting", False),
            fp16=True,
            stage="large",
            hypothesis=f"The large version of {name} improves validation macro-F1 by >=1 point.",
        ))
        if len(out) >= top_k:
            break
    return out


def seed_repeats(results: list[dict], seeds=(13, 2024)) -> list[ExperimentConfig]:
    """Stage 4: repeat the single best configuration with extra seeds."""
    eligible = [r for r in results if r["config"].get("stage") != "reproduce"]
    best = max(eligible, key=lambda r: r["validation"]["macro_f1"])
    base = {**best["config"], "stage": "seeds"}
    return [
        ExperimentConfig(**{
            **base, "seed": s, "name": f"s4_{best['name']}_seed{s}",
            "hypothesis": "Seed variance check for the best configuration.",
        })
        for s in seeds
    ]
