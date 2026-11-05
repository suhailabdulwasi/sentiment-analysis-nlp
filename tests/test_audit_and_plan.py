import numpy as np
import pandas as pd

from sentiment.audit import cross_split_overlap, overlap_rows, within_split_duplicates
from sentiment.plan import large_models, lr_sweep, seed_repeats
from sentiment.selection import choose_final, comparison_table, to_markdown


def test_within_split_duplicates():
    df = pd.DataFrame({"text": ["a b", "a b", "A B!", "c"], "label": [0, 0, 1, 2]})
    exact = within_split_duplicates(df)
    assert exact["redundant_rows"] == 1 and exact["conflicting_label_groups"] == 0
    norm = within_split_duplicates(df, key="normalized")
    assert norm["redundant_rows"] == 2 and norm["conflicting_label_groups"] == 1


def test_cross_split_overlap():
    tr = pd.DataFrame({"text": ["hello there", "x"], "label": [1, 0]})
    te = pd.DataFrame({"text": ["Hello there!", "new"], "label": [2, 1]})
    assert cross_split_overlap({"train": tr, "test": te})["train->test"]["overlapping_rows_in_test"] == 0
    res = cross_split_overlap({"train": tr, "test": te}, key="normalized")["train->test"]
    assert res["overlapping_rows_in_test"] == 1 and res["same_label"] == 0
    assert overlap_rows(tr, te).tolist() == [True, False]


def _result(name, model, f1, stage="screen", lr=2e-5, seed=42):
    return {
        "name": name, "train_seconds": 60, "epochs_completed": 2, "log_history": [],
        "config": {"name": name, "model_name": model, "learning_rate": lr, "num_train_epochs": 4,
                   "per_device_train_batch_size": 32, "gradient_accumulation_steps": 1, "seed": seed,
                   "stage": stage, "class_weighting": False, "hypothesis": "h"},
        "validation": {"macro_f1": f1, "accuracy": f1, "macro_recall": f1, "ece": 0.05},
    }


RESULTS = [
    _result("s0", "distilbert/distilroberta-base", 0.99, stage="reproduce"),
    _result("a", "vinai/bertweet-base", 0.72),
    _result("a2", "vinai/bertweet-base", 0.71),
    _result("b", "cardiffnlp/twitter-roberta-base-2022-154m", 0.715),
    _result("c", "FacebookAI/roberta-base", 0.70),
]


def test_lr_sweep_uses_distinct_top_architectures():
    names = [c.name for c in lr_sweep(RESULTS)]
    assert names == ["s2_bertweet_base_lr1e-5", "s2_bertweet_base_lr3e-5", "s2_bertweet_base_classweight",
                     "s2_twitter_roberta_base_2022_154m_lr1e-5", "s2_twitter_roberta_base_2022_154m_lr3e-5",
                     "s2_twitter_roberta_base_2022_154m_classweight"]
    assert all(c.stage == "lr_sweep" for c in lr_sweep(RESULTS))


def test_large_models_mapping():
    large = large_models(RESULTS)
    assert [c.model_name for c in large] == ["vinai/bertweet-large", "cardiffnlp/twitter-roberta-large-2022-154m"]
    assert all(c.effective_batch_size == 32 for c in large)


def test_seed_repeats_ignore_reproduction_run():
    seeds = seed_repeats(RESULTS)
    assert [c.seed for c in seeds] == [13, 2024]
    assert all(c.model_name == "vinai/bertweet-base" for c in seeds)


def test_comparison_table_sorted():
    t = comparison_table(RESULTS)
    assert t["experiment"].tolist()[0] == "s0" and t["val_macro_f1"].is_monotonic_decreasing
    md = to_markdown(t.head(2))
    assert md.startswith("| experiment |") and md.count("\n") == 4


def test_choose_final_prefers_simple_unless_clearly_better():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 2000)
    noisy = lambda acc: np.eye(3)[np.where(rng.random(2000) < acc, y, (y + 1) % 3)]  # noqa: E731
    single = {"k": 1, "val_probs": noisy(0.70), "members": ["a"]}
    tiny_gain = {"k": 2, "val_probs": np.where(rng.random((2000, 1)) < 0.999, single["val_probs"], noisy(0.7)),
                 "members": ["a", "b"]}
    big_gain = {"k": 3, "val_probs": noisy(0.80), "members": ["a", "b", "c"]}
    assert choose_final([single, tiny_gain], y)["k"] == 1
    assert choose_final([single, tiny_gain, big_gain], y)["k"] == 3
