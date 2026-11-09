"""High-level orchestration used by the Colab notebook (thin cells, testable logic)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sentiment.config import ExperimentConfig
from sentiment.data import GOP_HIGH_AGREEMENT_THRESHOLD, load_gop_debate, load_tweeteval_split
from sentiment.final_eval import (
    apply_temperature,
    check_lock,
    fit_temperature,
    predict_probs,
    save_predictions,
    summarize,
)
from sentiment.metrics import classification_metrics, expected_calibration_error, paired_bootstrap
from sentiment.selection import (
    choose_final,
    comparison_table,
    ensemble_candidates,
    load_results,
    seed_summary,
    to_markdown,
)


def reference_eval(model_ids: list[str], data_root: str | Path, out_path: str | Path) -> pd.DataFrame:
    """Validation-only evaluation of existing checkpoints (no training)."""
    val = load_tweeteval_split(data_root, "validation")
    y = val["label"].values
    rows, store = [], {}
    for mid in model_ids:
        probs = predict_probs([mid], val["text"].tolist())
        m = classification_metrics(y, probs.argmax(1))
        store[mid] = {**m, "ece": expected_calibration_error(probs, y)}
        rows.append({"model": mid, "val_accuracy": round(m["accuracy"], 4),
                     "val_macro_f1": round(m["macro_f1"], 4),
                     "val_macro_recall": round(m["macro_recall"], 4)})
        np.save(Path(out_path).with_name(mid.replace("/", "__") + "_val_probs.npy"), probs)
    Path(out_path).write_text(json.dumps(store, indent=2))
    return pd.DataFrame(rows)


def run_stage(configs: list[ExperimentConfig], data_root, run_root, keep_checkpoints=True) -> list[dict]:
    from sentiment.train import run_experiment

    out = []
    for cfg in configs:
        try:
            out.append(run_experiment(cfg, data_root, run_root, keep_checkpoint=keep_checkpoints))
        except Exception as exc:  # keep the sweep alive; record the failure
            err_dir = Path(run_root) / cfg.name
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}")
            print(f"[FAILED] {cfg.name}: {type(exc).__name__}: {exc}")
    return out


def prune_checkpoints(run_root, keep: int = 4) -> list[str]:
    """Delete 'best/' weights of all but the top-``keep`` runs per val macro-F1 (Drive space)."""
    import shutil

    results = sorted(load_results(run_root), key=lambda r: r["validation"]["macro_f1"], reverse=True)
    keep_models, kept, removed = set(), 0, []
    for r in results:
        d = Path(r["_dir"]) / "best"
        if not d.exists():
            continue
        # keep best run per architecture first, up to `keep`
        if r["config"]["model_name"] not in keep_models and kept < keep:
            keep_models.add(r["config"]["model_name"])
            kept += 1
            continue
        shutil.rmtree(d)
        removed.append(r["name"])
    return removed


def select(run_root, data_root, report_dir, max_members: int = 4) -> dict:
    """Freeze the final selection using validation data only."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    results = [r for r in load_results(run_root) if r["config"].get("stage") != "smoke"]
    table = comparison_table(results)
    table.to_csv(report_dir / "experiment_log.csv", index=False)
    (report_dir / "experiment_log.md").write_text(to_markdown(table.drop(columns=["hypothesis"])))
    seed_summary(table).to_csv(report_dir / "seed_summary.csv", index=False)

    y_val = load_tweeteval_split(data_root, "validation")["label"].values
    cands = ensemble_candidates(results, y_val, max_members=max_members)
    final = choose_final(cands, y_val)
    selection = {
        "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "members": final["members"],
        "member_dirs": final["member_dirs"],
        "val_macro_f1": final["val_macro_f1"],
        "val_accuracy": final["val_accuracy"],
        "candidates": [{k: v for k, v in c.items() if k not in ("val_probs", "member_dirs")} for c in cands],
        "rule": "smallest top-k ensemble unless a larger one gains >=0.5pt val macro-F1 with paired-bootstrap CI > 0",
    }
    if "vs_previous" in final:
        selection["vs_previous"] = final["vs_previous"]
    (report_dir / "selection.json").write_text(json.dumps(selection, indent=2))
    np.save(report_dir / "selected_val_probs.npy", final["val_probs"])
    return selection


def final_evaluation(
    report_dir, data_root, gop_path=None, baseline_model: str | None = None, force: bool = False
) -> dict:
    """Evaluate the frozen selection ONCE on TweetEval test and the GOP benchmark."""
    from sentiment import error_analysis as ea

    report_dir = Path(report_dir)
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sel = json.loads((report_dir / "selection.json").read_text())
    check_lock(report_dir / "test_lock.json", sel["members"], force)
    (report_dir / "test_lock.json").write_text(json.dumps({"members": sel["members"],
                                                           "at": time.strftime("%Y-%m-%d %H:%M:%S")}))
    members = [str(Path(d) / "best") for d in sel["member_dirs"]]

    val = load_tweeteval_split(data_root, "validation")
    val_probs = np.load(report_dir / "selected_val_probs.npy")
    temperature = fit_temperature(val_probs, val["label"].values)

    evals = {"tweeteval_test": load_tweeteval_split(data_root, "test")}
    if gop_path and Path(gop_path).exists():
        gop = load_gop_debate(gop_path)
        evals["gop_full_dedup"] = gop
        evals["gop_conf1"] = gop[gop["confidence"] >= GOP_HIGH_AGREEMENT_THRESHOLD].reset_index(drop=True)

    out = {"selection": sel, "temperature": temperature}
    for name, df in evals.items():
        probs = predict_probs(members, df["text"].tolist())
        y = df["label"].values
        res = summarize(name, y, probs)
        res["ece_after_temperature"] = expected_calibration_error(apply_temperature(probs, temperature), y)
        if baseline_model:
            bprobs = predict_probs([baseline_model], df["text"].tolist())
            res["baseline"] = summarize(name, y, bprobs)
            res["paired_vs_baseline_macro_f1"] = paired_bootstrap(y, bprobs.argmax(1), probs.argmax(1))
            res["paired_vs_baseline_accuracy"] = paired_bootstrap(
                y, bprobs.argmax(1), probs.argmax(1), metric="accuracy")
            ea.plot_confusion(res["baseline"]["confusion_matrix"], f"Original model - {name}",
                              fig_dir / f"confusion_baseline_{name}.png")
        out[name] = res
        save_predictions(report_dir / f"predictions_{name}.csv", df, probs)
        pred_df = df[["text", "label"]].assign(pred=probs.argmax(1), confidence=probs.max(1))
        ea.slice_report(pred_df).to_csv(report_dir / f"slices_{name}.csv", index=False)
        ea.confusion_pairs(pred_df).to_csv(report_dir / f"confusion_pairs_{name}.csv", index=False)
        ea.confident_disagreements(pred_df).to_csv(report_dir / f"confident_errors_{name}.csv", index=False)
        ea.plot_confusion(res["confusion_matrix"], f"Selected model - {name}", fig_dir / f"confusion_{name}.png")
        ea.plot_reliability(probs, y, f"Calibration - {name}", fig_dir / f"reliability_{name}.png")
    if "gop_full_dedup" in evals:
        g = evals["gop_full_dedup"].assign(
            pred=pd.read_csv(report_dir / "predictions_gop_full_dedup.csv")["pred"].values)
        per_cand = (g.groupby("candidate")
                    .apply(lambda s: pd.Series({"n": len(s), "accuracy": (s.label == s.pred).mean()}),
                           include_groups=False)
                    .sort_values("n", ascending=False))
        per_cand.to_csv(report_dir / "gop_per_candidate.csv")
    results = load_results(Path(sel["member_dirs"][0]).parent)
    ea.plot_learning_curves(results, fig_dir / "learning_curves.png")
    ea.plot_model_comparison(comparison_table(results), fig_dir / "model_comparison.png")
    (report_dir / "final_results.json").write_text(json.dumps(out, indent=2, default=float))
    return out
