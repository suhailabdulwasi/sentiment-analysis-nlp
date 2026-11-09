import json
import os
import sys
import zipfile

import pytest

from sentiment.dashboard import held_out_table, load_experiment_log, load_json

from .conftest import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from ingest_results import ingest  # noqa: E402

FINAL = {
    "tweeteval_test": {
        "n": 12284, "accuracy": 0.73, "accuracy_ci95": [0.72, 0.74], "macro_f1": 0.72,
        "macro_f1_ci95": [0.71, 0.73],
        "baseline": {"accuracy": 0.7041, "macro_f1": 0.7044},
        "paired_vs_baseline_macro_f1": {"diff": 0.0156, "ci95": [0.01, 0.02], "p_not_better": 0.0},
    },
    "gop_conf1": {"n": 3589, "accuracy": 0.7, "accuracy_ci95": [0.68, 0.72], "macro_f1": 0.6,
                  "macro_f1_ci95": [0.58, 0.62]},
}


@pytest.fixture
def report_dir(tmp_path):
    d = tmp_path / "experiments"
    (d / "figures").mkdir(parents=True)
    (d / "experiment_log.csv").write_text(
        "experiment,stage,model,lr,batch,seed,class_weighting,val_accuracy,val_macro_f1,val_macro_recall,train_min,extra\n"
        "a,screen,m1,2e-05,32,42,False,0.75,0.74,0.74,5,x\n"
        "b,screen,m2,2e-05,32,42,False,0.77,0.757,0.76,4,y\n"
    )
    (d / "selection.json").write_text(json.dumps({"members": ["b"], "val_macro_f1": 0.757}))
    (d / "final_results.json").write_text(json.dumps(FINAL))
    return d


def test_load_experiment_log_sorted_and_trimmed(report_dir, tmp_path):
    log = load_experiment_log(report_dir)
    assert log["experiment"].tolist() == ["b", "a"] and "extra" not in log.columns
    assert load_experiment_log(tmp_path / "nope") is None


def test_held_out_table(report_dir):
    t = held_out_table(load_json(report_dir, "final_results.json"))
    assert t["dataset"].tolist() == ["TweetEval test", "GOP election (high-agreement)"]
    row = t.iloc[0]
    assert row["delta_macro_f1"] == pytest.approx(0.0156) and row["delta_ci95"] == "[0.010, 0.020]"
    assert row["macro_f1_ci95"] == "[0.710, 0.730]"


def test_load_json_bad_file(tmp_path):
    (tmp_path / "x.json").write_text("{not json")
    assert load_json(tmp_path, "x.json") is None and load_json(tmp_path, "missing.json") is None


def _zip(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_ingest_extracts_whitelisted_and_skips_predictions(tmp_path):
    z = _zip(tmp_path / "r.zip", {
        "selection.json": json.dumps({"members": ["b"], "val_macro_f1": 0.757}),
        "final_results.json": json.dumps(FINAL),
        "figures/model_comparison.png": b"\x89PNG",
        "runs/b/result.json": "{}",
        "predictions_tweeteval_test.csv": "text,label\n",
        "notes.exe": b"MZ",
    })
    out = tmp_path / "out"
    summary = ingest(z, out)
    assert summary["selected"] == ["b"] and summary["tweeteval_test"]["macro_f1"] == 0.72
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    assert files == ["figures/model_comparison.png", "final_results.json", "runs/b/result.json", "selection.json"]


@pytest.mark.parametrize("bad", ["../evil.json", "/abs/evil.json", "a/../../evil.json"])
def test_ingest_rejects_path_traversal(tmp_path, bad):
    z = _zip(tmp_path / "r.zip", {bad: "{}"})
    with pytest.raises(ValueError, match="unsafe path"):
        ingest(z, tmp_path / "out")
    assert not (tmp_path / "evil.json").exists()


def test_app_comparison_tab(report_dir, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SENTIMENT_FAKE_MODEL", "1")
    monkeypatch.setenv("SENTIMENT_REPORTS_DIR", str(report_dir))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert len(at.tabs) == 4
    assert any("Selected (frozen before testing): b" in s.value for s in at.success)
    assert len(at.dataframe) >= 2


def test_app_comparison_tab_without_results(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SENTIMENT_FAKE_MODEL", "1")
    monkeypatch.setenv("SENTIMENT_REPORTS_DIR", str(tmp_path / "none"))
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("No experiment results found yet" in i.value for i in at.info)
    os.environ.pop("SENTIMENT_REPORTS_DIR", None)
