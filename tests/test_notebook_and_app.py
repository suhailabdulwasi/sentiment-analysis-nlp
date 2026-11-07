import ast
import json
import os
import subprocess
import sys

import pytest

from .conftest import ROOT


def test_notebook_is_current_and_valid():
    """The committed notebook must embed exactly the current source (rebuild if this fails)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_colab_notebook as b

    nb = json.loads((ROOT / "notebooks" / "colab_t4_experiments.ipynb").read_text())
    _, digest = b.bundle()
    src = "".join(c["source"] for c in nb["cells"] if c["cell_type"] == "code")
    assert digest in src, "notebook is stale: run python scripts/build_colab_notebook.py"
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(cell["source"])
    assert nb["metadata"]["colab"]["gpuType"] == "T4"
    # the compressed bundle decodes to exactly the current source files
    import base64
    import hashlib
    import lzma
    import re

    packed = re.search(r'BUNDLE_B64 = "([^"]+)"', src).group(1)
    files = json.loads(lzma.decompress(base64.b64decode(packed)))
    assert files == b.bundle()[0]
    assert hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() == digest
    # final test evaluation must come strictly after selection is frozen
    order = [c["source"] for c in nb["cells"] if c["cell_type"] == "code" and "BUNDLE_B64 = " not in c["source"]]
    i_select = next(i for i, s in enumerate(order) if "select(RUN_ROOT" in s)
    i_final = next(i for i, s in enumerate(order) if "final_evaluation(" in s)
    assert i_select < i_final


def test_streamlit_app_with_fake_backend():
    from streamlit.testing.v1 import AppTest

    os.environ["SENTIMENT_FAKE_MODEL"] = "1"
    try:
        at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        assert not at.exception
        at.text_area[0].input("I love this amazing candidate").run()
        at.button[0].click().run()
        assert not at.exception
        assert any("Positive" in s.value for s in at.success)
        at.text_area[0].input("   ").run()
        at.button[0].click().run()
        assert any("Please enter some text" in w.value for w in at.warning)
        assert len(at.tabs) == 3
    finally:
        os.environ.pop("SENTIMENT_FAKE_MODEL", None)


@pytest.mark.integration
def test_end_to_end_tiny_model(tmp_path):
    """Train a tiny random RoBERTa on 400 tweets, freeze selection, evaluate, check the test lock."""
    script = f"""
import json, pathlib, sys
sys.path.insert(0, {str(ROOT / 'scripts')!r})
from make_tiny_model import build
from sentiment.config import ExperimentConfig
from sentiment.pipeline import run_stage, select, final_evaluation
from sentiment.data import download_tweeteval
T = build({str(tmp_path / 'tiny')!r}, ["the debate was great", "terrible speech", "election in november"] * 30)
D = {str(ROOT / 'data' / 'raw' / 'tweeteval')!r}
R, REP = {str(tmp_path / 'runs')!r}, {str(tmp_path / 'report')!r}
run_stage([ExperimentConfig(name=f"t{{i}}", model_name=str(T), num_train_epochs=1, max_train_samples=400,
                            learning_rate=lr, stage="screen") for i, lr in enumerate([1e-3, 3e-3])], D, R)
sel = select(R, D, REP)
out = final_evaluation(REP, D, None)
assert out["tweeteval_test"]["n"] == 12284
p = pathlib.Path(REP) / "selection.json"; s = json.loads(p.read_text()); s["members"] = ["zzz"]; p.write_text(json.dumps(s))
try:
    final_evaluation(REP, D); raise SystemExit("lock did not trigger")
except RuntimeError:
    pass
print("E2E_OK", sel["members"])
"""
    if not (ROOT / "data" / "raw" / "tweeteval" / "test_text.txt").exists():
        pytest.skip("TweetEval not downloaded (python scripts/audit_data.py downloads it)")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=900)
    assert "E2E_OK" in r.stdout, r.stderr[-3000:]


def test_fp16_checkpoint_loads_with_fp32_master_weights(tmp_path):
    """Regression: DeBERTa-v3 failed with 'Attempting to unscale FP16 gradients'
    because transformers>=5 kept the checkpoint's fp16 dtype."""
    import torch
    from transformers import AutoConfig, RobertaForMaskedLM

    sys.path.insert(0, str(ROOT / "scripts"))
    from make_tiny_model import build

    from sentiment.labels import ID2LABEL, LABEL2ID
    from sentiment.train import load_model_for_training

    path = build(tmp_path / "tiny", ["a b c d"] * 20)
    RobertaForMaskedLM.from_pretrained(path).half().save_pretrained(path)
    cfg = AutoConfig.from_pretrained(path, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID)
    model = load_model_for_training(str(path), cfg)
    assert {p.dtype for p in model.parameters()} == {torch.float32}


@pytest.mark.integration
def test_checkpoints_never_written_to_results_folder(tmp_path, monkeypatch):
    """Regression: intermediate checkpoints on the Drive-mounted results folder filled
    Colab's disk. They must go to SENTIMENT_CKPT_DIR and be deleted after training."""
    data = ROOT / "data" / "raw" / "tweeteval"
    if not (data / "train_text.txt").exists():
        pytest.skip("TweetEval not downloaded")
    sys.path.insert(0, str(ROOT / "scripts"))
    from make_tiny_model import build

    from sentiment.config import ExperimentConfig
    from sentiment.train import run_experiment

    tiny = build(tmp_path / "tiny", ["good bad neutral words"] * 20)
    local = tmp_path / "local_ckpt"
    monkeypatch.setenv("SENTIMENT_CKPT_DIR", str(local))
    runs = tmp_path / "runs"
    (runs / "x" / "checkpoints" / "checkpoint-1").mkdir(parents=True)  # stale leftover
    cfg = ExperimentConfig(name="x", model_name=str(tiny), num_train_epochs=1, max_train_samples=96,
                           learning_rate=1e-3)
    run_experiment(cfg, data, runs)
    produced = sorted(p.name for p in (runs / "x").iterdir())
    assert produced == ["best", "result.json", "val_probs.npy"]
    assert not (local / "x").exists()
    best_files = {p.name for p in (runs / "x" / "best").iterdir()}
    assert "optimizer.pt" not in best_files and "model.safetensors" in best_files
