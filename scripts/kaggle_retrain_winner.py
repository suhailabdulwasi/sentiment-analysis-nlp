"""Reproduce the selected TweetEval model in Kaggle and export its checkpoint.

Run from Kaggle:
    %run /kaggle/working/sentiment-project/scripts/kaggle_retrain_winner.py

The checkpoint ZIP is saved in /kaggle/working for download. Do not
publish or deploy it until its validation metrics have been inspected.
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path("/kaggle/working/sentiment-project")
WORK = Path("/kaggle/working")
sys.path.insert(0, str(ROOT / "src"))

import torch

from sentiment.config import load_experiments
from sentiment.data import download_tweeteval
from sentiment.train import run_experiment

NAME = "s1_twitter_roberta_sentiment_latest"
CONFIG_FILE = ROOT / "configs" / "experiments.yaml"
DATA_DIR = WORK / "sentiment-retrain-data"
RUN_DIR = WORK / "sentiment-retrain-runs"
ZIP_PATH = WORK / "sentiment-cardiff-finetuned-checkpoint.zip"

if not torch.cuda.is_available():
    raise RuntimeError("GPU unavailable. In Kaggle, select Accelerator: GPU T4 x2.")

configs = {cfg.name: cfg for cfg in load_experiments(CONFIG_FILE)}
cfg = configs[NAME]
if cfg.model_name != "cardiffnlp/twitter-roberta-base-sentiment-latest":
    raise RuntimeError(f"Unexpected base model: {cfg.model_name}")

print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
print(f"Training {cfg.name} using {cfg.model_name}", flush=True)
print(f"Configuration: {json.dumps(cfg.to_dict(), indent=2)}", flush=True)
print("Downloading official TweetEval dataset...", flush=True)
dataset_root = download_tweeteval(DATA_DIR)
print(f"Dataset ready: {dataset_root}", flush=True)

out = RUN_DIR / NAME
if (out / "result.json").exists() and not (out / "best" / "config.json").exists():
    (out / "result.json").unlink()

result = run_experiment(cfg, dataset_root, RUN_DIR, keep_checkpoint=True)
best = out / "best"
if not (best / "config.json").exists():
    raise RuntimeError(f"Trained model not found at {best}")

print("\nValidation results:", json.dumps(result["validation"], indent=2), flush=True)
print("Note: retraining may not exactly reproduce prior results.", flush=True)

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
    for file in sorted(best.rglob("*")):
        if file.is_file():
            archive.write(file, arcname=str(Path("model") / file.relative_to(best)))
    archive.write(out / "result.json", arcname="result.json")
print(f"Checkpoint saved: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB)", flush=True)
print("Do not publish model weights to GitHub; use Hugging Face model hosting.", flush=True)
try:
    from IPython.display import FileLink, display
    display(FileLink(str(ZIP_PATH)))
except ImportError:
    pass
