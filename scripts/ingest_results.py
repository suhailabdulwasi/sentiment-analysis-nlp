"""Import a results_bundle.zip produced by the Colab/Kaggle notebook into reports/experiments/.

Usage: python scripts/ingest_results.py path/to/results_bundle.zip [--out reports/experiments]

The archive is treated as untrusted: only whitelisted file types with safe relative
paths are extracted, and size limits apply. Full prediction CSVs (which contain every
test tweet) are skipped to avoid redistributing the dataset.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path, PurePosixPath

ALLOWED_SUFFIXES = {".json", ".csv", ".md", ".png", ".txt"}
SKIP_PREFIXES = ("predictions_",)
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_TOTAL_BYTES = 200 * 1024 * 1024


def safe_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    out, total = [], 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
            raise ValueError(f"unsafe path in archive: {info.filename!r}")
        if path.suffix.lower() not in ALLOWED_SUFFIXES or path.name.startswith(SKIP_PREFIXES):
            continue
        if info.file_size > MAX_FILE_BYTES:
            raise ValueError(f"file too large: {info.filename} ({info.file_size} bytes)")
        total += info.file_size
        if total > MAX_TOTAL_BYTES:
            raise ValueError("archive exceeds total size limit")
        out.append(info)
    return out


def ingest(zip_path: str | Path, out_dir: str | Path) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in safe_members(zf):
            dest = out / PurePosixPath(info.filename)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(info))
            written.append(info.filename)
    summary = {"files": len(written)}
    sel = out / "selection.json"
    if sel.exists():
        s = json.loads(sel.read_text())
        summary["selected"] = s.get("members")
        summary["val_macro_f1"] = s.get("val_macro_f1")
    fin = out / "final_results.json"
    if fin.exists():
        f = json.loads(fin.read_text())
        for k in ("tweeteval_test", "gop_full_dedup", "gop_conf1"):
            if k in f:
                summary[k] = {"accuracy": round(f[k]["accuracy"], 4), "macro_f1": round(f[k]["macro_f1"], 4)}
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("zip_path")
    p.add_argument("--out", default="reports/experiments")
    a = p.parse_args()
    print(json.dumps(ingest(a.zip_path, a.out), indent=2))
