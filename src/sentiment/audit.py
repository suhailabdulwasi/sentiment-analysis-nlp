"""Duplicate, conflicting-label and cross-split leakage checks."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from sentiment.preprocess import dedup_key


def within_split_duplicates(df: pd.DataFrame, key: str = "exact") -> dict:
    """Count duplicate texts inside one split and how many carry conflicting labels."""
    k = df["text"] if key == "exact" else df["text"].map(dedup_key)
    k = k[k.str.len() > 0]
    dup_mask = k.duplicated(keep=False)
    groups = df.loc[k.index[dup_mask]].assign(_k=k[dup_mask]).groupby("_k")["label"]
    n_conflict_groups = int((groups.nunique() > 1).sum())
    return {
        "rows": int(len(df)),
        "rows_in_duplicate_groups": int(dup_mask.sum()),
        "redundant_rows": int(k.duplicated().sum()),
        "duplicate_groups": int(groups.ngroups),
        "conflicting_label_groups": n_conflict_groups,
    }


def cross_split_overlap(splits: Mapping[str, pd.DataFrame], key: str = "exact") -> dict:
    """For each pair of splits, count rows of the second whose text appears in the first."""
    keys = {
        name: (df["text"] if key == "exact" else df["text"].map(dedup_key))
        for name, df in splits.items()
    }
    names = list(splits)
    out: dict[str, dict] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ka = set(keys[a][keys[a].str.len() > 0])
            mask = keys[b].isin(ka) & (keys[b].str.len() > 0)
            overlap_rows = splits[b][mask]
            # label agreement for overlapping rows (first label seen in split a)
            first_label = (
                splits[a].assign(_k=keys[a]).drop_duplicates("_k").set_index("_k")["label"]
            )
            agree = (overlap_rows["label"].values == first_label.loc[keys[b][mask]].values).sum()
            out[f"{a}->{b}"] = {
                "overlapping_rows_in_" + b: int(mask.sum()),
                "fraction_of_" + b: round(float(mask.mean()), 5),
                "same_label": int(agree),
            }
    return out


def overlap_rows(reference: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """Boolean mask over ``target`` rows whose dedup_key occurs in ``reference``."""
    ref = set(reference["text"].map(dedup_key))
    tk = target["text"].map(dedup_key)
    return tk.isin(ref) & (tk.str.len() > 0)
