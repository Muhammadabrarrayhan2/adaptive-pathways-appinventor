"""Stage 4 — cluster skill profiles (proposal §3.2, step 2).

Two simple, well-understood methods on the same profiles:
  * hierarchical clustering (Ward) on log-scaled size features + binary "uses_*" flags, standardised;
  * k-prototypes (mixed numeric/categorical), run over several seeds.
For every k in `k_range` we report silhouette (both methods), k-prototypes seed stability (mean
pairwise ARI across seeds) and hierarchical bootstrap stability (ARI on 80% resamples). The primary
(method, k) maximises silhouette among candidates whose stability clears 0.6 with no cluster smaller
than 3; if none qualifies, the most stable candidate is used and `fallback_used` is set. The choice and
every number behind it are written to data/processed/cluster_selection.json so it can be audited.
"""
from __future__ import annotations

import itertools
import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

NUMERIC = ["n_blocks", "n_components", "n_screens", "n_distinct_component_types", "n_distinct_block_types",
           "max_block_depth", "max_statement_nesting", "n_event_handlers", "n_conditionals", "n_loops",
           "n_procedures", "n_list_blocks"]
FLAGS = ["uses_events", "uses_conditionals", "uses_loops", "uses_procedures", "uses_proc_params", "uses_variables",
         "uses_lists", "uses_multiscreen", "uses_sensors", "uses_timer", "uses_storage", "uses_cloud_web", "uses_media",
         "uses_speech", "uses_animation", "uses_maps", "uses_data_science", "uses_ai"]
STABLE = 0.6


def design_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num = np.log1p(df[NUMERIC].astype(float).to_numpy())
    flags = df[FLAGS].astype(bool).astype(float).to_numpy()
    X = StandardScaler().fit_transform(np.hstack([num, flags]))
    X = np.nan_to_num(X)  # constant columns (e.g. no AI projects yet) standardise to NaN
    return X, num, flags


def ward_labels(X: np.ndarray, k: int) -> np.ndarray:
    return fcluster(linkage(X, method="ward"), k, criterion="maxclust") - 1


def kproto_labels(num: np.ndarray, flags: np.ndarray, k: int, seed: int) -> np.ndarray | None:
    from kmodes.kprototypes import KPrototypes

    data = np.hstack([StandardScaler().fit_transform(num), flags.astype(int)])
    cat_idx = list(range(num.shape[1], data.shape[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Cao init is deterministic but fails when there are fewer distinct flag patterns than k.
        for init in ("Cao", "Huang"):
            try:
                model = KPrototypes(n_clusters=k, init=init, n_init=3, random_state=seed, max_iter=50)
                return model.fit_predict(data, categorical=cat_idx)
            except ValueError:
                continue
    return None


def _safe_sil(X: np.ndarray, labels: np.ndarray) -> float:
    return float(silhouette_score(X, labels)) if 1 < len(set(labels)) < len(labels) else float("nan")


def select_k(X, num, flags, cfg) -> list[dict[str, Any]]:
    ccfg, seed = cfg["clustering"], cfg["seed"]
    rng = np.random.default_rng(seed)
    lo, hi = ccfg["k_range"]
    rows = []
    for k in range(lo, min(hi, len(X) - 1) + 1):
        hier = ward_labels(X, k)
        kp_runs = [r for s in range(ccfg["n_seeds"]) if (r := kproto_labels(num, flags, k, seed + s)) is not None]
        if len(kp_runs) < 2:
            continue  # too few distinct profiles for this many clusters
        boot = []
        for _ in range(ccfg["bootstrap_rounds"]):
            idx = rng.choice(len(X), int(len(X) * ccfg["bootstrap_frac"]), replace=False)
            boot.append(adjusted_rand_score(hier[idx], ward_labels(X[idx], k)))
        rows.append({
            "k": k,
            "silhouette_hier": _safe_sil(X, hier),
            "silhouette_kproto": _safe_sil(X, kp_runs[0]),
            "kproto_seed_stability_ari": float(np.mean(
                [adjusted_rand_score(a, b) for a, b in itertools.combinations(kp_runs, 2)])),
            "hier_bootstrap_stability_ari": float(np.mean(boot)),
            "hier_vs_kproto_ari": float(adjusted_rand_score(hier, kp_runs[0])),
            "min_cluster_size_hier": int(np.bincount(hier).min()),
            "min_cluster_size_kproto": int(np.bincount(kp_runs[0]).min()),
        })
    return rows


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    out = cfg["paths"]["processed"]
    prof = pd.read_csv(out / "profiles.csv")
    df = prof[prof["in_analysis"]].reset_index(drop=True)
    if len(df) < 10:
        raise RuntimeError(f"only {len(df)} analysable projects — run ingest first")
    X, num, flags = design_matrix(df)
    sel = select_k(X, num, flags, cfg)

    cands = [{"method": m, "k": r["k"], "silhouette": r[f"silhouette_{m}"],
              "stability": r["hier_bootstrap_stability_ari" if m == "hier" else "kproto_seed_stability_ari"],
              "min_size": r[f"min_cluster_size_{m}"]} for r in sel for m in ("hier", "kproto")]
    ok = [c for c in cands if c["stability"] >= STABLE and c["min_size"] >= 3]
    best = max(ok, key=lambda c: c["silhouette"]) if ok else max(cands, key=lambda c: c["stability"])
    k = best["k"]
    df["cluster_hier"] = ward_labels(X, k)
    kp = kproto_labels(num, flags, k, cfg["seed"])
    df["cluster_kproto"] = kp if kp is not None else -1
    df["cluster"] = df[f"cluster_{best['method']}"]

    # Exemplars = projects closest to each cluster centroid, for the teacher review of cluster labels.
    exemplars = {}
    for c in sorted(df["cluster"].unique()):
        m = (df["cluster"] == c).to_numpy()
        d = np.linalg.norm(X[m] - X[m].mean(axis=0), axis=1)
        exemplars[int(c)] = df.loc[m, ["project_id", "source", "title"]].iloc[np.argsort(d)[:5]].to_dict("records")

    cols = ["project_id", "cluster", "cluster_hier", "cluster_kproto"]
    df[cols].to_csv(out / "clusters.csv", index=False)
    summary = df.groupby("cluster").agg(
        n=("project_id", "size"), **{f"median_{c}": (c, "median") for c in NUMERIC[:4]},
        **{f"pct_{f}": (f, "mean") for f in FLAGS},
    ).reset_index()
    summary.to_csv(out / "cluster_profiles.csv", index=False)
    (out / "cluster_selection.json").write_text(json.dumps(
        {"chosen_k": k, "primary_method": best["method"], "fallback_used": not ok,
         "chosen": best, "rule": f"max silhouette over (method, k) with stability ARI >= {STABLE} and min cluster size >= 3",
         "n_projects": len(df), "features": {"numeric_log1p": NUMERIC, "binary": FLAGS},
         "candidates": sel, "exemplars": exemplars}, indent=2), encoding="utf-8")
    return {"n": len(df), "method": best["method"], "chosen_k": k, "silhouette": round(best["silhouette"], 3),
            "stability_ari": round(best["stability"], 3), "fallback_used": not ok}
