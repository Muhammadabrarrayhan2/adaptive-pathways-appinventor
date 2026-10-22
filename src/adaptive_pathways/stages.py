"""Stage 5 — skill-gap taxonomy v0: named stages, project placement, and cluster alignment (proposal §3.2).

The proposal labels data-driven clusters *by hand*, then has two teachers review the labels. This
stage prepares that review instead of replacing it:
  1. every project gets a draft stage from transparent, ordered rules (the highest stage it demonstrates);
  2. each cluster gets the draft stage of the majority of its members, with purity, so the reviewer
     sees where data-driven structure and the draft stage definitions agree or disagree (RQ1 evidence);
  3. taxonomy/stage_labels.yaml is written for the reviewers. If they edit `cluster_to_stage` there,
     the edited mapping is kept on later runs and reported next to the rule-based one.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score

from .common import ROOT

# Ordered from most to least advanced: a project is placed at the first stage whose rule it satisfies.
STAGES: list[dict[str, Any]] = [
    {"id": 5, "key": "ai_component", "name": "AI-component app",
     "rule": "uses an AI component or AI extension (classifier, PoseNet, FaceMesh, ChatBot, ImageBot...)",
     "test": lambda r: r["uses_ai"]},
    {"id": 4, "key": "sensor_aware", "name": "Sensor-aware app",
     "rule": "uses a hardware sensor or location/maps, with event logic reacting to it",
     "test": lambda r: (r["uses_sensors"] or r["uses_maps"]) and r["uses_events"]},
    {"id": 3, "key": "data_aware", "name": "Data-aware app",
     "rule": "stores or structures data: lists/dictionaries, TinyDB/File, CloudDB/Web, or charts",
     "test": lambda r: r["uses_lists"] or r["uses_dictionaries"] or r["uses_storage"] or r["uses_data_science"]},
    {"id": 2, "key": "event_driven", "name": "Event-driven app",
     "rule": "has event handlers plus logic (conditionals/variables), multiple screens, or 15+ blocks",
     "test": lambda r: r["uses_events"] and (r["uses_conditionals"] or r["uses_variables"] or r["uses_multiscreen"]
                                            or r["n_blocks"] >= 15)},
    {"id": 1, "key": "ui_first_app", "name": "UI-only first app",
     "rule": "designer components with few or no blocks and no logic beyond simple event handlers",
     "test": lambda r: True},
]
STAGE_BY_ID = {s["id"]: s for s in STAGES}


def assign_stage(row: dict[str, Any]) -> int:
    for s in STAGES:
        if s["test"](row):
            return s["id"]
    return 1


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    out = cfg["paths"]["processed"]
    prof = pd.read_csv(out / "profiles.csv")
    ok = prof["status"] == "ok"
    prof.loc[ok, "stage_id"] = [assign_stage(r) for r in prof[ok].to_dict("records")]
    prof["stage_name"] = prof["stage_id"].map(lambda i: STAGE_BY_ID[int(i)]["name"] if pd.notna(i) else None)
    prof.to_csv(out / "profiles.csv", index=False)

    cl = pd.read_csv(out / "clusters.csv").merge(prof[["project_id", "stage_id"]], on="project_id")
    alignment = []
    for c, g in cl.groupby("cluster"):
        counts = g["stage_id"].value_counts()
        alignment.append({"cluster": int(c), "n": len(g), "majority_stage": int(counts.index[0]),
                          "majority_stage_name": STAGE_BY_ID[int(counts.index[0])]["name"],
                          "purity": round(float(counts.iloc[0] / len(g)), 3),
                          "stage_mix": {STAGE_BY_ID[int(k)]["key"]: int(v) for k, v in counts.items()}})
    ari = adjusted_rand_score(cl["cluster"], cl["stage_id"])

    tax_dir = ROOT / "taxonomy"
    tax_dir.mkdir(exist_ok=True)
    labels_path = tax_dir / "stage_labels.yaml"
    reviewed = {}
    if labels_path.exists():
        prev = yaml.safe_load(labels_path.read_text(encoding="utf-8")) or {}
        reviewed = prev.get("reviewer_cluster_to_stage") or {}
    labels_path.write_text(yaml.safe_dump({
        "taxonomy_version": "v0-draft (rule-based, awaiting teacher review)",
        "stages": [{k: v for k, v in s.items() if k != "test"} for s in STAGES],
        "rule_cluster_to_stage": {a["cluster"]: a["majority_stage_name"] for a in alignment},
        "reviewer_cluster_to_stage": reviewed,
        "reviewer_notes": "Teachers: set reviewer_cluster_to_stage {cluster: stage name} after inspecting exemplars "
                          "in data/processed/cluster_selection.json. Kept across pipeline runs.",
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (out / "stage_alignment.json").write_text(json.dumps(
        {"ari_clusters_vs_rule_stages": ari, "clusters": alignment}, indent=2), encoding="utf-8")
    dist = prof.loc[prof["in_analysis"] == True, "stage_name"].value_counts().to_dict()  # noqa: E712
    return {"stage_distribution": dist, "ari_clusters_vs_stages": round(ari, 3)}
