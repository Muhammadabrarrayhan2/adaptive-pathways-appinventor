"""Stage 7 — data-quality and corpus report (reports/data_report.md + reports/figures/*.png).

Feeds the "structural findings memo" checkpoint (proposal §6, weeks 3–4). Corpus statistics are set
next to the published baseline the proposal cites (Gresse von Wangenheim et al., ~88k Gallery projects)
so differences in sampling are visible rather than hidden.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .common import read_jsonl  # noqa: E402
from .recommend import CONCEPTS  # noqa: E402

INK, MUTED, GRID, SERIES = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"
BASELINE = {  # Gresse von Wangenheim et al., as summarised in the proposal §1.1
    "mean_blocks": 162.5, "pct_under_15_blocks": 0.25, "pct_with_events": 0.90, "pct_with_loops": 0.06,
    "pct_with_button_or_label": 0.85,
}


def _style(ax) -> None:
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def hbar(series: pd.Series, title: str, path, pct: bool = True) -> None:
    series = series.sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(series) + 1.2), dpi=150)
    ax.barh(series.index, series.values, color=SERIES, height=0.62)
    for y, v in enumerate(series.values):
        ax.text(v, y, f" {v:.0%}" if pct else f" {int(v)}", va="center", color=INK, fontsize=8)
    _style(ax)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    if pct:
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_xlim(0, 1.08)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def hist_blocks(blocks: pd.Series, path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=150)
    bins = np.logspace(0, np.log10(max(blocks.max(), 10)), 25)
    ax.hist(blocks.clip(lower=1), bins=bins, color=SERIES, edgecolor="white", linewidth=1)
    ax.set_xscale("log")
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.axvline(15, color=MUTED, linestyle="--", linewidth=1)
    ax.text(15, ax.get_ylim()[1] * 0.92, " 15 blocks", color=MUTED, fontsize=8)
    ax.set_title("Project size (blocks, log scale)", loc="left", color=INK, fontsize=11)
    ax.set_ylabel("projects", color=MUTED)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.astype(object).itertuples(index=False):  # object dtype keeps ints from becoming floats
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    P, R = cfg["paths"]["processed"], cfg["paths"]["reports"]
    fig_dir = R / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    prof = pd.read_csv(P / "profiles.csv")
    parsed = list(read_jsonl(cfg["paths"]["interim"] / "parsed.jsonl"))
    rejects = list(read_jsonl(cfg["paths"]["raw"] / "rejected.jsonl"))
    sel = json.loads((P / "cluster_selection.json").read_text(encoding="utf-8"))
    align = json.loads((P / "stage_alignment.json").read_text(encoding="utf-8"))
    catalog = json.loads((cfg["paths"]["catalog"] / "resources.json").read_text(encoding="utf-8"))
    A = prof[prof["in_analysis"] == True]  # noqa: E712

    at = [r["aiatools"] for r in parsed if r.get("status") == "ok"]
    at_ok = [a for a in at if a.get("ok")]
    parsed_comp = {r["project_id"]: r["component_types"] for r in parsed if r.get("status") == "ok"}
    has_btn_lbl = A["project_id"].map(lambda i: bool({"Button", "Label"} & set(parsed_comp.get(i, {})))).mean()
    ours = {"mean_blocks": A["n_blocks"].mean(), "pct_under_15_blocks": (A["n_blocks"] < 15).mean(),
            "pct_with_events": A["uses_events"].mean(), "pct_with_loops": A["uses_loops"].mean(),
            "pct_with_button_or_label": has_btn_lbl}
    summary = {
        "corpus_total": len(prof), "in_analysis": len(A),
        "by_source_total": prof["source"].value_counts().to_dict(),
        "by_source_in_analysis": A["source"].value_counts().to_dict(),
        "near_duplicates": int(prof["is_near_duplicate"].sum()), "empty": int(prof["is_empty"].sum()),
        "parse_errors": int((prof["status"] != "ok").sum()),
        "rejected_downloads": dict(Counter(r["reason"] for r in rejects)),
        "aiatools_success_rate": len(at_ok) / max(len(at), 1),
        "aiatools_block_agreement": sum(a["blocks_agree"] for a in at_ok) / max(len(at_ok), 1),
        "aiatools_component_agreement": sum(a["components_agree"] for a in at_ok) / max(len(at_ok), 1),
        "legacy_format_projects": sum(1 for r in parsed if r.get("n_legacy_blocks_normalised", 0) > 0),
        "licenses": prof["license"].fillna("unknown").value_counts().head(10).to_dict(),
        "baseline_comparison": {k: {"this_corpus": round(float(ours[k]), 3), "gresse_von_wangenheim": v}
                                for k, v in BASELINE.items()},
        "stage_distribution": A["stage_name"].value_counts().to_dict(),
        "clustering": {k: sel[k] for k in ("chosen_k", "primary_method", "fallback_used", "chosen", "rule", "n_projects")},
        "ari_clusters_vs_rule_stages": align["ari_clusters_vs_rule_stages"],
        "catalog_resources": len(catalog),
    }
    (P / "corpus_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    prev = A[list(CONCEPTS)].astype(bool).mean().rename(index=CONCEPTS)
    hbar(prev, f"Share of projects using each concept (n={len(A)})", fig_dir / "concept_prevalence.png")
    hist_blocks(A["n_blocks"], fig_dir / "block_distribution.png")
    stage_counts = A["stage_name"].value_counts()
    hbar(stage_counts / stage_counts.sum(), "Draft stage placement (taxonomy v0)", fig_dir / "stage_distribution.png")

    cand = pd.DataFrame(sel["candidates"])
    base_df = pd.DataFrame([{"metric": k, "this corpus": round(float(ours[k]), 3), "Gresse von Wangenheim et al.": v}
                            for k, v in BASELINE.items()])
    align_df = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in a.items()}
                             for a in align["clusters"]])
    src_df = pd.DataFrame({"total": prof["source"].value_counts(), "in_analysis": A["source"].value_counts()}).fillna(0).astype(int)
    src_df.insert(0, "source", src_df.index)

    md = f"""# Adaptive Pathways — data report

_Generated by `python run_pipeline.py report`. All numbers below are recomputed from `data/` on every run._

## 1. Corpus

{md_table(src_df)}

* Projects stored: **{len(prof)}**; used in analysis: **{len(A)}**
  (excluded: {summary['near_duplicates']} structural near-duplicates, {summary['empty']} empty, {summary['parse_errors']} parse errors).
* Downloads rejected before storage: `{summary['rejected_downloads']}`
  (`not_app_inventor` = other software's `.aia` files, e.g. Adobe Illustrator actions).
* Licences (top): `{summary['licenses']}`

## 2. Parser quality

* aiatools parsed **{summary['aiatools_success_rate']:.1%}** of projects; the rest used the direct XML/JSON reader.
* Agreement between aiatools and the direct reader: blocks **{summary['aiatools_block_agreement']:.1%}**,
  components **{summary['aiatools_component_agreement']:.1%}**.
* {summary['legacy_format_projects']} projects use the pre-2013 block format (`Button1_Click`); they are normalised to the modern vocabulary.
* Known aiatools issue handled: project-level `AIAFile.blocks()` returns only one screen's blocks in multi-screen
  projects; the pipeline aggregates `screen.blocks()` per screen instead.

## 3. Comparison with the published Gallery baseline

{md_table(base_df)}

Differences are expected: this corpus over-samples curated sources (official tutorials, featured Gallery apps,
GitHub repositories), so it skews larger and more structured than the full Gallery.

![block distribution](figures/block_distribution.png)

![concept prevalence](figures/concept_prevalence.png)

## 4. Clustering (RQ1)

Primary solution: **{sel['primary_method']}, k = {sel['chosen_k']}** (silhouette {sel['chosen']['silhouette']:.3f},
stability ARI {sel['chosen']['stability']:.3f}) — rule: {sel['rule']}.
{"**No candidate met the stability threshold; the most stable one is used — treat the clusters as tentative.**" if sel['fallback_used'] else ""}
Silhouette values around 0.1 indicate overlapping rather than sharply separated groups: learner projects form a
continuum, which is consistent with treating stages as a progression rather than discrete types.

{md_table(cand.round(3))}

## 5. Draft stages vs. data-driven clusters

ARI between the primary clusters and rule-based draft stages: **{align['ari_clusters_vs_rule_stages']:.3f}**
(1 = identical partitions, 0 = chance). Low purity clusters are where the teacher review should focus.

{md_table(align_df)}

![stage distribution](figures/stage_distribution.png)

## 6. Next-step catalog

{len(catalog)} resources ({sum(r['evidence'] == 'solution_aia' for r in catalog)} with concepts measured from their own solution `.aia`).

## 7. Limitations

* Gallery listings are only partly reachable without login (featured apps); Gallery projects are collected
  only through public listings. GitHub repositories (open licences only) over-represent classroom and tutorial work.
* Stage labels are a rule-based draft (taxonomy v0) awaiting the two-teacher review in `taxonomy/stage_labels.yaml`.
* No telemetry, no learner outcomes: everything here is structural and descriptive.
"""
    (R / "data_report.md").write_text(md, encoding="utf-8")
    return {"report": str(R / "data_report.md"), "in_analysis": len(A)}
