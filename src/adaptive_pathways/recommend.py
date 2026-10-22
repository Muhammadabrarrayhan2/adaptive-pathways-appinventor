# Copyright (c) 2026 Muhammad Abrar Rayhan (https://github.com/Muhammadabrarrayhan2)
# SPDX-License-Identifier: MIT
"""Explainable next-step recommender over the pipeline outputs (proposal §3.3).

Given an .aia: parse -> profile -> place in a taxonomy stage -> score catalog resources -> top 3,
each with a templated rationale built only from structural facts and corpus statistics (no generative
text, so every sentence can be traced to a number in the response's `signals`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .common import read_jsonl
from .features import profile
from .parse import parse_one
from .stages import STAGE_BY_ID, assign_stage

CONCEPTS = {  # flag -> label, in rough pedagogical order (earlier = more foundational)
    "uses_events": "event handlers",
    "uses_conditionals": "conditionals (if/else)",
    "uses_variables": "variables",
    "uses_multiscreen": "multiple screens",
    "uses_lists": "Lists",
    "uses_procedures": "procedures",
    "uses_loops": "loops",
    "uses_timer": "a Clock timer",
    "uses_storage": "data storage (TinyDB/File/CloudDB)",
    "uses_proc_params": "procedures with parameters",
    "uses_animation": "Canvas and sprites",
    "uses_media": "camera/sound/video media",
    "uses_sensors": "hardware sensors",
    "uses_maps": "maps and location",
    "uses_cloud_web": "cloud or web data",
    "uses_speech": "speech and translation",
    "uses_data_science": "charts and data science",
    "uses_ai": "an AI component",
    "uses_dictionaries": "dictionaries",
    "uses_generic": "'any component' blocks",
}
ORDER = {f: i for i, f in enumerate(CONCEPTS)}
DIFF_FIT = {1: {"beginner": 1.0, "intermediate": 0.3, "advanced": 0.0},
            2: {"beginner": 0.8, "intermediate": 1.0, "advanced": 0.3},
            3: {"beginner": 0.5, "intermediate": 1.0, "advanced": 0.7},
            4: {"beginner": 0.4, "intermediate": 1.0, "advanced": 1.0},
            5: {"beginner": 0.4, "intermediate": 0.8, "advanced": 1.0}}
SOURCE_NAMES = {"github": "GitHub", "gallery": "the App Inventor Gallery", "official": "an official tutorial",
                "classroom": "a consented classroom project"}
SIZE_COLS = ["n_blocks", "n_components", "n_screens", "n_event_handlers", "n_procedures", "n_list_blocks"]


class Recommender:
    def __init__(self, cfg: dict[str, Any], pseudonymize: Callable[[str], str] | None = None):
        """`pseudonymize` maps a project id to a display pseudonym. When given, peer examples carry only the
        pseudonym and structural facts (no title or URL, which can identify the author)."""
        self.pseudonymize = pseudonymize
        self.resources = json.loads((cfg["paths"]["catalog"] / "resources.json").read_text(encoding="utf-8"))
        prof = pd.read_csv(cfg["paths"]["processed"] / "profiles.csv")
        self.corpus = prof[prof["in_analysis"] == True].copy()  # noqa: E712
        urls = {r["project_id"]: r.get("public_url") for r in read_jsonl(cfg["paths"]["raw"] / "manifest.jsonl")}
        self.corpus["public_url"] = self.corpus["project_id"].map(urls)

    def _similar(self, p: dict[str, Any]) -> pd.DataFrame:
        """Corpus projects of comparable size. Comparing within stages would be circular, because stages
        are defined by some of the same concepts; size-matched peers are the non-circular reference."""
        n = p["n_blocks"]
        return self.corpus[self.corpus["n_blocks"].between(max(1, n // 2), 2 * n + 10)]

    def _peer(self, p: dict[str, Any], stage: int, flag: str, exclude: set[str]) -> dict[str, Any] | None:
        cand = self.corpus[(self.corpus["stage_id"] >= stage) & self.corpus[flag].astype(bool)
                           & self.corpus["public_url"].notna() & ~self.corpus["project_id"].isin(exclude)]
        if cand.empty:
            return None
        me = np.log1p(np.array([p[c] for c in SIZE_COLS], dtype=float))
        dist = np.linalg.norm(np.log1p(cand[SIZE_COLS].to_numpy(dtype=float)) - me, axis=1)
        best = cand.iloc[int(np.argmin(dist))]
        if self.pseudonymize:
            return {"project_id": best["project_id"], "pseudonym": self.pseudonymize(best["project_id"]),
                    "source": best["source"], "license": best.get("license"), "stage": int(best["stage_id"]),
                    "n_blocks": int(best["n_blocks"])}
        return {"project_id": best["project_id"], "title": best["title"], "source": best["source"],
                "public_url": best["public_url"], "stage": int(best["stage_id"])}

    def recommend(self, p: dict[str, Any], k: int = 3) -> dict[str, Any]:
        stage = assign_stage(p)
        used = {f for f in CONCEPTS if p[f]}
        gaps = sorted(set(CONCEPTS) - used, key=ORDER.get)
        similar = self._similar(p)
        peer_rate = {f: float(similar[f].astype(bool).mean()) if len(similar) else 0.0 for f in CONCEPTS}
        scored = []
        for r in self.resources:
            new = sorted(set(r["introduces"]) & set(gaps), key=ORDER.get)
            if not new or set(r["requires"]) - used:
                continue  # nothing new to learn, or prerequisites the learner has not shown yet
            step = r["target_stage"] - stage
            stage_fit = {0: 0.6, 1: 1.0, 2: 0.5}.get(step, 0.1 if step > 2 else 0.3)
            friction = {1: 1.0, 2: 0.7, 3: 0.4}.get(len(new), 0.2)
            lift_flag = max(new, key=peer_rate.get)  # the new concept most common among size-matched peers
            lift = peer_rate[lift_flag]
            familiar = sorted(set(r["components"]) & set(p["_component_types"]))
            foundational = 1.0 - ORDER[new[0]] / len(ORDER)
            score = (2.0 * stage_fit + 1.0 * friction + 1.5 * lift + 0.8 * foundational
                     + 0.3 * min(len(familiar), 2) + 0.7 * DIFF_FIT[stage].get(r["difficulty"], 0.5)
                     + 0.3 * (r["evidence"] == "solution_aia"))
            scored.append((score, r, new, familiar, lift, lift_flag))
        scored.sort(key=lambda t: -t[0])

        picks, seen_primary, peers_used = [], set(), set()
        for score, r, new, familiar, lift, lift_flag in scored:
            if new[0] in seen_primary:
                continue  # diversity: each suggestion opens a different first new concept
            seen_primary.add(new[0])
            peer = self._peer(p, min(stage + 1, 5), new[0], peers_used)
            if peer:
                peers_used.add(peer["project_id"])
            picks.append(self._explain(p, stage, used, gaps, r, new, familiar, lift, lift_flag, score, peer,
                                       len(similar)))
            if len(picks) == k:
                break
        return {"stage": {"id": stage, "name": STAGE_BY_ID[stage]["name"], "rule": STAGE_BY_ID[stage]["rule"]},
                "signals": {"used": sorted(used, key=ORDER.get), "gaps": gaps,
                            **{c: p[c] for c in SIZE_COLS + ["max_block_depth"]}},
                "recommendations": picks}

    def _explain(self, p, stage, used, gaps, r, new, familiar, lift, lift_flag, score, peer, n_similar) -> dict[str, Any]:
        def lbl(fs: list[str], cap: int = 3) -> str:
            names = [CONCEPTS[f] for f in fs]
            if not names:
                return "no additional concepts"
            return ", ".join(names[:cap]) + (f" and {len(names) - cap} more" if len(names) > cap else "")

        shown_used = [f for f in sorted(used, key=ORDER.get)][:4]
        builds_on = (f"builds on the {', '.join(familiar)} component(s) already in the project" if familiar
                     else "builds on the event-driven structure already present" if "uses_events" in used
                     else "starts from the designer screen the learner already has")
        size = "a low-friction" if len(new) == 1 else "a next"
        text = (f"This project has {p['n_blocks']} blocks across {p['n_screens']} screen(s) and uses "
                f"{lbl(shown_used)} (consistent with stage {stage}: {STAGE_BY_ID[stage]['name']}). "
                f"It does not yet use {lbl(gaps[:3])}. '{r['title']}' is suggested as {size} step because it "
                f"{builds_on} and introduces {lbl(new)}.")
        if n_similar >= 5:
            text += f" Among {n_similar} reference projects of similar size, {lift:.0%} already use {CONCEPTS[lift_flag]}."
        if peer and "pseudonym" in peer:
            text += (f" A comparable project at stage {peer['stage']}: {peer['pseudonym']} "
                     f"({peer['n_blocks']} blocks, from {SOURCE_NAMES.get(peer['source'], peer['source'])}).")
        elif peer:
            text += f" A peer example: '{peer['title']}' ({peer['public_url']})."
        return {"resource_id": r["resource_id"], "title": r["title"], "url": r["url"], "kind": r["kind"],
                "difficulty": r["difficulty"], "introduces": new, "target_stage": r["target_stage"],
                "score": round(float(score), 3), "peer_adoption": round(float(lift), 3), "peer_example": peer,
                "rationale": text}


def profile_file(path: Path) -> dict[str, Any]:
    rec = parse_one(str(path))
    p = profile(rec)
    p["_component_types"] = list(rec["component_types"])
    return p


def recommend_file(cfg: dict[str, Any], path: Path, k: int = 3) -> dict[str, Any]:
    out = Recommender(cfg).recommend(profile_file(path), k)
    out["project"] = path.name
    return out


def format_text(res: dict[str, Any]) -> str:
    lines = [f"Project: {res['project']}", f"Stage {res['stage']['id']}: {res['stage']['name']}",
             f"  rule: {res['stage']['rule']}", ""]
    for i, r in enumerate(res["recommendations"], 1):
        lines += [f"{i}. {r['title']}  [{r['kind']}, {r['difficulty']}, score {r['score']}]", f"   {r['url']}",
                  f"   {r['rationale']}", ""]
    return "\n".join(lines)
