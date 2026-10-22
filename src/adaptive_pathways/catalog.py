"""Stage 6 — next-step resource catalog (data/catalog/resources.json).

Each official tutorial / AI curriculum unit becomes a candidate next step with:
  * `components`   — canonical component types from the index page;
  * `introduces`   — the skill flags the resource exercises. When the resource ships a starter/solution
                     .aia that the pipeline downloaded and parsed, these come from that project's actual
                     profile ("evidence": "solution_aia"); otherwise from index metadata and keywords;
  * `target_stage` — the taxonomy stage the resource leads to.
Block-reference pages for core concepts (lists, procedures, loops, dictionaries) are added so that
concept gaps not covered by a tutorial still have a next step; every URL is checked at build time.
"""
from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from .common import PoliteSession, log, read_jsonl, short_hash
from .features import FAMILIES
from .stages import STAGE_BY_ID, assign_stage

KNOWN = {c for fam in FAMILIES.values() for c in fam}
ALIASES = {"DrawingCanvas": "Canvas", "Canvas": "Canvas", "Sprite": "ImageSprite", "Accelerometer": "AccelerometerSensor",
           "Orientation": "OrientationSensor", "Gyroscope": "GyroscopeSensor", "Location": "LocationSensor",
           "TextToSpeech": "TextToSpeech", "Speech": "SpeechRecognizer", "Timer": "Clock", "Database": "TinyDB",
           "Sprites": "ImageSprite", "ImageSprites": "ImageSprite", "Video": "VideoPlayer", "FileSharing": "Sharing",
           "ActivityStarter": "ActivityStarter"}
FLAG_FOR_FAMILY = {"sensors": "uses_sensors", "timer": "uses_timer", "storage_local": "uses_storage",
                   "cloud_web": "uses_cloud_web", "media": "uses_media", "speech_translate": "uses_speech",
                   "animation": "uses_animation", "maps": "uses_maps", "data_science": "uses_data_science",
                   "ai_builtin": "uses_ai"}
KEYWORD_FLAGS = [
    (r"classif|machine learning|\bml\b|chatgpt|gpt|imagebot|chatbot|facemesh|posenet|pose|teachable|"
     r"reinforcement|neural|\bai\b|deepfake|alexa|generative|genai|\bbot\b", "uses_ai"),
    (r"voice|speech|speak", "uses_speech"),
    (r"data science|chart|graph|dataset", "uses_data_science"),
    (r"\bmap\b|location|gps", "uses_maps"),
    (r"sensor|accelerometer|shake|tilt|iot", "uses_sensors"),
    (r"cloud|chat app|web\b", "uses_cloud_web"),
    (r"\blist|quiz", "uses_lists"),
]
CORE_PAGES = [
    ("Lists blocks reference", "https://appinventor.mit.edu/explore/ai2/support/blocks/lists", ["uses_lists"], 3, "beginner"),
    ("Procedures blocks reference", "https://appinventor.mit.edu/explore/ai2/support/blocks/procedures",
     ["uses_procedures", "uses_proc_params"], 2, "intermediate"),
    ("Control blocks (loops & conditionals) reference", "https://appinventor.mit.edu/explore/ai2/support/blocks/control",
     ["uses_loops", "uses_conditionals"], 2, "beginner"),
    ("Dictionaries blocks reference", "https://ai2.appinventor.mit.edu/reference/blocks/dictionaries.html",
     ["uses_dictionaries"], 3, "intermediate"),
    ("Variables blocks reference", "https://appinventor.mit.edu/explore/ai2/support/blocks/variables",
     ["uses_variables"], 2, "beginner"),
    ("Building apps with many screens", "https://ai2.appinventor.mit.edu/reference/other/manyscreens.html",
     ["uses_multiscreen"], 2, "intermediate"),
]
PREREQS = {"beginner": [], "intermediate": ["uses_events"], "advanced": ["uses_events", "uses_conditionals", "uses_variables"]}


def canon_component(name: str) -> str | None:
    key = re.sub(r"[^A-Za-z0-9]", "", name)
    if key in KNOWN:
        return key
    return ALIASES.get(key) or next((k for k in KNOWN if k.lower() == key.lower()), None)


def flags_from_components(comps: list[str]) -> set[str]:
    out = set()
    for c in comps:
        for fam, members in FAMILIES.items():
            if c in members and fam in FLAG_FOR_FAMILY:
                out.add(FLAG_FOR_FAMILY[fam])
    return out


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    cat_dir = cfg["paths"]["catalog"]
    raw = json.loads((cat_dir / "resources_raw.json").read_text(encoding="utf-8"))
    manifest = list(read_jsonl(cfg["paths"]["raw"] / "manifest.jsonl"))
    url_to_pid = {u: r["project_id"] for r in manifest for u in r["source_urls"]}
    prof = pd.read_csv(cfg["paths"]["processed"] / "profiles.csv").set_index("project_id")
    flag_cols = [c for c in prof.columns if c.startswith("uses_")]

    resources = []
    for r in raw:
        comps = [c for c in (canon_component(x) for x in r.get("components", [])) if c]
        text = f"{r['title']} {r.get('summary') or ''} {' '.join(r.get('components', []))}".lower()
        intro = flags_from_components(comps)
        if any(re.sub(r"[^a-z]", "", x.lower()) == "anycomponent" for x in r.get("components", [])):
            intro.add("uses_generic")
        intro |= {flag for pat, flag in KEYWORD_FLAGS if re.search(pat, text)}
        if r["kind"] == "ai_curriculum":
            intro.add("uses_ai")
        evidence = "index_metadata"
        pids = [url_to_pid[u] for u in r.get("aia_urls", []) if u in url_to_pid and url_to_pid[u] in prof.index]
        stage = None
        if pids:
            rows = prof.loc[pids]
            aia_flags = {c for c in flag_cols if rows[c].astype(bool).any()}
            intro |= aia_flags - {"uses_events"}
            stage = max(assign_stage(row) for row in rows.to_dict("records"))
            evidence = "solution_aia"
        if stage is None:
            stage = assign_stage({f: f in intro for f in flag_cols} | {"n_blocks": 30, "uses_events": True})
        diff = (r.get("difficulty") or "beginner").split()[0]
        resources.append({
            "resource_id": "R" + short_hash(r["url"], 8), "title": r["title"], "url": r["url"], "kind": r["kind"],
            "difficulty": diff if diff in PREREQS else "beginner", "components": comps,
            "introduces": sorted(intro), "requires": PREREQS.get(diff, []),
            "target_stage": stage, "target_stage_name": STAGE_BY_ID[stage]["name"],
            "evidence": evidence, "solution_project_ids": pids, "grade_levels": r.get("grade_levels"),
        })

    sess = PoliteSession(cfg)
    for title, url, intro, stage, diff in CORE_PAGES:
        status = sess.s.head(url, timeout=cfg["http"]["timeout_s"], allow_redirects=True).status_code
        if status != 200:
            log.warning("catalog: core page %s returned %s - skipped", url, status)
            continue
        resources.append({"resource_id": "R" + short_hash(url, 8), "title": title, "url": url, "kind": "concept_reference",
                          "difficulty": diff, "components": [], "introduces": intro, "requires": PREREQS[diff],
                          "target_stage": stage, "target_stage_name": STAGE_BY_ID[stage]["name"],
                          "evidence": "curated", "solution_project_ids": [], "grade_levels": None})

    (cat_dir / "resources.json").write_text(json.dumps(resources, indent=2, ensure_ascii=False), encoding="utf-8")
    by_stage = pd.Series([r["target_stage_name"] for r in resources]).value_counts().to_dict()
    return {"resources": len(resources), "with_solution_aia": sum(r["evidence"] == "solution_aia" for r in resources),
            "by_target_stage": by_stage}
