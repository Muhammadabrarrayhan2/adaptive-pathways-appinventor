# Copyright (c) 2026 Muhammad Abrar Rayhan (https://github.com/Muhammadabrarrayhan2)
# SPDX-License-Identifier: MIT
"""Stage 1 — ingestion from all configured sources into the content-addressed raw store."""
from __future__ import annotations

import json
from typing import Any

from ..common import log
from . import classroom, gallery, github, official
from .store import RawStore

# GitHub runs last: it tops the corpus up to `target_corpus_size` after the curated sources.
SOURCES = {"gallery": gallery, "official": official, "classroom": classroom, "github": github}


def enforce_policy(cfg: dict[str, Any], store: RawStore) -> int:
    """Remove stored projects that the current source policy no longer allows, e.g. after a licence
    allow-list change or a source being withdrawn. Runs before every ingest so the store always matches it."""
    allowed = set(cfg["sources"]["github"]["allowed_licenses"])
    drop = [sha for sha, r in store.rows.items()
            if "discovered_via" in r  # discovered through the community forum, whose ToS forbids automated access
            or (r["source"] == "github" and r.get("license") not in allowed)]
    for sha in drop:
        store.remove(sha)
    store.rejects = [r for r in store.rejects if r["source"] != "forum"]
    if drop:
        log.info("policy: removed %d stored projects not permitted by the source policy", len(drop))
    return len(drop)


def run(cfg: dict[str, Any], only: list[str] | None = None) -> dict[str, Any]:
    store = RawStore(cfg)
    summary: dict[str, Any] = {"removed_by_policy": enforce_policy(cfg, store)}
    store.save()
    for name, mod in SOURCES.items():
        if not cfg["sources"][name]["enabled"] or (only and name not in only):
            continue
        log.info("ingest[%s] starting (corpus=%d)", name, len(store))
        try:
            summary[name] = mod.run(cfg, store)
        except Exception as e:  # one failing source must not lose the others' work
            log.exception("ingest[%s] failed", name)
            summary[name] = {"error": repr(e)}
        store.save()
        log.info("ingest[%s] done: %s", name, summary[name])
    summary["corpus_size"] = len(store)
    summary["by_source"] = {s: store.count(s) for s in ("gallery", "official", "classroom", "github")}
    summary["rejected"] = len(store.rejects)
    (cfg["paths"]["raw"] / "ingest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
