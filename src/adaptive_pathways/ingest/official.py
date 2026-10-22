# Copyright (c) 2026 Muhammad Abrar Rayhan (https://github.com/Muhammadabrarrayhan2)
# SPDX-License-Identifier: MIT
"""Official MIT App Inventor materials: tutorial index, AI curriculum index, and their starter/solution .aia files.

These pages define the space of "next steps" the recommender may suggest (proposal §3.1), so their
metadata (components, difficulty, grade level) is saved to data/catalog/resources_raw.json.
"""
from __future__ import annotations

import html
import json
import re
from typing import Any
from urllib.parse import urljoin

from ..common import PoliteSession, log
from .store import RawStore

TUT_CARD_RE = re.compile(
    r'<h3 class="card-title">(?P<title>.*?)</h3>.*?Component\(s\):&nbsp;(?P<comps>.*?)</b>.*?'
    r"Difficulty:&nbsp;(?P<diff>.*?)</b>.*?href=\"(?P<href>[^\"]+)\"",
    re.S,
)
AI_CARD_RE = re.compile(r'<div class="col-xl-12 card card-resources">(.*?)More Details</a>', re.S)
AIA_HREF_RE = re.compile(r'href="([^"]+?\.aia)(?:\?[^"]*)?"', re.I)


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def parse_tutorials(page: str, base: str) -> list[dict[str, Any]]:
    out = []
    for m in TUT_CARD_RE.finditer(page):
        out.append(
            {
                "kind": "tutorial",
                "title": _clean(m["title"]),
                "components": [c.strip() for c in _clean(m["comps"]).split(",") if c.strip()],
                "difficulty": _clean(m["diff"]).lower(),
                "url": urljoin(base, m["href"]),
            }
        )
    return out


def parse_ai_resources(page: str, base: str) -> list[dict[str, Any]]:
    out = []
    for block in AI_CARD_RE.findall(page):
        title = re.search(r'card-title[^>]*><a href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not title:
            continue

        def field(name: str) -> str | None:
            m = re.search(rf"<strong>{name}:\s*</strong>(.*?)</p>", block, re.S)
            return _clean(m.group(1)) if m else None

        grades = re.findall(r"<li>(.*?)</li>", block)
        summary = _clean(block.split("</ul>")[-1]) if "</ul>" in block else None
        out.append(
            {
                "kind": "ai_curriculum",
                "title": _clean(title.group(2)),
                "url": urljoin(base, title.group(1)),
                "difficulty": (field("Difficulty") or "").lower() or None,
                "resource_type": field("Resource Type"),
                "subject": field("Subject"),
                "grade_levels": [_clean(g) for g in grades],
                "summary": summary,
            }
        )
    return out


def run(cfg: dict[str, Any], store: RawStore) -> dict[str, int]:
    ocfg = cfg["sources"]["official"]
    base = ocfg["base_url"]
    sess = PoliteSession(cfg)
    fetch = lambda url: sess.get(url).content.decode("utf-8", "replace")  # noqa: E731 — server omits charset
    resources = parse_tutorials(fetch(base + ocfg["tutorial_index"]), base)
    resources += parse_ai_resources(fetch(base + ocfg["ai_index"]), base)
    log.info("official: %d tutorial/curriculum resources indexed", len(resources))

    ok = 0
    for res in resources:
        r = sess.get(res["url"])
        if r.status_code != 200:
            res["aia_urls"] = []
            continue
        page = r.content.decode("utf-8", "replace")
        res["aia_urls"] = sorted({urljoin(res["url"], h) for h in AIA_HREF_RE.findall(page)})
        if res["kind"] == "ai_curriculum" and not res.get("summary"):
            m = re.search(r'<meta name="description" content="([^"]+)"', page)
            res["summary"] = html.unescape(m.group(1)) if m else None
        for url in res["aia_urls"]:
            if url in store.seen_urls:
                continue
            a = sess.get(url)
            if a.status_code != 200:
                store.rejects.append({"url": url, "source": "official", "reason": f"http_{a.status_code}"})
                continue
            sha = store.add(
                a.content,
                source="official",
                url=url,
                meta={
                    "title": url.rsplit("/", 1)[-1][:-4],
                    "public_url": res["url"],
                    "license": "CC BY-SA 4.0 (MIT App Inventor)",
                    "official": {"resource": res["title"], "difficulty": res.get("difficulty")},
                },
            )
            ok += sha is not None

    out = cfg["paths"]["catalog"] / "resources_raw.json"
    out.write_text(json.dumps(resources, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"resources": len(resources), "downloaded": ok}
