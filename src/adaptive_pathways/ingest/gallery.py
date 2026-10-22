"""MIT App Inventor public Gallery: crawl public listings and download each project's .aia.

Anonymous access exposes the featured listings (with sort variants) and per-app permalinks; the
project archive itself is served from the public bucket at gallery/apps/<gid>/aia. Only the public
Gallery URL is retained — no author identity (proposal §3.1).
"""
from __future__ import annotations

import html
import re
from typing import Any

from ..common import PoliteSession, log
from .store import RawStore

CARD_RE = re.compile(r'<div class="card shadow".*?</div>\s*</div>', re.S)
TITLE_RE = re.compile(r'<h4 class="card-title">(.*?)</h4>', re.S)
GID_RE = re.compile(r'id="like_(\d+)">(\d+)</span>')
DATE_RE = re.compile(r"<i>(.*?)</i>&nbsp;Likes")
DESC_RE = re.compile(r'width="300px"/>\s*</p>\s*<p class="card-text">(.*?)</p>', re.S)


def parse_cards(page: str) -> list[dict[str, Any]]:
    out = []
    for card in CARD_RE.findall(page):
        m = GID_RE.search(card)
        if not m:
            continue
        t, d, desc = TITLE_RE.search(card), DATE_RE.search(card), DESC_RE.search(card)
        out.append(
            {
                "gallery_id": m.group(1),
                "likes": int(m.group(2)),
                "title": html.unescape(t.group(1).strip()) if t else None,
                "published": d.group(1).strip() if d else None,
                "description": html.unescape(re.sub(r"<[^>]+>", " ", desc.group(1))).strip() if desc else None,
            }
        )
    return out


def crawl_listing(sess: PoliteSession, base: str, path: str, max_pages: int) -> list[dict[str, Any]]:
    cards, offset = [], 0
    for _ in range(max_pages):
        url = f"{base}{path}" + (f"?offset={offset}" if offset else "")
        r = sess.get(url)
        if r.status_code != 200:
            break
        page_cards = parse_cards(r.text)
        if not page_cards:
            break
        cards += page_cards
        nxt = re.search(r'href="\?offset=(\d+)">Next', r.text)
        if not nxt or int(nxt.group(1)) <= offset:
            break
        offset = int(nxt.group(1))
    return cards


def run(cfg: dict[str, Any], store: RawStore) -> dict[str, int]:
    gcfg = cfg["sources"]["gallery"]
    sess = PoliteSession(cfg)
    apps: dict[str, dict[str, Any]] = {}
    for path in gcfg["listings"]:
        for c in crawl_listing(sess, gcfg["base_url"], path, gcfg["max_pages"]):
            apps.setdefault(c["gallery_id"], c)
    # "Other projects by same author" pages are keyed by gallery id, so no author identity is needed.
    for gid in list(apps):
        for c in crawl_listing(sess, gcfg["base_url"], f"/byowner/{gid}", 5):
            apps.setdefault(c["gallery_id"], c)
    log.info("gallery: %d public apps discovered", len(apps))

    ok = 0
    for gid, meta in apps.items():
        url = gcfg["aia_url"].format(gid=gid)
        if url in store.seen_urls:
            continue
        r = sess.get(url)
        if r.status_code != 200:
            store.rejects.append({"url": url, "source": "gallery", "reason": f"http_{r.status_code}"})
            continue
        sha = store.add(
            r.content,
            source="gallery",
            url=url,
            meta={
                "title": meta["title"],
                "public_url": f"{gcfg['base_url']}/?galleryid={gid}",
                "license": "CC BY-SA 4.0 (App Inventor Gallery terms)",
                "gallery": {k: meta[k] for k in ("likes", "published", "description")},
            },
        )
        ok += sha is not None
    return {"discovered": len(apps), "downloaded": ok}
