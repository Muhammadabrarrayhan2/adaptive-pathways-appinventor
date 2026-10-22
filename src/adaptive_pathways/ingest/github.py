"""Public GitHub repositories containing App Inventor .aia projects.

GitHub code search does not index binary files, so discovery is two-step: repository search over
App Inventor-related queries, then a recursive git-tree listing per repo to find *.aia files.
Only repositories with an explicitly open licence (`allowed_licenses`) are used. Forks are skipped and each repo contributes at most `max_aia_per_repo` projects (seeded sample),
so one course repo cannot dominate the corpus.
"""
from __future__ import annotations

import os
import random
import subprocess
from typing import Any
from urllib.parse import quote

from ..common import PoliteSession, log
from .store import RawStore

API = "https://api.github.com"


def _token() -> str | None:
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok
    try:
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=15).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def search_repos(sess: PoliteSession, queries: list[str], max_repos: int) -> list[dict[str, Any]]:
    repos: dict[str, dict[str, Any]] = {}
    for q in queries:
        for page in range(1, 11):
            r = sess.get(f"{API}/search/repositories", params={"q": q, "per_page": 100, "page": page})
            if r.status_code != 200:
                log.warning("github search %r page %d -> %s", q, page, r.status_code)
                break
            items = r.json().get("items", [])
            for it in items:
                if not it["fork"]:
                    repos.setdefault(it["full_name"], it)
            if len(items) < 100 or len(repos) >= max_repos * 3:
                break
    log.info("github: %d candidate repositories from %d queries", len(repos), len(queries))
    return list(repos.values())


def run(cfg: dict[str, Any], store: RawStore) -> dict[str, int]:
    gcfg = cfg["sources"]["github"]
    tok = _token()
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    else:
        log.warning("github: no token found (GITHUB_TOKEN or `gh auth login`); unauthenticated limits are tight")
    api = PoliteSession(cfg, headers)
    raw = PoliteSession(cfg)
    rng = random.Random(cfg["seed"])
    target = cfg["limits"]["target_corpus_size"]

    allowed = set(gcfg["allowed_licenses"])
    repos = [r for r in search_repos(api, gcfg["queries"], gcfg["max_repos"])
             if ((r.get("license") or {}).get("spdx_id")) in allowed]
    log.info("github: %d repositories carry an allowed open licence", len(repos))
    scanned = with_aia = ok = 0
    for repo in repos:
        if len(store) >= target or scanned >= gcfg["max_repos"]:
            break
        scanned += 1
        full, branch = repo["full_name"], repo["default_branch"]
        r = api.get(f"{API}/repos/{full}/git/trees/{quote(branch, safe='')}", params={"recursive": "1"})
        if r.status_code != 200:
            continue
        paths = [
            t["path"]
            for t in r.json().get("tree", [])
            if t["type"] == "blob" and t["path"].lower().endswith(".aia") and t.get("size", 0) <= cfg["limits"]["max_aia_bytes"]
        ]
        if not paths:
            continue
        with_aia += 1
        paths.sort()
        for path in rng.sample(paths, min(len(paths), gcfg["max_aia_per_repo"])):
            url = f"https://raw.githubusercontent.com/{full}/{quote(branch)}/{quote(path)}"
            if url in store.seen_urls:
                continue
            a = raw.get(url)
            if a.status_code != 200:
                store.rejects.append({"url": url, "source": "github", "reason": f"http_{a.status_code}"})
                continue
            sha = store.add(
                a.content,
                source="github",
                url=url,
                meta={
                    "title": path.rsplit("/", 1)[-1][:-4],
                    "public_url": f"{repo['html_url']}/blob/{branch}/{quote(path)}",
                    "license": repo["license"]["spdx_id"],
                    "github": {"stars": repo["stargazers_count"], "repo_created": repo["created_at"][:10]},
                },
            )
            ok += sha is not None
        if scanned % 25 == 0:
            store.save()
            log.info("github: scanned %d repos, %d with .aia, corpus now %d", scanned, with_aia, len(store))
    return {"repos_scanned": scanned, "repos_with_aia": with_aia, "downloaded": ok}
