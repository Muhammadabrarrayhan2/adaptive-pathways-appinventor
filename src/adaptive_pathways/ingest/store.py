"""Content-addressed raw store for .aia files plus a provenance manifest.

Each accepted archive is written once to data/raw/aia/<sha256>.aia, so byte-identical copies found
through different sources are de-duplicated automatically; extra sightings are kept as provenance.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from ..common import check_aia_bytes, log, read_jsonl, sha256_bytes, write_jsonl


class RawStore:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.dir = cfg["paths"]["raw"] / "aia"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = cfg["paths"]["raw"] / "manifest.jsonl"
        self.rejects_path = cfg["paths"]["raw"] / "rejected.jsonl"
        self.rows: dict[str, dict[str, Any]] = {r["sha256"]: r for r in read_jsonl(self.manifest_path)}
        self.rejects: list[dict[str, Any]] = list(read_jsonl(self.rejects_path))
        self.seen_urls = {u for r in self.rows.values() for u in r["source_urls"]}
        self.seen_urls |= {r["url"] for r in self.rejects}

    def __len__(self) -> int:
        return len(self.rows)

    def count(self, source: str) -> int:
        return sum(1 for r in self.rows.values() if r["source"] == source)

    def add(self, data: bytes, *, source: str, url: str, meta: dict[str, Any] | None = None) -> str | None:
        """Validate and store one archive. Returns its sha256, or None if rejected."""
        self.seen_urls.add(url)
        reason = check_aia_bytes(data, self.cfg)
        if reason:
            self.rejects.append({"url": url, "source": source, "reason": reason})
            log.debug("reject %s (%s)", url, reason)
            return None
        sha = sha256_bytes(data)
        if sha in self.rows:
            row = self.rows[sha]
            if url not in row["source_urls"]:
                row["source_urls"].append(url)
            return sha
        (self.dir / f"{sha}.aia").write_bytes(data)
        self.rows[sha] = {
            "project_id": sha[:16],
            "sha256": sha,
            "source": source,
            "source_urls": [url],
            "bytes": len(data),
            "retrieved_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            **(meta or {}),
        }
        return sha

    def remove(self, sha: str) -> None:
        self.rows.pop(sha, None)
        self.path(sha).unlink(missing_ok=True)

    def path(self, sha: str) -> Path:
        return self.dir / f"{sha}.aia"

    def save(self) -> None:
        write_jsonl(self.manifest_path, self.rows.values())
        write_jsonl(self.rejects_path, self.rejects)
