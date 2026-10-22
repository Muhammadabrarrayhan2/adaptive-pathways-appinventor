# Copyright (c) 2026 Muhammad Abrar Rayhan (https://github.com/Muhammadabrarrayhan2)
# SPDX-License-Identifier: MIT
"""Shared helpers: config, logging, polite HTTP, hashing, JSONL I/O and .aia safety checks."""
from __future__ import annotations

import hashlib
import json
import logging
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlparse

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[2]

log = logging.getLogger("adaptive_pathways")


def setup_logging(level: int = logging.INFO) -> None:
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[logging.StreamHandler(), logging.FileHandler(logs / "pipeline.log", encoding="utf-8")],
        force=True,
    )


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load((path or ROOT / "config.yaml").read_text(encoding="utf-8"))
    cfg["paths"] = {k: ROOT / v for k, v in cfg["paths"].items()}
    for p in cfg["paths"].values():
        p.mkdir(parents=True, exist_ok=True)
    return cfg


class PoliteSession:
    """requests.Session with retries/backoff and a per-host minimum interval between requests."""

    def __init__(self, cfg: dict[str, Any], headers: dict[str, str] | None = None):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = cfg["user_agent"]
        if headers:
            self.s.headers.update(headers)
        retry = Retry(
            total=cfg["http"]["retries"],
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "HEAD"),
            respect_retry_after_header=True,
        )
        self.s.mount("https://", HTTPAdapter(max_retries=retry))
        self.s.mount("http://", HTTPAdapter(max_retries=retry))
        self.timeout = cfg["http"]["timeout_s"]
        self.min_interval = cfg["http"]["min_interval_s"]
        self._last: dict[str, float] = {}

    def get(self, url: str, **kw) -> requests.Response:
        host = urlparse(url).netloc
        wait = self.min_interval - (time.monotonic() - self._last.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)
        try:
            return self.s.get(url, timeout=self.timeout, **kw)
        finally:
            self._last[host] = time.monotonic()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def short_hash(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def check_aia_bytes(data: bytes, cfg: dict[str, Any]) -> str | None:
    """Return None if `data` looks like a safe, genuine App Inventor project, else a rejection reason.

    Downloaded archives are untrusted: they are only ever read through zipfile, never extracted
    or executed, and are size-checked against a zip-bomb limit first.
    """
    if len(data) > cfg["limits"]["max_aia_bytes"]:
        return "too_large"
    if data[:2] != b"PK":
        return "not_zip"
    try:
        import io

        with zipfile.ZipFile(io.BytesIO(data)) as z:
            infos = z.infolist()
            if sum(i.file_size for i in infos) > cfg["limits"]["max_uncompressed_bytes"]:
                return "zip_bomb_guard"
            names = [i.filename for i in infos]
    except zipfile.BadZipFile:
        return "bad_zip"
    if not any(n.endswith("youngandroidproject/project.properties") for n in names):
        return "not_app_inventor"  # e.g. Adobe Illustrator .aia action files share the extension
    if not any(n.endswith(".scm") for n in names):
        return "no_screens"
    return None
