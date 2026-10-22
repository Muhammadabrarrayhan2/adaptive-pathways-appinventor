"""Consented classroom artifacts (proposal §3.1, §4).

Drop learner .aia files into data/raw/classroom_inbox/ together with consent.csv:

    filename,consent_id,learner_consent,parental_consent,is_minor
    alice_project.aia,C-001,yes,yes,yes

A file is ingested only if learner consent is "yes" and, for minors, parental consent is "yes".
Before storage the archive is rebuilt with only the code files (.scm/.bky/project.properties);
media assets are dropped (they may contain photos or voices of children) and identifying
properties (author package path, project/app name) are replaced by the consent id.
"""
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from typing import Any

from ..common import log
from .store import RawStore

ID_PROPS = ("main", "name", "aname")


def deidentify(data: bytes, pseudonym: str) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            name = info.filename
            if not (name.endswith((".scm", ".bky")) or name.endswith("project.properties")):
                continue
            body = zin.read(info)
            # Rename the per-user package path src/appinventor/ai_<user>/<project>/ -> src/anon/<pseudonym>/
            new_name = re.sub(r"^src/.+/([^/]+\.(?:scm|bky))$", rf"src/anon/{pseudonym}/\1", name)
            if name.endswith("project.properties"):
                lines = []
                for line in body.decode("utf-8", "replace").splitlines():
                    key = line.split("=", 1)[0]
                    if key == "main":
                        line = f"main=anon.{pseudonym}.Screen1"
                    elif key in ID_PROPS:
                        line = f"{key}={pseudonym}"
                    lines.append(line)
                body = "\n".join(lines).encode()
            elif name.endswith(".scm"):
                text = body.decode("utf-8", "replace")
                text = re.sub(r'"AppName"\s*:\s*"[^"]*"', f'"AppName":"{pseudonym}"', text)
                body = text.encode()
            zout.writestr(new_name, body)
    return out.getvalue()


def run(cfg: dict[str, Any], store: RawStore) -> dict[str, int]:
    inbox = cfg["paths"]["raw"] / "classroom_inbox"
    inbox.mkdir(parents=True, exist_ok=True)
    consent_file = inbox / "consent.csv"
    if not consent_file.exists():
        consent_file.write_text("filename,consent_id,learner_consent,parental_consent,is_minor\n", encoding="utf-8")
        log.info("classroom: created empty consent register at %s", consent_file)
        return {"consented": 0, "skipped_no_consent": 0, "downloaded": 0}

    with consent_file.open(encoding="utf-8") as f:
        consent = {row["filename"]: row for row in csv.DictReader(f)}
    ok = skipped = 0
    for path in sorted(inbox.glob("*.aia")):
        row = consent.get(path.name)
        yes = lambda k: (row or {}).get(k, "").strip().lower() == "yes"  # noqa: E731
        if not row or not yes("learner_consent") or (yes("is_minor") and not yes("parental_consent")):
            skipped += 1
            log.warning("classroom: %s skipped — no valid consent row", path.name)
            continue
        pseudonym = re.sub(r"[^A-Za-z0-9_]", "_", row["consent_id"])
        sha = store.add(
            deidentify(path.read_bytes(), pseudonym),
            source="classroom",
            url=f"classroom://{pseudonym}",
            meta={"title": pseudonym, "public_url": None, "license": "consented research use only",
                  "classroom": {"consent_id": row["consent_id"]}},
        )
        ok += sha is not None
    (cfg["paths"]["raw"] / "classroom_ingest_log.json").write_text(
        json.dumps({"ingested": ok, "skipped_no_consent": skipped}, indent=2), encoding="utf-8"
    )
    return {"consented": ok, "skipped_no_consent": skipped, "downloaded": ok}
