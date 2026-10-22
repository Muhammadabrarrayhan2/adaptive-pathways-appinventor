"""Stage 2 — parse every stored .aia into a structural record (data/interim/parsed.jsonl).

Two passes per project:
  * aiatools (mit-cml/aiatools, the official parser named in the proposal) gives the component and
    block inventories;
  * a direct read of the .scm JSON / .bky XML gives what aiatools does not expose: block-tree depth,
    procedure parameter counts, and extension components (e.g. PersonalImageClassifier). It also
    serves as a cross-check: the agreement rate between the two passes is reported for data quality.
If aiatools fails on a project (old/new language versions, unknown extensions), the direct read is used
alone and the record says so (`parser`).

No identifying data is kept: project.properties `main` (which embeds the author's account name),
screen names and app names are not written to the output.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from typing import Any

from tqdm import tqdm

from .common import log, read_jsonl, write_jsonl


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(el: ET.Element, name: str) -> list[ET.Element]:
    return [c for c in el if _local(c.tag) == name]


def _load_scm(text: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {}


def _walk_components(node: dict[str, Any], out: list[str]) -> None:
    for c in node.get("$Components", []) or []:
        out.append(c.get("$Type", "?"))
        _walk_components(c, out)


class BlockStats:
    """Accumulates structural statistics from Blockly XML."""

    def __init__(self) -> None:
        self.instances: dict[str, str] = {}  # component instance name -> type, for the current screen
        self.legacy: int = 0
        self.types: Counter[str] = Counter()
        self.events: Counter[str] = Counter()
        self.methods: Counter[str] = Counter()
        self.setget: Counter[str] = Counter()
        self.generic: int = 0
        self.disabled: int = 0
        self.top_level: int = 0
        self.max_depth: int = 0
        self.max_nesting: int = 0
        self.proc_params: list[int] = []
        self.comments: int = 0

    def normalise_legacy(self, btype: str, top: bool) -> tuple[str, str | None]:
        """Map pre-2013 block types ("Button1_Click", "Ball_setproperty") to the modern vocabulary."""
        if "_" not in btype or btype.split("_", 1)[0] in LANG_PREFIXES:
            return btype, None
        head, tail = btype.split("_", 1)
        ctype = self.instances.get(head) or (head if head in self.instances.values() else None)
        if ctype is None:
            return btype, None
        self.legacy += 1
        if tail in ("setproperty", "getproperty"):
            return "component_set_get", ctype
        if tail == "component":
            return "component_component_block", ctype
        if top:
            self.events[f"{ctype}.{tail}"] += 1
            return "component_event", ctype
        self.methods[f"{ctype}.{tail}"] += 1
        return "component_method", ctype

    def visit(self, block: ET.Element, depth: int, nesting: int, disabled: bool) -> None:
        btype, _ = self.normalise_legacy(block.get("type", "?"), depth == 1)
        self.types[btype] += 1
        disabled = disabled or block.get("disabled") == "true"
        self.disabled += disabled
        self.max_depth = max(self.max_depth, depth)
        self.max_nesting = max(self.max_nesting, nesting)
        mut = next(iter(_children(block, "mutation")), None)
        if mut is not None:
            ctype = mut.get("component_type")
            if mut.get("is_generic") == "true":
                self.generic += 1
            if btype == "component_event" and ctype:
                self.events[f"{ctype}.{mut.get('event_name')}"] += 1
            elif btype == "component_method" and ctype:
                self.methods[f"{ctype}.{mut.get('method_name')}"] += 1
            elif btype == "component_set_get" and ctype:
                self.setget[f"{ctype}.{mut.get('property_name')}.{mut.get('set_or_get')}"] += 1
            if btype.startswith("procedures_def"):
                self.proc_params.append(len(_children(mut, "arg")))
        if _children(block, "comment"):
            self.comments += 1
        for child in block:
            tag = _local(child.tag)
            if tag in ("value", "statement"):
                for b in _children(child, "block"):  # shadow blocks are placeholders, not learner code
                    self.visit(b, depth + 1, nesting + (tag == "statement"), disabled)
            elif tag == "next":
                for b in _children(child, "block"):
                    self.visit(b, depth, nesting, disabled)


LANG_PREFIXES = {"controls", "logic", "math", "text", "lists", "color", "lexical", "global", "local", "procedures",
                 "component", "dictionaries", "obfuscated", "helpers", "pair", "obsufcated"}


def _walk_instances(node: dict[str, Any], out: dict[str, str]) -> None:
    for c in node.get("$Components", []) or []:
        out[c.get("$Name", "")] = c.get("$Type", "?")
        _walk_instances(c, out)


def raw_parse(zf: zipfile.ZipFile) -> dict[str, Any]:
    names = zf.namelist()
    instances: dict[str, dict[str, str]] = {}
    comps: list[str] = []
    yaversions = []
    screens = 0
    for n in names:
        if n.endswith(".scm"):
            screens += 1
            scm = _load_scm(zf.read(n).decode("utf-8", "replace"))
            if scm.get("YaVersion"):
                yaversions.append(int(scm["YaVersion"]))
            props = scm.get("Properties", {})
            comps.append(props.get("$Type", "Form"))
            _walk_components(props, comps)
            inst = {props.get("$Name", "Screen1"): "Form"}
            _walk_instances(props, inst)
            instances[n[:-4]] = inst
    stats = BlockStats()
    for n in names:
        if not n.endswith(".bky"):
            continue
        body = zf.read(n)
        if not body.strip():
            continue
        try:
            root = ET.fromstring(body)
        except ET.ParseError:
            continue
        stats.instances = instances.get(n[:-4], {})
        for b in _children(root, "block"):
            stats.top_level += 1
            stats.visit(b, 1, 0, False)
    extensions = sorted({n.split("/")[2] for n in names if n.startswith("assets/external_comps/") and n.count("/") >= 3})
    return {
        "n_screens": screens,
        "yaversion": max(yaversions) if yaversions else None,
        "component_types": dict(Counter(comps)),
        "extension_packages": extensions,
        "block_types": dict(stats.types),
        "events": dict(stats.events),
        "methods": dict(stats.methods),
        "setget": dict(stats.setget),
        "n_generic_blocks": stats.generic,
        "n_disabled_blocks": stats.disabled,
        "n_top_level_blocks": stats.top_level,
        "max_block_depth": stats.max_depth,
        "max_statement_nesting": stats.max_nesting,
        "procedure_param_counts": stats.proc_params,
        "n_block_comments": stats.comments,
        "n_legacy_blocks_normalised": stats.legacy,
    }


def aiatools_parse(path: str) -> dict[str, Any]:
    from aiatools import AIAFile

    with AIAFile(path) as aia:
        comps = Counter(c.type.name for c in aia.components())
        # Aggregate per screen: in aiatools 0.x, project-level AIAFile.blocks() returns the blocks of only
        # one screen for multi-screen projects (e.g. 309 of 606 blocks), while per-screen counts are exact.
        blocks: Counter[str] = Counter()
        for screen in aia.screens:
            blocks.update(b.type for b in screen.blocks())
        return {"component_types": dict(comps), "block_types": dict(blocks), "n_screens": len(list(aia.screens))}


def parse_one(path: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as zf:
        rec = raw_parse(zf)
    rec["parser"] = "xml"
    try:
        at = aiatools_parse(path)
        raw_blocks = sum(rec["block_types"].values())
        at_blocks = sum(at["block_types"].values())
        raw_comps = sum(rec["component_types"].values())
        at_comps = sum(at["component_types"].values())
        rec["aiatools"] = {
            "ok": True,
            "n_blocks": at_blocks,
            "n_components": at_comps,
            "blocks_agree": at_blocks == raw_blocks,
            "components_agree": at_comps == raw_comps,
        }
        # aiatools is authoritative for the block inventory when it parses the project; the direct read
        # stays authoritative for component types because it keeps extension names that aiatools generalises.
        rec["block_types"] = at["block_types"]
        rec["parser"] = "aiatools+xml"
    except Exception as e:  # noqa: BLE001 — any aiatools failure falls back to the direct read
        rec["aiatools"] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    return rec


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    manifest = list(read_jsonl(cfg["paths"]["raw"] / "manifest.jsonl"))
    out, errors = [], 0
    for row in tqdm(manifest, desc="parse", unit="aia"):
        base = {"project_id": row["project_id"], "sha256": row["sha256"], "source": row["source"], "title": row.get("title")}
        try:
            rec = parse_one(str(cfg["paths"]["raw"] / "aia" / f"{row['sha256']}.aia"))
            out.append({**base, "status": "ok", **rec})
        except Exception as e:  # noqa: BLE001
            errors += 1
            log.warning("parse failed for %s: %s", row["project_id"], e)
            out.append({**base, "status": "error", "error": f"{type(e).__name__}: {e}"})
    write_jsonl(cfg["paths"]["interim"] / "parsed.jsonl", out)
    at_ok = sum(1 for r in out if r.get("aiatools", {}).get("ok"))
    return {"parsed": len(out) - errors, "errors": errors, "aiatools_ok": at_ok}
