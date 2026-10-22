# Copyright (c) 2026 Muhammad Abrar Rayhan (https://github.com/Muhammadabrarrayhan2)
# SPDX-License-Identifier: MIT
"""Adaptive Pathways data pipeline — single entry point.

    python run_pipeline.py all                 # ingest -> parse -> features -> cluster -> stages -> catalog -> report
    python run_pipeline.py ingest --only gallery official
    python run_pipeline.py parse
    python run_pipeline.py recommend path/to/project.aia

Each stage reads the previous stage's files from data/, so stages can be re-run independently.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from adaptive_pathways.common import load_config, log, setup_logging  # noqa: E402

STAGES = ["ingest", "parse", "features", "cluster", "stages", "catalog", "report"]


def run_stage(name: str, cfg: dict, args: argparse.Namespace) -> object:
    if name == "ingest":
        from adaptive_pathways import ingest

        return ingest.run(cfg, args.only)
    if name == "parse":
        from adaptive_pathways import parse

        return parse.run(cfg)
    if name == "features":
        from adaptive_pathways import features

        return features.run(cfg)
    if name == "cluster":
        from adaptive_pathways import cluster

        return cluster.run(cfg)
    if name == "stages":
        from adaptive_pathways import stages

        return stages.run(cfg)
    if name == "catalog":
        from adaptive_pathways import catalog

        return catalog.run(cfg)
    if name == "report":
        from adaptive_pathways import report

        return report.run(cfg)
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=STAGES + ["all", "recommend"])
    ap.add_argument("aia", nargs="?", help="project .aia for `recommend`")
    ap.add_argument("--only", nargs="*", help="ingest: restrict to these sources")
    ap.add_argument("--from", dest="start", choices=STAGES, help="all: resume from this stage")
    ap.add_argument("--json", action="store_true", help="recommend: print JSON instead of text")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config()

    if args.stage == "recommend":
        from adaptive_pathways import recommend

        if not args.aia:
            ap.error("recommend needs a path to an .aia file")
        result = recommend.recommend_file(cfg, Path(args.aia))
        print(json.dumps(result, indent=2, ensure_ascii=False) if args.json else recommend.format_text(result))
        return

    todo = STAGES if args.stage == "all" else [args.stage]
    if args.start:
        todo = STAGES[STAGES.index(args.start):]
    for name in todo:
        t0 = time.time()
        log.info("=== stage %s ===", name)
        out = run_stage(name, cfg, args)
        log.info("=== stage %s finished in %.1fs: %s", name, time.time() - t0, json.dumps(out, default=str)[:600])


if __name__ == "__main__":
    main()
