# Copyright (c) 2026 Muhammad Abrar Rayhan (https://github.com/Muhammadabrarrayhan2)
# SPDX-License-Identifier: MIT
"""Stage 3 — turn parsed records into one structured skill profile per project (data/processed/profiles.csv).

The profile is the "small structured profile" of proposal §3.2: size, event handling, control flow,
procedures (with/without parameters), data structures, and which component families are present
(UI, sensors, storage, cloud/web, media, animation, maps, data science, AI). Every feature is
documented in data/processed/feature_spec.json so a teacher or reviewer can audit what a number means.

Quality flags are added here: parse errors, empty projects, and structural near-duplicates (e.g. an
unmodified tutorial re-uploaded to GitHub). Only rows with `in_analysis == True` feed clustering.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import pandas as pd

from .common import read_jsonl

# Component families, following the App Inventor designer palette.
FAMILIES: dict[str, set[str]] = {
    "ui": {"Button", "Label", "TextBox", "PasswordTextBox", "CheckBox", "Switch", "Slider", "Spinner", "ListView",
           "ListPicker", "DatePicker", "TimePicker", "Image", "Notifier", "WebViewer", "CircularProgress", "LinearProgress"},
    "layout": {"HorizontalArrangement", "VerticalArrangement", "HorizontalScrollArrangement", "VerticalScrollArrangement",
               "TableArrangement", "AbsoluteArrangement"},
    "media": {"Camera", "Camcorder", "ImagePicker", "Player", "Sound", "SoundRecorder", "VideoPlayer", "MediaStore"},
    "speech_translate": {"SpeechRecognizer", "TextToSpeech", "Translator", "YandexTranslate"},
    "animation": {"Canvas", "Ball", "ImageSprite"},
    "maps": {"Map", "Marker", "Circle", "Rectangle", "Polygon", "LineString", "FeatureCollection", "Navigation"},
    "sensors": {"AccelerometerSensor", "BarcodeScanner", "Barometer", "GyroscopeSensor", "Hygrometer", "LightSensor",
                "LocationSensor", "MagneticFieldSensor", "NearField", "OrientationSensor", "Pedometer", "ProximitySensor",
                "Thermometer"},
    "timer": {"Clock"},
    "social": {"ContactPicker", "EmailPicker", "PhoneCall", "PhoneNumberPicker", "Sharing", "Texting", "Twitter"},
    "storage_local": {"TinyDB", "File", "DataFile"},
    "cloud_web": {"CloudDB", "TinyWebDB", "FirebaseDB", "Web", "Spreadsheet", "FusiontablesControl", "GameClient", "Voting"},
    "connectivity": {"ActivityStarter", "BluetoothClient", "BluetoothServer", "Serial"},
    "robotics": {"NxtDrive", "NxtColorSensor", "NxtLightSensor", "NxtSoundSensor", "NxtTouchSensor", "NxtUltrasonicSensor",
                 "NxtDirectCommands", "Ev3Motors", "Ev3ColorSensor", "Ev3GyroSensor", "Ev3TouchSensor",
                 "Ev3UltrasonicSensor", "Ev3Sound", "Ev3UI", "Ev3Commands"},
    "data_science": {"Chart", "ChartData2D", "Trendline", "Regression", "AnomalyDetection", "DataFile", "Spreadsheet"},
    "ai_builtin": {"ChatBot", "ImageBot"},
}
# AI extensions are distributed as .aix packages with varied class names; match them by pattern.
AI_EXT_RE = re.compile(
    r"classifier|posenet|facemesh|handpose|teachable|tensorflow|tflite|objectdetect|chatbot|imagebot|"
    r"gpt|openai|gemini|llm|machinelearning|lookextension|^look$|emotion|faceapi|mediapipe|\bml",
    re.I,
)
NON_VISIBLE_IGNORE = {"Form"}

LOOP_BLOCKS = {"controls_forEach", "controls_forRange", "controls_while", "controls_for_each_dict", "controls_forEachDict"}
COND_BLOCKS = {"controls_if", "controls_choose"}
PROC_DEF = {"procedures_defnoreturn", "procedures_defreturn"}
PROC_CALL = {"procedures_callnoreturn", "procedures_callreturn"}
SOURCE_PRIORITY = {"classroom": 0, "official": 1, "gallery": 2, "github": 3}


def _prefix_sum(counter: dict[str, int], *prefixes: str) -> int:
    return sum(v for k, v in counter.items() if k.startswith(prefixes))


def profile(rec: dict[str, Any]) -> dict[str, Any]:
    comps: dict[str, int] = {k: v for k, v in rec["component_types"].items() if k not in NON_VISIBLE_IGNORE}
    blocks: dict[str, int] = rec["block_types"]
    ext_names = set(rec.get("extension_packages", [])) | {c for c in comps if not any(c in f for f in FAMILIES.values())}
    params = rec.get("procedure_param_counts", [])

    p: dict[str, Any] = {
        "n_blocks": sum(blocks.values()),
        "n_components": sum(comps.values()),
        "n_screens": rec["n_screens"],
        "n_distinct_component_types": len(comps),
        "n_distinct_block_types": len(blocks),
        "n_top_level_blocks": rec["n_top_level_blocks"],
        "max_block_depth": rec["max_block_depth"],
        "max_statement_nesting": rec["max_statement_nesting"],
        "n_event_handlers": blocks.get("component_event", 0),
        "n_distinct_events": len(rec.get("events", {})),
        "n_conditionals": sum(blocks.get(b, 0) for b in COND_BLOCKS),
        "n_loops": sum(blocks.get(b, 0) for b in LOOP_BLOCKS),
        "n_procedures": sum(blocks.get(b, 0) for b in PROC_DEF),
        "n_procedures_with_params": sum(1 for x in params if x > 0),
        "n_procedure_calls": sum(blocks.get(b, 0) for b in PROC_CALL),
        "n_global_vars": blocks.get("global_declaration", 0),
        "n_local_vars": _prefix_sum(blocks, "local_declaration"),
        "n_list_blocks": _prefix_sum(blocks, "lists_"),
        "n_dict_blocks": _prefix_sum(blocks, "dictionaries_", "pair"),
        "n_text_blocks": _prefix_sum(blocks, "text"),
        "n_math_blocks": _prefix_sum(blocks, "math_"),
        "n_logic_blocks": _prefix_sum(blocks, "logic_"),
        "n_generic_blocks": rec.get("n_generic_blocks", 0),
        "n_component_methods": blocks.get("component_method", 0),
        "n_disabled_blocks": rec.get("n_disabled_blocks", 0),
        "n_extensions": len(rec.get("extension_packages", [])),
    }
    for fam, members in FAMILIES.items():
        p[f"n_{fam}"] = sum(v for k, v in comps.items() if k in members)
    p["n_ai_extension"] = sum(v for k, v in comps.items() if k in ext_names and AI_EXT_RE.search(k)) + sum(
        1 for e in rec.get("extension_packages", []) if AI_EXT_RE.search(e.rsplit(".", 1)[-1])
    )

    # Binary "has the learner used X yet?" signals that drive the taxonomy and the rationales.
    p.update(
        uses_events=p["n_event_handlers"] > 0,
        uses_conditionals=p["n_conditionals"] > 0,
        uses_loops=p["n_loops"] > 0,
        uses_procedures=p["n_procedures"] > 0,
        uses_proc_params=p["n_procedures_with_params"] > 0,
        uses_variables=(p["n_global_vars"] + p["n_local_vars"]) > 0,
        uses_lists=p["n_list_blocks"] > 0,
        uses_dictionaries=p["n_dict_blocks"] > 0,
        uses_multiscreen=p["n_screens"] > 1,
        uses_sensors=p["n_sensors"] > 0,
        uses_timer=p["n_timer"] > 0,
        uses_storage=(p["n_storage_local"] + p["n_cloud_web"]) > 0,
        uses_cloud_web=p["n_cloud_web"] > 0,
        uses_media=p["n_media"] > 0,
        uses_speech=p["n_speech_translate"] > 0,
        uses_animation=p["n_animation"] > 0,
        uses_maps=p["n_maps"] > 0,
        uses_data_science=p["n_data_science"] > 0,
        uses_ai=(p["n_ai_builtin"] + p["n_ai_extension"]) > 0,
        uses_generic=p["n_generic_blocks"] > 0,
    )
    return p


def fingerprint(rec: dict[str, Any]) -> str:
    key = json.dumps([sorted(rec["component_types"].items()), sorted(rec["block_types"].items())])
    return hashlib.sha256(key.encode()).hexdigest()[:16]


FEATURE_DOCS = {
    "n_blocks": "Total blocks across all screens (aiatools, aggregated per screen).",
    "n_components": "Designer components excluding the Screen (Form) itself.",
    "max_block_depth": "Deepest chain of nested blocks (value + statement sockets), a proxy for expression complexity.",
    "max_statement_nesting": "Deepest nesting of statement sockets (if-inside-loop-inside-event...).",
    "n_procedures_with_params": "Procedures declaring at least one parameter (abstraction beyond copy-paste).",
    "n_generic_blocks": "'Any component' blocks — an advanced abstraction over component instances.",
    "n_ai_extension": "Components/extensions recognised as AI (image/audio classifiers, PoseNet, FaceMesh, LLM bots...).",
    "uses_*": "Boolean: the project contains at least one of the corresponding blocks/components.",
    "n_<family>": "Count of components in a palette family: " + ", ".join(FAMILIES),
    "in_analysis": "True if parsed OK, not empty and not a structural near-duplicate of a higher-priority project.",
}


def run(cfg: dict[str, Any]) -> dict[str, Any]:
    parsed = list(read_jsonl(cfg["paths"]["interim"] / "parsed.jsonl"))
    manifest = {r["sha256"]: r for r in read_jsonl(cfg["paths"]["raw"] / "manifest.jsonl")}
    rows = []
    for rec in parsed:
        base = {"project_id": rec["project_id"], "source": rec["source"], "title": rec.get("title"),
                "license": manifest.get(rec["sha256"], {}).get("license"), "status": rec["status"]}
        if rec["status"] != "ok":
            rows.append(base)
            continue
        rows.append({**base, "parser": rec["parser"], "yaversion": rec.get("yaversion"),
                     "fingerprint": fingerprint(rec), **profile(rec)})
    df = pd.DataFrame(rows)

    ok = df["status"] == "ok"
    df["is_empty"] = ok & (df["n_blocks"].fillna(0) == 0) & (df["n_components"].fillna(0) == 0)
    df["_prio"] = df["source"].map(SOURCE_PRIORITY)
    df = df.sort_values(["_prio", "project_id"]).reset_index(drop=True)
    df["is_near_duplicate"] = ok & df.duplicated("fingerprint", keep="first") & df["fingerprint"].notna()
    df["in_analysis"] = ok & ~df["is_empty"] & ~df["is_near_duplicate"]
    df = df.drop(columns="_prio")

    out = cfg["paths"]["processed"]
    df.to_csv(out / "profiles.csv", index=False)
    (out / "feature_spec.json").write_text(json.dumps(
        {"families": {k: sorted(v) for k, v in FAMILIES.items()}, "ai_extension_pattern": AI_EXT_RE.pattern,
         "docs": FEATURE_DOCS}, indent=2), encoding="utf-8")
    return {"profiles": len(df), "in_analysis": int(df["in_analysis"].sum()),
            "near_duplicates": int(df["is_near_duplicate"].sum()), "empty": int(df["is_empty"].sum()),
            "errors": int((~ok).sum())}
