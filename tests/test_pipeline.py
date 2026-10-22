# Copyright (c) 2026 Muhammad Abrar Rayhan (https://github.com/Muhammadabrarrayhan2)
# SPDX-License-Identifier: MIT
"""Offline tests on synthetic .aia archives (no network, no downloaded data needed)."""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_pathways.common import check_aia_bytes  # noqa: E402
from adaptive_pathways.features import profile  # noqa: E402
from adaptive_pathways.ingest.classroom import deidentify  # noqa: E402
from adaptive_pathways.parse import parse_one, raw_parse  # noqa: E402
from adaptive_pathways.stages import assign_stage  # noqa: E402

CFG = {"limits": {"max_aia_bytes": 25_000_000, "max_uncompressed_bytes": 200_000_000}}
PKG = "src/appinventor/ai_secretuser/Demo"


def scm(screen: str, comps: list[tuple[str, str]]) -> str:
    body = {"YaVersion": "208", "Source": "Form", "Properties": {
        "$Name": screen, "$Type": "Form", "$Version": "27", "AppName": "SecretApp", "Title": screen, "Uuid": "0",
        "$Components": [{"$Name": n, "$Type": t, "$Version": "1", "Uuid": str(i + 1)} for i, (n, t) in enumerate(comps)]}}
    return "#|\n$JSON\n" + json.dumps(body) + "\n|#"


EVENT_WITH_LIST = """<xml xmlns="https://developers.google.com/blockly/xml">
  <block type="component_event" id="a" x="0" y="0">
    <mutation component_type="Button" is_generic="false" instance_name="Button1" event_name="Click"></mutation>
    <field name="COMPONENT_SELECTOR">Button1</field>
    <statement name="DO">
      <block type="controls_if" id="b">
        <value name="IF0"><block type="logic_boolean" id="c"><field name="BOOL">TRUE</field></block></value>
        <statement name="DO0">
          <block type="lists_add_items" id="d"><mutation items="1"></mutation></block>
        </statement>
      </block>
    </statement>
  </block>
  <block type="procedures_defnoreturn" id="e" x="0" y="200">
    <mutation><arg name="x"></arg></mutation><field name="NAME">doIt</field>
  </block>
</xml>"""
LEGACY = """<xml><block type="Button1_Click" x="0" y="0"><title name="COMPONENT_SELECTOR">Button1</title>
  <statement name="DO"><block type="Label1_setproperty"><mutation yailtype="text"></mutation></block></statement>
</block></xml>"""


def make_aia(tmp_path: Path, screens: dict[str, tuple[list[tuple[str, str]], str]], name="p.aia") -> Path:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("youngandroidproject/project.properties", f"main=appinventor.ai_secretuser.Demo.Screen1\nname=Demo\naname=Demo\n")
        for s, (comps, bky) in screens.items():
            z.writestr(f"{PKG}/{s}.scm", scm(s, comps))
            z.writestr(f"{PKG}/{s}.bky", bky)
        z.writestr("assets/kid_photo.jpg", b"\xff\xd8fakejpeg")
    p = tmp_path / name
    p.write_bytes(buf.getvalue())
    return p


def test_rejects_non_app_inventor_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("actions.xml", "<x/>")
    assert check_aia_bytes(buf.getvalue(), CFG) == "not_app_inventor"
    assert check_aia_bytes(b"not a zip", CFG) == "not_zip"


def test_multiscreen_blocks_counted_on_every_screen(tmp_path):
    p = make_aia(tmp_path, {"Screen1": ([("Button1", "Button")], EVENT_WITH_LIST),
                            "Screen2": ([("Button1", "Button")], EVENT_WITH_LIST)})
    rec = parse_one(str(p))
    assert sum(rec["block_types"].values()) == 10  # 5 per screen, both screens
    assert rec["n_screens"] == 2
    if rec["aiatools"]["ok"]:
        assert rec["aiatools"]["blocks_agree"]


def test_structure_depth_and_params(tmp_path):
    p = make_aia(tmp_path, {"Screen1": ([("Button1", "Button")], EVENT_WITH_LIST)})
    with zipfile.ZipFile(p) as z:
        rec = raw_parse(z)
    assert rec["max_block_depth"] == 3          # event > if > (boolean | add-items)
    assert rec["max_statement_nesting"] == 2    # DO > DO0
    assert rec["procedure_param_counts"] == [1]
    assert rec["events"] == {"Button.Click": 1}


def test_legacy_block_format_is_normalised(tmp_path):
    p = make_aia(tmp_path, {"Screen1": ([("Button1", "Button"), ("Label1", "Label")], LEGACY)})
    with zipfile.ZipFile(p) as z:
        rec = raw_parse(z)
    assert rec["block_types"] == {"component_event": 1, "component_set_get": 1}
    assert rec["events"] == {"Button.Click": 1}


def test_profile_and_stage(tmp_path):
    p = make_aia(tmp_path, {"Screen1": ([("Button1", "Button"), ("Acc", "AccelerometerSensor")], EVENT_WITH_LIST)})
    prof = profile(parse_one(str(p)))
    assert prof["uses_lists"] and prof["uses_proc_params"] and prof["uses_sensors"] and not prof["uses_ai"]
    assert assign_stage(prof) == 4  # sensor-aware outranks data-aware
    prof["uses_ai"] = True
    assert assign_stage(prof) == 5


def test_ai_extension_detected(tmp_path):
    p = make_aia(tmp_path, {"Screen1": ([("Btn", "Button"), ("PIC1", "PersonalImageClassifier")], EVENT_WITH_LIST)})
    assert profile(parse_one(str(p)))["uses_ai"]


def test_classroom_deidentification_removes_identity_and_media(tmp_path):
    p = make_aia(tmp_path, {"Screen1": ([("Button1", "Button")], EVENT_WITH_LIST)})
    clean = deidentify(p.read_bytes(), "C001")
    with zipfile.ZipFile(io.BytesIO(clean)) as z:
        names = z.namelist()
        blob = b"".join(z.read(n) for n in names)
    assert not any(n.startswith("assets/") for n in names)
    assert b"secretuser" not in blob and b"SecretApp" not in blob
    assert check_aia_bytes(clean, CFG) is None
    out = tmp_path / "clean.aia"
    out.write_bytes(clean)
    assert sum(parse_one(str(out))["block_types"].values()) == 5  # code survives de-identification


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "data/catalog/resources.json").exists(),
                    reason="needs a pipeline run")
def test_recommender_end_to_end(tmp_path):
    from adaptive_pathways.common import load_config
    from adaptive_pathways.recommend import recommend_file

    p = make_aia(tmp_path, {"Screen1": ([("Button1", "Button"), ("Label1", "Label")], LEGACY)})
    res = recommend_file(load_config(), p)
    recs = res["recommendations"]
    assert 1 <= len(recs) <= 3
    assert len({r["introduces"][0] for r in recs}) == len(recs)  # diverse first concepts
    for r in recs:
        assert r["rationale"] and set(r["introduces"]) <= set(res["signals"]["gaps"])

