#!/usr/bin/env python3
"""Generate docs/setup-config.schema.json from the setup-config dataclasses.

Reads the dataclass definitions in imswitch/imcontrol/model/SetupInfo.py and
imswitch/imcontrol/view/guitools/ViewSetupInfo.py via the ``ast`` module (not
by importing them) so it has no runtime dependency on the rest of the
package, and turns them into a JSON Schema (draft-07) describing the setup
JSON files under imswitch/_data/user_defaults/imcontrol_setups/.

This is a documentation aid, not a strict/authoritative contract: the setup
JSON is parsed with ``dataclasses_json`` in ``Undefined.INCLUDE`` mode, so in
practice unknown top-level keys are accepted, and several fields declared as
required here are in fact omitted or passed as ``null`` in real configs (see
docs/SETUP_JSON_SCHEMA_REFERENCE.md for the list). Re-run this script and
diff the output whenever SetupInfo.py / ViewSetupInfo.py change.

Usage:
    python scripts/generate_setup_schema.py [docs/setup-config.schema.json]
"""
import ast
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SETUPINFO_PY = REPO_ROOT / "imswitch/imcontrol/model/SetupInfo.py"
VIEWSETUPINFO_PY = REPO_ROOT / "imswitch/imcontrol/view/guitools/ViewSetupInfo.py"
DEFAULT_OUT = REPO_ROOT / "docs/setup-config.schema.json"

PRIMITIVE = {
    "str": {"type": "string"},
    "int": {"type": "integer"},
    "float": {"type": "number"},
    "bool": {"type": "boolean"},
    "Any": {},
    "dict": {"type": "object"},
    "list": {"type": "array"},
    "List": {"type": "array"},
}


def parse_classes(path):
    """AST-walk a module and collect dataclass field name/type/default/doc."""
    tree = ast.parse(path.read_text())
    classes = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        fields = []
        body = node.body
        i = 0
        while i < len(body):
            stmt = body[i]
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                doc = None
                if i + 1 < len(body):
                    nxt = body[i + 1]
                    if (isinstance(nxt, ast.Expr) and isinstance(nxt.value, ast.Constant)
                            and isinstance(nxt.value.value, str)):
                        doc = nxt.value.value.strip()
                fields.append({
                    "name": stmt.target.id,
                    "type": ast.unparse(stmt.annotation),
                    "default": ast.unparse(stmt.value) if stmt.value is not None else None,
                    "doc": doc,
                })
            i += 1
        classes[node.name] = {"fields": fields, "bases": [ast.unparse(b) for b in node.bases]}
    return classes


def split_top_level(s):
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def parse_type(t, classes):
    t = t.strip()

    m = re.fullmatch(r"Optional\[(.*)\]", t)
    if m:
        return {"anyOf": [parse_type(m.group(1), classes), {"type": "null"}]}

    m = re.fullmatch(r"Union\[(.*)\]", t)
    if m:
        options = [parse_type(p, classes) for p in split_top_level(m.group(1))]
        if {"type": "null"} in options:
            options = [o for o in options if o != {"type": "null"}]
            return {"anyOf": options + [{"type": "null"}]}
        return {"anyOf": options}

    m = re.fullmatch(r"Dict\[(.*)\]", t)
    if m:
        _, val_t = split_top_level(m.group(1))
        return {"type": "object", "additionalProperties": parse_type(val_t, classes)}

    m = re.fullmatch(r"List\[(.*)\]", t)
    if m:
        return {"type": "array", "items": parse_type(m.group(1), classes)}

    if t in PRIMITIVE:
        return dict(PRIMITIVE[t])

    t_unquoted = t.strip("'\"")
    if t_unquoted in classes:
        return {"$ref": f"#/$defs/{t_unquoted}"}

    return {}


def default_literal(default_src):
    if default_src in (None, "None", "field(default_factory=lambda: None)"):
        return None
    try:
        return eval(default_src, {"__builtins__": {}})
    except Exception:
        return None


def build_class_schema(info, classes):
    props, required = {}, []
    for f in info["fields"]:
        if f["name"] == "_catchAll":
            continue
        schema = parse_type(f["type"], classes)
        if f["doc"]:
            schema = {**schema, "description": re.sub(r"\s+", " ", f["doc"]).strip()}
        dv = default_literal(f["default"])
        if dv is not None and not isinstance(dv, (dict, list)):
            schema = {**schema, "default": dv}
        props[f["name"]] = schema
        if f["default"] is None:
            required.append(f["name"])
    out = {"type": "object", "properties": props}
    if required:
        out["required"] = required
    return out


def main():
    classes = {}
    classes.update(parse_classes(SETUPINFO_PY))
    classes.update(parse_classes(VIEWSETUPINFO_PY))

    defs = {
        cname: build_class_schema(info, classes)
        for cname, info in classes.items()
        if cname not in ("SetupInfo", "ViewSetupInfo")
    }

    root = build_class_schema(classes["ViewSetupInfo"], classes)
    base = build_class_schema(classes["SetupInfo"], classes)
    root["properties"] = {**base["properties"], **root["properties"]}
    root["required"] = sorted(set(base.get("required", [])) | set(root.get("required", [])))

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "https://github.com/openUC2/ImSwitch/setup-config.schema.json",
        "title": "ImSwitch setup configuration",
        "description": (
            "Schema for the hardware-control setup JSON consumed by ImSwitch "
            "(imswitch/imcontrol/model/SetupInfo.py + "
            "imswitch/imcontrol/view/guitools/ViewSetupInfo.py). additionalProperties "
            "is true at the top level because SetupInfo is parsed with "
            "dataclasses_json Undefined.INCLUDE, and per-device 'managerProperties' "
            "dicts are intentionally free-form / driver-specific and not modeled "
            "here. See docs/SETUP_JSON_SCHEMA_REFERENCE.md for the field-by-field "
            "reference and a list of known divergences between this schema and "
            "real-world config files."
        ),
        "type": "object",
        "additionalProperties": True,
        "properties": root["properties"],
        "$defs": defs,
    }

    outpath = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(schema, indent=2) + "\n")
    print(f"wrote {outpath} ({len(defs)} $defs, {len(root['properties'])} top-level properties)")


if __name__ == "__main__":
    main()
