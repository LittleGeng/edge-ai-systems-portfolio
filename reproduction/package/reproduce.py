#!/usr/bin/env python3
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

provenance = json.loads((ROOT / "provenance.json").read_text())
for item in provenance["inputs"]:
    path = ROOT / item["package_path"]
    if sha256(path) != item["sha256"]:
        raise SystemExit(f"input hash mismatch: {path}")

def load(name):
    return json.loads((ROOT / "input" / name).read_text())

edge = load("edge-quality.json")
fixed = load("edge-fixed-node.json")
orin = load("edge-orin.json")
smol = load("smolvla-orin.json")
replay = load("control-replay.json")
actual = {
    "edge_high_quality_completion_rate": edge["high_quality_completion_rate"],
    "edge_4090_denoising_speedup": fixed["denoising_100_steps"]["eager"]["p50_ms"] / fixed["denoising_100_steps"]["torch_tensorrt"]["p50_ms"],
    "edge_orin_p50_ms": orin["wall_p50_ms"],
    "edge_orin_p95_ms": orin["wall_p95_ms"],
    "edge_orin_deadline_misses": orin["deadline_misses"],
    "smolvla_orin_control_rate_hz": smol["control_rate_hz"],
    "smolvla_orin_deadline_misses": smol["deadline_misses"],
    "smolvla_replay_fresh_action_rate": replay["smolvla_30hz"]["fresh_action_rate"],
}
expected = json.loads((ROOT / "expected.json").read_text())
checks = {}
for name, spec in expected["claims"].items():
    value = actual[name]
    target = spec["value"]
    tolerance = spec["absolute_tolerance"]
    checks[name] = {"actual": value, "expected": target, "absolute_error": abs(value - target), "passed": abs(value - target) <= tolerance}
result = {
    "schema_version": 1,
    "status": "passed" if all(item["passed"] for item in checks.values()) else "failed",
    "scope": "Self-contained reproduction of frozen result aggregation and acceptance gates; no model inference is executed.",
    "checks": checks,
}
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if result["status"] == "passed" else 2)
