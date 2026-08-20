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
edge_steps = load("edge-step-quality.json")
latency_envelope = load("smolvla-latency-envelope.json")
resize_quality = load("smolvla-resize-quality-gate.json")
resize384_finetune = load("smolvla-resize384-finetune-quality-gate.json")
quantization = load("orin-smolvla-quantization.json")
dataset_audit = load("smolvla-dataset-audit.json")
edge_orin_steps = load("edge-orin-denoising-steps.json")
strengthening = load("orin-smolvla-strengthening.json")
chunk_05 = load("orin-chunk-scheduler-05.json")
chunk_08 = load("orin-chunk-scheduler-08.json")
quant = {item["variant"]: item for item in quantization["comparisons"]}
edge_orin_step = {item["step_count"]: item for item in edge_orin_steps["results"]}
actual = {
    "edge_high_quality_completion_rate": edge["high_quality_completion_rate"],
    "edge_4090_denoising_speedup": fixed["denoising_100_steps"]["eager"]["p50_ms"] / fixed["denoising_100_steps"]["torch_tensorrt"]["p50_ms"],
    "edge_orin_p50_ms": orin["wall_p50_ms"],
    "edge_orin_p95_ms": orin["wall_p95_ms"],
    "edge_orin_deadline_misses": orin["deadline_misses"],
    "smolvla_orin_control_rate_hz": smol["control_rate_hz"],
    "smolvla_orin_deadline_misses": smol["deadline_misses"],
    "smolvla_replay_fresh_action_rate": replay["smolvla_30hz"]["fresh_action_rate"],
    "edge_step_100_completion_rate": edge_steps["variants"]["100"]["high_quality_completion_rate"],
    "edge_step_6_controller_p50_ms": edge_steps["variants"]["6"]["controller_latency_ms"]["p50"],
    "edge_lower_step_quality_gates_passed": sum(1 for step in ("20", "10", "8", "6") if edge_steps["variants"][step]["quality_gate"]["passed"]),
    "smolvla_latency_envelope_05_async_success": latency_envelope["comparison"]["threshold_0.5"]["async_success_rate"],
    "smolvla_latency_envelope_08_async_success": latency_envelope["comparison"]["threshold_0.8"]["async_success_rate"],
    "smolvla_latency_envelope_05_fallback": latency_envelope["comparison"]["threshold_0.5"]["fallback_total"],
    "smolvla_latency_envelope_08_fallback": latency_envelope["comparison"]["threshold_0.8"]["fallback_total"],
    "smolvla_resize256_success_rate": resize_quality["candidate"]["success_rate"],
    "smolvla_resize256_success_delta": resize_quality["paired_analysis"]["success_rate_delta_resize256_minus_eager"],
    "smolvla_resize384_finetune_success_rate": resize384_finetune["candidate"]["success_rate"],
    "smolvla_resize384_finetune_success_delta": resize384_finetune["paired_analysis"]["success_rate_delta_candidate_minus_baseline"],
    "smolvla_orin_int8_weight_only_p50_ms": quant["int8-weight-only"]["first_action_p50_ms"],
    "smolvla_orin_int8_dynamic_p50_ms": quant["int8-dynamic"]["first_action_p50_ms"],
    "smolvla_dataset_original_audit_passed": int(dataset_audit["original_audit_verification"]["status"] == "passed"),
    "smolvla_dataset_shard_hash_count": dataset_audit["original_audit_verification"]["data_shard_hash_count"],
    "edge_orin_loop_100_p50_ms": edge_orin_step[100]["p50_ms"],
    "edge_orin_loop_6_p50_ms": edge_orin_step[6]["p50_ms"],
    "edge_orin_loop_6_deadline_misses": edge_orin_step[6]["deadline_misses"],
    "smolvla_stage_denoise_vlm_share_pct": strengthening["stage_profile"]["stages"]["denoise_vlm_forward"]["share_of_full_inference_mean_pct"],
    "smolvla_stage_prefix_embedding_share_pct": strengthening["stage_profile"]["stages"]["prefix_embedding"]["share_of_full_inference_mean_pct"],
    "smolvla_chunk_05_fresh_action_rate": chunk_05["fresh_action_tick_rate"],
    "smolvla_chunk_08_fresh_action_rate": chunk_08["fresh_action_tick_rate"],
    "smolvla_chunk_08_stale_ticks": chunk_08["stale_ticks"],
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
