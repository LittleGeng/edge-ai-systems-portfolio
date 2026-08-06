# Failure analysis and stop decisions

1. **SmolVLA misses the target by construction.** Measured steady throughput is 0.837 Hz versus 30 Hz. A queue can hide jitter but cannot create fresh actions; the 10 s replay produced 8/300 fresh ticks and 284 safe-fallback ticks.
2. **Energy is not defensible.** Tegrastats exposed component rails but no confirmed total-module rail. `energy_j_per_action` remains null instead of summing overlapping rails.
3. **VLA task success is modest.** Main-condition success is 13.3%; this is a feasibility result, not production readiness.
4. **The dataset audit has a provenance gap.** Deterministic splits were reconstructed from the same pinned official metadata and matched the frozen training command, but the original per-frame audit artifact was not synchronized.
5. **Scope is deliberately narrow.** Orin performance uses synthetic inputs; no real sensor, actuator, network, or safety-loop latency is included.

Next optimization gate: profile vision encoder, language backbone and action expert separately; proceed to TensorRT/quantization only if a path to at least 10 Hz is credible without violating the quality suite.
