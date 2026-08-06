# Public claim evidence map

| Claim | Value | Scope | Evidence |
|---|---:|---|---|
| `EDGE-SUCCESS` | 0.95 ratio | 60 official-checkpoint simulator episodes | [`../evidence/edge-quality.json`](../evidence/edge-quality.json) |
| `EDGE-SUCCESS-RERUN` | 0.9 ratio | 20 same-seed official-checkpoint episodes; independent rerun | [`../evidence/independent-rerun.json`](../evidence/independent-rerun.json) |
| `EDGE-4090-EAGER` | 566.709 ms | fixed RTX 4090 node | [`../evidence/edge-fixed-node.json`](../evidence/edge-fixed-node.json) |
| `EDGE-4090-TRT` | 235.798 ms | fixed RTX 4090 node | [`../evidence/edge-fixed-node.json`](../evidence/edge-fixed-node.json) |
| `EDGE-4090-SPEEDUP` | 2.40337 x | P50 ratio on same node | [`../evidence/edge-fixed-node.json`](../evidence/edge-fixed-node.json) |
| `EDGE-ORIN-P50` | 3.37094 ms | U-Net inference only; no simulator/robot I/O | [`../evidence/edge-orin.json`](../evidence/edge-orin.json) |
| `EDGE-ORIN-P95` | 4.41489 ms | U-Net inference only; no simulator/robot I/O | [`../evidence/edge-orin.json`](../evidence/edge-orin.json) |
| `EDGE-ORIN-MISS` | 0 count/500 | measured board inference | [`../evidence/edge-orin.json`](../evidence/edge-orin.json) |
| `VLA-MAIN` | 0.133333 ratio | 3 seeds, 45 runs; no zero-shot claim | [`../evidence/smolvla-quality.json`](../evidence/smolvla-quality.json) |
| `VLA-GEN` | 0.180952 ratio | tasks excluded from selection but included in final fine-tuning | [`../evidence/smolvla-quality.json`](../evidence/smolvla-quality.json) |
| `VLA-ORIN-P50` | 1191.74 ms | synthetic input, PyTorch path | [`../evidence/smolvla-orin.json`](../evidence/smolvla-orin.json) |
| `VLA-ORIN-RATE` | 0.836851 Hz | synthetic input, PyTorch path | [`../evidence/smolvla-orin.json`](../evidence/smolvla-orin.json) |
| `VLA-ORIN-MISS` | 500 count/500 | every measured inference missed | [`../evidence/smolvla-orin.json`](../evidence/smolvla-orin.json) |
| `VLA-REPLAY-FRESH` | 0.0266667 ratio | offline timing replay, not task execution | [`../evidence/control-replay.json`](../evidence/control-replay.json) |
