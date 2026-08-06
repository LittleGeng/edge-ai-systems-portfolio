# Embodied intelligence resume entries

- Audited a pinned LeRobot LIBERO dataset, trained SmolVLA LoRA policies across three seeds, and reported 13.3% main-condition success with two-stage bootstrap intervals and explicit non-zero-shot scope.
- Deployed the frozen SmolVLA PyTorch path to Jetson AGX Orin and measured 1190 ms first-action P50, 1192 ms steady P50 and 0.837 Hz; used deadline replay to reject a misleading 30 Hz real-time claim and define the next profiling gate.

Interview boundary: Orin used synthetic observations; no physical-robot or simulator closed-loop success is claimed.
