#!/usr/bin/env python3
"""Fail closed when a public portfolio contains private operational data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


REQUIRED = {
    "README.md",
    "PUBLICATION.json",
    "SHA256SUMS",
    "projects/edge-diffusion.md",
    "projects/smolvla-edge.md",
    "docs/evidence-map.md",
    "docs/failure-analysis.md",
    "results/results.csv",
    "figures/latency-comparison.svg",
    "demos/edge-diffusion-evidence-walkthrough.mp4",
    "demos/smolvla-edge-evidence-walkthrough.mp4",
    "reproduction/package/reproduce.py",
}
FORBIDDEN_PATHS = {
    "docs/interview-qa.json",
    "docs/interview-qa.md",
}
FORBIDDEN_TEXT = {
    "/home/geng": "private home path",
    "/root/": "remote runtime path",
    "connect.bj": "cloud SSH endpoint",
    "seetacloud.com": "cloud SSH endpoint",
    "robot.kiring.cn": "private relay endpoint",
    "bitbucket": "private SSH key name",
    "ssh_endpoints.local": "private endpoint registry",
    '"instance_id"': "cloud instance identifier",
    '"gpu_uuid"': "GPU hardware identifier",
    '"balance"': "billing balance",
    '"hourly_rate_cny"': "cloud billing rate",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "GitHub classic token": re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    "GitHub fine-grained token": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "OpenAI-style key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
}
TEXT_SUFFIXES = {"", ".csv", ".json", ".md", ".py", ".svg", ".txt"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    files = [path for path in root.rglob("*") if path.is_file() and ".git" not in path.parts]
    relative = {str(path.relative_to(root)) for path in files}
    errors.extend(f"missing required file: {name}" for name in sorted(REQUIRED - relative))
    errors.extend(f"interview Q&A must remain private: {name}" for name in sorted(FORBIDDEN_PATHS & relative))

    for path in files:
        rel = str(path.relative_to(root))
        if path.is_symlink():
            errors.append(f"symlink is not allowed: {rel}")
            continue
        if path.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"file exceeds 10 MiB: {rel}")
        if path.suffix.lower() not in TEXT_SUFFIXES or rel == "scripts/verify_repository.py":
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"unexpected binary text file: {rel}")
            continue
        for needle, reason in FORBIDDEN_TEXT.items():
            if needle.lower() in content.lower():
                errors.append(f"{reason}: {rel}")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                errors.append(f"{name}: {rel}")

    manifest = root / "SHA256SUMS"
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            expected, name = line.split(maxsplit=1)
            target = root / name.lstrip("*")
            if not target.is_file() or sha256(target) != expected:
                errors.append(f"hash mismatch: {name}")

    reproduction = subprocess.run(
        ["python3", str(root / "reproduction/package/reproduce.py")],
        capture_output=True,
        text=True,
    )
    if reproduction.returncode != 0:
        errors.append("frozen reproduction failed")

    result = {
        "status": "passed" if not errors else "failed",
        "files_checked": len(files),
        "errors": errors,
        "reproduction": reproduction.stdout.strip(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
