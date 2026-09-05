# pipeline/sampler.py
"""
Stage 1 — Sampling & Preprocessing (CPU Pass)

Re-exports `run_cpu_pass` as `run_sampling` for backwards compatibility.
"""

from __future__ import annotations

from pipeline.cpu_pass import run_cpu_pass, run_sampling

__all__ = ["run_cpu_pass", "run_sampling"]
