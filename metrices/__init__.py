"""
DreamCloth evaluation metrics package.

This package supports "single-image -> simulated rollout" evaluation:
- Compute physics/temporal plausibility metrics from Phase2 mesh sequences.
- Optionally compute geometry and render metrics when GT meshes/images are available.
"""

from .config import EvalConfig
from .evaluator import Evaluator

__all__ = ["EvalConfig", "Evaluator"]

