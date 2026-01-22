from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_float(value: Any) -> Any:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            if value != value:
                return None
            if value in (float("inf"), float("-inf")):
                return None
            return float(value)
        return value
    except Exception:
        return None


def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_sanitize(v) for v in obj]
    return _safe_float(obj)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_sanitize(payload), f, indent=2, sort_keys=True)


def summarize_series(values: List[float]) -> Dict[str, float]:
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }

