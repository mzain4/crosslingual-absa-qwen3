"""Convert generated JSON into the M-ABSA evaluator's tuple format.

The conversion changes serialization only. It never guesses missing targets,
corrects labels or adjusts target spans. Invalid JSON follows the original
experiment harness's format-only fallback and is marked in the run record.
"""

from __future__ import annotations

import json
import re

from . import constants as C


def to_official_string(targets, task: str) -> str:
    """Render tuples as ``(a, c, s); ...`` or ``(a, s); ...``."""
    if task not in C.TASKS:
        raise ValueError(f"unknown task {task!r}")
    return "; ".join("(" + ", ".join(target) + ")" for target in targets)


def format_only_cleanup(text: str) -> str:
    """Remove presentation wrappers without changing predicted fields."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:text|json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    cleaned = re.sub(r"^(?:answer|output|result|final answer)\s*:\s*",
                     "", cleaned, flags=re.I)
    if re.match(r"^none\b", cleaned, flags=re.I):
        return "None"
    groups = re.findall(r"\([^()]*\)", cleaned, flags=re.S)
    if groups:
        return "; ".join(" ".join(group.split()) for group in groups)
    return " ".join(cleaned.split())


def process_prediction(text: str, task: str, categories=()) -> tuple[str, str]:
    """Return scorer text and diagnostic status, following the frozen harness."""
    if task not in C.TASKS:
        raise ValueError(f"unknown task {task!r}")
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    cleaned = re.sub(r"^(?:answer|output|result|final answer|json)\s*:\s*",
                     "", cleaned, flags=re.I)
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return format_only_cleanup(text), "invalid_json"
    if not isinstance(payload, list):
        return format_only_cleanup(text), "json_not_array"
    required = (("entity", "category", "sentiment") if task == "tasd"
                else ("entity", "sentiment"))
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            return format_only_cleanup(text), "json_item_not_object"
        if set(item) != set(required):
            return format_only_cleanup(text), "json_wrong_keys"
        values = tuple(item[key].strip() if isinstance(item[key], str) else ""
                       for key in required)
        if any(not value for value in values):
            return format_only_cleanup(text), "json_non_string_or_empty_value"
        rows.append(values)
    if not rows:
        return ("" if task == "tasd" else "None"), "valid_empty_json"
    prediction = to_official_string(rows, task)
    if task == "tasd" and any(row[1] not in categories for row in rows):
        return prediction, "invalid_category"
    if any(row[-1] not in C.SENTIMENTS for row in rows):
        return prediction, "invalid_sentiment"
    return prediction, "valid_json"
