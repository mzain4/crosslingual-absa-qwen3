"""Exact-tuple micro scoring, following the M-ABSA evaluator.

The accumulation in :func:`compute_f1_scores` and the prediction parser in
:func:`extract_spans_extraction` are ports of the released evaluator
(``eval_baseline_mT5/eval_utils.py`` in the M-ABSA repository). Two properties
are preserved on purpose, because the paper's numbers depend on them:

* matching is exact string equality on whole tuples, with no normalisation;
* the true-positive count is a membership test per predicted tuple, so the
  evaluator's duplicate-handling behaviour is inherited unchanged.

The edit-distance repair (``fix_pred_with_editdistance``) is intentionally **not**
implemented. The paper reports the raw score only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from . import constants as C

Target = tuple[str, ...]


def extract_spans_extraction(
    task: str,
    seq: str,
    *,
    correct_empty_tasd: bool = False,
) -> list[Target]:
    """Parse the evaluator's extraction format.

    The default follows the released M-ABSA parser exactly, including its
    task-specific handling of empty output. Set ``correct_empty_tasd`` to treat
    an empty TASD prediction as no tuples instead of one empty triplet.
    """
    if task not in ("uabsa", "aope", "tasd", "aste"):
        raise ValueError(f"unknown task {task!r}; expected one of {C.TASKS}")

    text = seq or ""
    if task == "uabsa" and text.lower() == "none":
        return []
    if correct_empty_tasd and task == "tasd" and (
        not text.strip() or text.strip().lower() == "none"
    ):
        return []

    targets: list[Target] = []
    for chunk in text.split("; "):
        body = chunk[1:-1]
        if task in ("uabsa", "aope"):
            try:
                first, second = body.split(", ")
            except ValueError:
                first, second = "", ""
            targets.append((first, second))
        else:
            try:
                first, second, third = body.split(", ")
            except ValueError:
                first, second, third = "", "", ""
            targets.append((first, second, third))
    return targets


def compute_f1_scores(
    predictions: Sequence[Sequence[Target]],
    gold: Sequence[Sequence[Target]],
) -> dict[str, float]:
    """Micro precision, recall and F1 over exact tuples."""
    if len(predictions) != len(gold):
        raise ValueError(
            f"{len(predictions)} prediction groups for {len(gold)} gold groups"
        )

    true_positive = 0
    n_predicted = 0
    n_gold = 0

    for predicted, reference in zip(predictions, gold):
        n_gold += len(reference)
        n_predicted += len(predicted)
        for target in predicted:
            if target in reference:
                true_positive += 1

    precision = true_positive / n_predicted if n_predicted else 0.0
    recall = true_positive / n_gold if n_gold else 0.0
    if precision or recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def score_strings(
    predictions: Iterable[str],
    gold: Iterable[str],
    task: str,
    *,
    correct_empty_tasd: bool = False,
) -> dict[str, float]:
    """Score prediction strings exactly as the released evaluator does."""
    predicted_groups = [
        extract_spans_extraction(
            task, text, correct_empty_tasd=correct_empty_tasd
        )
        for text in predictions
    ]
    gold_groups = [extract_spans_extraction(task, text) for text in gold]
    return compute_f1_scores(predicted_groups, gold_groups)


def to_percentage(scores: dict[str, float]) -> dict[str, float]:
    """Rescale a metric dict onto the 0-100 range used in the paper."""
    return {
        "precision": 100.0 * scores["precision"],
        "recall": 100.0 * scores["recall"],
        "f1": 100.0 * scores["f1"],
    }


def mean_f1(per_group: Sequence[dict[str, float]]) -> float:
    """Unweighted mean of group F1 scores, in percent.

    Every reported headline number is an average of per-group F1 values rather
    than a pooled micro-F1, so a language with more sentences does not dominate.
    """
    if not per_group:
        raise ValueError("no groups to average")
    return sum(group["f1"] for group in per_group) / len(per_group)
