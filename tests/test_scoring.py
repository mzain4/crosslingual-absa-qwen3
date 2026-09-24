"""The exact-tuple micro metric, including the evaluator behaviours it inherits."""

from __future__ import annotations

import pytest

from clabsa.scoring import (
    compute_f1_scores,
    extract_spans_extraction,
    mean_f1,
    score_strings,
    to_percentage,
)


# --- parser ----------------------------------------------------------------

def test_parses_triplets():
    parsed = extract_spans_extraction(
        "tasd", "(a, cat, positive); (b, cat, negative)"
    )
    assert parsed == [("a", "cat", "positive"), ("b", "cat", "negative")]


def test_parses_pairs():
    parsed = extract_spans_extraction("uabsa", "(a, positive)")
    assert parsed == [("a", "positive")]


def test_empty_prediction_matches_released_evaluator_by_default():
    assert extract_spans_extraction("tasd", "") == [("", "", "")]
    assert extract_spans_extraction("uabsa", "") == [("", "")]
    assert extract_spans_extraction("uabsa", "none") == []
    assert extract_spans_extraction("tasd", "None") == [("", "", "")]


def test_empty_tasd_correction_is_explicitly_opt_in():
    assert extract_spans_extraction(
        "tasd", "", correct_empty_tasd=True
    ) == []
    assert extract_spans_extraction(
        "tasd", "None", correct_empty_tasd=True
    ) == []


def test_wrong_field_count_matches_released_evaluator():
    assert extract_spans_extraction(
        "tasd", "(a, b, positive, extra)"
    ) == [("", "", "")]


def test_malformed_chunk_becomes_an_empty_tuple():
    assert extract_spans_extraction("tasd", "(a, b)") == [("", "", "")]


# --- metric ----------------------------------------------------------------

def test_perfect_match_scores_one():
    gold = [[("a", "c", "positive")]]
    assert compute_f1_scores(gold, gold)["f1"] == 1.0


def test_partial_match():
    predicted = [[("a", "c", "positive"), ("x", "c", "negative")]]
    gold = [[("a", "c", "positive"), ("b", "c", "negative")]]
    scores = compute_f1_scores(predicted, gold)
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["f1"] == 0.5


def test_missing_prediction_lowers_recall_not_precision():
    scores = compute_f1_scores([[]], [[("a", "c", "positive")]])
    assert scores == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_exact_matching_is_case_sensitive():
    """No normalisation is applied, by design."""
    scores = compute_f1_scores([[("a", "c", "Positive")]],
                               [[("a", "c", "positive")]])
    assert scores["f1"] == 0.0


def test_duplicate_predictions_inflate_recall():
    """Inherited quirk: true positives are a membership test per prediction.

    With one gold tuple and the same tuple predicted twice the metric reports
    recall 2.0. The paper's scores were produced with this behaviour, so it is
    preserved rather than silently corrected.
    """
    scores = compute_f1_scores(
        [[("a", "c", "positive"), ("a", "c", "positive")]],
        [[("a", "c", "positive")]],
    )
    assert scores["precision"] == 1.0
    assert scores["recall"] == 2.0


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        compute_f1_scores([[("a", "c", "positive")]], [])


def test_string_scoring_matches_tuple_scoring():
    predicted = ["(a, c, positive)"]
    gold = ["(a, c, positive); (b, c, negative)"]
    from_strings = score_strings(predicted, gold, "tasd")
    from_tuples = compute_f1_scores([[("a", "c", "positive")]],
                                    [[("a", "c", "positive"),
                                      ("b", "c", "negative")]])
    assert from_strings == from_tuples


def test_to_percentage_scales_by_one_hundred():
    scaled = to_percentage({"precision": 0.5, "recall": 0.25, "f1": 0.4})
    assert scaled == {"precision": 50.0, "recall": 25.0, "f1": 40.0}


def test_mean_f1_is_unweighted():
    groups = [{"f1": 100.0}, {"f1": 0.0}, {"f1": 0.0}]
    assert mean_f1(groups) == pytest.approx(33.3333, abs=1e-3)


def test_mean_f1_rejects_empty_input():
    with pytest.raises(ValueError):
        mean_f1([])
