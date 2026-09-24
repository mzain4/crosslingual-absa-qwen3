"""Dataset parsing and task projection."""

from __future__ import annotations

import pytest

from clabsa.data import Example, parse_labels, parse_line, split_path

REAL_LINE = (
    "this unit is ` ` pretty ` ` and stylish , so my high school daughter was "
    "attracted to it for that reason .####[['unit', 'LAPTOP#DESIGN_FEATURES', "
    "'positive'], ['unit', 'LAPTOP#DESIGN_FEATURES', 'positive']]"
)


def test_parse_labels_reads_triplets():
    labels = parse_labels(
        "[['unit', 'LAPTOP#DESIGN_FEATURES', 'positive'], "
        "['screen', 'DISPLAY#GENERAL', 'negative']]"
    )
    assert labels == (
        ("unit", "LAPTOP#DESIGN_FEATURES", "positive"),
        ("screen", "DISPLAY#GENERAL", "negative"),
    )


def test_parse_labels_preserves_duplicates():
    """M-ABSA ships repeated identical triplets and the metric counts them."""
    labels = parse_labels(
        "[['unit', 'LAPTOP#DESIGN_FEATURES', 'positive'], "
        "['unit', 'LAPTOP#DESIGN_FEATURES', 'positive']]"
    )
    assert len(labels) == 2
    assert labels[0] == labels[1]


def test_parse_labels_keeps_null_marker():
    assert parse_labels("[['NULL', 'food quality', 'positive']]") == (
        ("NULL", "food quality", "positive"),
    )


def test_parse_labels_rejects_malformed():
    with pytest.raises(ValueError):
        parse_labels("[['only', 'two']]")
    with pytest.raises(ValueError):
        parse_labels("not a list")


def test_parse_line_splits_on_separator():
    example = parse_line(REAL_LINE)
    assert example is not None
    assert example.sentence.startswith("this unit is")
    assert len(example.triplets) == 2


def test_parse_line_skips_blank_lines():
    assert parse_line("") is None
    assert parse_line("   ") is None


def test_parse_line_requires_separator():
    with pytest.raises(ValueError):
        parse_line("a sentence with no label separator")


def test_targets_project_per_task():
    example = Example(
        sentence="x",
        triplets=(("unit", "LAPTOP#DESIGN_FEATURES", "positive"),),
    )
    assert example.targets("tasd") == [
        ("unit", "LAPTOP#DESIGN_FEATURES", "positive")
    ]
    assert example.targets("uabsa") == [("unit", "positive")]


def test_targets_reject_unknown_task():
    example = Example(sentence="x", triplets=())
    with pytest.raises(ValueError):
        example.targets("aste")


def test_split_path_builds_expected_location():
    path = split_path("vendor/M-ABSA", "food", "de", "test")
    assert path.parts[-4:] == ("data", "food", "de", "test.txt")


def test_split_path_validates_arguments():
    with pytest.raises(ValueError):
        split_path("vendor/M-ABSA", "restaurant", "en", "test")
    with pytest.raises(ValueError):
        split_path("vendor/M-ABSA", "food", "en", "validation")
