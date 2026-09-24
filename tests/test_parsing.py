"""Frozen JSON-to-tuple conversion and evaluator transport."""

import pytest

from clabsa.parsing import process_prediction, to_official_string
from clabsa.scoring import score_strings


def test_valid_tasd_json_becomes_exact_tuple():
    raw = ('[{"entity":"screen","category":"DISPLAY#GENERAL",'
           '"sentiment":"positive"}]')
    assert process_prediction(raw, "tasd", ["DISPLAY#GENERAL"]) == (
        "(screen, DISPLAY#GENERAL, positive)", "valid_json")


def test_valid_uabsa_json_becomes_exact_pair():
    assert process_prediction('[{"entity":"screen","sentiment":"negative"}]',
                              "uabsa") == ("(screen, negative)", "valid_json")


def test_empty_json_uses_task_specific_transport():
    assert process_prediction("[]", "tasd") == ("", "valid_empty_json")
    assert process_prediction("[]", "uabsa") == ("None", "valid_empty_json")


def test_no_semantic_repair_of_invalid_category():
    text, status = process_prediction(
        '[{"entity":"screen","category":"WRONG","sentiment":"positive"}]',
        "tasd", ["DISPLAY#GENERAL"])
    assert text == "(screen, WRONG, positive)"
    assert status == "invalid_category"


def test_missing_json_field_is_not_filled_in():
    text, status = process_prediction('[{"entity":"screen"}]', "tasd")
    assert status == "json_wrong_keys"
    assert text == '[{"entity":"screen"}]'


def test_json_null_is_not_changed_to_benchmark_null_marker():
    _, status = process_prediction('[{"entity":null,"sentiment":"positive"}]',
                                   "uabsa")
    assert status == "json_non_string_or_empty_value"


def test_prose_after_json_is_invalid():
    _, status = process_prediction('[] hope this helps', "uabsa")
    assert status == "invalid_json"


def test_official_string_round_trips_through_evaluator():
    targets = (("screen", "DISPLAY#GENERAL", "positive"),
               ("battery", "BATTERY#QUALITY", "negative"))
    text = to_official_string(targets, "tasd")
    assert text == ("(screen, DISPLAY#GENERAL, positive); "
                    "(battery, BATTERY#QUALITY, negative)")
    assert score_strings([text], [text], "tasd")["f1"] == 1.0


def test_unknown_task_is_rejected():
    with pytest.raises(ValueError):
        to_official_string((), "aste")
