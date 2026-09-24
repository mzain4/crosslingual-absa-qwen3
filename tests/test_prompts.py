"""Prompt construction."""

from __future__ import annotations

import hashlib
import json
import pytest

from clabsa import constants as C
from clabsa.categories import categories_for
from clabsa.data import Example
from clabsa.prompts import (
    build_messages,
    build_prompt,
    instruction_for,
    render_targets,
    select_demonstrations,
)


def test_category_counts_match_the_published_table():
    """Table 1 of the paper reports 13 / 114 / 88 categories."""
    assert len(categories_for("food")) == 13
    assert len(categories_for("laptop")) == 114
    assert len(categories_for("phone")) == 88


def test_instruction_lists_every_category_for_tasd():
    text = instruction_for("tasd", "food")
    for category in categories_for("food"):
        assert category in text


def test_uabsa_instruction_omits_the_category_inventory():
    text = instruction_for("uabsa", "food")
    assert "food quality" not in text
    assert "sentiment" in text


def test_instruction_rejects_unknown_task():
    with pytest.raises(ValueError):
        instruction_for("aste", "food")


def test_zero_shot_prompt_has_no_demonstrations():
    messages = build_messages("tasd", "food", "the food was great")
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "the food was great" in messages[-1]["content"]


def test_frozen_p04_prompt_fingerprint():
    # Compared against the prompt in the original final-test harness.
    prompt = build_prompt("nice keyboard .", "tasd", "laptop")
    assert hashlib.sha256(prompt.encode()).hexdigest() == (
        "2581eb93e653971e6d7369444f0bf9db7737978e68f25f7615e69d07a64de99c"
    )


def test_few_shot_prompt_adds_a_demonstration_turn():
    demos = [("a great screen", [("screen", "DISPLAY#GENERAL", "positive")])]
    messages = build_messages("tasd", "laptop", "bad battery", demos)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "a great screen" in messages[1]["content"]
    assert '"entity":"screen"' in messages[1]["content"]


def test_render_targets_formats_both_arities():
    assert render_targets([("s", "c", "positive")]) == (
        '[{"entity":"s","category":"c","sentiment":"positive"}]'
    )
    assert render_targets([("s", "negative")]) == (
        '[{"entity":"s","sentiment":"negative"}]'
    )
    assert render_targets([]) == "[]"
    assert json.loads(render_targets([("screen", "DISPLAY#GENERAL", "positive")])) == [
        {"entity": "screen", "category": "DISPLAY#GENERAL", "sentiment": "positive"}
    ]


def test_render_targets_rejects_bad_arity():
    with pytest.raises(ValueError):
        render_targets([("only-one",)])


def test_select_demonstrations_uses_the_given_indices():
    examples = [
        Example(sentence=f"sentence {i}", triplets=(("a", "c", "positive"),))
        for i in range(10)
    ]
    chosen = select_demonstrations(examples, "tasd", (1, 4, 7))
    assert [sentence for sentence, _ in chosen] == [
        "sentence 1", "sentence 4", "sentence 7"
    ]


def test_select_demonstrations_rejects_out_of_range_index():
    examples = [Example(sentence="only", triplets=())]
    with pytest.raises(IndexError):
        select_demonstrations(examples, "tasd", (5,))


def test_configured_few_shot_indices_are_three_per_domain():
    assert set(C.THREE_SHOT_INDICES) == set(C.DOMAINS)
    for indices in C.THREE_SHOT_INDICES.values():
        assert len(indices) == 3
        assert len(set(indices)) == 3
