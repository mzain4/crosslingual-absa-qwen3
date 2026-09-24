"""Prompt construction for the reported zero-shot, three-shot and QLoRA runs."""

from __future__ import annotations

import json
from collections.abc import Sequence

from . import constants as C
from .categories import category_block

SYSTEM_PROMPT = (
    "You are a deterministic multilingual information-extraction system. "
    "Follow the requested schema exactly and return no explanation."
)


def build_prompt(sentence: str, task: str, domain: str) -> str:
    """Build the frozen P04 instruction from the original experiment harness."""
    if task not in C.TASKS:
        raise ValueError(f"unknown task {task!r}")
    name = ("Target-Aspect-Sentiment Detection (TASD)" if task == "tasd"
            else "Unified Aspect-Based Sentiment Analysis (UABSA)")
    keys = ("`entity`, `category`, and `sentiment`" if task == "tasd"
            else "`entity` and `sentiment`")
    prompt = f"""Task: {name} for M-ABSA.

Extract every sentiment-bearing target and return only one valid JSON array.
Each array item must be an object with exactly these {'three' if task == 'tasd' else 'two'} keys:
{keys}.

Mandatory extraction decision:
- The sentence may be in any language. Analyze it in its original language.
- If the sentence expresses any positive, negative, or neutral opinion, do not return `[]`.
- If an opinion target is explicitly named, copy its exact target span in the original language.
- If the target is absent or referred to only by a pronoun, use the literal string `NULL`.
- Return `[]` only when the sentence expresses no positive, negative, or neutral opinion.

Entity-boundary decision:
- Copy the minimal noun or name span that names the opinion target; omit surrounding articles or determiners when the noun remains an exact span.
- Do not copy a pronoun or demonstrative as the entity. If the target is only pronominal or omitted, use `NULL`.
- Do not replace the annotated product or object target with a praised or criticized property phrase.

Field rules:
- `entity`: the shortest exact target span copied from the sentence, or `NULL` under the rule above.
"""
    if task == "tasd":
        prompt += "- Never omit `entity`; a category and sentiment alone are not a valid item.\n"
        prompt += "- `category`: exactly one value from the allowed inventory below.\n"
    else:
        prompt += "- Never omit `entity`.\n"
    prompt += """- `sentiment`: exactly `positive`, `negative`, or `neutral`.
- Keep separate sentiment-bearing targets as separate objects.
- Do not translate, paraphrase, stem, repair, or add words to an entity span.

Output contract:
- Output one JSON array. For a non-empty answer, use this schema:
"""
    if task == "tasd":
        prompt += '  [{"entity":"<exact span or NULL>","category":"<allowed category>","sentiment":"<positive|negative|neutral>"}]\n'
    else:
        prompt += '  [{"entity":"<exact span or NULL>","sentiment":"<positive|negative|neutral>"}]\n'
    prompt += """- The angle-bracket values above are placeholders, not an answer to copy.
- Output no Markdown, code fence, explanation, labels-only answer, or commentary.

"""
    if task == "tasd":
        prompt += f"Allowed category inventory:\n{category_block(domain)}\n\n"
        prompt += "Final check: identify every opinion first; then verify every array item contains entity, category, and sentiment.\n"
    else:
        prompt += "Final check: every non-empty array item must contain entity and sentiment.\n"
    return prompt + f"Sentence: {sentence}\nJSON:"


def instruction_for(task: str, domain: str) -> str:
    """Return the task instruction without a review (for inspection only)."""
    return build_prompt("", task, domain).removesuffix("Sentence: \nJSON:")


def render_targets(targets: Sequence[tuple[str, ...]]) -> str:
    """Serialize gold tuples as valid JSON, preserving the original labels."""
    objects = []
    for target in targets:
        if len(target) == 3:
            objects.append(dict(zip(("entity", "category", "sentiment"), target)))
        elif len(target) == 2:
            objects.append(dict(zip(("entity", "sentiment"), target)))
        else:
            raise ValueError(f"unexpected target arity: {target!r}")
    return json.dumps(objects, ensure_ascii=False, separators=(",", ":"))


def build_messages(
    task: str,
    domain: str,
    sentence: str,
    demonstrations: Sequence[tuple[str, Sequence[tuple[str, ...]]]] = (),
) -> list[dict[str, str]]:
    """Use the system/user chat format of the recorded experiments."""
    prompt = build_prompt(sentence, task, domain)
    if demonstrations:
        marker = f"Sentence: {sentence}\nJSON:"
        lines = [
            "Labeled English training demonstrations:",
            "Use them to learn the extraction format and decisions. Do not copy an "
            "answer unless it applies to the target sentence.",
        ]
        for number, (example_sentence, targets) in enumerate(demonstrations, 1):
            lines.extend(("", f"Demonstration {number}",
                          f"Sentence: {example_sentence}",
                          f"JSON: {render_targets(targets)}"))
        lines.extend(("", "Target sentence:", marker))
        prompt = prompt.replace(marker, "\n".join(lines), 1)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]


def select_demonstrations(examples: Sequence, task: str, indices: Sequence[int]):
    """Select fixed zero-based English training indices."""
    selected = []
    for index in indices:
        if not 0 <= index < len(examples):
            raise IndexError(f"demonstration index {index} outside {len(examples)} rows")
        example = examples[index]
        selected.append((example.sentence, example.targets(task)))
    return selected
