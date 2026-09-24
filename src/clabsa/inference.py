"""Prompting-based inference.

Imports torch lazily so that ``import clabsa`` and the test suite work on a
machine without a GPU.

Decoding is greedy and the output length is capped per task, matching the
protocol in the paper. Thinking mode is not enabled: Qwen3-4B-Instruct-2507
supports non-thinking generation only.
"""

from __future__ import annotations

import json
import itertools
from collections.abc import Iterator, Sequence
from pathlib import Path

from . import constants as C
from .parsing import process_prediction, to_official_string
from .categories import categories_for
from .prompts import build_messages, select_demonstrations


def load_model(
    model_id: str = C.MODEL_ID,
    revision: str = C.MODEL_REVISION,
    load_in_4bit: bool = True,
    adapter_path: str | Path | None = None,
):
    """Load the base model, optionally attaching a trained LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    kwargs: dict = {"revision": revision}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return model, tokenizer


def render_prompt(tokenizer, messages: Sequence[dict[str, str]]) -> str:
    """Apply the model's chat template with a generation prompt."""
    return tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )


def generate(
    model,
    tokenizer,
    messages: Sequence[dict[str, str]],
    max_new_tokens: int,
) -> str:
    """Greedy single-example generation, returning only the new text."""
    import torch

    prompt = render_prompt(tokenizer, messages)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(completion, skip_special_tokens=True)


def predict_one(
    model,
    tokenizer,
    task: str,
    domain: str,
    sentence: str,
    demonstrations: Sequence = (),
) -> tuple[str, str]:
    """Run one example and return ``(prediction_string, parse_status)``."""
    messages = build_messages(task, domain, sentence, demonstrations)
    text = generate(model, tokenizer, messages, C.MAX_NEW_TOKENS[task])
    return process_prediction(text, task,
                              categories_for(domain) if task == "tasd" else ())


def iter_shard(
    checkout: Path,
    domain: str,
    language: str,
    task: str,
    strategy: str,
):
    """Yield the examples of one domain/language/test shard.

    Few-shot demonstrations are drawn from the English training split, so the
    same three examples are reused for every target language.
    """
    from .data import read_split

    examples = read_split(checkout, domain, language, "test")

    demonstrations: list = []
    if strategy == "three-shot":
        train = read_split(checkout, domain, C.SOURCE_LANGUAGE, "train")
        demonstrations = select_demonstrations(
            train, task, C.THREE_SHOT_INDICES[domain]
        )
    elif strategy not in ("zero-shot", "qlora"):
        raise ValueError(
            f"unknown prompting strategy {strategy!r}; expected 'zero-shot' or "
            "'three-shot'"
        )

    for example in examples:
        yield example, demonstrations


def run_shard(
    model,
    tokenizer,
    checkout: Path,
    domain: str,
    language: str,
    task: str,
    strategy: str,
    output_path: Path,
    max_examples: int | None = None,
    correct_empty_tasd: bool = False,
) -> dict:
    """Run one shard, writing one JSON record per sentence.

    Predictions are written incrementally so an interrupted run can be
    inspected without losing completed work.
    """
    from .scoring import score_strings, to_percentage

    predictions: list[str] = []
    gold: list[str] = []
    statuses: dict[str, int] = {}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        source = iter_shard(checkout, domain, language, task, strategy)
        for example, demonstrations in itertools.islice(source, max_examples):
            predicted, status = predict_one(
                model, tokenizer, task, domain, example.sentence, demonstrations
            )
            statuses[status] = statuses.get(status, 0) + 1

            predictions.append(predicted)
            gold.append(to_official_string(example.targets(task), task))

            handle.write(json.dumps({
                "sentence": example.sentence,
                "gold": gold[-1],
                "prediction": predicted,
                "status": status,
            }, ensure_ascii=False) + "\n")

    scores = to_percentage(score_strings(
        predictions,
        gold,
        task,
        correct_empty_tasd=correct_empty_tasd,
    ))
    return {
        "domain": domain,
        "language": language,
        "task": task,
        "strategy": strategy,
        "max_examples": max_examples,
        "empty_tasd_mode": (
            "corrected" if correct_empty_tasd else "released_evaluator"
        ),
        "n": len(predictions),
        "statuses": statuses,
        **scores,
    }


def prediction_iterator(path: Path) -> Iterator[dict]:
    """Read back a predictions file written by :func:`run_shard`."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
