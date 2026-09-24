"""QLoRA adapter training.

One adapter is trained per domain and task on the English training split only.
Target languages are never seen during training; they measure transfer.

The base weights stay frozen. Only the low-rank matrices are updated, and the
loss is computed on answer tokens so prompt tokens do not contribute.
"""

from __future__ import annotations

from pathlib import Path

from . import constants as C
from .prompts import build_messages, render_targets


def load_base_for_training(
    model_id: str = C.MODEL_ID,
    revision: str = C.MODEL_REVISION,
):
    """Load the base model in 4-bit NF4 with gradient checkpointing enabled."""
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
        device_map={"": 0},
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tokenizer


def attach_adapter(model):
    """Attach the LoRA adapter described in the paper and report its size."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=C.LORA_R,
        lora_alpha=C.LORA_ALPHA,
        lora_dropout=C.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(C.LORA_TARGET_MODULES),
    )
    model = get_peft_model(model, config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    share = 100.0 * trainable / total
    print(f"trainable parameters: {trainable:,} ({share:.2f}% of {total:,})")
    if trainable != C.ADAPTER_TRAINABLE_PARAMS:
        print(
            "WARNING: trainable parameter count differs from the value reported "
            f"in the paper ({C.ADAPTER_TRAINABLE_PARAMS:,}). Check the LoRA "
            "target module list before comparing numbers."
        )
    return model


def encode_example(tokenizer, task: str, domain: str, sentence: str, targets) -> dict:
    """Use the inference chat prompt and supervise only its JSON answer."""
    messages = build_messages(task, domain, sentence)
    answer = render_targets(targets)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": answer}],
        tokenize=False, add_generation_prompt=False, enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    encoded = tokenizer(full_text, add_special_tokens=False, truncation=True,
                        max_length=C.TRAIN_MAX_SEQ_LEN)
    labels = list(encoded["input_ids"])
    labels[:min(len(prompt_ids), len(labels))] = [-100] * min(
        len(prompt_ids), len(labels))
    if not any(value != -100 for value in labels):
        raise ValueError("no supervised answer tokens; increase sequence length")
    encoded["labels"] = labels
    return encoded


class AnswerCollator:
    """Pad variable-length examples; ignore pad positions in the loss."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        import torch

        width = max(len(row["input_ids"]) for row in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in features:
            pad = width - len(row["input_ids"])
            batch["input_ids"].append(
                row["input_ids"] + [self.tokenizer.pad_token_id] * pad)
            batch["attention_mask"].append(row["attention_mask"] + [0] * pad)
            batch["labels"].append(row["labels"] + [-100] * pad)
        return {key: torch.tensor(value, dtype=torch.long)
                for key, value in batch.items()}


def train_adapter(
    checkout: Path,
    domain: str,
    task: str,
    output_dir: Path,
    model_id: str = C.MODEL_ID,
    revision: str = C.MODEL_REVISION,
) -> Path:
    """Train and save one adapter. Returns the directory it was written to."""
    import torch
    from transformers import Trainer, TrainingArguments

    from .data import read_split

    torch.manual_seed(C.INITIALIZATION_SEED)

    model, tokenizer = load_base_for_training(model_id, revision)
    model = attach_adapter(model)

    examples = read_split(checkout, domain, C.SOURCE_LANGUAGE, "train")
    encoded = [
        encode_example(tokenizer, task, domain, example.sentence,
                       example.targets(task))
        for example in examples
    ]

    output_dir = Path(output_dir)
    arguments = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=C.TRAIN_EPOCHS,
        learning_rate=C.TRAIN_LEARNING_RATE,
        per_device_train_batch_size=C.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=C.TRAIN_GRAD_ACCUM,
        optim="paged_adamw_8bit",
        lr_scheduler_type="constant",
        logging_steps=10,
        save_strategy="no",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=C.TRAINER_SEED,
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(model=model, args=arguments, train_dataset=encoded,
                      data_collator=AnswerCollator(tokenizer))
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"saved adapter to {output_dir}")
    return output_dir
