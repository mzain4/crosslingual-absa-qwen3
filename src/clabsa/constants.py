"""Fixed experimental constants.

Everything here is a decision that was frozen before the final test run. Keeping
them in one place makes the diff between the paper and the code auditable.
"""

from __future__ import annotations

# --- Dataset ---------------------------------------------------------------

#: Domains evaluated in the paper (Table 1).
DOMAINS: tuple[str, ...] = ("food", "laptop", "phone")

#: Target languages. English is the source language.
LANGUAGES: tuple[str, ...] = ("en", "de", "fr", "ar", "hi", "sw")

#: The only language used for demonstrations and adapter training.
SOURCE_LANGUAGE: str = "en"

#: Task formulations.
TASKS: tuple[str, ...] = ("tasd", "uabsa")

#: Splits available in the M-ABSA checkout.
SPLITS: tuple[str, ...] = ("train", "dev", "test")

#: Sentiment labels for the categorical setting used here.
SENTIMENTS: tuple[str, ...] = ("positive", "negative", "neutral")

#: Marker used for an implicit target, following the M-ABSA convention.
NULL_TARGET: str = "NULL"

#: M-ABSA repository pinned to the revision used for every reported number.
MABSA_REPOSITORY: str = "https://github.com/swaggy66/M-ABSA.git"
MABSA_REVISION: str = "dadb0ccd55aa6a7e5de1b38ecaa059e64417c37e"

# --- Model -----------------------------------------------------------------

MODEL_ID: str = "Qwen/Qwen3-4B-Instruct-2507"

#: Resolved model revision recorded in the paper (leading characters of the
#: Hugging Face commit). Use "main" to follow the moving head.
MODEL_REVISION: str = "cdbee75f17c01a7cc42f958dc650907174af0554"

#: Greedy decoding length limits, per task.
MAX_NEW_TOKENS: dict[str, int] = {"tasd": 128, "uabsa": 96}

# --- Few-shot selection ----------------------------------------------------

#: Zero-based indices into the domain's English training split, chosen before
#: the final test run. One tuple of three per domain.
THREE_SHOT_INDICES: dict[str, tuple[int, int, int]] = {
    "food": (13, 32, 44),
    "laptop": (135, 185, 37),
    "phone": (4, 7, 12),
}

# --- QLoRA -----------------------------------------------------------------

#: Adapter rank. Also the value chosen by the small configuration study.
LORA_R: int = 8
LORA_ALPHA: int = 16
LORA_DROPOUT: float = 0.05

#: Projections updated by the adapter.
LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

#: Training schedule, identical for every domain and task adapter.
TRAIN_EPOCHS: int = 3
TRAIN_LEARNING_RATE: float = 2e-4
TRAIN_BATCH_SIZE: int = 1
TRAIN_GRAD_ACCUM: int = 8
TRAIN_MAX_SEQ_LEN: int = 1536

#: Seed used before loading the base model and attaching each adapter.
INITIALIZATION_SEED: int = 13

#: Seed passed to Hugging Face Trainer for the training process.
TRAINER_SEED: int = 42

#: Expected trainable parameters per adapter, used by the test suite to catch
#: an accidental change to the LoRA target list.
ADAPTER_TRAINABLE_PARAMS: int = 16_515_072

#: Approximate base-model parameter count; the paper reports this as 4.02B.
BASE_MODEL_PARAMS: int = 4_022_000_000
