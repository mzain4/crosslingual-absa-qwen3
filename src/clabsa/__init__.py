"""Cross-lingual ABSA adaptation with a small instruction model.

Three strategies are compared on identical multilingual test sets, scored with
the M-ABSA evaluator's raw exact-tuple micro-F1:

* zero-shot prompting
* three-shot prompting
* QLoRA adapter training on English data only

Only ``data``, ``prompts``, ``parsing``, ``scoring``, ``categories`` and
``constants`` are import-safe without a GPU. ``inference`` and ``train_qlora``
import torch lazily so the test suite runs anywhere.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
