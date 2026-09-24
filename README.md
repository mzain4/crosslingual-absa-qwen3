# Cross-Lingual ABSA with a Small Instruction Model

Companion code and results for the manuscript **Cross-Lingual Aspect-Based
Sentiment Analysis of E-Commerce Reviews with Memory-Efficient QLoRA-Adapted
Qwen3-4B**.

The experiment asks a practical question: if you only have one consumer GPU, how
much does a 4B instruction model gain from examples or from adapter training?
Three strategies are compared on identical multilingual test sets, scored with
the M-ABSA evaluator:

| Strategy     | Supervision                 | What it measures                             |
| ------------ | --------------------------- | -------------------------------------------- |
| `zero-shot`  | task instructions only      | what the model already knows                 |
| `three-shot` | 3 fixed English examples    | the value of a handful of demonstrations     |
| `qlora`      | full English training split | the value of parameter-efficient fine-tuning |

Only English labels are ever used for demonstrations or training. Every other
language measures transfer, never adaptation.

## Headline results

Six-language mean F1, averaged over 18 domain-language groups.

| Task  | zero-shot | three-shot | QLoRA     | mT5-base (published) |
| ----- | --------- | ---------- | --------- | -------------------- |
| TASD  | 12.89     | 14.49      | **34.51** | 25.93                |
| UABSA | 26.39     | 30.33      | **45.06** | 43.52                |

QLoRA trains 0.41% of the parameters and exceeds the published mT5-base average
on TASD. Food UABSA stays below mT5, and Swahili remains weak across setups;
both are analysed in the paper rather than tuned away.

## Requirements

- One CUDA GPU with **8 GB** of memory (an RTX 3070 Ti was used for all local experiments reported in the paper).
- Python 3.11 (the recorded Windows runs used 3.11.5).
- `git`, to fetch the pinned M-ABSA checkout.

The recorded prompting environment used PyTorch 2.8.0+cu126, Transformers
4.57.1, Accelerate 1.11.0 and bitsandbytes 0.48.1. The requirements file
allows newer compatible versions; record your installed versions when
comparing a rerun with the reported scores.

The test suite and the data preparation step run on CPU. Only `run` and
`train` need a GPU.

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick start

```bash
# 1. Fetch M-ABSA at the pinned revision and print the split sizes.
clabsa prepare --checkout vendor/M-ABSA

# 2. Prompting strategies. Each command walks the full 18-group matrix.
clabsa run --strategy zero-shot  --out runs/zero-shot
clabsa run --strategy three-shot --out runs/three-shot

# 3. QLoRA: one adapter per domain and task, trained on English only.
clabsa train --out adapters

# 4. Evaluate the adapters on the same test sets.
clabsa run --strategy qlora --adapter-dir adapters --out runs/qlora

# 5. Per-group and overall scores.
clabsa evaluate --predictions runs/qlora
```

The default evaluator mode exactly preserves the released M-ABSA parser. To
instead treat an empty TASD prediction as no predicted tuples, add
`--correct-empty-tasd` to either `clabsa run` or `clabsa evaluate`. That
corrected mode is provided for analysis and does not reproduce the paper's
reported scoring behavior.

To smoke-test without a full run, narrow the matrix:

```text
clabsa run --strategy zero-shot --out runs/smoke --domains laptop --languages en --tasks tasd --max-examples 2
```

The smoke-test score is **not** comparable to the paper's full-test scores.
Omit `--max-examples` for a complete shard. Each run writes fresh prediction
files; it does not resume partially completed files.

The recorded adapter training took about 2.16 h for Food, 4.22 h for Laptop
and 4.19 h for Phone on the reference machine. Each figure sums the two task
adapters; generation time is extra.

## Repository layout

```text
crosslingual-absa-qwen3/
|-- configs/                 Frozen configuration for each strategy
|-- data/categories.json     Published TASD category inventories
|-- outputs/                 Recorded results backing every number in the paper
|-- src/clabsa/
|   |-- constants.py         Every frozen decision (pins, seeds, indices, hyperparameters)
|   |-- categories.py        Category inventory lookup
|   |-- data.py              M-ABSA checkout, split reading, task projection
|   |-- prompts.py           Prompt construction for all strategies
|   |-- parsing.py           Model output -> scored tuples (format only)
|   |-- scoring.py           Exact-tuple micro metric
|   |-- inference.py         Greedy generation over the test matrix
|   |-- train_qlora.py       4-bit NF4 + LoRA training
|   `-- cli.py               Command line entry point
`-- tests/                   Fast, model-free unit tests
```

The importable package is `clabsa`, short for cross-lingual ABSA, and
`pip install -e .` installs a console script of the same name.

Model weights, adapters, the M-ABSA checkout and generated predictions are all
ignored by git.

## Recorded results

This repository provides the code, configurations, prompts, evaluation scripts,
and recorded results for the experiments reported in the paper. `outputs/`
contains per-group scores for all 288 system/domain/language/task combinations,
the aggregated means, dataset statistics, and the error-analysis counts.
Averaging `raw_f1` over the six languages reproduces every headline mean in the
paper, so the reported figures can be re-derived without rerunning anything. See
[`outputs/README.md`](outputs/README.md) for the column meanings and the mapping
to individual tables.

## About the dataset

The M-ABSA data is **not** committed to this repository. `clabsa prepare`
clones it at the pinned revision recorded in `src/clabsa/constants.py`, which is
better practice than vendoring for three reasons: the benchmark is Wu et al.'s
work and carries its own terms, a pinned revision gives an exact and auditable
provenance, and it keeps this repository small.

The category inventories are the one exception. They are stored in
`data/categories.json` because they are needed to build prompts offline and are
short enough to audit by eye.

## How scoring works

The reported metric is the M-ABSA evaluator's **raw exact-tuple micro-F1**. A
tuple scores only when every field matches an annotated tuple in the same
sentence. `precision`, `recall` and `f1` are computed with the same accumulation
as the released evaluator, so two of its behaviours are inherited deliberately:

- matching is exact string equality, with no normalisation;
- the true-positive count is a membership test per predicted tuple, so the
  evaluator's duplicate-handling behaviour carries through unchanged.

The evaluator's optional **edit-distance repair is not applied**. That repaired
score is a separate, more forgiving metric and is not reported in the paper.

The default parser preserves the released evaluator's task-specific empty-output
behavior exactly: UABSA treats the literal `None` as no tuples, while an empty
TASD prediction produces `('', '', '')` and is counted as a false positive. The
optional `--correct-empty-tasd` mode treats an empty TASD prediction as no tuples
instead. See `scoring.extract_spans_extraction`.

## Reproducibility

The main fixed settings are gathered in `src/clabsa/constants.py`:

- **Data** — M-ABSA pinned at `dadb0ccd55aa6a7e5de1b38ecaa059e64417c37e`. The
  pinned revision does not report exactly the sentence counts printed in the
  M-ABSA paper; the paper's limitations section discloses the difference.
- **Model** — `Qwen/Qwen3-4B-Instruct-2507`, resolved revision `cdbee75f17c01a7cc42f958dc650907174af0554`,
  non-thinking generation, greedy decoding, 128 new tokens for TASD and 96 for
  UABSA.
- **Few-shot examples** — fixed training indices, chosen before the final test
  run: Food `[13, 32, 44]`, Laptop `[135, 185, 37]`, Phone `[4, 7, 12]`.
- **Adapters** — rank 8, alpha 16, dropout 0.05, on the query, key, value,
  output, gate, up and down projections; 3 epochs, learning rate 2e-4, batch size
  1 with gradient accumulation over 8 steps, maximum sequence length 1536, paged
  8-bit AdamW, initialization/launcher seed 13, and Trainer seed 42.

Published mT5-base and zero-shot LLM baselines are **not** reproduced here. They
are cited values from the M-ABSA paper and are shown for context only.

## Tests

```bash
pytest
```

The suite is model-free. It covers dataset
parsing, task projection, prompt construction, output parsing and the metric
itself, including the inherited duplicate-handling behaviour and the
empty-prediction case.

## Acknowledgements

The benchmark, its category inventories and the evaluator come from
[M-ABSA](https://github.com/swaggy66/M-ABSA) by Wu et al. (EMNLP 2025). Please
cite their work alongside this one. QLoRA is due to Dettmers et al. and LoRA to
Hu et al.

## Citation

```bibtex
@misc{zain2026crosslingual,
  title  = {Cross-Lingual Aspect-Based Sentiment Analysis of E-Commerce Reviews
            with Memory-Efficient {QLoRA}-Adapted {Qwen3-4B}},
  author = {Zain Ul Abidin, Muhammad and Wasim, Muhammad},
  year   = {2026},
  note   = {Manuscript and companion software; publication details pending}
}
```

## Licence

MIT. The M-ABSA dataset and evaluator are the work of Wu et al. and are
downloaded at a pinned revision rather than redistributed here.
