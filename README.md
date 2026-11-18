# Memory-Efficient Cross-Lingual and Cross-Domain ABSA

Experiment code for **Memory-Efficient Cross-Lingual and Cross-Domain
Aspect-Based Sentiment Analysis of E-Commerce Reviews**.

The study asks how far English-only training can support structured extraction
across languages and product domains. Qwen adaptation runs on one RTX 3070 Ti
with 8 GB memory. mT5 and Gemma runs use cloud GPUs; the hardware claim applies
to Qwen. This repository contains experiment code and numerical records.

## What was evaluated

| Comparison | English training | Evaluation | Seeds |
|---|---|---|---|
| Qwen3-4B QLoRA and reproduced mT5-base | Food, Laptop, Phone; TASD and UABSA | Each domain in six languages | 13, 42, 87 |
| Gemma-4-E4B-it QLoRA | Laptop UABSA | Six languages | 13, 42, 87 |
| Qwen original versus empty-label-weighted loss | Food and Phone UABSA | Six languages | 13, 42, 87 |
| Repeated cross-domain Qwen comparison | Reuse the preceding Food/Phone adapters | Food to Phone and Phone to Food; English/Swahili; both objectives | 13, 42, 87 |
| Earlier Qwen zero-shot, three-shot and QLoRA | No examples / three examples / full English split | Three domains, two tasks, six languages | One fitted adapter per task/domain |
| Earlier complete cross-domain Qwen grid | Reuse three earlier UABSA adapters | All six directed domain pairs in six languages | One run |

Languages are English (`en`), German (`de`), French (`fr`), Arabic (`ar`),
Hindi (`hi`) and Swahili (`sw`). English test results measure the training
language; the other five measure cross-lingual transfer. TASD predicts target,
category and sentiment; UABSA omits the category. Domain transfer uses UABSA
because its output schema is shared across domains.

## Results

F1 is on a 0–100 scale. Each language is scored by pooled exact-tuple micro-F1;
the reported means average language scores and then fitted seeds.

| Three-seed comparison | TASD | UABSA |
|---|---:|---:|
| Qwen3-4B, three domains | 34.39 | 45.59 |
| Reproduced mT5-base, three domains | 23.97 | 32.47 |
| Gemma, Laptop only | — | 64.02 |

Empty-label weighting changes Food UABSA from **39.08 to 45.78** and Phone
from **41.61 to 42.43**. English/Swahili repeated Food–Phone transfer increases
from **7.56 to 8.74**. Fewer omissions accompany more false positives on reviews
with no annotated opinion. This is a change to the training objective, not a
new QLoRA architecture or a complete solution to domain transfer.

The earlier single-run Qwen means were 34.51 TASD and 45.06 UABSA. Its complete
cross-domain mean of 24.52 covers different fitted adapters and conditions from
the repeated subset. Keep these studies separate. Original summaries remain in
`outputs/`; current records are in `outputs/repeated-study/`.

## Install and verify without a GPU

Use a separate Python environment for each model family. Install a suitable
CUDA PyTorch build before the training dependencies. See `environments.md` for
the recorded versions; installing the broad package requirements alone does not
recreate a recorded environment.

```bash
pip install -e ".[dev]"
python -m pytest
clabsa verify-recorded
clabsa prepare --checkout vendor/M-ABSA
```

`verify-recorded` recomputes all 306 condition scores from the committed
per-review tuple counts, without downloading models. The 158,052 evaluation
records include repeated baseline, weighted, Gemma, transfer and Phone
development conditions. Original prediction-file identities are recorded in
`conditions.json`; derived counts support numerical verification without
redistributing review corpora. Count verification is distinct from regenerating
predictions from model weights.

## Repeated experiments

The default stage prints a plan and does not start training:

```bash
clabsa experiment --model qwen
clabsa experiment --model mt5
clabsa experiment --model gemma
```

To train and evaluate all 18 Qwen models, use the same output directory:

```bash
clabsa experiment --model qwen --stage train --out runs/qwen
clabsa experiment --model qwen --stage evaluate --out runs/qwen
```

Use `--domains food --tasks uabsa --seeds 13` for one recorded condition first.
Replace `qwen` with `mt5` for its 18-model matrix or `gemma` for its three-seed
Laptop UABSA comparison. Gemma requires its supported Transformers version,
model access and sufficient VRAM; its recorded full runs used an L4.

Weighted Qwen training keeps every review and multiplies the mean answer-token
loss by 0.25 when its gold tuple list is empty. Nonempty reviews, including
implicit `NULL` targets, retain weight one. Microbatch one prevents the review
weight from being normalized away.

```bash
clabsa experiment --model qwen --empty-weight 0.25 --stage train --out runs/weighted
clabsa experiment --model qwen --empty-weight 0.25 --stage evaluate --out runs/weighted
```

Repeated transfer needs no additional training. It uses Food and Phone UABSA
adapters from each training directory:

```bash
clabsa experiment --model qwen --cross-domain --stage evaluate --adapter-root runs/qwen --out runs/transfer-control
clabsa experiment --model qwen --empty-weight 0.25 --cross-domain --stage evaluate --adapter-root runs/weighted --out runs/transfer-weighted
```

These commands cover 12 conditions per objective: two directions, two languages
and three seeds. Together they reproduce the 24-condition scope.

### Checkpoints and interrupted work

Rerun the same command to resume. Each job has an isolated output directory and
an immutable configuration identity. Resume checkpoints include optimizer,
scheduler, random state, training step and file hashes; a checkpoint is eligible
only after its completion record verifies. Final model files receive a separate
completion record. Inference flushes each prediction to disk, checks the saved
prefix against the pinned test split and continues after that prefix.

Keep completed model directories if further inference is needed. Scores alone
cannot regenerate predictions. Back up completed results separately from the
larger training checkpoints. A local flush does not establish that a mounted
cloud drive has synchronized, so retain an independent downloaded backup.

Only one worker can enter a job directory. A hard process termination can leave
`worker.lock`; remove it only after confirming that its previous worker has
stopped. Incomplete checkpoints are never used, and the runner refuses to
silently restart training when checkpoint directories exist but none verifies.

The consolidated runner is CPU-tested and follows the recorded settings. It
has not been subjected to another full GPU campaign after packaging. The
completed study used the archived model-specific launchers. A small training,
reload and resume check in the intended environment should precede a new full
campaign; cross-platform numerical identity is not promised.

## Frozen prompting and earlier experiments

The original prompting protocol remains `M1-ZS-P04`. Demonstrations are fixed
English training indices: Food `[13, 32, 44]`, Laptop `[135, 185, 37]`, Phone
`[4, 7, 12]`. No target-language demonstrations or test-based prompt tuning are
used.

```bash
clabsa run --strategy zero-shot --out runs/zero-shot
clabsa run --strategy three-shot --out runs/three-shot
clabsa train --out adapters/historical
clabsa run --strategy qlora --adapter-dir adapters/historical --out runs/historical
clabsa evaluate --predictions runs/historical
```

`train` retains the earlier single-run configuration: initialization seed 13,
Trainer seed 42 and the historical constant scheduler. The repeated
`experiment` command uses matching initialization/data seeds and a linear
scheduler. Neither command silently substitutes one protocol for the other.

## Metric and provenance

### Original complete cross-domain grid

With the original single-run UABSA adapters in
adapters/historical/food-uabsa, laptop-uabsa and phone-uabsa, run:

~~~bash
clabsa historical-transfer --adapter-dir adapters/historical --out runs/historical-transfer
~~~

This evaluates all six directed pairs of Food, Laptop and Phone in all six
languages: 36 full-test conditions. The prompt uses the target domain while
the adapter comes from the source domain. There is no further training.
The command refuses to overwrite existing prediction files. It requires the
original adapters to reproduce the recorded historical grid; newly trained
adapters constitute a rerun. It is separate from the repeated Food–Phone
comparison. This newly packaged interface has not been GPU-replayed.

Primary scoring follows the released M-ABSA tuple-membership matching, without
edit-distance span repair. Repeated predicted tuples retain the released
matching behavior. The paper's blank-TASD convention treats an empty prediction
or gold string as zero tuples; it is the default. Use `--released-empty-tasd`
with `run` or `evaluate` only for the upstream empty-string sensitivity check.
An annotated `NULL` opinion is always a tuple. Capped multiset matching is
secondary and is labeled in the committed analyses.

Model and dataset revisions are pinned in `src/clabsa/constants.py`. Recorded
protocol settings, data hashes and the Food development selection IDs are in
`configs/recorded-study/`. Each rerun records its own installed versions and
prediction hashes. A changed training identity requires a new output directory.

The pinned English train/development/test counts are Food 1278/312/522, Laptop
1264/326/532 and Phone 1276/307/522. M-ABSA's printed table lists different Food
development and Phone train/test counts; runs here consistently use the pinned
repository. Food denotes food-product reviews, not the Restaurant domain.

The recorded causal-model training helper converts annotations through tuple
text before constructing JSON. Its comma/parenthesis parsing and whitespace
stripping differ from direct annotation-to-JSON conversion on ten English
training reviews (one Laptop, nine Phone; none Food). The original and weighted
Qwen runs share this behavior. It is retained explicitly in
recorded_training_answer for replication; its independent effect was not
tested. A different serializer is a changed training protocol, not a correction
that can be applied to the saved predictions.

Five of those reviews contain delimiter-bearing targets; the other five differ
only by field whitespace normalization. The record is in
configs/recorded-study/training_transport_audit.json. Frozen prompt fingerprints
cover both tasks in all three domains.

## Files and sources

- `src/clabsa/`: data, frozen prompts, training, inference and scoring.
- `configs/`: earlier settings and recorded repeated-study protocols.
- `outputs/`: numerical summaries and independently checked repeated-study counts.
- `tests/`: CPU checks of parsing, prompts, scoring, scopes and loss weighting.

M-ABSA data and evaluator: [official repository](https://github.com/swaggy66/M-ABSA)
and [paper](https://aclanthology.org/2025.emnlp-main.128/).
Model identities: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
[mT5-base](https://huggingface.co/google/mt5-base),
[Gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it).
Code license: MIT. Data and model weights retain their providers' licenses and
are obtained separately. Citation metadata is in `CITATION.cff`.
