# Recorded environments

Keep model environments separate. These are provenance records, not a claim
that the model families used identical software or hardware.

| Campaign | Torch | Transformers | PEFT | bitsandbytes | Accelerate | Hardware |
|---|---|---|---|---|---|---|
| Repeated Qwen and weighted controls | 2.11.0+cu128 | 5.5.4 | 0.19.1 | 0.49.2 | 1.13.0 | RTX 3070 Ti 8 GB |
| Gemma v5 | 2.14.1+cu130 | 5.5.4 | 0.19.1 | 0.49.2 | 1.13.0 | NVIDIA L4 |
| mT5 suite | 2.14.1+cu130 | 4.57.1 | Not used | Not used | 1.11.0 | Colab cloud GPUs |
| Earlier prompting | 2.8.0+cu126 | 4.57.1 | As archived for adapter runs | 0.48.1 | 1.11.0 | RTX 3070 Ti 8 GB |

mT5 also used SentencePiece 0.2.1 and huggingface-hub 0.36.0. Its training was
FP32, with microbatch four and accumulation four. Qwen used NF4 double
quantization, BF16 compute, microbatch one and accumulation eight. Gemma used
NF4 and the recorded BF16 frozen-matrix/FP32-vector policy at training and
reload. The effective batch sizes are 16 for mT5 and eight for Qwen/Gemma.

Install the CUDA Torch build supported by the chosen runtime first. Then pin
the remaining packages to the appropriate row and install this project with
`pip install -e . --no-deps`. Do not install the Qwen/Gemma dependency set over
the mT5 environment. GPU execution is explicit; plan and recorded-score
verification commands do not allocate a GPU.
