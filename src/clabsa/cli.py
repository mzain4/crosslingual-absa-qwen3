"""Command line entry point.

    clabsa prepare  --checkout vendor/M-ABSA
    clabsa run      --strategy three-shot --out runs/three-shot
    clabsa train    --out adapters
    clabsa run      --strategy qlora --adapter-dir adapters --out runs/qlora
    clabsa evaluate --predictions runs/qlora
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

from . import constants as C


def _select(values: list[str] | None, default: tuple[str, ...]) -> list[str]:
    return list(values) if values else list(default)


# --- prepare ---------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> int:
    from .data import ensure_checkout, language_matrix

    checkout = ensure_checkout(Path(args.checkout))
    print(f"checkout ready at {checkout}")
    print(f"revision {C.MABSA_REVISION}")

    print("\nsentence counts per domain and language:")
    header = "domain    " + "".join(f"{lang:>7}" for lang in C.LANGUAGES)
    print(header)
    for split in ("train", "test"):
        counts = language_matrix(checkout, split)
        print(f"\n[{split}]")
        for domain, per_language in counts.items():
            row = f"{domain:<10}"
            row += "".join(f"{per_language.get(lang, 0):>7}" for lang in C.LANGUAGES)
            print(row)
    return 0


# --- prompt ----------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    from .inference import load_model, run_shard

    checkout = Path(args.checkout)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    domains = _select(args.domains, C.DOMAINS)
    languages = _select(args.languages, C.LANGUAGES)
    tasks = _select(args.tasks, C.TASKS)
    if args.max_examples is not None and args.max_examples <= 0:
        raise ValueError("--max-examples must be a positive integer")
    for values, allowed, name in ((domains, C.DOMAINS, "domains"),
                                  (languages, C.LANGUAGES, "languages"),
                                  (tasks, C.TASKS, "tasks")):
        unknown = set(values) - set(allowed)
        if unknown:
            raise ValueError(f"unknown {name}: {sorted(unknown)}")

    if args.strategy == "qlora":
        missing = [str(Path(args.adapter_dir) / f"{domain}-{task}")
                   for domain in domains for task in tasks
                   if not (Path(args.adapter_dir) / f"{domain}-{task}" /
                           "adapter_config.json").is_file()]
        if missing:
            raise FileNotFoundError("missing adapters: " + ", ".join(missing))

    results = []
    prompting_model = None
    for domain in domains:
        for task in tasks:
            if args.strategy == "qlora":
                adapter = Path(args.adapter_dir) / f"{domain}-{task}"
                print(f"loading base model with adapter {adapter}", flush=True)
                model, tokenizer = load_model(adapter_path=adapter)
            else:
                if prompting_model is None:
                    print("loading base model for prompting")
                    prompting_model = load_model()
                model, tokenizer = prompting_model

            for language in languages:
                target = out_root / domain / language / f"{task}.jsonl"
                print(f"{args.strategy} | {domain} | {language} | {task}", flush=True)
                summary = run_shard(
                    model, tokenizer, checkout,
                    domain, language, task, args.strategy, target,
                    max_examples=args.max_examples,
                    correct_empty_tasd=args.correct_empty_tasd,
                )
                results.append(summary)
                print(f"    F1 {summary['f1']:.2f}  (n={summary['n']})")

            if args.strategy == "qlora":
                del model, tokenizer
                gc.collect()
                import torch
                torch.cuda.empty_cache()

    record = out_root / "summary.json"
    record.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {record}")
    return 0


# --- train -----------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> int:
    from .train_qlora import train_adapter

    checkout = Path(args.checkout)
    domains = _select(args.domains, C.DOMAINS)
    tasks = _select(args.tasks, C.TASKS)

    for domain in domains:
        for task in tasks:
            destination = Path(args.out) / f"{domain}-{task}"
            print(f"training {domain} / {task} -> {destination}", flush=True)
            train_adapter(checkout, domain, task, destination)
            gc.collect()
            import torch
            torch.cuda.empty_cache()
    return 0


# --- evaluate --------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> int:
    from .scoring import score_strings, to_percentage

    root = Path(args.predictions)
    files = sorted(root.rglob("*.jsonl"))
    if not files:
        print(f"no prediction files under {root}", file=sys.stderr)
        return 1

    rows = []
    for path in files:
        records = [json.loads(line) for line in
                   path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not records:
            continue
        parts = path.relative_to(root).parts
        domain, language, task = parts[0], parts[1], parts[2].removesuffix(".jsonl")
        scores = to_percentage(score_strings(
            [r["prediction"] for r in records],
            [r["gold"] for r in records],
            task,
            correct_empty_tasd=args.correct_empty_tasd,
        ))
        rows.append({"domain": domain, "language": language, "task": task,
                     "n": len(records), **scores})

    for task in C.TASKS:
        subset = [r for r in rows if r["task"] == task]
        if not subset:
            continue
        print(f"\n{task.upper()}")
        print(f"{'domain':<10}{'language':<10}{'n':>6}{'P':>9}{'R':>9}{'F1':>9}")
        for domain in C.DOMAINS:
            for row in [r for r in subset if r["domain"] == domain]:
                print(f"{row['domain']:<10}{row['language']:<10}{row['n']:>6}"
                      f"{row['precision']:>9.2f}{row['recall']:>9.2f}{row['f1']:>9.2f}")

        overall = sum(r["f1"] for r in subset) / len(subset)
        print(f"\n{task.upper()} mean over {len(subset)} groups: {overall:.2f}")
        for domain in C.DOMAINS:
            per_domain = [r["f1"] for r in subset if r["domain"] == domain]
            if per_domain:
                label = f"{task.upper()} {domain} mean over {len(per_domain)}:"
                print(f"  {label:<34}{sum(per_domain) / len(per_domain):.2f}")
    return 0


# --- wiring ----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clabsa",
        description="Cross-lingual ABSA adaptation on M-ABSA.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkout", default="vendor/M-ABSA",
                        help="path to the pinned M-ABSA checkout")

    p = sub.add_parser("prepare", parents=[common],
                       help="fetch the M-ABSA checkout and report split sizes")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("run", parents=[common],
                       help="run a strategy over the test matrix")
    p.add_argument("--strategy", required=True,
                   choices=["zero-shot", "three-shot", "qlora"])
    p.add_argument("--out", required=True, help="directory for predictions")
    p.add_argument("--adapter-dir", default="adapters",
                   help="used when --strategy qlora: holds <domain>-<task> dirs")
    p.add_argument("--domains", nargs="*", default=None)
    p.add_argument("--languages", nargs="*", default=None)
    p.add_argument("--tasks", nargs="*", default=None)
    p.add_argument("--max-examples", type=int, default=None,
                   help="first N examples per shard for a smoke test; omit for full test")
    p.add_argument(
        "--correct-empty-tasd",
        action="store_true",
        help="treat empty TASD output as no tuples (not paper-compatible)",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("train", parents=[common],
                       help="train QLoRA adapters on English data")
    p.add_argument("--out", required=True, help="directory for adapters")
    p.add_argument("--domains", nargs="*", default=None)
    p.add_argument("--tasks", nargs="*", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="score prediction files")
    p.add_argument("--predictions", required=True)
    p.add_argument(
        "--correct-empty-tasd",
        action="store_true",
        help="treat empty TASD output as no tuples (not paper-compatible)",
    )
    p.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
