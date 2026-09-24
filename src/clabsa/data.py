"""Reading and preparing M-ABSA splits.

Data layout inside the checkout::

    data/<domain>/<language>/{train,dev,test}.txt

Each line is ``sentence####[[aspect, category, sentiment], ...]``.

The file stores triplets for every task. TASD consumes the triplets directly;
UABSA drops the category field, which is the task definition rather than a
transformation of the data.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import constants as C

Triple = tuple[str, str, str]
Pair = tuple[str, str]
Item = Triple | Pair


@dataclass(frozen=True)
class Example:
    """One annotated sentence."""

    sentence: str
    triplets: tuple[Triple, ...]  # always stored as triplets

    def targets(self, task: str) -> list[Item]:
        """Project the stored triplets onto the requested task."""
        if task == "tasd":
            return list(self.triplets)
        if task == "uabsa":
            return [(aspect, sentiment) for aspect, _category, sentiment
                    in self.triplets]
        raise ValueError(f"unknown task {task!r}; expected one of {C.TASKS}")


# --- parsing ---------------------------------------------------------------

def parse_labels(raw: str) -> tuple[Triple, ...]:
    """Parse the ``[[a, c, s], ...]`` label field into triplets.

    Literal evaluation is used because that is what the benchmark ships and what
    the official pipeline does. ``ast.literal_eval`` is used instead of ``eval``
    so a malformed line cannot execute code.
    """
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError) as error:
        raise ValueError(
            f"could not parse label field: {raw[:60]!r}"
        ) from error

    if not isinstance(parsed, list):
        raise ValueError(f"expected a list of triplets, got {type(parsed).__name__}")

    triplets: list[Triple] = []
    for item in parsed:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"expected a 3-element triplet, got {item!r}")
        aspect, category, sentiment = (str(field) for field in item)
        triplets.append((aspect, category, sentiment))
    return tuple(triplets)


def parse_line(line: str) -> Example | None:
    """Parse one dataset line, returning ``None`` for blank lines."""
    line = line.strip()
    if not line:
        return None
    if "####" not in line:
        raise ValueError(f"line has no '####' separator: {line[:80]!r}")
    sentence, raw_labels = line.split("####", 1)
    # The recorded harness tokenizes with split() and joins with one space.
    # Preserve that normalization for prompts and exact target-span scoring.
    return Example(sentence=" ".join(sentence.split()),
                   triplets=parse_labels(raw_labels))


# --- reading ---------------------------------------------------------------

def split_path(checkout: Path, domain: str, language: str, split: str) -> Path:
    """Return the path of one split file, validating the arguments."""
    if domain not in C.DOMAINS:
        raise ValueError(f"unsupported domain {domain!r}; expected {C.DOMAINS}")
    if split not in C.SPLITS:
        raise ValueError(f"unsupported split {split!r}; expected {C.SPLITS}")
    return Path(checkout) / "data" / domain / language / f"{split}.txt"


def read_split(
    checkout: Path,
    domain: str,
    language: str,
    split: str,
) -> list[Example]:
    """Read one split file into memory."""
    path = split_path(checkout, domain, language, split)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing. The M-ABSA checkout is incomplete; run "
            "`clabsa prepare` to fetch it."
        )
    examples: list[Example] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_line(line)
            if parsed is not None:
                examples.append(parsed)
    return examples


def language_matrix(checkout: Path, split: str) -> dict[str, dict[str, int]]:
    """Sentence counts for every domain/language pair, for a provenance record."""
    counts: dict[str, dict[str, int]] = {}
    for domain in C.DOMAINS:
        counts[domain] = {}
        for language in C.LANGUAGES:
            path = split_path(checkout, domain, language, split)
            if path.is_file():
                counts[domain][language] = sum(
                    1 for line in path.read_text(
                        encoding="utf-8", errors="replace").splitlines()
                    if line.strip()
                )
    return counts


# --- checkout --------------------------------------------------------------

def _run_git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed:\n{result.stderr.strip()}"
        )
    return result.stdout


def ensure_checkout(
    destination: Path,
    revision: str = C.MABSA_REVISION,
    url: str = C.MABSA_REPOSITORY,
) -> Path:
    """Clone (or reuse) the M-ABSA checkout at the pinned revision.

    The revision is pinned because the pinned repository revision does not
    report exactly the sentence counts printed in the M-ABSA paper, and that
    difference is disclosed in the paper's limitations section.
    """
    destination = Path(destination)
    if (destination / ".git").is_dir():
        _run_git("fetch", "--tags", "--quiet", "origin", cwd=destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _run_git("clone", "--quiet", url, str(destination))

    _run_git("checkout", "--quiet", revision, cwd=destination)

    missing = [str(split_path(destination, d, "en", "test"))
               for d in C.DOMAINS
               if not split_path(destination, d, "en", "test").is_file()]
    if missing:
        raise RuntimeError(
            "checkout is missing expected files:\n  " + "\n  ".join(missing)
        )
    return destination
