"""The published TASD category inventories.

The inventories are part of the M-ABSA benchmark definition, not something this
work invents, so they are stored verbatim in ``data/categories.json`` and are
shown to the model in the prompt. Counts match Table 1 of the paper:
Food 13, Laptop 114, Phone 88.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_CATEGORIES_FILE = "data/categories.json"


def _find_repository_root(start: Path) -> Path:
    """Walk upwards until the categories file is found.

    The CLI is supported from a source checkout (including an editable
    install), where the category inventory is kept as a visible data file.
    """
    for candidate in (start, *start.parents):
        if (candidate / _CATEGORIES_FILE).is_file():
            return candidate
    raise FileNotFoundError(
        f"could not locate {_CATEGORIES_FILE} above {start}. "
        "Run from a checkout of this repository."
    )


@lru_cache(maxsize=1)
def _payload() -> dict:
    root = _find_repository_root(Path(__file__).resolve())
    with open(root / _CATEGORIES_FILE, encoding="utf-8") as handle:
        return json.load(handle)


def categories_for(domain: str) -> list[str]:
    """Return the published TASD category inventory for a domain."""
    table = _payload()["categories"]
    if domain not in table:
        raise KeyError(
            f"no category inventory for {domain!r}; known domains are "
            f"{sorted(table)}"
        )
    return list(table[domain])


def category_block(domain: str) -> str:
    """Format the inventory as a bullet list for inclusion in a prompt."""
    return "\n".join(f"- {name}" for name in categories_for(domain))


def sentiments() -> list[str]:
    """Return the sentiment labels declared by the benchmark metadata."""
    return list(_payload()["sentiments"])


def source_note() -> str:
    """Provenance string recorded in run metadata."""
    return str(_payload()["source"])
