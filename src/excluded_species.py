"""Utilities for consistently omitting species flagged for exclusion."""

from __future__ import annotations

import csv
from pathlib import Path


def normalise_species_name(name: object) -> str:
    """Return the underscore-separated form used by trees and exclusion data."""
    return str(name).strip().replace(" ", "_")


def load_excluded_species(path: str | Path | None = None) -> tuple[set[int], set[str]]:
    """Load excluded class indices and scientific names from the project list."""
    exclusion_path = Path(path) if path is not None else Path(__file__).with_name("excluded_species.csv")
    excluded_indices: set[int] = set()
    excluded_names: set[str] = set()

    with exclusion_path.open(newline="", encoding="utf-8") as file:
        for row in csv.reader(file):
            if not row:
                continue
            excluded_indices.add(int(row[0]))
            if len(row) > 1:
                excluded_names.add(normalise_species_name(row[1]))

    return excluded_indices, excluded_names


def is_excluded_species(
    excluded_indices: set[int],
    excluded_names: set[str],
    *,
    index: int | None = None,
    name: object | None = None,
) -> bool:
    """Check an original class index and/or a scientific name against the list."""
    return (
        index is not None and index in excluded_indices
    ) or (
        name is not None and normalise_species_name(name) in excluded_names
    )
