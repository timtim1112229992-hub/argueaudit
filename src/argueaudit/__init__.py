"""argueaudit: a field-separated census of argument components in school science
writing.

Where an instrument stores the parts of an argument in separate fields rather
than as continuous prose, an empty field records an omission rather than an
inarticulate attempt. This package codes such fields by deterministic ruleset,
estimates component rates with exact and resampled intervals, and writes the
provenance needed to re-execute the whole of it.
"""
from __future__ import annotations

__version__ = "1.0.0"

from .config import SETTINGS  # noqa: F401
from .pipeline import run  # noqa: F401

__all__ = ["SETTINGS", "run", "__version__"]
