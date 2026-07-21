"""
Persistent per-paper context library.

Stores structured summaries from Stage 5 indexed by arxiv_id so
subsequent queries about the same paper can skip re-analysis.
"""

from astrorag.library.library     import ContextLibrary, get_library
from astrorag.library.models      import LibraryEntry, LibraryStats
from astrorag.library.persistence import (
    load_entry, save_entry, list_entries, delete_entry,
)

__all__ = [
    "ContextLibrary",
    "get_library",
    "LibraryEntry",
    "LibraryStats",
    "load_entry",
    "save_entry",
    "list_entries",
    "delete_entry",
]