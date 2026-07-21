"""
Centralised path management for AstroRAG.

Defines and creates the project directory structure. All modules
import paths from here rather than hardcoding them, ensuring the
project works from any working directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib     import Path

# ── project root detection ────────────────────────────────
# resolve to the parent of the astrorag package
_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_ROOT.parent


@dataclass(frozen=True)
class ProjectPaths:
    """
    Canonical paths for the AstroRAG project.

    All paths are absolute Path objects. Directories are created
    automatically on instantiation.
    """

    # ── top-level ────────────────────────────────────────
    project_root: Path = field(default_factory=lambda: _PROJECT_ROOT)
    package_root: Path = field(default_factory=lambda: _PACKAGE_ROOT)

    # ── working directories ──────────────────────────────
    data_dir:    Path = field(default_factory=lambda: _PROJECT_ROOT / "data")
    pdf_dir:     Path = field(default_factory=lambda: _PROJECT_ROOT / "pdfs")
    results_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "results")
    logs_dir:    Path = field(default_factory=lambda: _PROJECT_ROOT / "logs")

    # ── configuration ────────────────────────────────────
    tests_dir:   Path = field(default_factory=lambda: _PROJECT_ROOT / "tests")
    scripts_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "scripts")

    def __post_init__(self) -> None:
        """Create all working directories if they don't exist."""
        for directory in (
            self.data_dir,
            self.pdf_dir,
            self.results_dir,
            self.logs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    # ── convenience methods ──────────────────────────────
    def pdf_path(self, arxiv_id: str) -> Path:
        """Return the full path where a paper's PDF should be stored."""
        return self.pdf_dir / f"{arxiv_id}.pdf"

    def result_path(self, filename: str) -> Path:
        """Return the full path for a result file."""
        return self.results_dir / filename

    def log_path(self, filename: str = "astrorag.log") -> Path:
        """Return the full path for a log file."""
        return self.logs_dir / filename

    def as_dict(self) -> dict[str, str]:
        """Return all paths as strings for JSON serialisation."""
        return {
            "project_root": str(self.project_root),
            "package_root": str(self.package_root),
            "data_dir":     str(self.data_dir),
            "pdf_dir":      str(self.pdf_dir),
            "results_dir":  str(self.results_dir),
            "logs_dir":     str(self.logs_dir),
            "tests_dir":    str(self.tests_dir),
            "scripts_dir":  str(self.scripts_dir),
        }


# ── singleton accessor ────────────────────────────────────
_paths_instance: ProjectPaths | None = None


def get_paths() -> ProjectPaths:
    """
    Return the singleton ProjectPaths instance.

    Created lazily on first call. Use this instead of instantiating
    ProjectPaths directly, ensuring all modules share the same paths.
    """
    global _paths_instance
    if _paths_instance is None:
        _paths_instance = ProjectPaths()
    return _paths_instance