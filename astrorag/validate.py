"""
Environment and dataset validation for AstroRAG.

Verifies before running the pipeline:
- Python version
- All required dependencies installed and importable
- Dataset files exist and are accessible
- Groq API key is set (optional check)
- Directory structure is correct
- Configuration is valid

Run with:
    python -m astrorag.validate
    astrorag-validate       # if installed as CLI
"""

from __future__ import annotations

import gzip
import importlib
import json
import sys
from   pathlib import Path

from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel

from astrorag.config import get_settings
from astrorag.logger import get_logger
from astrorag.paths  import get_paths

console = Console()
logger  = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# individual validation functions
# ══════════════════════════════════════════════════════════

def validate_python_version(min_major: int = 3, min_minor: int = 10) -> bool:
    """Check that Python version meets minimum requirement."""
    v = sys.version_info
    ok = (v.major > min_major) or (v.major == min_major and v.minor >= min_minor)
    if ok:
        logger.info(f"✓ Python {v.major}.{v.minor}.{v.micro}")
    else:
        logger.error(
            f"✗ Python {v.major}.{v.minor}.{v.micro} — "
            f"requires >= {min_major}.{min_minor}"
        )
    return ok


def validate_dependencies() -> dict[str, bool]:
    """
    Verify all core dependencies are importable.

    Returns a dict mapping package name → import success.
    """
    required = [
        "numpy", "pandas", "scipy", "sklearn",
        "networkx", "rank_bm25",
        "fitz",              # pymupdf
        "pdfplumber",
        "pypdf",
        "pdfminer",
        "groq",
        "requests",
        "dotenv",            # python-dotenv
        "pydantic",
        "pydantic_settings",
        "ipywidgets",
        "tqdm",
        "rich",
    ]

    results: dict[str, bool] = {}
    for pkg in required:
        try:
            importlib.import_module(pkg)
            results[pkg] = True
            logger.debug(f"✓ {pkg}")
        except ImportError as e:
            results[pkg] = False
            logger.error(f"✗ {pkg} — {e}")

    return results


def validate_directories() -> bool:
    """Verify project directory structure exists."""
    paths       = get_paths()
    all_present = True

    for name, path in (
        ("data",    paths.data_dir),
        ("pdfs",    paths.pdf_dir),
        ("results", paths.results_dir),
        ("logs",    paths.logs_dir),
    ):
        if path.exists() and path.is_dir():
            logger.info(f"✓ {name} directory: {path}")
        else:
            logger.error(f"✗ {name} directory missing: {path}")
            all_present = False

    return all_present


def validate_dataset() -> dict[str, dict]:
    """
    Verify all required dataset files exist and are readable.

    For each file returns a dict with:
        exists:      bool
        size_bytes:  int
        size_mb:     float
        readable:    bool
        head_ok:     bool  (can read first line/record without error)
    """
    settings  = get_settings()
    files     = settings.dataset_files
    results:  dict[str, dict] = {}

    for name, path in files.items():
        info: dict = {
            "path":       str(path),
            "exists":     False,
            "size_bytes": 0,
            "size_mb":    0.0,
            "readable":   False,
            "head_ok":    False,
        }

        if not path.exists():
            logger.error(f"✗ dataset file missing: {name} @ {path}")
            results[name] = info
            continue

        info["exists"]     = True
        info["size_bytes"] = path.stat().st_size
        info["size_mb"]    = info["size_bytes"] / (1024 * 1024)

        try:
            if path.suffix == ".gz":
                with gzip.open(path, "rt", encoding="utf-8",
                               errors="replace") as fh:
                    first_line = fh.readline()
                    if path.name.endswith(".jsonl.gz"):
                        json.loads(first_line.strip())
                info["head_ok"] = True
            elif path.suffix == ".npz":
                import numpy as np
                arr = np.load(path)
                _   = arr.files
                info["head_ok"] = True
            elif path.suffix == ".npy":
                import numpy as np
                arr = np.load(path)
                _   = arr.shape
                info["head_ok"] = True
            info["readable"] = True
            logger.info(
                f"✓ {name:<20} {info['size_mb']:>7.1f} MB @ {path.name}"
            )
        except Exception as e:
            logger.error(f"✗ {name} unreadable: {e}")

        results[name] = info

    return results


def validate_api_key() -> bool:
    """Check that GROQ_API_KEY is set (does not test the key itself)."""
    settings = get_settings()
    if settings.groq_api_key and len(settings.groq_api_key) > 20:
        logger.info(f"✓ GROQ_API_KEY set (length={len(settings.groq_api_key)})")
        return True
    logger.error("✗ GROQ_API_KEY not set or invalid")
    logger.error("  Set in .env file or export GROQ_API_KEY=gsk_...")
    return False


def validate_config() -> bool:
    """Verify configuration weights sum correctly."""
    settings = get_settings()
    ok       = True

    if not settings.signal_weights_sum_valid:
        total = (
            settings.w_s1_concept + settings.w_s2_biblio
            + settings.w_s3_cocitation + settings.w_s4_domain
        )
        logger.error(f"✗ Signal weights sum to {total:.4f}, expected 1.0")
        ok = False
    else:
        logger.info("✓ Signal weights sum to 1.0")

    if not settings.quality_weights_sum_valid:
        total = (
            settings.q_weight_faithfulness
            + settings.q_weight_coverage
            + settings.q_weight_consistency
        )
        logger.error(f"✗ Quality weights sum to {total:.4f}, expected 1.0")
        ok = False
    else:
        logger.info("✓ Quality weights sum to 1.0")

    return ok


# ══════════════════════════════════════════════════════════
# full validation
# ══════════════════════════════════════════════════════════

def run_full_validation(check_api: bool = True) -> dict:
    """
    Run all validation checks and return summary dict.

    Args:
        check_api: If True, check GROQ_API_KEY is set.

    Returns:
        Dict with:
            all_passed:   bool
            python:       bool
            dependencies: dict[str, bool]
            directories:  bool
            dataset:      dict[str, dict]
            api_key:      bool | None
            config:       bool
    """
    console.print()
    console.print(Panel.fit(
        "[bold cyan]AstroRAG Environment Validation[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    console.print("[bold]1. Python version[/bold]")
    python_ok = validate_python_version()

    console.print()
    console.print("[bold]2. Dependencies[/bold]")
    deps         = validate_dependencies()
    deps_all_ok  = all(deps.values())

    console.print()
    console.print("[bold]3. Directory structure[/bold]")
    dirs_ok = validate_directories()

    console.print()
    console.print("[bold]4. Dataset files[/bold]")
    dataset      = validate_dataset()
    dataset_ok   = all(f["exists"] and f["readable"] for f in dataset.values())

    console.print()
    console.print("[bold]5. Configuration[/bold]")
    config_ok = validate_config()

    api_ok: bool | None = None
    if check_api:
        console.print()
        console.print("[bold]6. API keys[/bold]")
        api_ok = validate_api_key()

    all_passed = (
        python_ok and deps_all_ok and dirs_ok
        and dataset_ok and config_ok
        and (api_ok is True or api_ok is None)
    )

    # ── summary table ────────────────────────────────────
    console.print()
    table = Table(title="Validation Summary", show_header=True,
                  header_style="bold cyan")
    table.add_column("Check", style="cyan")
    table.add_column("Result", justify="center")

    def _mark(ok: bool | None) -> str:
        if ok is True:  return "[bold green]PASS[/bold green]"
        if ok is False: return "[bold red]FAIL[/bold red]"
        return "[bold yellow]SKIP[/bold yellow]"

    table.add_row("Python version",        _mark(python_ok))
    table.add_row("Dependencies",           _mark(deps_all_ok))
    table.add_row("Directory structure",    _mark(dirs_ok))
    table.add_row("Dataset files",          _mark(dataset_ok))
    table.add_row("Configuration",          _mark(config_ok))
    if check_api:
        table.add_row("API key",            _mark(api_ok))

    console.print(table)

    console.print()
    if all_passed:
        console.print(Panel.fit(
            "[bold green]✓ All checks passed — ready to run pipeline[/bold green]",
            border_style="green",
        ))
    else:
        console.print(Panel.fit(
            "[bold red]✗ Some checks failed — fix issues before running pipeline[/bold red]",
            border_style="red",
        ))
    console.print()

    return {
        "all_passed":   all_passed,
        "python":       python_ok,
        "dependencies": deps,
        "directories":  dirs_ok,
        "dataset":      dataset,
        "api_key":      api_ok,
        "config":       config_ok,
    }


def main() -> int:
    """CLI entry point — return 0 on success, 1 on failure."""
    result = run_full_validation()
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())