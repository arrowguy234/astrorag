"""
Step 1 tests — verify foundation modules work correctly.

Tests:
  - Paths singleton returns consistent instance
  - All expected directories are created
  - Settings loads from environment and validates
  - Logger is idempotent and creates log file
  - Validation functions run without crashing
"""

from __future__ import annotations

import logging
from   pathlib import Path

import pytest


# ══════════════════════════════════════════════════════════
# paths tests
# ══════════════════════════════════════════════════════════

@pytest.mark.step1
class TestPaths:
    def test_get_paths_returns_singleton(self):
        from astrorag.paths import get_paths
        p1 = get_paths()
        p2 = get_paths()
        assert p1 is p2

    def test_all_paths_are_absolute(self):
        from astrorag.paths import get_paths
        paths = get_paths()
        for name, attr in (
            ("project_root", paths.project_root),
            ("package_root", paths.package_root),
            ("data_dir",     paths.data_dir),
            ("pdf_dir",      paths.pdf_dir),
            ("results_dir",  paths.results_dir),
            ("logs_dir",     paths.logs_dir),
        ):
            assert isinstance(attr, Path), f"{name} not a Path"
            assert attr.is_absolute(),     f"{name} not absolute"

    def test_directories_created_automatically(self):
        from astrorag.paths import get_paths
        paths = get_paths()
        for name, directory in (
            ("data",    paths.data_dir),
            ("pdfs",    paths.pdf_dir),
            ("results", paths.results_dir),
            ("logs",    paths.logs_dir),
        ):
            assert directory.exists(), f"{name} directory not created"
            assert directory.is_dir(),  f"{name} not a directory"

    def test_pdf_path_helper(self):
        from astrorag.paths import get_paths
        paths = get_paths()
        p     = paths.pdf_path("0709.2152")
        assert p.name       == "0709.2152.pdf"
        assert p.parent     == paths.pdf_dir

    def test_result_path_helper(self):
        from astrorag.paths import get_paths
        paths = get_paths()
        p     = paths.result_path("test.json")
        assert p.name       == "test.json"
        assert p.parent     == paths.results_dir

    def test_log_path_helper(self):
        from astrorag.paths import get_paths
        paths = get_paths()
        p     = paths.log_path("test.log")
        assert p.name       == "test.log"
        assert p.parent     == paths.logs_dir

    def test_as_dict_returns_strings(self):
        from astrorag.paths import get_paths
        d = get_paths().as_dict()
        assert isinstance(d, dict)
        for value in d.values():
            assert isinstance(value, str)


# ══════════════════════════════════════════════════════════
# settings tests
# ══════════════════════════════════════════════════════════

@pytest.mark.step1
class TestSettings:
    def test_get_settings_returns_singleton(self):
        from astrorag.config import get_settings
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_defaults_present(self):
        from astrorag.config import get_settings
        s = get_settings()
        assert s.top_k        == 50
        assert s.n_clusters   == 3
        assert s.ppr_alpha    == 0.85
        assert s.bm25_k1      == 1.5
        assert s.bm25_b       == 0.75

    def test_signal_weights_sum_to_one(self):
        from astrorag.config import get_settings
        s = get_settings()
        total = (
            s.w_s1_concept + s.w_s2_biblio
            + s.w_s3_cocitation + s.w_s4_domain
        )
        assert abs(total - 1.0) < 0.01
        assert s.signal_weights_sum_valid

    def test_quality_weights_sum_to_one(self):
        from astrorag.config import get_settings
        s = get_settings()
        total = (
            s.q_weight_faithfulness + s.q_weight_coverage
            + s.q_weight_consistency
        )
        assert abs(total - 1.0) < 0.01
        assert s.quality_weights_sum_valid

    def test_data_dir_expands_tilde(self):
        from astrorag.config import Settings
        s = Settings(data_dir="~/test_dir")
        assert "~" not in s.data_dir
        assert s.data_dir.startswith("/")

    def test_dataset_files_returns_dict(self):
        from astrorag.config import get_settings
        s     = get_settings()
        files = s.dataset_files
        expected = {
            "abstracts", "concept_emb", "paper_concepts",
            "vocabulary", "citations", "index_mapping",
            "years", "identifier_map",
        }
        assert set(files.keys()) == expected
        for path in files.values():
            assert isinstance(path, Path)
            assert path.is_absolute()

    def test_log_level_validation(self):
        from astrorag.config import Settings
        with pytest.raises(ValueError):
            Settings(log_level="INVALID")

    def test_summary_string_generated(self):
        from astrorag.config import get_settings
        s = get_settings()
        text = s.summary()
        assert "AstroRAG Configuration Summary" in text
        assert "Sample size" in text
        assert "PPR alpha"   in text


# ══════════════════════════════════════════════════════════
# logger tests
# ══════════════════════════════════════════════════════════

@pytest.mark.step1
class TestLogger:
    def test_get_logger_returns_logger(self):
        from astrorag.logger import get_logger
        log = get_logger("test.module")
        assert isinstance(log, logging.Logger)
        assert log.name == "test.module"

    def test_setup_logging_idempotent(self):
        from astrorag.logger import setup_logging
        setup_logging(level="DEBUG")
        setup_logging(level="INFO")
        setup_logging(level="WARNING")

    def test_log_file_created(self):
        from astrorag.logger import get_logger
        from astrorag.paths  import get_paths
        log = get_logger("test.log.creation")
        log.info("test message")
        log_path = get_paths().log_path("astrorag.log")
        assert log_path.exists()

    def test_noisy_loggers_silenced(self):
        from astrorag.logger import setup_logging
        setup_logging()
        for name in ("urllib3", "httpx", "openai", "httpcore"):
            assert logging.getLogger(name).level >= logging.WARNING


# ══════════════════════════════════════════════════════════
# validation module tests
# ══════════════════════════════════════════════════════════

@pytest.mark.step1
class TestValidation:
    def test_python_version_check(self):
        from astrorag.validate import validate_python_version
        assert validate_python_version(min_major=3, min_minor=10)

    def test_dependencies_check_returns_dict(self):
        from astrorag.validate import validate_dependencies
        deps = validate_dependencies()
        assert isinstance(deps, dict)
        assert "numpy"    in deps
        assert "pandas"   in deps
        assert "networkx" in deps

    def test_directories_check(self):
        from astrorag.validate import validate_directories
        assert validate_directories()

    def test_config_check(self):
        from astrorag.validate import validate_config
        assert validate_config()

    def test_run_full_validation_returns_dict(self):
        from astrorag.validate import run_full_validation
        result = run_full_validation(check_api=False)
        assert isinstance(result, dict)
        assert "all_passed"   in result
        assert "python"       in result
        assert "dependencies" in result
        assert "directories"  in result
        assert "dataset"      in result
        assert "config"       in result


# ══════════════════════════════════════════════════════════
# package-level tests
# ══════════════════════════════════════════════════════════

@pytest.mark.step1
class TestPackage:
    def test_package_imports(self):
        import astrorag
        assert hasattr(astrorag, "__version__")
        assert astrorag.__version__ == "1.0.0"

    def test_public_api_exposed(self):
        import astrorag
        for name in (
            "Settings", "get_settings",
            "get_logger", "setup_logging",
            "ProjectPaths", "get_paths",
        ):
            assert hasattr(astrorag, name), f"Missing: {name}"