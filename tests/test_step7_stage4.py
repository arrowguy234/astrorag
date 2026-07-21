"""
Step 7 tests — Stage 4 PDF fetch and section parsing.
"""

from __future__ import annotations

import pytest


# ══════════════════════════════════════════════════════════
# URL construction
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestURLConstruction:
    def test_normalize_removes_version(self):
        from astrorag.pdf import normalize_arxiv_id_for_url
        assert normalize_arxiv_id_for_url("0704.0007v2") == "0704.0007"
        assert normalize_arxiv_id_for_url("0704.0007v1") == "0704.0007"

    def test_normalize_preserves_new_style(self):
        from astrorag.pdf import normalize_arxiv_id_for_url
        assert normalize_arxiv_id_for_url("0704.0007") == "0704.0007"

    def test_normalize_preserves_old_style(self):
        from astrorag.pdf import normalize_arxiv_id_for_url
        assert normalize_arxiv_id_for_url("astro-ph/0703001") == "astro-ph/0703001"

    def test_build_urls_returns_list(self):
        from astrorag.pdf import build_arxiv_urls
        urls = build_arxiv_urls("0704.0007")
        assert isinstance(urls, list)
        assert len(urls) >= 2
        assert all(u.startswith("https://arxiv.org") for u in urls)


# ══════════════════════════════════════════════════════════
# section detection
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestSectionDetection:
    def test_detect_basic_sections(self):
        from astrorag.pdf.sections import detect_sections
        text = (
            "Introduction\nThis paper describes...\n"
            "Methods\nWe use MCMC...\n"
            "Results\nWe find that...\n"
            "Conclusion\nIn summary...\n"
        )
        positions = detect_sections(text)
        names = [n for _, n in positions]
        assert "Introduction" in names
        assert "Methods"      in names
        assert "Results"      in names

    def test_detect_empty_returns_empty(self):
        from astrorag.pdf.sections import detect_sections
        assert detect_sections("") == []

    def test_split_by_sections_produces_dict(self):
        from astrorag.pdf.sections import split_by_sections
        text = (
            "Introduction\nThis paper describes AGN feedback.\n\n"
            "Methods\nWe use MCMC sampling.\n\n"
            "Results\nWe find that cavity power scales with mass.\n\n"
        )
        sections = split_by_sections(text)
        assert "Introduction" in sections
        assert "Methods"      in sections
        assert "Results"      in sections

    def test_results_gets_larger_budget(self):
        from astrorag.pdf.sections import split_by_sections
        # results section should get up to 4000 chars, others 2000
        long_results = "x" * 5000
        text = f"Introduction\nshort intro.\nResults\n{long_results}\n"
        sections = split_by_sections(text, max_default=1000, max_important=3000)
        assert sections["Results"].n_chars <= 3000
        assert sections["Results"].n_chars > 1000

    def test_no_sections_returns_full_text_bucket(self):
        from astrorag.pdf.sections import split_by_sections
        text = "just some text with no headers whatsoever"
        sections = split_by_sections(text)
        assert "Full text" in sections


# ══════════════════════════════════════════════════════════
# model tests
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestModels:
    def test_section_word_count(self):
        from astrorag.pdf.models import Section
        s = Section(name="Results", text="one two three four")
        assert s.n_words == 4
        assert s.n_chars == len("one two three four")

    def test_pdf_document_has_results_true(self):
        from astrorag.pdf.models import PDFDocument, Section
        from pathlib             import Path
        doc = PDFDocument(
            arxiv_id  = "test",
            pdf_path  = Path("/tmp/x.pdf"),
            full_text = "x",
            sections  = {"Results": Section(name="Results", text="found stuff")},
        )
        assert doc.has_results
        assert doc.get_section("Results") == "found stuff"

    def test_pdf_document_case_insensitive_lookup(self):
        from astrorag.pdf.models import PDFDocument, Section
        from pathlib             import Path
        doc = PDFDocument(
            arxiv_id  = "test",
            pdf_path  = Path("/tmp/x.pdf"),
            full_text = "x",
            sections  = {"RESULTS": Section(name="RESULTS", text="found")},
        )
        assert doc.get_section("results") == "found"


# ══════════════════════════════════════════════════════════
# fetcher integration — requires internet
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestFetcherIntegration:
    """These need arxiv.org reachable — use a small well-known paper."""

    def test_fetch_real_paper(self, tmp_path):
        from astrorag.pdf import fetch_arxiv_pdf
        # a small, always-available paper
        path, from_cache, err = fetch_arxiv_pdf(
            arxiv_id  = "0704.0007",
            cache_dir = tmp_path,
        )
        assert err == "" or path is not None
        if path is not None:
            assert path.exists()
            assert path.stat().st_size > 1000

    def test_cache_hit_second_time(self, tmp_path):
        from astrorag.pdf import fetch_arxiv_pdf
        # first fetch
        p1, from_cache1, _ = fetch_arxiv_pdf(
            arxiv_id="0704.0007", cache_dir=tmp_path
        )
        if p1 is None:
            pytest.skip("Could not reach arXiv")
        # second should be cache hit
        p2, from_cache2, _ = fetch_arxiv_pdf(
            arxiv_id="0704.0007", cache_dir=tmp_path
        )
        assert from_cache2 is True
        assert p1 == p2


# ══════════════════════════════════════════════════════════
# Stage 4 end-to-end integration
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestStage4EndToEnd:
    def test_stage4_on_real_paper(self):
        from astrorag.retrieval import RetrievalResult
        from astrorag.stages    import Stage4PDF

        stage4  = Stage4PDF()
        paper   = RetrievalResult(
            arxiv_id  = "0704.0007",
            paper_idx = 0,
        )
        pdf_doc = stage4.run(paper)

        if not pdf_doc.success:
            pytest.skip(f"PDF fetch failed: {pdf_doc.error}")

        assert pdf_doc.n_chars_total > 1000
        assert pdf_doc.n_pages > 0
        assert len(pdf_doc.sections) >= 1
        assert pdf_doc.extractor in {"pymupdf", "pdfplumber"}