"""
Step 8 tests — Stage 5 deep summarisation and quality gate.
"""

from __future__ import annotations

import pytest


# ══════════════════════════════════════════════════════════
# equation extraction
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestEquationExtraction:
    def test_extract_inline_math(self):
        from astrorag.extraction import extract_equations
        text = "The mass is defined as $M = 4 \\pi r^2 \\rho$ in units of..."
        eqs = extract_equations(text)
        assert any("M" in e for e in eqs)

    def test_extract_variable_assignment(self):
        from astrorag.extraction import extract_equations
        text = "We define P_jet = 4 P V / t_buoy for the cavity power."
        eqs = extract_equations(text)
        assert any("P_jet" in e or "4 P V" in e for e in eqs)

    def test_extract_measurement_with_uncertainty(self):
        from astrorag.extraction import extract_equations
        text = "The temperature is 2.5 ± 0.3 keV in the cluster core."
        eqs = extract_equations(text)
        assert any("2.5" in e for e in eqs)

    def test_extract_measurements_with_units(self):
        from astrorag.extraction import extract_measurements
        text = "We find L_x = 1.2e44 erg/s and R = 5.3 kpc in the core."
        ms = extract_measurements(text)
        assert any("erg" in m for m in ms)

    def test_empty_input(self):
        from astrorag.extraction import extract_equations, extract_measurements
        assert extract_equations("") == []
        assert extract_measurements("") == []


# ══════════════════════════════════════════════════════════
# table detection
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestTableDetection:
    def test_numeric_row_detected(self):
        from astrorag.extraction import is_table_row
        assert is_table_row("A2052   15.2   1.2e44   3.5")

    def test_pipe_row_detected(self):
        from astrorag.extraction import is_table_row
        assert is_table_row("| Galaxy | Mass | Redshift | Age |")

    def test_prose_not_detected(self):
        from astrorag.extraction import is_table_row
        assert not is_table_row("This is prose text.")

    def test_extract_tables(self):
        from astrorag.extraction import extract_tables
        text = (
            "Table 1: Cluster properties\n"
            "A2052   15.2   1.2e44   3.5\n"
            "M87     2.1    8.5e43   0.8\n"
            "\nSome prose here.\n"
            "Later table\n"
            "X1      1.0    2.0    3.0\n"
            "X2      4.0    5.0    6.0\n"
        )
        tables = extract_tables(text)
        assert len(tables) >= 1


# ══════════════════════════════════════════════════════════
# section routing
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestSectionRouting:
    def test_equation_query_routes_to_methods(self):
        from astrorag.extraction import detect_question_type
        sections = detect_question_type("What is the equation for cavity power?")
        assert sections[0] in ("Methods", "Methodology")

    def test_measurement_query_routes_to_results(self):
        from astrorag.extraction import detect_question_type
        sections = detect_question_type("What are the measurement values?")
        assert "Results" in sections[:3]

    def test_default_priority(self):
        from astrorag.extraction import detect_question_type
        sections = detect_question_type("Random query with no keywords")
        assert "Results" in sections


# ══════════════════════════════════════════════════════════
# quality gate
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestQualityGate:
    def _make_summary(self, **overrides):
        from astrorag.llm.models import (
            StructuredSummary, SubQuestionAnswer,
            KeyEquation, NumericalResult,
        )
        defaults = {
            "paper_overview": "This paper is about AGN feedback.",
            "sub_question_answers": {
                "Q1": SubQuestionAnswer(
                    answered=True,
                    answer_text="AGN jets deposit energy into the ICM.",
                    section="Introduction",
                ),
                "Q2": SubQuestionAnswer(
                    answered=True,
                    answer_text="Chandra observations show X-ray cavities.",
                    section="Observations",
                ),
                "Q3": SubQuestionAnswer(
                    answered=True,
                    answer_text="Jet power is 1.2e44 erg per second.",
                    section="Results",
                ),
            },
            "evidence_type": "observational",
            "instruments":   ["Chandra"],
            "key_equations": [
                KeyEquation(equation="P = 4PV/t", variables="P pressure", section="Methods"),
            ],
            "numerical_results": [
                NumericalResult(quantity="jet power", value="1.2e44",
                                uncertainty="0.3e44", unit="erg/s"),
            ],
            "key_snippet": "AGN jets deposit energy into the ICM through cavities.",
        }
        defaults.update(overrides)
        return StructuredSummary(**defaults)

    def test_high_quality_accepted(self):
        from astrorag.extraction import assess_quality, QualityDecision
        summary = self._make_summary()
        paper_text = (
            "AGN jets deposit energy into the ICM through cavities. "
            "Chandra observations show X-ray cavities in massive clusters. "
            "Jet power is 1.2e44 erg per second for typical systems. "
            "The pressure and cavity volume determine the cavity enthalpy."
        )
        qa = assess_quality(summary, paper_text)
        assert qa.scores.Q_total > 0.6
        assert qa.decision in (QualityDecision.ACCEPT, QualityDecision.RETRY)

    def test_missing_equations_penalized(self):
        from astrorag.extraction import assess_quality
        summary = self._make_summary(key_equations=[], numerical_results=[])
        paper_text = "AGN jets deposit energy into the ICM through cavities. " * 5
        qa = assess_quality(summary, paper_text)
        # Q_i should be lower
        assert qa.scores.Q_i < 1.0

    def test_low_coverage_reduces_qc(self):
        from astrorag.extraction import assess_quality
        from astrorag.llm.models import SubQuestionAnswer
        summary = self._make_summary()
        summary.sub_question_answers["Q2"] = SubQuestionAnswer(
            answered=False, answer_text="", section=""
        )
        summary.sub_question_answers["Q3"] = SubQuestionAnswer(
            answered=False, answer_text="", section=""
        )
        paper_text = "AGN jets deposit energy through cavities."
        qa = assess_quality(summary, paper_text)
        assert qa.scores.Q_c == pytest.approx(1/3, abs=0.01)


# ══════════════════════════════════════════════════════════
# integration test — needs API
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
@pytest.mark.requires_api
@pytest.mark.requires_data
class TestStage5Integration:
    N = 2000

    def test_full_pipeline_stage_5(self):
        from astrorag.data       import DataLoader
        from astrorag.data.models import LoadConfig
        from astrorag.stages     import (
            Stage0Decompose, Stage1BM25, Stage2Graph,
            Stage3Rerank, Stage4PDF, Stage5Summarise,
        )

        corpus = DataLoader(config=LoadConfig(
            sample_size=self.N, use_cache=True, show_progress=False,
        )).load()

        query = "How do AGN jets suppress star formation?"

        s0 = Stage0Decompose().run(query)
        s1 = Stage1BM25(corpus=corpus).run(query, top_k=30)
        s2 = Stage2Graph(corpus=corpus).run(s1)
        s3 = Stage3Rerank().run(retrieval=s1, graph_context=s2,
                                 decomposition=s0.decomposition)
        s4 = Stage4PDF().run(s3)
        if not s4.success:
            pytest.skip("PDF fetch failed")

        s5 = Stage5Summarise().run(
            decomposition = s0.decomposition,
            retrieval     = s1,
            stage3_result = s3,
            initial_pdf   = s4,
        )
        assert s5.summary is not None
        assert 0.0 <= s5.quality.scores.Q_total <= 1.0
        assert s5.n_attempts >= 1