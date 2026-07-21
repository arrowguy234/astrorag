"""
Step 3 tests — Stage 0 query decomposition.
"""

from __future__ import annotations

import pytest


# ══════════════════════════════════════════════════════════
# rule-based detection tests (no LLM needed)
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestWavelengthDetection:
    def test_xray_detected(self):
        from astrorag.stages.stage0_decompose import detect_wavelength
        assert detect_wavelength(
            "How do AGN jets appear in Chandra X-ray images?"
        ) == "X-ray"

    def test_radio_detected(self):
        from astrorag.stages.stage0_decompose import detect_wavelength
        assert detect_wavelength(
            "What does ALMA reveal about star formation?"
        ) == "radio"

    def test_optical_detected(self):
        from astrorag.stages.stage0_decompose import detect_wavelength
        assert detect_wavelength(
            "How is HST used to measure redshift?"
        ) == "optical"

    def test_infrared_detected(self):
        from astrorag.stages.stage0_decompose import detect_wavelength
        assert detect_wavelength(
            "What does JWST see at 5 microns?"
        ) == "infrared"

    def test_default_is_multi_wavelength(self):
        from astrorag.stages.stage0_decompose import detect_wavelength
        assert detect_wavelength(
            "What causes galaxy quenching?"
        ) == "multi-wavelength"


class TestCatalogDetection:
    def test_chandra_detected(self):
        from astrorag.stages.stage0_decompose import detect_catalogs
        result = detect_catalogs("Chandra observations of AGN cavities")
        assert "CHANDRA" in result

    def test_multiple_catalogs(self):
        from astrorag.stages.stage0_decompose import detect_catalogs
        result = detect_catalogs("Combined Chandra and XMM analysis of A2052")
        assert "CHANDRA" in result

    def test_no_catalogs(self):
        from astrorag.stages.stage0_decompose import detect_catalogs
        assert detect_catalogs("What causes AGN feedback?") == []


class TestQueryTypeDetection:
    def test_observational(self):
        from astrorag.stages.stage0_decompose import detect_query_type
        assert detect_query_type(
            "What observations show jet feedback?"
        ) == "observational"

    def test_theoretical(self):
        from astrorag.stages.stage0_decompose import detect_query_type
        assert detect_query_type(
            "What simulations model AGN feedback?"
        ) == "theoretical"

    def test_default_general(self):
        from astrorag.stages.stage0_decompose import detect_query_type
        assert detect_query_type("How does gravity work?") == "general"


# ══════════════════════════════════════════════════════════
# rule-based decomposition (no LLM)
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestRuleBasedDecomposition:
    def test_returns_three_sub_questions(self):
        from astrorag.stages.stage0_decompose import _rule_based_decompose
        d = _rule_based_decompose("How do AGN jets suppress star formation?")
        assert set(d.sub_questions.keys()) == {"Q1", "Q2", "Q3"}
        for q in d.sub_questions.values():
            assert len(q) > 0

    def test_preserves_original_query(self):
        from astrorag.stages.stage0_decompose import _rule_based_decompose
        q = "How do supernovae enrich the ISM with heavy elements?"
        d = _rule_based_decompose(q)
        assert d.original_query == q

    def test_wavelength_included(self):
        from astrorag.stages.stage0_decompose import _rule_based_decompose
        d = _rule_based_decompose("Chandra observations of AGN cavities")
        assert d.wavelength == "X-ray"


# ══════════════════════════════════════════════════════════
# stage0 class — rule-based path (no LLM)
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestStage0RuleBased:
    def test_stage0_rule_based_only(self):
        from astrorag.stages.stage0_decompose import Stage0Decompose
        stage  = Stage0Decompose()
        result = stage.run(
            query           = "How do AGN jets suppress star formation?",
            rule_based_only = True,
        )
        assert result.fallback_used is True
        assert result.llm_response  is None
        assert len(result.decomposition.sub_questions) == 3

    def test_empty_query_raises(self):
        from astrorag.stages.stage0_decompose import Stage0Decompose
        stage = Stage0Decompose()
        with pytest.raises(ValueError, match="empty"):
            stage.run(query="", rule_based_only=True)

    def test_whitespace_only_raises(self):
        from astrorag.stages.stage0_decompose import Stage0Decompose
        stage = Stage0Decompose()
        with pytest.raises(ValueError, match="empty"):
            stage.run(query="   \n\t  ", rule_based_only=True)


# ══════════════════════════════════════════════════════════
# query decomposition model validation
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestQueryDecompositionModel:
    def test_valid_decomposition_accepted(self):
        from astrorag.llm.models import QueryDecomposition
        d = QueryDecomposition(
            original_query = "test",
            sub_questions  = {
                "Q1": "Q1 text",
                "Q2": "Q2 text",
                "Q3": "Q3 text",
            },
        )
        assert d.wavelength == "multi-wavelength"

    def test_missing_q1_raises(self):
        from astrorag.llm.models import QueryDecomposition
        with pytest.raises(ValueError, match="Q1"):
            QueryDecomposition(
                original_query = "test",
                sub_questions  = {"Q2": "x", "Q3": "y"},
            )

    def test_empty_sub_question_raises(self):
        from astrorag.llm.models import QueryDecomposition
        with pytest.raises(ValueError, match="empty"):
            QueryDecomposition(
                original_query = "test",
                sub_questions  = {"Q1": "", "Q2": "y", "Q3": "z"},
            )

    def test_summary_string_formatted(self):
        from astrorag.llm.models import QueryDecomposition
        d = QueryDecomposition(
            original_query = "test query",
            sub_questions  = {"Q1": "a", "Q2": "b", "Q3": "c"},
            wavelength     = "X-ray",
        )
        text = d.summary()
        assert "test query"        in text
        assert "X-ray"              in text


# ══════════════════════════════════════════════════════════
# LLM tests — require GROQ_API_KEY
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
@pytest.mark.requires_api
class TestStage0WithLLM:
    def test_LLM_decomposition_returns_three_qs(self):
        from astrorag.stages.stage0_decompose import Stage0Decompose
        stage  = Stage0Decompose()
        result = stage.run(
            query   = "How do AGN jets suppress star formation in massive elliptical galaxies?",
            use_llm = True,
        )
        d = result.decomposition
        assert len(d.sub_questions) == 3
        for q in d.sub_questions.values():
            assert len(q) > 10

    def test_LLM_decomposition_captures_wavelength(self):
        from astrorag.stages.stage0_decompose import Stage0Decompose
        stage  = Stage0Decompose()
        result = stage.run(
            query   = "What do Chandra observations reveal about X-ray cavities?",
            use_llm = True,
        )
        assert result.decomposition.wavelength == "X-ray"
        assert "CHANDRA" in result.decomposition.catalogs

    def test_LLM_response_has_telemetry(self):
        from astrorag.stages.stage0_decompose import Stage0Decompose
        stage  = Stage0Decompose()
        result = stage.run(
            query   = "How is star formation regulated in dwarf galaxies?",
            use_llm = True,
        )
        if not result.fallback_used:
            assert result.llm_response is not None
            assert result.llm_response.latency_seconds > 0

    def test_convenience_function(self):
        from astrorag.stages import decompose_query
        d = decompose_query(
            "What causes cooling flows in galaxy clusters?",
            use_llm = True,
        )
        assert len(d.sub_questions) == 3


# ══════════════════════════════════════════════════════════
# LLM client tests
# ══════════════════════════════════════════════════════════

@pytest.mark.step2
class TestLLMClientNoAPI:
    def test_missing_api_key_raises(self, monkeypatch):
        from astrorag.llm.client import LLMClient
        from astrorag.config     import Settings
        # bypass .env by constructing Settings with empty key directly
        empty_settings = Settings(_env_file=None, groq_api_key="")
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            LLMClient(api_key="", settings=empty_settings)

    def test_strip_json_fences_basic(self):
        from astrorag.llm.client import _strip_json_fences
        assert _strip_json_fences('```json\n{"a":1}\n```') == '{"a":1}'
        assert _strip_json_fences('```\n{"a":1}\n```')     == '{"a":1}'
        assert _strip_json_fences('{"a":1}')               == '{"a":1}'
        assert _strip_json_fences('  {"a":1}  ')           == '{"a":1}'


@pytest.mark.step2
@pytest.mark.requires_api
class TestLLMClientWithAPI:
    def test_client_instantiates(self):
        from astrorag.llm import get_llm_client
        client = get_llm_client()
        assert client is not None

    def test_chat_json_valid_response(self):
        from astrorag.llm         import get_llm_client
        from astrorag.llm.models  import QueryDecomposition

        client = get_llm_client()
        response = client.chat_json(
            system = "Return valid JSON only.",
            user   = (
                'Return this JSON exactly: '
                '{"original_query":"test","sub_questions":'
                '{"Q1":"a","Q2":"b","Q3":"c"},'
                '"wavelength":"X-ray","catalogs":[],"query_type":"general"}'
            ),
            schema = QueryDecomposition,
            max_tokens = 300,
            stage_name = "test",
        )
        assert response.data.wavelength == "X-ray"