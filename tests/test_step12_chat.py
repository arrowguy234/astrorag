"""Step 12 tests — chat interface and context library."""

from __future__ import annotations

import pytest


@pytest.mark.step2
class TestChatModels:
    def test_chat_message_defaults(self):
        from astrorag.chat.models import ChatMessage
        m = ChatMessage(role="user", content="hi")
        assert m.role == "user"
        assert m.timestamp != ""

    def test_chat_session_add(self):
        from astrorag.chat.models import ChatSession
        s = ChatSession(
            arxiv_id="1102.1481",
            paper_title="Test",
            original_query="q",
        )
        s.add_user("hello")
        s.add_assistant("hi")
        assert len(s.messages) == 2


@pytest.mark.step2
class TestLibrary:
    def test_library_empty_stats(self, tmp_path):
        from astrorag.chat.library import ContextLibrary
        lib = ContextLibrary(path=tmp_path / "lib.json")
        stats = lib.stats()
        assert stats["n_entries"] == 0

    def test_add_and_get_entry(self, tmp_path):
        from astrorag.chat.library import ContextLibrary, LibraryEntry
        lib = ContextLibrary(path=tmp_path / "lib.json")
        e = LibraryEntry(
            arxiv_id="test.001",
            title="Test Paper",
            abstract="abc",
            original_query="q",
            q_total=0.9,
            decision="ACCEPT",
        )
        lib.add(e)
        retrieved = lib.get("test.001")
        assert retrieved is not None
        assert retrieved.title == "Test Paper"

    def test_search(self, tmp_path):
        from astrorag.chat.library import ContextLibrary, LibraryEntry
        lib = ContextLibrary(path=tmp_path / "lib.json")
        for i in range(3):
            lib.add(LibraryEntry(
                arxiv_id=f"test.{i:03d}",
                title=f"Paper {i} about AGN",
                abstract="abc",
                original_query="q",
            ))
        results = lib.search("AGN")
        assert len(results) == 3

    def test_persistence(self, tmp_path):
        from astrorag.chat.library import ContextLibrary, LibraryEntry
        p = tmp_path / "lib.json"
        lib1 = ContextLibrary(path=p)
        lib1.add(LibraryEntry(
            arxiv_id="x",
            title="T",
            abstract="a",
            original_query="q",
        ))
        # reload
        lib2 = ContextLibrary(path=p)
        assert lib2.get("x") is not None


@pytest.mark.step2
class TestFormatter:
    def _make_entry(self):
        from astrorag.chat.library import LibraryEntry
        return LibraryEntry(
            arxiv_id="1102.1481",
            title="Test Paper on AGN Jets",
            abstract="",
            original_query="How do AGN jets suppress star formation?",
            paper_overview="This paper studies AGN feedback.",
            evidence_type="computational",
            instruments=["Chandra", "XMM-Newton"],
            key_equations=[
                {"equation": "E_cav = 4PV", "variables": "P pressure, V volume"},
            ],
            numerical_results=[
                {"quantity": "jet power", "value": "1.2e44",
                 "uncertainty": "0.3e44", "unit": "erg/s"},
            ],
            sub_question_answers={
                "Q1": {"answered": True, "answer_text": "AGN jets deposit energy",
                       "section": "Methods"},
            },
            q_total=0.98, q_f=0.97, q_c=1.0, q_i=0.96,
            decision="ACCEPT",
        )

    def test_format_summary_markdown(self):
        from astrorag.chat.formatter import format_summary_markdown
        e = self._make_entry()
        md = format_summary_markdown(e)
        assert "1102.1481" in md
        assert "AGN" in md

    def test_format_equations_table(self):
        from astrorag.chat.formatter import format_equations_table
        e = self._make_entry()
        html = format_equations_table(e)
        assert "E_cav" in html

    def test_format_numerical_results(self):
        from astrorag.chat.formatter import format_numerical_results_table
        e = self._make_entry()
        html = format_numerical_results_table(e)
        assert "1.2e44" in html

    def test_format_quality_scores(self):
        from astrorag.chat.formatter import format_quality_scores
        e = self._make_entry()
        html = format_quality_scores(e)
        assert "ACCEPT" in html
        assert "0.98" in html