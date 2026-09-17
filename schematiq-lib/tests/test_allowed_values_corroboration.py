"""Tests for PaperProcessor._enforceable_allowed_values /
_record_allowed_value_confirmations and their effect on cross-document
allowed_values enforcement.

Reproduces the actual bug: a schema's allowed_values, auto-discovered from
one document, must not be force-applied to a different, unrelated
document's real answers until corroborated by enough independent
documents. Follows the repo pattern of driving PaperProcessor methods
unbound with a plain object standing in for `self` (see
test_figure_citations.py), so no real LLM or on-disk pipeline is needed.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from schematiq.value_extraction.core.json_parser import JSONResponseParser
from schematiq.value_extraction.core.paper_processor import PaperProcessor


@dataclass
class FakeColumn:
    name: str
    allowed_values: Optional[List[str]] = None
    auto_expand_threshold: Optional[int] = 2


class FakeSelf:
    """Minimal stand-in for a PaperProcessor instance."""

    def __init__(self):
        self.json_parser = JSONResponseParser()
        self._allowed_value_confirmations: Dict[str, Dict[str, Set[str]]] = {}


def _enforceable(fake_self: FakeSelf, column: FakeColumn) -> List[str]:
    return PaperProcessor._enforceable_allowed_values(fake_self, column)


def _record(fake_self: FakeSelf, parsed: dict, columns: List[FakeColumn], document: str) -> None:
    PaperProcessor._record_allowed_value_confirmations(fake_self, parsed, columns, document)


class TestEnforceableAllowedValues:
    def test_freshly_seeded_value_not_yet_enforceable(self):
        """A value seen in only the single document that produced it (no
        recorded confirmations yet) is below threshold 2 -- not enforceable."""
        s = FakeSelf()
        column = FakeColumn(name="key_finding", allowed_values=["CD45 D1D2 tandem domains"])
        assert _enforceable(s, column) == []

    def test_value_becomes_enforceable_once_corroborated(self):
        """Once a second, distinct document has independently produced the
        same value, it graduates to enforceable."""
        s = FakeSelf()
        column = FakeColumn(name="graph_type", allowed_values=["bar chart"])
        s._allowed_value_confirmations["graph_type"] = {"bar chart": {"paper2.pdf"}}
        assert _enforceable(s, column) == ["bar chart"]

    def test_only_corroborated_values_pass_through_others_stay_suppressed(self):
        """Filtering is per-value, not all-or-nothing for the column."""
        s = FakeSelf()
        column = FakeColumn(
            name="key_finding",
            allowed_values=["bar chart finding", "CD45 D1D2 tandem domains"],
        )
        s._allowed_value_confirmations["key_finding"] = {"bar chart finding": {"paper2.pdf"}}
        assert _enforceable(s, column) == ["bar chart finding"]

    def test_higher_threshold_needs_more_corroboration(self):
        s = FakeSelf()
        column = FakeColumn(name="col", allowed_values=["v"], auto_expand_threshold=3)
        s._allowed_value_confirmations["col"] = {"v": {"paper2.pdf"}}
        assert _enforceable(s, column) == []  # only 1 extra doc, needs 2

        s._allowed_value_confirmations["col"]["v"].add("paper3.pdf")
        assert _enforceable(s, column) == ["v"]

    def test_disabled_threshold_passes_through_unchanged(self):
        """auto_expand_threshold None/0 disables the gate -- today's behavior."""
        s = FakeSelf()
        column = FakeColumn(name="col", allowed_values=["v"], auto_expand_threshold=None)
        assert _enforceable(s, column) == ["v"]

        column2 = FakeColumn(name="col", allowed_values=["v"], auto_expand_threshold=0)
        assert _enforceable(s, column2) == ["v"]

    def test_no_allowed_values_returns_empty(self):
        s = FakeSelf()
        column = FakeColumn(name="col", allowed_values=None)
        assert _enforceable(s, column) == []


class TestRecordAllowedValueConfirmations:
    def test_matching_answer_recorded_under_canonical_value(self):
        """A document whose raw answer fuzzy-matches an allowed_value gets
        recorded under the CANONICAL allowed_value, not its own raw
        wording, so slightly different phrasings all count toward the same
        corroboration bucket."""
        s = FakeSelf()
        column = FakeColumn(name="graph_type", allowed_values=["bar chart"])
        parsed = {"graph_type": {"answer": "Bar Chart", "excerpts": []}}

        _record(s, parsed, [column], "doc1")

        assert s._allowed_value_confirmations["graph_type"]["bar chart"] == {"doc1"}

    def test_non_matching_answer_not_recorded(self):
        s = FakeSelf()
        column = FakeColumn(name="key_finding", allowed_values=["CD45 D1D2 tandem domains"])
        parsed = {"key_finding": {"answer": "Expression of B3/25 glycoprotein", "excerpts": []}}

        _record(s, parsed, [column], "doc1")

        assert s._allowed_value_confirmations == {}

    def test_column_without_allowed_values_skipped(self):
        s = FakeSelf()
        column = FakeColumn(name="free_text_col", allowed_values=None)
        parsed = {"free_text_col": {"answer": "anything", "excerpts": []}}

        _record(s, parsed, [column], "doc1")

        assert s._allowed_value_confirmations == {}


class TestEndToEndCrossDocumentContamination:
    """Reproduces the actual observed bug end-to-end, driving the same
    _enforceable_allowed_values -> postprocess -> _record_allowed_value_confirmations
    sequence every real call site uses."""

    def test_second_documents_real_answer_is_not_overwritten(self):
        s = FakeSelf()
        column = FakeColumn(
            name="key_features_visualized",
            allowed_values=["Diffraction resolution, space groups, and crystallographic R-factors"],
        )

        # Document 2 is a real, unrelated document with its own genuine
        # answer -- the CD45-specific value above was seeded by a different
        # document entirely and has no independent corroboration yet.
        doc2_answer = "Expression of B3/25 glycoprotein on induced HL-60 cells"
        parsed = {
            column.name: {
                "answer": doc2_answer,
                "excerpts": [{"text": "B3/25 glycoprotein expression assay", "source": "doc2"}],
            }
        }
        enforceable = PaperProcessor._enforceable_allowed_values(s, column)
        out, _ = s.json_parser.postprocess(parsed, [column.name], {column.name: enforceable})
        _record(s, parsed, [column], "doc2")

        # Bug reproduction check: doc2's real answer must survive untouched.
        assert out[column.name]["answer"] == doc2_answer
        assert out[column.name]["excerpts"] == [
            {"text": "B3/25 glycoprotein expression assay", "source": "doc2"}
        ]
        assert "normalized_to_allowed" not in out[column.name]

    def test_value_confirmed_by_a_second_document_then_enforces_on_a_third(self):
        s = FakeSelf()
        column = FakeColumn(name="graph_type", allowed_values=["bar chart"])

        # Doc 1 (the document that seeded this value): not yet corroborated
        # by anyone else, so its own answer isn't force-enforced either --
        # it's recorded as a real, matched-but-untrusted occurrence instead.
        enforceable = PaperProcessor._enforceable_allowed_values(s, column)
        assert enforceable == []
        parsed1 = {column.name: {"answer": "bar chart", "excerpts": [{"text": "a bar chart", "source": "doc1"}]}}
        out1, _ = s.json_parser.postprocess(parsed1, [column.name], {column.name: enforceable})
        assert "normalized_to_allowed" not in out1[column.name]
        _record(s, parsed1, [column], "doc1")

        # Doc 1's occurrence is now on record -- threshold (2) is met, so the
        # value graduates to enforceable for doc 2 onward.
        enforceable = PaperProcessor._enforceable_allowed_values(s, column)
        assert enforceable == ["bar chart"]
        parsed2 = {column.name: {"answer": "Bar Chart", "excerpts": [{"text": "shown as bars", "source": "doc2"}]}}
        out2, _ = s.json_parser.postprocess(parsed2, [column.name], {column.name: enforceable})
        assert out2[column.name]["answer"] == "bar chart"
        assert out2[column.name].get("normalized_to_allowed") is True
