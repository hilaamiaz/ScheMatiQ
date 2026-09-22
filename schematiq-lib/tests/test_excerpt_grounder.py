"""Tests for excerpt grounding / hallucination detection."""

import pytest

from schematiq.value_extraction.utils.excerpt_grounder import ExcerptGrounder


class TestExcerptGrounder:
    def setup_method(self):
        self.grounder = ExcerptGrounder()

    def test_exact_match(self):
        source = "GPT-4 achieves 86.4% accuracy on the MMLU benchmark."
        start, end, status = self.grounder.ground_excerpt("86.4% accuracy", source)
        assert status == "exact"
        assert source[start:end] == "86.4% accuracy"

    def test_case_insensitive_match(self):
        source = "The BERT model was fine-tuned on SQuAD."
        start, end, status = self.grounder.ground_excerpt("the bert model", source)
        assert status == "case_insensitive"

    def test_fuzzy_match(self):
        source = "The model achieved an accuracy of 86.4 percent on MMLU."
        start, end, status = self.grounder.ground_excerpt(
            "accuracy of 86.4% on MMLU", source
        )
        assert status == "fuzzy"

    def test_not_found(self):
        source = "This paper discusses climate change."
        start, end, status = self.grounder.ground_excerpt(
            "GPT-4 achieves 86.4%", source
        )
        assert status == "not_found"
        assert start is None
        assert end is None

    def test_empty_excerpt(self):
        start, end, status = self.grounder.ground_excerpt("", "some text")
        assert status == "not_found"

    def test_empty_source(self):
        start, end, status = self.grounder.ground_excerpt("test", "")
        assert status == "not_found"

    def test_short_excerpt_no_fuzzy(self):
        """Excerpts shorter than 3 words skip fuzzy matching."""
        source = "The cat sat on the mat."
        start, end, status = self.grounder.ground_excerpt("dog", source)
        assert status == "not_found"

    def test_fuzzy_match_with_irregular_spacing(self):
        """A real fuzzy match must not be lost to double-spaced source text.

        Docling leaves PDF-justification double-space artifacts in extracted
        text (e.g. "Fig.  1 a, Stimulation  by  SEM..."). Rejoining the
        matched word window with single spaces and re-`.find()`-ing it
        against the original (irregularly spaced) source used to make a real
        match report "not_found" purely on whitespace.
        """
        source = "The  model  achieved  an  accuracy  of  86.4  percent  on  MMLU."
        start, end, status = self.grounder.ground_excerpt(
            "accuracy of 86.4% on MMLU", source
        )
        assert status == "fuzzy"
        assert start is not None and end is not None
        # 5-word window (matching the excerpt's own 5 tokens); the excerpt's
        # "86.4%" folds source's "86.4"+"percent" into one token, so the
        # window can't also reach "MMLU" -- the important thing is the
        # double-spacing between real words was preserved, not collapsed or
        # mis-searched.
        assert source[start:end] == "accuracy  of  86.4  percent  on"

    def test_fuzzy_match_offsets_point_at_real_span(self):
        """The returned (start, end) must slice out the actual matched words,
        not an offset recovered by re-searching a differently-spaced string."""
        source = "Intro text. The model achieved an accuracy of 86.4 percent on MMLU. Outro."
        start, end, status = self.grounder.ground_excerpt(
            "accuracy of 86.4% on MMLU", source
        )
        assert status == "fuzzy"
        assert source[start:end] == "accuracy of 86.4 percent on"


class TestGroundAllExcerpts:
    def setup_method(self):
        self.grounder = ExcerptGrounder()

    def test_ground_string_excerpts(self):
        source = "GPT-4 achieves 86.4% accuracy on MMLU."
        result = {
            "model": {
                "answer": "GPT-4",
                "excerpts": ["GPT-4 achieves 86.4% accuracy"],
            },
        }
        stats = self.grounder.ground_all_excerpts(result, source)
        assert stats["exact"] == 1
        # Excerpts should now be dicts with grounding info
        exc = result["model"]["excerpts"][0]
        assert isinstance(exc, dict)
        assert exc["grounding_status"] == "exact"
        assert exc["char_start"] is not None

    def test_skip_metadata_keys(self):
        result = {
            "_source_document": "test.pdf",
            "model": {"answer": "GPT-4", "excerpts": ["GPT-4"]},
        }
        stats = self.grounder.ground_all_excerpts(result, "GPT-4 is great")
        assert stats["exact"] == 1
        # Metadata key should be unchanged
        assert result["_source_document"] == "test.pdf"

    def test_mixed_grounding_statuses(self):
        source = "GPT-4 achieves 86.4% accuracy on MMLU benchmark."
        result = {
            "model": {
                "answer": "GPT-4",
                "excerpts": [
                    "GPT-4 achieves 86.4%",
                    "This excerpt does not exist in the source at all whatsoever",
                ],
            },
        }
        stats = self.grounder.ground_all_excerpts(result, source)
        assert stats["exact"] >= 1
        assert stats["not_found"] >= 1

    def test_empty_extraction(self):
        stats = self.grounder.ground_all_excerpts({}, "some source text")
        assert stats == {"exact": 0, "case_insensitive": 0, "fuzzy": 0, "not_found": 0}

    def test_duplicate_excerpt_text_resolves_to_different_occurrences(self):
        """Two different columns citing the identical phrase, which really
        appears twice in the source, must not both collapse onto the
        document's first occurrence."""
        source = (
            "Results showed wild-type vs ST3Gal-I deficient mice differed. "
            "Figure 2 legend: wild-type vs ST3Gal-I deficient comparison shown."
        )
        result = {
            "finding_1": {
                "answer": "difference observed",
                "excerpts": ["wild-type vs ST3Gal-I deficient"],
            },
            "finding_2": {
                "answer": "figure comparison",
                "excerpts": ["wild-type vs ST3Gal-I deficient"],
            },
        }
        self.grounder.ground_all_excerpts(result, source)
        first_start = result["finding_1"]["excerpts"][0]["char_start"]
        second_start = result["finding_2"]["excerpts"][0]["char_start"]
        assert first_start is not None and second_start is not None
        assert first_start != second_start
        assert first_start == source.index("wild-type vs ST3Gal-I deficient")
        assert second_start == source.rindex("wild-type vs ST3Gal-I deficient")

    def test_duplicate_excerpt_more_repeats_than_occurrences_falls_back(self):
        """A third identical-text excerpt, with no remaining occurrence left
        to claim, still grounds to a real position instead of not_found."""
        source = "The gene was upregulated in the treated group. The treated group showed changes."
        result = {
            "col_a": {"answer": "a", "excerpts": ["the treated group"]},
            "col_b": {"answer": "b", "excerpts": ["the treated group"]},
            "col_c": {"answer": "c", "excerpts": ["the treated group"]},
        }
        stats = self.grounder.ground_all_excerpts(result, source)
        assert stats["not_found"] == 0
        for col in ("col_a", "col_b", "col_c"):
            exc = result[col]["excerpts"][0]
            assert exc["grounding_status"] != "not_found"
            assert exc["char_start"] is not None

    def test_short_excerpt_with_irregular_spacing_still_grounds(self):
        """A short (<3-word) excerpt that only differs from the source by
        whitespace can't reach the fuzzy phase (which requires >= 3 words),
        so it needs the whitespace-normalized phase to ground at all."""
        source = "The comparison of ST3Gal-I  deficient mice was notable."
        start, end, status = self.grounder.ground_excerpt("ST3Gal-I deficient", source)
        assert status != "not_found"
        assert start is not None and end is not None
        assert source[start:end] == "ST3Gal-I  deficient"
