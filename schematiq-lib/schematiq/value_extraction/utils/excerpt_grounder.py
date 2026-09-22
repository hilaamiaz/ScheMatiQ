"""Verify extraction excerpts against source text for hallucination detection."""

import difflib
import re
from typing import Dict, List, Optional, Tuple


class ExcerptGrounder:
    """Verify extraction excerpts against source text.

    Uses exact matching first, then fuzzy sliding-window matching
    for paraphrased excerpts.
    """

    def __init__(self, fuzzy_threshold: float = 0.6):
        self.fuzzy_threshold = fuzzy_threshold

    def ground_excerpt(
        self,
        excerpt: str,
        source_text: str,
        source_lower: Optional[str] = None,
        search_start: int = 0,
    ) -> Tuple[Optional[int], Optional[int], str]:
        """Find excerpt location in source text.

        Args:
            excerpt: The excerpt text to locate.
            source_text: The original source document text.
            source_lower: Optional pre-lowercased source_text to avoid
                recomputing it for every excerpt.
            search_start: Only consider matches at or after this offset.
                Used by ground_all_excerpts so that a second excerpt with
                text identical to an earlier one resolves to the
                document's *next* occurrence instead of colliding with
                the earlier excerpt's match (plain .find() always returns
                the first occurrence, with no memory of what's already
                been claimed). If nothing is found from search_start
                onward, this falls back to an unconstrained search from
                position 0 -- so a duplicate excerpt with no remaining
                occurrence still grounds to *a* real position rather than
                reporting not_found.

        Returns:
            (start_pos, end_pos, status) where status is
            "exact", "case_insensitive", "fuzzy", or "not_found".
        """
        if not excerpt or not source_text:
            return None, None, "not_found"

        if source_lower is None:
            source_lower = source_text.lower()

        result = self._search_from(excerpt, source_text, source_lower, search_start)
        if result[2] != "not_found" or search_start == 0:
            return result
        return self._search_from(excerpt, source_text, source_lower, 0)

    def _search_from(
        self,
        excerpt: str,
        source_text: str,
        source_lower: str,
        min_pos: int,
    ) -> Tuple[Optional[int], Optional[int], str]:
        """Run the exact / case-insensitive / whitespace-normalized / fuzzy
        phases, considering only matches starting at or after min_pos."""
        # Phase 1: Exact substring search
        pos = source_text.find(excerpt, min_pos)
        if pos >= 0:
            return pos, pos + len(excerpt), "exact"

        # Phase 2: Case-insensitive search
        pos = source_lower.find(excerpt.lower(), min_pos)
        if pos >= 0:
            return pos, pos + len(excerpt), "case_insensitive"

        # Phase 2b: Whitespace-normalized search. PDF-justification
        # artifacts (e.g. Docling's double-spaces) or a quoted excerpt
        # collapsing the model's own whitespace can defeat a byte-literal
        # find() above -- and a short excerpt (under 3 words, e.g. a
        # terse comparison phrase) is too short to reach the fuzzy phase
        # below, so without this it has no fallback at all and silently
        # reports not_found.
        norm_result = self._search_normalized(excerpt, source_text, min_pos)
        if norm_result[2] != "not_found":
            return norm_result

        # Phase 3: Fuzzy sliding window with Jaccard pre-filter
        excerpt_words = excerpt.split()
        if len(excerpt_words) < 3:
            return None, None, "not_found"

        # Tokenize with each word's own (start, end) span in source_text, so
        # the matched window's real character range can be read off directly
        # instead of rejoining words with single spaces and re-`.find()`-ing
        # that string in source_text — which silently fails (reports
        # "not_found") whenever the source has irregular spacing, e.g.
        # Docling's PDF-justification double-spaces.
        word_spans = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", source_text)]
        source_words = [w for w, _, _ in word_spans]
        window_size = len(excerpt_words)

        if window_size > len(source_words):
            return None, None, "not_found"

        best_ratio = 0.0
        best_pos = None

        # Pre-compute lowercased token set for Jaccard filter
        excerpt_token_set = set(excerpt.lower().split())

        for i in range(len(source_words) - window_size + 1):
            if word_spans[i][1] < min_pos:
                continue
            # Jaccard pre-filter: skip windows with low token overlap
            window_token_set_lower = set(
                w.lower() for w in source_words[i : i + window_size]
            )
            intersection = len(excerpt_token_set & window_token_set_lower)
            union = len(excerpt_token_set | window_token_set_lower)
            if union > 0 and intersection / union < 0.3:
                continue

            window = " ".join(source_words[i : i + window_size])
            ratio = difflib.SequenceMatcher(
                None, excerpt.lower(), window.lower()
            ).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_pos = i

        if best_ratio >= self.fuzzy_threshold and best_pos is not None:
            char_start = word_spans[best_pos][1]
            char_end = word_spans[best_pos + window_size - 1][2]
            return char_start, char_end, "fuzzy"

        return None, None, "not_found"

    @staticmethod
    def _collapse_whitespace_with_map(text: str) -> Tuple[str, List[int]]:
        """Collapse runs of whitespace to a single space, returning the
        collapsed string plus a map from each collapsed-string index back
        to its original-string index."""
        out_chars: List[str] = []
        index_map: List[int] = []
        prev_was_space = False
        for i, ch in enumerate(text):
            if ch.isspace():
                if prev_was_space:
                    continue
                out_chars.append(" ")
                index_map.append(i)
                prev_was_space = True
            else:
                out_chars.append(ch)
                index_map.append(i)
                prev_was_space = False
        return "".join(out_chars), index_map

    def _search_normalized(
        self, excerpt: str, source_text: str, min_pos: int
    ) -> Tuple[Optional[int], Optional[int], str]:
        """Case-insensitive search with whitespace collapsed on both sides,
        mapping the match back to offsets in the original source_text."""
        norm_excerpt, _ = self._collapse_whitespace_with_map(excerpt.strip())
        norm_excerpt = norm_excerpt.lower()
        if not norm_excerpt:
            return None, None, "not_found"
        norm_source, index_map = self._collapse_whitespace_with_map(source_text)
        norm_source_lower = norm_source.lower()

        search_pos = 0
        while True:
            pos = norm_source_lower.find(norm_excerpt, search_pos)
            if pos < 0:
                return None, None, "not_found"
            start = index_map[pos]
            end = index_map[pos + len(norm_excerpt) - 1] + 1
            if start >= min_pos:
                return start, end, "case_insensitive"
            search_pos = pos + 1

    def ground_all_excerpts(self, extraction_result: dict, source_text: str) -> dict:
        """Add grounding info to all excerpts in an extraction result.

        Modifies extraction_result in-place, converting string excerpts
        to dicts with grounding metadata.

        Returns:
            Dict with grounding statistics
            {"exact": N, "case_insensitive": N, "fuzzy": N, "not_found": N}.
        """
        stats = {"exact": 0, "case_insensitive": 0, "fuzzy": 0, "not_found": 0}

        # Pre-compute lowercased source once for all excerpts
        source_lower = source_text.lower()

        # Where the next search for a given excerpt TEXT should start.
        # Two different excerpts (e.g. from two different columns) can
        # carry the exact same text while referring to two different
        # mentions of that phrase in the document (a results sentence and
        # a figure caption, say). Without this, both would independently
        # resolve to the document's first occurrence via plain .find().
        # Local to this call, not stored on self -- self.excerpt_grounder
        # is a single instance shared across concurrently-processed
        # documents, so any occurrence state must stay scoped to the one
        # document being grounded here.
        next_search_pos: Dict[str, int] = {}

        for col_name, col_data in extraction_result.items():
            if col_name.startswith("_"):
                continue
            if not isinstance(col_data, dict):
                continue
            excerpts = col_data.get("excerpts", [])
            grounded_excerpts = []
            for exc in excerpts:
                text = exc["text"] if isinstance(exc, dict) else exc
                start, end, status = self.ground_excerpt(
                    text,
                    source_text,
                    source_lower=source_lower,
                    search_start=next_search_pos.get(text, 0),
                )
                if end is not None:
                    next_search_pos[text] = end
                stats[status] += 1
                if isinstance(exc, dict):
                    exc["char_start"] = start
                    exc["char_end"] = end
                    exc["grounding_status"] = status
                    grounded_excerpts.append(exc)
                else:
                    grounded_excerpts.append(
                        {
                            "text": text,
                            "char_start": start,
                            "char_end": end,
                            "grounding_status": status,
                        }
                    )
            col_data["excerpts"] = grounded_excerpts

        return stats
