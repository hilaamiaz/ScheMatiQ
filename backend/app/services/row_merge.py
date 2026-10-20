"""Merge table rows that describe the same observation unit.

Each document is extracted on its own, so a unit reported by several papers
(e.g. the same NES peptide) ends up as several rows. This module folds such
rows into one:

* Identity: rows are the same unit when their identity value -- a peptide
  sequence read from the configured identity columns -- is exactly equal
  after normalization. A mutated variant has a different sequence and so
  stays a separate row.
* Cells: an empty cell takes the value another row found; equal values are
  kept once with all their excerpts; differing values are resolved by the
  LLM, which picks the best-supported answer. The other answers are kept on
  the cell under ``_alternatives``.
* Near matches (one sequence contains the other, or they differ in one or
  two residues) are never merged automatically; they are written to
  ``row_merge_suggestions.json`` for review.

Merging works on rows already extracted -- no document is read again -- and
the table is backed up before it is rewritten.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

SUGGESTIONS_FILENAME = "row_merge_suggestions.json"
UNIT_NAME_PSEUDO_COLUMN = "_unit_name"

# A peptide counts as an identity value only within this length range: shorter
# runs of capitals are acronyms, longer ones are whole-protein sequences (which
# several distinct NESs of one protein would share).
MIN_PEPTIDE_LENGTH = 8
MAX_PEPTIDE_LENGTH = 30
# Near-match threshold for suggestions: same length, at most this many residues differ.
NEAR_MATCH_MAX_MISMATCHES = 2

_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
_PEPTIDE_RE = re.compile(f"[{_AMINO_ACIDS}]{{{MIN_PEPTIDE_LENGTH},}}")
# "17 E A T D L N 31" -> letters written one by one.
_SPACED_LETTERS_RE = re.compile(r"\b(?:[A-Z]\s+){3,}[A-Z]\b")
_EMPTY_ANSWERS = {
    "", "none", "null", "n/a", "na", "not found", "not reported", "not specified",
    "not mentioned", "not available", "unknown", "not applicable", "-",
}


# ── Identity ────────────────────────────────────────────────────────────


def _normalize_column_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _answer(cell: Any) -> str:
    if isinstance(cell, dict):
        value = cell.get("answer")
    else:
        value = cell
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = "; ".join(str(v) for v in value)
    return str(value).strip()


def _is_empty(answer: str) -> bool:
    return answer.strip().strip(".").lower() in _EMPTY_ANSWERS


def extract_peptide(text: str) -> Optional[str]:
    """The NES-length peptide written in text, or None.

    Handles plain sequences ("LQLPPLERLTL"), position-annotated ones
    ("355DVKSLESAMKDLKIK369") and letters written one by one
    ("17 E A T D L N N S M A A L N V N 31"). When text holds several
    candidates the longest wins.
    """
    if not text:
        return None
    joined = _SPACED_LETTERS_RE.sub(lambda m: re.sub(r"\s+", "", m.group(0)), text)
    candidates = [
        run for run in _PEPTIDE_RE.findall(joined)
        if MIN_PEPTIDE_LENGTH <= len(run) <= MAX_PEPTIDE_LENGTH
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


def resolve_identity_columns(identity_columns: Iterable[str], row_keys: Iterable[str]) -> List[str]:
    """Map configured identity column names onto the table's actual keys
    (discovery may rename "NES Sequence" to "NES_Sequence")."""
    by_normalized = {_normalize_column_name(k): k for k in row_keys if not k.startswith("_")}
    resolved = []
    for name in identity_columns:
        if name == UNIT_NAME_PSEUDO_COLUMN:
            resolved.append(name)
            continue
        actual = by_normalized.get(_normalize_column_name(name))
        if actual and actual not in resolved:
            resolved.append(actual)
    return resolved


def row_identity(row: Dict[str, Any], identity_columns: List[str]) -> Optional[str]:
    """First peptide found in the identity columns, in priority order."""
    for column in identity_columns:
        if column == UNIT_NAME_PSEUDO_COLUMN:
            text = str(row.get("_unit_name") or row.get("_row_name") or "")
        else:
            text = _answer(row.get(column))
        peptide = extract_peptide(text)
        if peptide:
            return peptide
    return None


# ── Cell merging ────────────────────────────────────────────────────────


def _normalize_answer(answer: str) -> str:
    return re.sub(r"\s+", " ", answer).strip().rstrip(".;").casefold()


def _excerpt_key(excerpt: Any) -> str:
    if isinstance(excerpt, dict):
        if excerpt.get("type") == "figure":
            return f"figure:{excerpt.get('figure_id', '')}".lower()
        return f"{excerpt.get('source', '')}:{str(excerpt.get('text', '')).strip().lower()}"
    return str(excerpt).strip().lower()


def _union_excerpts(cells: Iterable[Any]) -> List[Any]:
    seen, merged = set(), []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        for excerpt in cell.get("excerpts") or []:
            key = _excerpt_key(excerpt)
            if key and key not in seen:
                seen.add(key)
                merged.append(excerpt)
    return merged


def _row_source(row: Dict[str, Any]) -> str:
    return str(row.get("_source_document") or ", ".join(row.get("_papers") or []) or row.get("_row_name") or "")


@dataclass
class Candidate:
    answer: str
    cells: List[Dict[str, Any]]
    sources: List[str]


@dataclass
class Conflict:
    column: str
    candidates: List[Candidate]


# chooser(conflicts, column_definitions) -> {column: index of the chosen candidate}
Chooser = Callable[[List[Conflict], Dict[str, str]], Dict[str, int]]


def _prior_alternatives(candidates: List[Candidate]) -> List[Dict[str, Any]]:
    """Alternatives recorded by an earlier merge on any of these cells."""
    seen, prior = set(), []
    for candidate in candidates:
        for cell in candidate.cells:
            for alt in cell.get("_alternatives") or []:
                key = _normalize_answer(str(alt.get("answer", "")))
                if key not in seen:
                    seen.add(key)
                    prior.append(alt)
    return prior


def _fallback_choice(candidates: List[Candidate]) -> int:
    """Best-supported answer when no LLM decision is available."""
    return max(
        range(len(candidates)),
        key=lambda i: (
            len(candidates[i].sources),
            len(_union_excerpts(candidates[i].cells)),
            len(candidates[i].answer),
        ),
    )


def merge_group(
    rows: List[Dict[str, Any]],
    chooser: Optional[Chooser] = None,
    column_definitions: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Fold rows describing the same unit into one row."""
    if len(rows) == 1:
        return rows[0]

    def filled(row):
        return sum(1 for k, v in row.items() if not k.startswith("_") and not _is_empty(_answer(v)))

    base = max(rows, key=filled)
    merged = {k: v for k, v in base.items() if k.startswith("_") or k == "document_directory"}

    papers: List[str] = []
    sources: List[str] = []
    directories: List[str] = []
    for row in rows:
        for paper in row.get("_papers") or []:
            if paper not in papers:
                papers.append(paper)
        src = row.get("_source_document")
        if src and src not in sources:
            sources.append(src)
        for d in (row.get("_metadata") or {}).get("source_directories") or []:
            if d not in directories:
                directories.append(d)
    merged["_papers"] = papers
    merged["_source_documents"] = sources
    # A member that is itself a merged row contributes its own members, so
    # merging again (e.g. after adding documents) keeps the full history.
    merged["_merged_from"] = []
    for r in rows:
        merged["_merged_from"].extend(r.get("_merged_from") or [{
            "_row_name": r.get("_row_name"),
            "_unit_name": r.get("_unit_name"),
            "_source_document": r.get("_source_document"),
            "_papers": r.get("_papers") or [],
        }])
    if "_metadata" in merged and directories:
        merged["_metadata"] = dict(merged["_metadata"], source_directories=directories)

    columns: List[str] = []
    for row in rows:
        for key in row:
            if not key.startswith("_") and key != "document_directory" and key not in columns:
                columns.append(key)

    conflicts: List[Conflict] = []
    for column in columns:
        cells = [(row, row.get(column)) for row in rows if column in row]
        groups: Dict[str, Candidate] = {}
        for row, cell in cells:
            answer = _answer(cell)
            if _is_empty(answer):
                continue
            key = _normalize_answer(answer)
            candidate = groups.setdefault(key, Candidate(answer=answer, cells=[], sources=[]))
            if isinstance(cell, dict):
                candidate.cells.append(cell)
            candidate.sources.append(_row_source(row))

        if not groups:
            # Every row is empty here: keep one, preserving a confirmed-empty marker.
            empties = [cell for _, cell in cells]
            confirmed = next((c for c in empties if isinstance(c, dict) and c.get("_confirmed_empty")), None)
            merged[column] = confirmed if confirmed is not None else empties[0]
            continue
        if len(groups) == 1:
            candidate = next(iter(groups.values()))
            first = candidate.cells[0] if candidate.cells else {"answer": candidate.answer}
            merged[column] = dict(first, excerpts=_union_excerpts(candidate.cells))
            merged[column].pop("_confirmed_empty", None)
            prior = _prior_alternatives([candidate])
            if prior:
                merged[column]["_alternatives"] = prior
            continue
        conflicts.append(Conflict(column=column, candidates=list(groups.values())))

    choices: Dict[str, int] = {}
    decided_by = "fallback"
    if conflicts and chooser is not None:
        try:
            choices = chooser(conflicts, column_definitions or {})
            decided_by = "llm"
        except Exception as e:  # never lose the merge over the tie-break
            logger.warning("Row merge: LLM choice failed, using best-supported answers: %s", e)
            choices = {}

    for conflict in conflicts:
        index = choices.get(conflict.column)
        by = decided_by
        if not isinstance(index, int) or not 0 <= index < len(conflict.candidates):
            index, by = _fallback_choice(conflict.candidates), "fallback"
        chosen = conflict.candidates[index]
        first = chosen.cells[0] if chosen.cells else {"answer": chosen.answer}
        cell = dict(first, excerpts=_union_excerpts(chosen.cells))
        cell.pop("_confirmed_empty", None)
        cell["_alternatives"] = [
            {
                "answer": c.answer,
                "sources": c.sources,
                "excerpts": _union_excerpts(c.cells),
            }
            for i, c in enumerate(conflict.candidates) if i != index
        ] + _prior_alternatives(conflict.candidates)
        cell["_merge_decided_by"] = by
        merged[conflict.column] = cell

    return merged


# ── LLM tie-break ───────────────────────────────────────────────────────


def _excerpt_texts(cells: List[Dict[str, Any]], limit: int = 3) -> List[str]:
    texts = []
    for excerpt in _union_excerpts(cells):
        if isinstance(excerpt, dict) and excerpt.get("text"):
            texts.append(str(excerpt["text"])[:300])
        elif isinstance(excerpt, str):
            texts.append(excerpt[:300])
        if len(texts) >= limit:
            break
    return texts


def build_choice_prompt(unit: str, conflicts: List[Conflict], definitions: Dict[str, str]) -> str:
    parts = [
        "Several papers report the same entity and were extracted into separate table rows. "
        f"The entity: {unit}\n"
        "For each field below the papers gave different answers. Choose the single best answer "
        "for each field: the one that is most accurate and specific for this entity and best "
        "supported by its evidence excerpts. Do not invent a new answer.\n",
    ]
    for conflict in conflicts:
        parts.append(f"\n### Field: {conflict.column}")
        if definitions.get(conflict.column):
            parts.append(f"Definition: {definitions[conflict.column]}")
        for i, candidate in enumerate(conflict.candidates, 1):
            parts.append(f"[{i}] {candidate.answer}")
            parts.append(f"    sources: {', '.join(candidate.sources)}")
            for text in _excerpt_texts(candidate.cells):
                parts.append(f"    evidence: \"{text}\"")
    parts.append(
        '\nRespond with JSON only, mapping each field name to the number of the chosen answer, '
        'e.g. {"choices": {"Field A": 2, "Field B": 1}}'
    )
    return "\n".join(parts)


def _parse_choices(text: str, conflicts: List[Conflict]) -> Dict[str, int]:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise ValueError(f"no JSON in LLM reply: {text[:200]!r}")
    data = json.loads(match.group(0))
    raw = data.get("choices", data) if isinstance(data, dict) else {}
    by_normalized = {_normalize_column_name(c.column): c.column for c in conflicts}
    choices = {}
    for name, number in raw.items():
        column = by_normalized.get(_normalize_column_name(str(name)))
        try:
            choices[column] = int(number) - 1
        except (TypeError, ValueError):
            continue
    choices.pop(None, None)
    return choices


def make_llm_chooser(llm, should_stop: Optional[Callable[[], bool]] = None) -> Chooser:
    """Chooser that asks llm once per merged row for all its conflicting fields."""
    from schematiq.core.network_resilience import recover_from_transient_error

    def choose(conflicts: List[Conflict], definitions: Dict[str, str], unit: str = "") -> Dict[str, int]:
        prompt = build_choice_prompt(unit or "(see sources)", conflicts, definitions)
        attempt = 0
        while True:
            try:
                return _parse_choices(llm.generate(prompt), conflicts)
            except Exception as e:
                if not recover_from_transient_error(e, attempt, should_stop, logger.warning):
                    raise
                attempt += 1

    return choose


# ── Table-level merge ───────────────────────────────────────────────────


@dataclass
class MergeResult:
    rows: List[Dict[str, Any]]
    merged_groups: int = 0
    rows_before: int = 0
    rows_without_identity: int = 0
    suggestions: List[Dict[str, Any]] = field(default_factory=list)
    llm_decisions: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "rows_before": self.rows_before,
            "rows_after": len(self.rows),
            "merged_groups": self.merged_groups,
            "rows_without_identity": self.rows_without_identity,
            "suggestions": len(self.suggestions),
            "llm_decisions": self.llm_decisions,
        }


def _edit_distance(a: str, b: str, limit: int) -> int:
    """Levenshtein distance, or limit + 1 once it is known to exceed limit."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        if min(current) > limit:
            return limit + 1
        previous = current
    return previous[-1]


def _near_match(a: str, b: str) -> Optional[str]:
    if a in b or b in a:
        return "one sequence contains the other"
    distance = _edit_distance(a, b, NEAR_MATCH_MAX_MISMATCHES)
    if distance <= NEAR_MATCH_MAX_MISMATCHES:
        return f"{distance} residue change(s) apart (possible mutant, variant or different boundaries)"
    return None


def merge_rows(
    rows: List[Dict[str, Any]],
    identity_columns: List[str],
    chooser: Optional[Callable[..., Dict[str, int]]] = None,
    column_definitions: Optional[Dict[str, str]] = None,
) -> MergeResult:
    """Merge exact identity matches; report near matches as suggestions."""
    all_keys = {k for row in rows for k in row}
    columns = resolve_identity_columns(identity_columns, all_keys)
    result = MergeResult(rows=[], rows_before=len(rows))
    if not columns:
        logger.warning("Row merge: none of the identity columns %s exist in the table", identity_columns)
        result.rows = list(rows)
        return result

    order: List[Any] = []  # identity key, or ("row", i) for rows without one
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for i, row in enumerate(rows):
        key = row_identity(row, columns)
        if key is None:
            result.rows_without_identity += 1
            order.append(("row", i))
            continue
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(row)

    for item in order:
        if isinstance(item, tuple):
            result.rows.append(rows[item[1]])
            continue
        members = groups[item]
        if len(members) == 1:
            result.rows.append(members[0])
            continue
        unit = str(members[0].get("_unit_name") or members[0].get("_row_name") or item)

        def bound_chooser(conflicts, definitions, _unit=f"{unit} (sequence {item})"):
            result.llm_decisions += 1
            return chooser(conflicts, definitions, _unit)

        merged = merge_group(members, bound_chooser if chooser else None, column_definitions)
        merged["_identity"] = item
        result.rows.append(merged)
        result.merged_groups += 1

    keys = list(groups)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            reason = _near_match(a, b)
            if reason:
                result.suggestions.append({
                    "sequences": [a, b],
                    "reason": reason,
                    "rows": [
                        [r.get("_unit_name") or r.get("_row_name") for r in groups[a]],
                        [r.get("_unit_name") or r.get("_row_name") for r in groups[b]],
                    ],
                })
    return result


# ── Files ───────────────────────────────────────────────────────────────


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Row merge: skipping unreadable line in %s", path.name)
    return rows


def write_jsonl_atomic(path: Path, rows: List[Dict[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_column_definitions(session_dir: Path) -> Dict[str, str]:
    for name in ("value_extraction_schema.json", "discovered_schema.json"):
        path = session_dir / name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        columns = data.get("columns") or data.get("schema") or []
        return {
            (c.get("column") or c.get("name")): (c.get("definition") or "")
            for c in columns if isinstance(c, dict) and (c.get("column") or c.get("name"))
        }
    return {}


def merge_table_file(
    table_path: Path,
    identity_columns: List[str],
    chooser=None,
    column_definitions: Optional[Dict[str, str]] = None,
    new_rows_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Merge duplicate rows in table_path, backing it up first.

    With new_rows_path (rows extracted from added documents), new rows that
    match a row of the table -- or each other -- are merged; groups that touch
    the table are written to table_path, the rest stay in new_rows_path to be
    appended as usual.
    """
    table_rows = read_jsonl(table_path)
    new_rows = read_jsonl(new_rows_path) if new_rows_path else []
    new_ids = {id(row) for row in new_rows}

    result = merge_rows(table_rows + new_rows, identity_columns, chooser, column_definitions)
    summary = result.summary()
    if result.merged_groups == 0:
        _write_suggestions(table_path.parent, result.suggestions)
        return summary

    backup = table_path.with_name(f"{table_path.stem}.before_merge_{int(time.time())}.jsonl")
    if table_path.exists():
        table_path.replace(backup)
    summary["backup"] = backup.name

    # A merged group that includes a table row belongs to the table; a group
    # made only of new rows, like any unmerged new row, stays with the new rows.
    columns = resolve_identity_columns(identity_columns, {k for r in table_rows + new_rows for k in r})
    table_keys = {row_identity(r, columns) for r in table_rows} - {None}
    to_table, to_new = [], []
    for row in result.rows:
        if "_identity" in row and "_merged_from" in row:
            (to_table if row["_identity"] in table_keys else to_new).append(row)
        else:
            (to_new if id(row) in new_ids else to_table).append(row)

    write_jsonl_atomic(table_path, to_table)
    if new_rows_path:
        write_jsonl_atomic(new_rows_path, to_new)
    _write_suggestions(table_path.parent, result.suggestions)
    logger.info("Row merge on %s: %s", table_path.name, summary)
    return summary


def _write_suggestions(session_dir: Path, suggestions: List[Dict[str, Any]]) -> None:
    path = session_dir / SUGGESTIONS_FILENAME
    try:
        path.write_text(json.dumps(suggestions, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        logger.warning("Row merge: could not write suggestions: %s", e)
