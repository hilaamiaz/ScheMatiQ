"""Merging rows that describe the same unit (identical NES peptide)."""

import json

import pytest

from app.services import row_merge as rm

IDENTITY = ["NES Sequence", "Functional Export Signals", "Sequence", "_unit_name"]


def cell(answer, *texts, source="p"):
    return {"answer": answer, "excerpts": [{"text": t, "source": source} for t in texts]}


def row(unit, paper, **cells):
    return {
        "_unit_name": unit, "_row_name": unit, "_source_document": paper,
        "_papers": [f"{paper}.txt"], "_metadata": {"source_directories": ["/d"]},
        **cells,
    }


@pytest.mark.parametrize("text, expected", [
    ("LQLPPLERLTL", "LQLPPLERLTL"),
    ("355DVKSLESAMKDLKIK369", "DVKSLESAMKDLKIK"),
    ("17 E A T D L N N S M A A L N V N 31", "EATDLNNSMAALNVN"),
    ("ADD1 NES (377FEALMRMLDWLGYRT391)", "FEALMRMLDWLGYRT"),
    ("Residues 188-198", None),
    ("C-terminal end of APOBEC1", None),        # acronym, too short
    ("M" + "ACDEFGHIKL" * 10, None),            # whole protein, too long
])
def test_extract_peptide(text, expected):
    assert rm.extract_peptide(text) == expected


def test_identity_falls_back_through_columns():
    cols = rm.resolve_identity_columns(IDENTITY, ["Functional_Export_Signals", "Sequence"])
    assert cols == ["Functional_Export_Signals", "Sequence", "_unit_name"]
    positions_only = row("Mek1 NES", "a", Functional_Export_Signals=cell("30-44"),
                         Sequence=cell("TNLEALQKKLEELELDEQ"))
    assert rm.row_identity(positions_only, cols) == "TNLEALQKKLEELELDEQ"


def test_exact_matches_merge_and_cells_follow_the_rules():
    rows = [
        row("HIV-1 Rev NES", "a", Sequence=cell("LQLPPLERLTL", "Rev NES LQLPPLERLTL", source="a"),
            Organism=cell("Viral", "HIV-1", source="a"), Mutations=cell(""),
            Evidence=cell("LMB sensitivity", "LMB blocked export", source="a")),
        row("Rev NES", "b", Sequence=cell("lqlpplerltl"),  # lowercase: not a peptide match
            Organism=cell("Viral")),
        row("Rev NES", "c", Sequence=cell("LQLPPLERLTL", "the LQLPPLERLTL motif", source="c"),
            Organism=cell("viral.", "virus", source="c"), Mutations=cell("L78A", "L78A abolished", source="c"),
            Evidence=cell("CRM1 binding assay", "bound CRM1", source="c")),
    ]
    chosen = {}

    def chooser(conflicts, definitions, unit):
        chosen["unit"] = unit
        chosen["columns"] = [c.column for c in conflicts]
        return {"Evidence": 1}  # 0-based: the second candidate

    result = rm.merge_rows(rows, IDENTITY, chooser)
    assert result.summary()["merged_groups"] == 1
    merged = next(r for r in result.rows if "_merged_from" in r)
    assert merged["_papers"] == ["a.txt", "c.txt"]
    assert [m["_source_document"] for m in merged["_merged_from"]] == ["a", "c"]
    # equal after normalization -> kept once, excerpts of both
    assert merged["Organism"]["answer"] == "Viral"
    assert len(merged["Organism"]["excerpts"]) == 2
    # empty in one row -> takes the other's value
    assert merged["Mutations"]["answer"] == "L78A"
    # conflict -> LLM decision, other answer kept as alternative
    assert chosen["columns"] == ["Evidence"]
    assert "LQLPPLERLTL" in chosen["unit"]
    assert merged["Evidence"]["answer"] == "CRM1 binding assay"
    assert merged["Evidence"]["_merge_decided_by"] == "llm"
    assert [a["answer"] for a in merged["Evidence"]["_alternatives"]] == ["LMB sensitivity"]


def test_mutants_and_near_matches_are_suggested_not_merged():
    rows = [
        row("PKI NES", "a", Sequence=cell("LALKLAGLDI")),
        row("PKI NES long", "b", Sequence=cell("CELALKLAGLDIN")),
        row("PKI mutant", "c", Sequence=cell("LALKAAGLDI")),
    ]
    result = rm.merge_rows(rows, IDENTITY, chooser=None)
    assert result.merged_groups == 0 and len(result.rows) == 3
    reasons = sorted(s["reason"] for s in result.suggestions)
    assert reasons[0].startswith("1 residue")
    assert "contains" in reasons[-1]


def test_boundary_and_point_differences_are_suggested():
    rows = [row("Rev a", "a", Sequence=cell("LQLPPLERLTL")), row("Rev b", "b", Sequence=cell("CLPPLERLTL"))]
    result = rm.merge_rows(rows, IDENTITY, chooser=None)
    assert result.merged_groups == 0
    assert [s["sequences"] for s in result.suggestions] == [["LQLPPLERLTL", "CLPPLERLTL"]]


def test_failed_llm_falls_back_to_best_supported_answer():
    rows = [
        row("X", "a", Sequence=cell("LQLPPLERLTL"), Note=cell("short")),
        row("X", "b", Sequence=cell("LQLPPLERLTL"), Note=cell("longer answer", "e1", "e2")),
    ]

    def broken(conflicts, definitions, unit):
        raise RuntimeError("model down")

    merged = next(r for r in rm.merge_rows(rows, IDENTITY, broken).rows if "_merged_from" in r)
    assert merged["Note"]["answer"] == "longer answer"
    assert merged["Note"]["_merge_decided_by"] == "fallback"


def test_llm_reply_parsing():
    conflicts = [rm.Conflict("Evidence_Type", []), rm.Conflict("Organism", [])]
    reply = 'Sure.\n```json\n{"choices": {"Evidence Type": 2, "Organism": "1"}}\n```'
    assert rm._parse_choices(reply, conflicts) == {"Evidence_Type": 1, "Organism": 0}


def test_merging_again_keeps_history(tmp_path):
    table = tmp_path / "extracted_data.jsonl"
    rows = [row("A", "a", Sequence=cell("LQLPPLERLTL"), Note=cell("one")),
            row("A", "b", Sequence=cell("LQLPPLERLTL"), Note=cell("two"))]
    table.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    pick_first = lambda conflicts, d, u: {c.column: 0 for c in conflicts}
    rm.merge_table_file(table, IDENTITY, pick_first)

    new = tmp_path / "additional_data.jsonl"
    new.write_text(json.dumps(row("A", "c", Sequence=cell("LQLPPLERLTL"), Note=cell("three"))) + "\n"
                   + json.dumps(row("B", "c", Sequence=cell("VAEKLEALSV"))) + "\n", encoding="utf-8")
    summary = rm.merge_table_file(table, IDENTITY, pick_first, new_rows_path=new)

    merged = rm.read_jsonl(table)
    assert len(merged) == 1
    assert [m["_source_document"] for m in merged[0]["_merged_from"]] == ["a", "b", "c"]
    assert {a["answer"] for a in merged[0]["Note"]["_alternatives"]} == {"two", "three"}
    # the unrelated new row stays to be appended by the normal path
    assert [r["_unit_name"] for r in rm.read_jsonl(new)] == ["B"]
    assert summary["backup"].startswith("extracted_data.before_merge_")
    assert any(p.name == summary["backup"] for p in tmp_path.iterdir())


def test_rows_without_identity_are_untouched(tmp_path):
    rows = [row("p53 NES", "a", Sequence=cell("")), row("p53 NES", "b", Sequence=cell("not reported"))]
    result = rm.merge_rows(rows, IDENTITY, None)
    assert result.rows_without_identity == 2 and len(result.rows) == 2


class _FakeLLM:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return '{"choices": {"Note": 2}}'


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_runner_merge_uses_the_llm_and_project_identity(tmp_path, monkeypatch):
    import asyncio
    from app.models.schematiq import ScheMatiQConfig
    from app.services.orchestration import schematiq_runner

    monkeypatch.setattr(schematiq_runner, "work_dir", tmp_path)
    session_dir = tmp_path / "s1"
    session_dir.mkdir()
    _write(session_dir / "extracted_data.jsonl", [
        row("Rev", "a", Sequence=cell("LQLPPLERLTL"), Note=cell("first")),
        row("Rev", "b", Sequence=cell("LQLPPLERLTL"), Note=cell("second")),
    ])
    config = ScheMatiQConfig(
        query="q", output_path="o",
        schema_creation_backend={"provider": "gemini"}, value_extraction_backend={"provider": "gemini"},
        merge_duplicate_rows=True, row_identity_columns=["NES Sequence", "Sequence"],
    )
    llm = _FakeLLM()
    summary = asyncio.run(schematiq_runner.merge_duplicate_rows("s1", config, llm))
    assert summary["merged_groups"] == 1 and summary["llm_decisions"] == 1
    merged = rm.read_jsonl(session_dir / "extracted_data.jsonl")
    assert len(merged) == 1 and merged[0]["Note"]["answer"] == "second"
    assert "LQLPPLERLTL" in llm.prompts[0]


def test_added_documents_fold_into_existing_rows(tmp_path, monkeypatch):
    import asyncio
    from app.core import config as app_config
    from app.services import data_utils
    from app.services.upload_document_processor import UploadDocumentProcessor

    monkeypatch.setattr(app_config, "DEFAULT_SCHEMATIQ_WORK_DIR", str(tmp_path))
    persisted = []

    async def fake_persist(session_id, path):
        persisted.append(path.name)

    monkeypatch.setattr(data_utils, "persist_session_data_file", fake_persist)
    work = tmp_path / "s1"
    work.mkdir()
    (work / "config.json").write_text(json.dumps({
        "merge_duplicate_rows": True, "row_identity_columns": ["Sequence"],
    }), encoding="utf-8")
    _write(work / "extracted_data.jsonl", [row("Rev", "a", Sequence=cell("LQLPPLERLTL"), Note=cell("old"))])
    new = tmp_path / "additional_data.jsonl"
    _write(new, [row("Rev", "z", Sequence=cell("LQLPPLERLTL"), Note=cell("new")),
                 row("PKI", "z", Sequence=cell("LALKLAGLDI"))])

    processor = UploadDocumentProcessor.__new__(UploadDocumentProcessor)

    async def no_progress(*args, **kwargs):
        return None

    processor.broadcast_progress = no_progress
    asyncio.run(processor._merge_duplicate_rows_into_table("s1", new, _FakeLLM(), lambda: False))

    table = rm.read_jsonl(work / "extracted_data.jsonl")
    assert len(table) == 1 and table[0]["_papers"] == ["a.txt", "z.txt"]
    assert [r["_unit_name"] for r in rm.read_jsonl(new)] == ["PKI"]
    assert persisted == ["extracted_data.jsonl"]
