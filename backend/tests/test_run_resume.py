"""Resuming an interrupted run: discovery checkpoint and conversion cache."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.models.session import SchemaEvolution
from app.services import conversion_cache
from app.services.pipeline import schema_discovery
from app.services.pipeline.run_checkpoint import RunCheckpoint, compute_fingerprint
from schematiq.core import schematiq as ScheMatiQ
from schematiq.core.schema import Column, Schema

CONFIG = {
    "query": "q",
    "max_keys_schema": 50,
    "schema_creation_backend": {"provider": "gemini", "model": "m", "api_key": "secret-1"},
    "initial_observation_unit": {"name": "Thing", "definition": "a thing"},
    "docs_path": "/a/pending_documents",
}


# ── Fingerprint ─────────────────────────────────────────────────────────


def test_fingerprint_ignores_paths_and_keys_but_not_settings():
    base = compute_fingerprint(CONFIG, ["b.txt", "a.txt"])
    moved = dict(CONFIG, docs_path="/a/documents",
                 schema_creation_backend=dict(CONFIG["schema_creation_backend"], api_key="other"))
    assert compute_fingerprint(moved, ["a.txt", "b.txt"]) == base
    assert compute_fingerprint(dict(CONFIG, query="other"), ["a.txt", "b.txt"]) != base
    assert compute_fingerprint(CONFIG, ["a.txt"]) != base


def test_checkpoint_from_other_config_is_ignored(tmp_path):
    cp = RunCheckpoint(tmp_path, "fp-1")
    cp.mark_extraction_started()
    assert RunCheckpoint(tmp_path, "fp-1").extraction_started
    assert not RunCheckpoint(tmp_path, "fp-2").has_progress


def test_checkpoint_never_stores_api_keys(tmp_path):
    cp = RunCheckpoint(tmp_path, compute_fingerprint(CONFIG, ["a.txt"]))
    cp.mark_extraction_started()
    assert "secret-1" not in cp.path.read_text(encoding="utf-8")


# ── Schema discovery resume ─────────────────────────────────────────────


@pytest.fixture
def fake_discovery(monkeypatch):
    """Three one-document batches; each generate_schema adds a column named after its doc."""
    from app.services.pipeline import batch_planner

    monkeypatch.setattr(batch_planner, "plan_batches", lambda docs, names, cfg: [
        SimpleNamespace(documents=[d], filenames=[n]) for d, n in zip(docs, names)
    ])
    monkeypatch.setattr(ScheMatiQ, "select_relevant_content", lambda docs, query, retriever: list(docs))
    monkeypatch.setattr(ScheMatiQ, "evaluate_schema_convergence", lambda a, b: False)
    state = {"calls": [], "fail_on": None}

    def generate_schema(passages, query, max_keys_schema, current_schema, llm, context_window_size=8192):
        doc = passages[0]
        state["calls"].append(doc)
        if doc == state["fail_on"]:
            raise httpx.ConnectError("network down")
        return Schema(query=query, columns=[Column(name=f"col_{doc}", definition="d")], max_keys=max_keys_schema)

    monkeypatch.setattr(ScheMatiQ, "generate_schema", generate_schema)
    # Do not wait for the network inside tests.
    monkeypatch.setattr(schema_discovery, "recover_from_transient_error", lambda *a, **k: False)
    return state


def _discover(checkpoint):
    async def progress(*args, **kwargs):
        return None

    return asyncio.run(schema_discovery.run_schema_discovery(
        "s1", ["d1", "d2", "d3"], ["d1.txt", "d2.txt", "d3.txt"], dict(CONFIG),
        llm=None, retriever=None, progress_callback=progress,
        is_stop_requested=lambda sid: False, checkpoint=checkpoint,
    ))


def test_discovery_resumes_from_first_unfinished_batch(tmp_path, fake_discovery):
    fake_discovery["fail_on"] = "d2"
    with pytest.raises(httpx.ConnectError):
        _discover(RunCheckpoint(tmp_path, "fp"))
    assert fake_discovery["calls"] == ["d1", "d2"]

    checkpoint = RunCheckpoint(tmp_path, "fp")
    assert checkpoint.discovery["next_batch"] == 1

    fake_discovery["calls"].clear()
    fake_discovery["fail_on"] = None
    schema, evolution = _discover(checkpoint)
    assert fake_discovery["calls"] == ["d2", "d3"]
    assert [c.name for c in schema.columns] == ["col_d1", "col_d2", "col_d3"]
    assert schema.observation_unit.name == "Thing"
    assert [s.iteration for s in evolution.snapshots] == [1, 2, 3]


def test_non_network_batch_failure_is_skipped(tmp_path, fake_discovery, monkeypatch):
    original = ScheMatiQ.generate_schema

    def bad_d2(passages, **kwargs):
        if passages[0] == "d2":
            raise ValueError("unparseable model output")
        return original(passages, **kwargs)

    monkeypatch.setattr(ScheMatiQ, "generate_schema", bad_d2)
    schema, _ = _discover(RunCheckpoint(tmp_path, "fp"))
    assert [c.name for c in schema.columns] == ["col_d1", "col_d3"]


def test_checkpoint_round_trips_evolution(tmp_path):
    evolution = SchemaEvolution(column_sources={"x": "a.txt"})
    schema = Schema(query="q", columns=[Column(name="x", definition="d")])
    RunCheckpoint(tmp_path, "fp").mark_discovery_done(schema.to_full_dict(), evolution.model_dump(mode="json"))
    cp = RunCheckpoint(tmp_path, "fp")
    assert cp.discovery_done
    assert Schema.from_dict(cp.discovery["schema"]).columns[0].name == "x"
    assert SchemaEvolution.model_validate(cp.discovery["evolution"]).column_sources == {"x": "a.txt"}


# ── Conversion cache ────────────────────────────────────────────────────


def test_reupload_reuses_cached_conversion(tmp_path, monkeypatch):
    from app.services import document_preprocessor, figure_extraction_service

    monkeypatch.setattr(conversion_cache, "CACHE_DIR", tmp_path / "cache")
    calls = {"convert": 0, "extract": 0}

    def fake_convert(source_path, out_dir, soffice, wid):
        calls["convert"] += 1
        (out_dir / f"{source_path.stem}.txt").write_text("paper text", encoding="utf-8")
        return True, "ok"

    def fake_extract(pdf_path, documents_dir, *, source_document):
        calls["extract"] += 1
        tmp = documents_dir / "figures" / ".extract_x"
        tmp.mkdir(parents=True)
        (tmp / "fig1.png").write_bytes(b"png")
        fig = SimpleNamespace(figure_id="fig1", image_filename="fig1.png")
        return figure_extraction_service._FigureBuildResult([fig], tmp, status="ok")

    def fake_persist(build, documents_dir, stem, *, source_document):
        target = documents_dir / "figures" / stem
        build.tmp_dir.rename(target)
        (target / "manifest.json").write_text("{}", encoding="utf-8")
        return target

    monkeypatch.setattr(document_preprocessor, "convert_file", fake_convert)
    monkeypatch.setattr(figure_extraction_service, "extract_figures", fake_extract)
    monkeypatch.setattr(figure_extraction_service, "persist_figures", fake_persist)

    def upload(session):
        pending = tmp_path / session / "pending_documents"
        pending.mkdir(parents=True)
        pdf = pending / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 same bytes")
        return pending, document_preprocessor.preprocess_uploaded_file(pdf, original_filename="paper.pdf")

    _, first = upload("s1")
    pending, second = upload("s2")
    assert first.success and second.success
    assert calls == {"convert": 1, "extract": 1}
    assert (pending / "paper.txt").read_text(encoding="utf-8") == "paper text"
    assert (pending / "figures" / "paper" / "fig1.png").read_bytes() == b"png"
    assert not (pending / "paper.pdf").exists()
    assert second.status == first.status


def test_degraded_figures_are_not_cached(tmp_path, monkeypatch):
    from app.services import document_preprocessor, figure_extraction_service

    monkeypatch.setattr(conversion_cache, "CACHE_DIR", tmp_path / "cache")

    def fake_convert(source_path, out_dir, soffice, wid):
        (out_dir / f"{source_path.stem}.txt").write_text("t", encoding="utf-8")
        return True, "ok"

    def fake_extract(pdf_path, documents_dir, *, source_document):
        tmp = documents_dir / "figures" / ".extract_x"
        tmp.mkdir(parents=True)
        return figure_extraction_service._FigureBuildResult([], tmp, status="ok", degraded=True)

    monkeypatch.setattr(document_preprocessor, "convert_file", fake_convert)
    monkeypatch.setattr(figure_extraction_service, "extract_figures", fake_extract)
    pending = tmp_path / "s" / "pending_documents"
    pending.mkdir(parents=True)
    pdf = pending / "p.pdf"
    pdf.write_bytes(b"%PDF")
    assert document_preprocessor.preprocess_uploaded_file(pdf, original_filename="p.pdf").success
    assert not any((tmp_path / "cache").glob("*/meta.json"))
