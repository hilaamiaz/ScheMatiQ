"""Resume an interrupted ScheMatiQ run instead of starting it over.

A run over hundreds of documents takes hours, and a network outage, a crash
or a backend restart part-way through used to throw all of it away. The
pipeline therefore records its progress in ``run_checkpoint.json`` in the
session's work dir:

* schema discovery saves the schema after every batch, so a rerun continues
  with the next batch (or skips discovery entirely once it finished);
* once value extraction starts, a rerun resumes it, keeping the documents
  already extracted (tracked by schematiq-lib next to extracted_data.jsonl).

The checkpoint is only honoured when the rerun has the same configuration and
document set (its fingerprint); otherwise the run starts clean. It is removed
when the run completes, so a later deliberate rerun starts from scratch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "run_checkpoint.json"
_VERSION = 1

# Config entries that can differ between the original run and its rerun
# without changing what the run computes.
_VOLATILE_CONFIG_KEYS = {"docs_path", "output_path", "review_observation_unit"}


def _strip_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: _strip_secrets(v) for k, v in value.items()
            if "api_key" not in k.lower()
        }
    if isinstance(value, list):
        return [_strip_secrets(v) for v in value]
    return value


def compute_fingerprint(schematiq_config: Dict[str, Any], filenames: Iterable[str]) -> str:
    relevant = {
        k: v for k, v in schematiq_config.items() if k not in _VOLATILE_CONFIG_KEYS
    }
    payload = json.dumps(
        {"config": _strip_secrets(relevant), "documents": sorted(filenames)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RunCheckpoint:
    """Progress record of one session's run. Every write is atomic."""

    def __init__(self, session_dir: Path, fingerprint: str):
        self.path = Path(session_dir) / CHECKPOINT_FILENAME
        self.fingerprint = fingerprint
        self._state: Dict[str, Any] = {}
        previous = self._load()
        if previous and previous.get("fingerprint") == fingerprint and previous.get("version") == _VERSION:
            self._state = previous
        elif previous:
            logger.info(
                "Ignoring run checkpoint %s: configuration or documents changed since it was written",
                self.path,
            )

    def _load(self) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            logger.warning("Unreadable run checkpoint %s, starting fresh: %s", self.path, e)
            return None

    def _save(self) -> None:
        self._state.update(
            version=_VERSION,
            fingerprint=self.fingerprint,
            updated_at=datetime.now().isoformat(),
        )
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(self._state, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as e:
            # Losing a checkpoint only costs redoing work; never fail the run over it.
            logger.warning("Could not write run checkpoint %s: %s", self.path, e)

    # ── Schema discovery ─────────────────────────────────────────────

    @property
    def discovery(self) -> Optional[Dict[str, Any]]:
        """Saved discovery state, or None when there is nothing to resume."""
        return self._state.get("discovery")

    @property
    def discovery_done(self) -> bool:
        return bool((self.discovery or {}).get("done"))

    def save_discovery(
        self,
        *,
        next_batch: int,
        schema_dict: Dict[str, Any],
        evolution_dict: Dict[str, Any],
        unchanged_count: int,
        cumulative_docs: int,
        completion_done: bool,
        done: bool = False,
    ) -> None:
        self._state["discovery"] = {
            "next_batch": next_batch,
            "schema": schema_dict,
            "evolution": evolution_dict,
            "unchanged_count": unchanged_count,
            "cumulative_docs": cumulative_docs,
            "completion_done": completion_done,
            "done": done,
        }
        self._save()

    def mark_discovery_done(self, schema_dict: Dict[str, Any], evolution_dict: Dict[str, Any]) -> None:
        state = dict(self.discovery or {})
        state.update(schema=schema_dict, evolution=evolution_dict, done=True)
        self._state["discovery"] = state
        self._save()

    # ── Value extraction ─────────────────────────────────────────────

    @property
    def extraction_started(self) -> bool:
        return bool(self._state.get("extraction_started"))

    def mark_extraction_started(self) -> None:
        self._state["extraction_started"] = True
        self._save()

    # ── Lifecycle ────────────────────────────────────────────────────

    @property
    def has_progress(self) -> bool:
        return bool(self.discovery) or self.extraction_started

    def clear(self) -> None:
        self._state = {}
        clear_checkpoint(self.path.parent)


def clear_checkpoint(session_dir: Path) -> None:
    try:
        (Path(session_dir) / CHECKPOINT_FILENAME).unlink(missing_ok=True)
    except OSError as e:
        logger.warning("Could not remove run checkpoint in %s: %s", session_dir, e)
