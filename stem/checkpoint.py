"""Atomic checkpointing of run state.

Writes go to a temp file and are then ``rename``d over the target — on POSIX filesystems
rename is atomic, so a crash mid-write leaves the previous checkpoint intact.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .models import RunState, Specialization


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(path)


def save_run(state: RunState, directory: Path) -> Path:
    """Persist a full run, plus a sidecar of just the genome for easy reuse."""
    state.updated_at = datetime.now(timezone.utc)
    run_path = Path(directory) / f"{state.run_id}.json"
    _atomic_write(run_path, state.model_dump_json(indent=2))

    genome_path = Path(directory) / f"{state.run_id}.genome.json"
    _atomic_write(genome_path, state.specialization.model_dump_json(indent=2))
    return run_path


def load_run(path: Path) -> RunState:
    return RunState.model_validate_json(Path(path).read_text())


def save_genome(genome: Specialization, path: Path) -> Path:
    _atomic_write(Path(path), genome.model_dump_json(indent=2))
    return Path(path)


def load_genome(path: Path) -> Specialization:
    return Specialization.model_validate_json(Path(path).read_text())
