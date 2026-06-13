"""Programmatic entry point for running the evaluation across domains.

Used by ``main.py eval`` and importable directly. Returns DomainReports; the CLI handles
printing and saving.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..envs import get_environment, list_environments
from .harness import DomainReport, Harness, render_markdown


def run_eval(
    llm: Any,
    domains: list[str] | None = None,
    generations: int = 3,
    verbose: bool = False,
) -> list[DomainReport]:
    domains = domains or list_environments()
    harness = Harness(llm, verbose=verbose)
    reports: list[DomainReport] = []
    for name in domains:
        env = get_environment(name)
        reports.append(harness.run_domain(env, generations=generations))
    return reports


def save_results(reports: list[DomainReport], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"results": [r.to_dict() for r in reports], "markdown": render_markdown(reports)}
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
