"""
3-pass task archaeology:
  Pass 1 — artifact analysis   (what exists and what patterns it reveals)
  Pass 2 — decision points     (where choices were made and why)
  Pass 3 — failure triangulation (what broke and what it implies about the real task)
"""

from __future__ import annotations

import json
from typing import Any

from .llm import LLMClient
from .models import ArtifactAnalysis

_PASS1_SYSTEM = """\
You are an expert task archaeologist. Given a set of artifacts (code, docs, notes, \
logs, or descriptions), extract:
- concrete artifacts present
- recurring patterns across them
- apparent bottlenecks or friction points

Respond ONLY with valid JSON matching this schema:
{"artifacts": [...], "patterns": [...], "bottlenecks": [...]}
"""

_PASS2_SYSTEM = """\
You are an expert task archaeologist. Given the artifact summary from Pass 1, \
identify the key decision points — moments where the practitioner had to choose \
between approaches — and infer the reasoning behind each choice.

Respond ONLY with valid JSON: {"decision_points": [...]}
"""

_PASS3_SYSTEM = """\
You are an expert task archaeologist. Given artifact patterns and decision points, \
analyze failure examples to triangulate:
- what actually failed (symptoms)
- what the root causes were
- what this implies about the real task structure

Respond ONLY with valid JSON: {"failure_modes": [...]}
"""


class TaskArchaeologist:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(
        self,
        artifact_descriptions: list[str],
        failure_examples: list[str] | None = None,
    ) -> ArtifactAnalysis:
        """Execute all three passes and return a merged ArtifactAnalysis."""
        pass1 = self._pass1(artifact_descriptions)
        pass2 = self._pass2(pass1)
        pass3 = self._pass3(pass1, pass2, failure_examples or [])
        return ArtifactAnalysis(
            artifacts=pass1.get("artifacts", []),
            patterns=pass1.get("patterns", []),
            bottlenecks=pass1.get("bottlenecks", []),
            decision_points=pass2.get("decision_points", []),
            failure_modes=pass3.get("failure_modes", []),
            raw={"pass1": pass1, "pass2": pass2, "pass3": pass3},
        )

    # ------------------------------------------------------------------
    # Passes
    # ------------------------------------------------------------------

    def _pass1(self, artifacts: list[str]) -> dict[str, Any]:
        user_content = "Artifacts to analyze:\n\n" + "\n\n---\n\n".join(artifacts)
        return self._json_completion(_PASS1_SYSTEM, user_content)

    def _pass2(self, pass1: dict[str, Any]) -> dict[str, Any]:
        user_content = f"Pass 1 summary:\n{json.dumps(pass1, indent=2)}"
        return self._json_completion(_PASS2_SYSTEM, user_content)

    def _pass3(
        self,
        pass1: dict[str, Any],
        pass2: dict[str, Any],
        failures: list[str],
    ) -> dict[str, Any]:
        user_content = (
            f"Pass 1 summary:\n{json.dumps(pass1, indent=2)}\n\n"
            f"Pass 2 summary:\n{json.dumps(pass2, indent=2)}\n\n"
            f"Failure examples:\n" + ("\n\n---\n\n".join(failures) if failures else "(none provided)")
        )
        return self._json_completion(_PASS3_SYSTEM, user_content)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _json_completion(self, system: str, user: str) -> dict[str, Any]:
        text = self.llm.complete(system=system, user=user)
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
