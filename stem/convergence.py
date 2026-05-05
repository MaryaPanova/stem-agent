"""Convergence detection and EXECUTION ↔ EVOLUTION gate."""

from __future__ import annotations

import json
from typing import Any

from .llm import LLMClient
from .models import SpecialistProfile
from .prompts import SCORE_TURN

DRIFT_THRESHOLD = 0.6   # rolling average below this triggers EVOLUTION
DRIFT_WINDOW = 5        # number of recent turns to average


class ConvergenceDetector:
    """
    Scores each specialist turn and signals drift when the rolling average
    over the last DRIFT_WINDOW turns falls below DRIFT_THRESHOLD.
    """

    def __init__(
        self,
        llm: LLMClient,
        drift_threshold: float = DRIFT_THRESHOLD,
        window: int = DRIFT_WINDOW,
    ) -> None:
        self.llm = llm
        self.drift_threshold = drift_threshold
        self.window = window

    def score_turn(
        self,
        user_message: str,
        assistant_reply: str,
        profile: SpecialistProfile,
    ) -> tuple[float, str]:
        """Score a single execution turn. Returns (score 0–1, one-line reason)."""
        result = self._json_completion(
            system=SCORE_TURN,
            user=(
                f"Specialist domain: {profile.domain}\n"
                f"Core competencies: {', '.join(profile.core_competencies)}\n\n"
                f"USER: {user_message}\n\n"
                f"ASSISTANT: {assistant_reply}"
            ),
        )
        # Default to 1.0 so a scoring failure never falsely triggers evolution
        return float(result.get("score", 1.0)), str(result.get("reason", ""))

    def is_drifting(self, scores: list[float]) -> bool:
        """True when the rolling average of recent scores falls below the threshold."""
        if len(scores) < self.window:
            return False
        recent = scores[-self.window:]
        return sum(recent) / len(recent) < self.drift_threshold

    def _json_completion(self, system: str, user: str) -> dict[str, Any]:
        text = self.llm.complete(system=system, user=user, max_tokens=256)
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
