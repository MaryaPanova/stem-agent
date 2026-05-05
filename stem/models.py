"""Shared data models for the Stem Agent system."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Phase(str, Enum):
    INTERVIEW = "interview"
    ARCHAEOLOGY = "archaeology"
    CRYSTALLIZATION = "crystallization"
    EXECUTION = "execution"
    EVOLUTION = "evolution"


class ConversationTurn(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ArtifactAnalysis(BaseModel):
    artifacts: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    bottlenecks: list[str] = Field(default_factory=list)
    decision_points: list[str] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class SpecialistProfile(BaseModel):
    domain: str
    core_competencies: list[str] = Field(default_factory=list)
    preferred_tools: list[str] = Field(default_factory=list)
    heuristics: list[str] = Field(default_factory=list)
    known_failure_modes: list[str] = Field(default_factory=list)
    convergence_score: float = 0.0  # 0.0–1.0; >=0.85 triggers crystallization


class Playbook(BaseModel):
    version: int = 1
    domain: str
    steps: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    guardrails: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.domain.replace(' ', '_')}_v{self.version}.json"
        path.write_text(self.model_dump_json(indent=2))
        return path


class AgentState(BaseModel):
    phase: Phase = Phase.INTERVIEW
    session_id: str = ""
    history: list[ConversationTurn] = Field(default_factory=list)
    artifact_analysis: ArtifactAnalysis | None = None
    specialist_profile: SpecialistProfile | None = None
    playbook: Playbook | None = None
    system_prompt: str = ""
    agent_code: str = ""
    evolution_count: int = 0
    execution_scores: list[float] = Field(default_factory=list)
    execution_feedback: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Checkpoint(BaseModel):
    state: AgentState
    checkpoint_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    note: str = ""

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.checkpoint_id}.json"
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> Checkpoint:
        return cls.model_validate_json(path.read_text())
