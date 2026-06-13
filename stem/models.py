"""Core data models for the Stem Agent.

The central object is the ``Specialization`` — the *genome* the agent mutates as it
discovers what it needs to become. Everything else (tasks, tool specs, trajectories,
results) is plumbing around producing and scoring that genome.

Nothing here knows about a "domain". A domain is something the agent discovers by poking
at an environment; it is never named in the genome's starting state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Tools & tasks (provided by an Environment)
# ---------------------------------------------------------------------------


class ToolSpec(BaseModel):
    """A tool the environment makes *available*. The agent does not start out using
    these — it discovers them and decides which to adopt."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)  # JSON-schema for arguments

    def to_anthropic(self) -> dict[str, Any]:
        """Render in the shape the Anthropic tool-use API expects."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters or {"type": "object", "properties": {}},
        }


class Task(BaseModel):
    """A single multi-step problem. The prompt states an *objective*, never an identity.

    Good:  "Maximise `portfolio_value` over 40 steps using the tools available."
    Bad:   "You are a trader. Maximise returns."   (that pre-specializes the agent)
    """

    id: str
    objective: str
    max_steps: int = 30
    split: str = "train"  # "train" tasks drive evolution; "test" tasks are held out
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Trajectories (what happened during a rollout)
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    error: str | None = None
    step: int = 0


class Trajectory(BaseModel):
    """Full record of one attempt at one task — the raw material for both scoring and
    evolution."""

    task_id: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # plan/verify/reflect text
    final_answer: str = ""
    subagent_runs: int = 0

    # --- process metrics, read by the eval harness --------------------------
    @property
    def num_tool_calls(self) -> int:
        return len(self.tool_calls)

    @property
    def num_errors(self) -> int:
        return sum(1 for c in self.tool_calls if c.error)

    @property
    def num_recoveries(self) -> int:
        """Errors that were followed by a later successful call to the same tool —
        a crude but honest 'did it recover from its mistake' signal."""
        recovered = 0
        seen_error: set[str] = set()
        for call in self.tool_calls:
            if call.error:
                seen_error.add(call.name)
            elif call.name in seen_error:
                recovered += 1
                seen_error.discard(call.name)
        return recovered


class TaskResult(BaseModel):
    task_id: str
    score: float  # 0..1, objective where possible
    detail: dict[str, Any] = Field(default_factory=dict)
    trajectory: Trajectory | None = None


# ---------------------------------------------------------------------------
# The genome: the evolution surface
# ---------------------------------------------------------------------------


class LoopConfig(BaseModel):
    """Which phases run during a rollout. `act` is always on; the rest are evolvable.
    Turning on `plan`/`verify`/`reflect` literally changes the agent's control flow."""

    plan: bool = False
    verify: bool = False
    reflect: bool = False


class AdoptedTool(BaseModel):
    name: str
    usage_notes: str = ""  # what the agent learned about how to call it well


class Skill(BaseModel):
    """A reusable procedure the agent accumulated — named so it can be referenced and
    reused across tasks, not re-derived each time."""

    name: str
    when: str  # when to apply it
    body: str  # the procedure itself


class SubagentSpec(BaseModel):
    name: str
    role: str  # focused instruction for the sub-agent
    tools: list[str] = Field(default_factory=list)  # subset of tool names it may use


class Specialization(BaseModel):
    """The mutable self. A blank genome is genuinely undifferentiated: a generic identity,
    no adopted tools, no skills, no extra loop phases, no sub-agents, no eval criteria.

    Every field here is a distinct *surface* the evolution engine can change."""

    identity: str = (
        "You are an undifferentiated agent. You have just been dropped into an unfamiliar "
        "environment with some tools and an objective. You do not yet know what kind of "
        "agent you need to be. Probe the tools, attempt the objective, and learn from what "
        "works and what fails."
    )
    adopted_tools: list[AdoptedTool] = Field(default_factory=list)
    skills: list[Skill] = Field(default_factory=list)
    loop: LoopConfig = Field(default_factory=LoopConfig)
    subagents: list[SubagentSpec] = Field(default_factory=list)
    eval_criteria: list[str] = Field(default_factory=list)

    generation: int = 0
    lineage: list[str] = Field(default_factory=list)  # human-readable change log

    def is_blank(self) -> bool:
        return (
            not self.adopted_tools
            and not self.skills
            and not self.subagents
            and not self.eval_criteria
            and self.loop == LoopConfig()
        )

    def render_system_prompt(self) -> str:
        """Turn the genome into the system prompt that drives a rollout."""
        parts = [self.identity.strip()]

        if self.adopted_tools:
            lines = [f"- {t.name}: {t.usage_notes}".rstrip(": ") for t in self.adopted_tools]
            parts.append("## Tools you have learned to use\n" + "\n".join(lines))

        if self.skills:
            lines = [f"### {s.name}\nWhen: {s.when}\n{s.body}" for s in self.skills]
            parts.append("## Skills (reusable procedures)\n" + "\n\n".join(lines))

        if self.eval_criteria:
            lines = [f"- {c}" for c in self.eval_criteria]
            parts.append("## How you judge your own work\n" + "\n".join(lines))

        if self.subagents:
            lines = [
                f"- {s.name}: {s.role} (tools: {', '.join(s.tools) or 'none'})"
                for s in self.subagents
            ]
            parts.append(
                "## Sub-agents you can spawn\nCall the `spawn_subagent` tool with a "
                "`name` and a focused `task` to delegate.\n" + "\n".join(lines)
            )

        active = [name for name, on in self.loop.model_dump().items() if on]
        if active:
            parts.append(
                "## Working style\nYou have found these phases worth doing: "
                + ", ".join(active)
                + "."
            )

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------


class MutationType(str, Enum):
    REWRITE_IDENTITY = "rewrite_identity"
    ADOPT_TOOL = "adopt_tool"
    ADD_SKILL = "add_skill"
    SET_LOOP = "set_loop"
    DEFINE_SUBAGENT = "define_subagent"
    UPDATE_EVAL_CRITERIA = "update_eval_criteria"


class Mutation(BaseModel):
    """A single change to one surface of the genome, with the reasoning that produced it."""

    type: MutationType
    rationale: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Run state (checkpointed)
# ---------------------------------------------------------------------------


class GenerationRecord(BaseModel):
    generation: int
    mean_train_score: float
    mutations: list[Mutation] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)


class RunState(BaseModel):
    """Everything needed to resume an evolution run."""

    run_id: str
    domain: str  # which environment, NOT a property of the agent — just bookkeeping
    specialization: Specialization = Field(default_factory=Specialization)
    history: list[GenerationRecord] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
