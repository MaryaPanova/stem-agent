"""The Environment interface — the *world* an undifferentiated agent is pointed at.

An environment is the only thing that differs between "becoming a trader" and "becoming a
security auditor". The agent code never changes. An environment provides three things,
matching the brief:

  * **tasks**   — multi-step objectives (no identity baked in)
  * **tools**   — what is *available* to discover (names + schemas only)
  * **feedback**— an objective ``score`` for a finished attempt

Subclass ``BaseEnvironment`` and register tools with handlers; the base handles tool
dispatch, error capture, and exposing schemas.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

from .models import Task, TaskResult, ToolSpec, Trajectory


@dataclass
class StepResult:
    """Outcome of executing one tool call against the environment."""

    observation: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


Handler = Callable[..., Any]


class ToolRegistry:
    """Maps tool name -> (spec, handler). Handlers may raise; dispatch turns the raised
    exception into a ``StepResult.error`` so the agent can observe and recover from it."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Handler] = {}

    def register(self, spec: ToolSpec, handler: Handler) -> None:
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def has(self, name: str) -> bool:
        return name in self._handlers

    def execute(self, name: str, arguments: dict[str, Any]) -> StepResult:
        if name not in self._handlers:
            return StepResult(error=f"unknown tool: {name!r}")
        try:
            return StepResult(observation=self._handlers[name](**arguments))
        except TypeError as exc:  # almost always wrong/missing arguments
            return StepResult(error=f"bad arguments for {name!r}: {exc}")
        except Exception as exc:  # noqa: BLE001 - env feedback should never crash the loop
            return StepResult(error=f"{type(exc).__name__}: {exc}")


class Environment(ABC):
    name: str = "environment"

    @abstractmethod
    def available_tools(self) -> list[ToolSpec]:
        """Tool schemas the agent may discover and call."""

    @abstractmethod
    def tasks(self) -> list[Task]:
        """All tasks; filter by ``Task.split`` for train/test."""

    @abstractmethod
    def reset(self, task: Task) -> str:
        """Prepare internal state for a task. Returns the opening observation text."""

    @abstractmethod
    def execute(self, name: str, arguments: dict[str, Any]) -> StepResult:
        """Run one tool call against the current task's state."""

    @abstractmethod
    def score(self, task: Task, trajectory: Trajectory) -> TaskResult:
        """Objective score in [0, 1] for a finished attempt."""

    def train_tasks(self) -> list[Task]:
        return [t for t in self.tasks() if t.split == "train"]

    def test_tasks(self) -> list[Task]:
        return [t for t in self.tasks() if t.split == "test"]


class BaseEnvironment(Environment):
    """Environment with tool-registry plumbing already wired up."""

    def __init__(self) -> None:
        self.registry = ToolRegistry()
        self._build_tools()

    @abstractmethod
    def _build_tools(self) -> None:
        """Register tools on ``self.registry``."""

    def available_tools(self) -> list[ToolSpec]:
        return self.registry.specs()

    def execute(self, name: str, arguments: dict[str, Any]) -> StepResult:
        return self.registry.execute(name, arguments)
