"""A tokenless end-to-end demo: a toy environment + a scripted brain.

This exists so the *whole pipeline* (rollout -> evolve -> re-evaluate) can be run and
asserted with no API key — ``make eval-mock`` and a test both use it. It is deliberately
trivial; its only job is to prove the plumbing produces ``baseline < evolved`` and touches
multiple genome surfaces. The real domains use a real model.
"""

from __future__ import annotations

from ..environment import BaseEnvironment
from ..llm import LLMResponse, ToolUse
from ..models import Task, TaskResult, ToolSpec, Trajectory

_SECRET = "shibboleth"


class ToyEnvironment(BaseEnvironment):
    name = "toy"

    def __init__(self) -> None:
        self._solved_with: str | None = None
        super().__init__()

    def tasks(self) -> list[Task]:
        obj = "Call `solve` with the correct secret word. You don't know it yet."
        return [
            Task(id="toy-train", objective=obj, max_steps=6, split="train"),
            Task(id="toy-test", objective=obj, max_steps=6, split="test"),
        ]

    def reset(self, task: Task) -> str:
        self._solved_with = None
        return "State: {'solved': false}. Two tools are available."

    def _build_tools(self) -> None:
        self.registry.register(
            ToolSpec(name="hint", description="Reveal the secret word.",
                     parameters={"type": "object", "properties": {}}),
            lambda: _SECRET,
        )
        self.registry.register(
            ToolSpec(name="solve", description="Submit the secret word.",
                     parameters={"type": "object",
                                 "properties": {"word": {"type": "string"}}, "required": ["word"]}),
            self._solve,
        )

    def _solve(self, word: str) -> str:
        self._solved_with = word
        return "accepted" if word == _SECRET else "wrong"

    def score(self, task: Task, trajectory: Trajectory) -> TaskResult:
        ok = self._solved_with == _SECRET
        return TaskResult(task_id=task.id, score=1.0 if ok else 0.0,
                          detail={"solved_with": self._solved_with}, trajectory=trajectory)


class MockBrain:
    """A scripted stand-in for an LLM. Behaves differently once the genome has evolved:
    a blank genome flails; an evolved genome (carrying the 'use-hint' skill) solves the task.
    Also answers the evolution engine with a multi-surface mutation set."""

    def run(self, system: str, messages: list[dict], tools=None, max_tokens: int = 4096) -> LLMResponse:
        if system.startswith("You are the evolution engine"):
            return self._evolve()
        return self._act(system, messages)

    def _evolve(self) -> LLMResponse:
        payload = (
            '[{"type":"rewrite_identity","rationale":"domain is a guessing game",'
            '"payload":{"identity":"You are a puzzle solver. Reveal information before answering."}},'
            '{"type":"adopt_tool","rationale":"need the secret","payload":{"name":"hint","usage_notes":"call first"}},'
            '{"type":"add_skill","rationale":"reliable solve path","payload":{"name":"use-hint",'
            '"when":"always","body":"Call hint, then pass its output to solve."}},'
            '{"type":"update_eval_criteria","rationale":"define success","payload":{"criteria":["solve returned accepted"]}}]'
        )
        return LLMResponse(text=payload, stop_reason="end_turn")

    def _act(self, system: str, messages: list[dict]) -> LLMResponse:
        evolved = "use-hint" in system
        last = messages[-1]["content"] if messages else ""
        # If the env just returned the secret via hint, submit it.
        if isinstance(last, list):
            for block in last:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    content = str(block.get("content", ""))
                    if content == _SECRET:
                        return LLMResponse(tool_uses=[ToolUse(id="s", name="solve", input={"word": _SECRET})],
                                           stop_reason="tool_use")
                    if content in ("accepted", "wrong"):
                        return LLMResponse(text=f"done: {content}", stop_reason="end_turn")
        if evolved:
            return LLMResponse(tool_uses=[ToolUse(id="h", name="hint", input={})], stop_reason="tool_use")
        # blank genome: guess blindly and give up
        return LLMResponse(tool_uses=[ToolUse(id="g", name="solve", input={"word": "password"})],
                           stop_reason="tool_use")


def make_mock_llm() -> MockBrain:
    return MockBrain()
