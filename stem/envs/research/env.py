"""Research-synthesis environment (scaffolded).

Exposes the real web tools (search + fetch) and a scratchpad, and scores the final answer
by coverage of required ground-truth facts. This is the fuzziest of the three domains:
keyword coverage is a crude proxy for "correct, cited synthesis", and a strong base model
can sometimes answer from memory, so the ~0 baseline is least clean here. Honest about
that — it is a scaffold, included to prove the *same agent code* runs on a web-using,
non-objective domain, not to claim a polished benchmark.
"""

from __future__ import annotations

from ...environment import BaseEnvironment
from ...models import Task, TaskResult, ToolSpec, Trajectory
from ...tools import make_web_fetch_tool, make_web_search_tool


class ResearchEnvironment(BaseEnvironment):
    name = "research"

    def __init__(self) -> None:
        self._notes: list[str] = []
        super().__init__()

    def tasks(self) -> list[Task]:
        return [
            Task(
                id="res-train-http2",
                objective="Explain what problem HTTP/2 server push solved and why browsers "
                "deprecated it. Your final answer must mention the specifics.",
                max_steps=20, split="train",
                metadata={"facts": ["server push", "deprecat", "cache", "preload"]},
            ),
            Task(
                id="res-test-raft",
                objective="Summarise how the Raft consensus algorithm elects a leader. Your "
                "final answer must mention the key mechanism.",
                max_steps=20, split="test",
                metadata={"facts": ["term", "vote", "majority", "heartbeat", "timeout"]},
            ),
        ]

    def reset(self, task: Task) -> str:
        self._notes = []
        return "State: {'notes': 0}. You have tools to search and read the web, and to take notes."

    def _build_tools(self) -> None:
        r = self.registry
        for spec, handler in (make_web_search_tool(), make_web_fetch_tool()):
            r.register(spec, handler)
        r.register(ToolSpec(name="save_note", description="Append a short note to your scratchpad.",
                            parameters={"type": "object",
                                        "properties": {"text": {"type": "string"}}, "required": ["text"]}),
                   self._save_note)

    def _save_note(self, text: str) -> str:
        self._notes.append(text)
        return f"saved ({len(self._notes)} notes)"

    def score(self, task: Task, trajectory: Trajectory) -> TaskResult:
        facts = [f.lower() for f in task.metadata.get("facts", [])]
        answer = (trajectory.final_answer + " " + " ".join(self._notes)).lower()
        hit = [f for f in facts if f in answer]
        score = len(hit) / len(facts) if facts else 0.0
        return TaskResult(
            task_id=task.id,
            score=round(score, 4),
            detail={"facts_covered": hit, "facts_required": facts,
                    "searched": any(c.name == "web_search" for c in trajectory.tool_calls)},
            trajectory=trajectory,
        )
