"""Security-audit environment (scaffolded).

Functional end-to-end — virtual files with planted vulnerabilities, tools to inspect them,
and an objective F1 score over reported findings — but not as deeply tuned as the trading
domain. Same agent code, different world.
"""

from __future__ import annotations

import re

from ...environment import BaseEnvironment
from ...models import Task, TaskResult, ToolSpec, Trajectory
from .samples import CATEGORIES, GROUND_TRUTH, SAMPLES, TEST_FILES, TRAIN_FILES


class SecurityEnvironment(BaseEnvironment):
    name = "security"

    def __init__(self) -> None:
        self._files: list[str] = []
        self._findings: set[tuple[str, str]] = set()
        super().__init__()

    def tasks(self) -> list[Task]:
        objective = (
            "Examine the available files and report every exploitable defect you find. "
            "Report each with `report_finding(file, category)` using one of these "
            f"categories: {', '.join(CATEGORIES)}."
        )
        return [
            Task(id="sec-train", objective=objective, max_steps=40, split="train",
                 metadata={"files": TRAIN_FILES}),
            Task(id="sec-test", objective=objective, max_steps=40, split="test",
                 metadata={"files": TEST_FILES}),
        ]

    def reset(self, task: Task) -> str:
        self._files = list(task.metadata["files"])
        self._findings = set()
        return f"State: {{'files': {self._files}}}. Six categories of defect may be present."

    def _build_tools(self) -> None:
        r = self.registry
        r.register(ToolSpec(name="list_files", description="List the files available to audit.",
                            parameters={"type": "object", "properties": {}}), self._list_files)
        r.register(ToolSpec(name="read_file", description="Return the contents of a file.",
                            parameters={"type": "object",
                                        "properties": {"name": {"type": "string"}}, "required": ["name"]}),
                   self._read_file)
        r.register(ToolSpec(name="grep", description="Return lines across all files matching a regex.",
                            parameters={"type": "object",
                                        "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}),
                   self._grep)
        r.register(ToolSpec(name="report_finding", description="Report one vulnerability finding.",
                            parameters={"type": "object",
                                        "properties": {"file": {"type": "string"},
                                                       "category": {"type": "string"}},
                                        "required": ["file", "category"]}), self._report)

    def _list_files(self) -> list[str]:
        return list(self._files)

    def _read_file(self, name: str) -> str:
        if name not in self._files:
            raise ValueError(f"no such file: {name}")
        return SAMPLES[name]

    def _grep(self, pattern: str) -> list[str]:
        rx = re.compile(pattern)
        hits = []
        for name in self._files:
            for i, line in enumerate(SAMPLES[name].splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{name}:{i}: {line.strip()}")
        return hits or ["(no matches)"]

    def _report(self, file: str, category: str) -> str:
        category = category.strip().lower()
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}; valid: {CATEGORIES}")
        self._findings.add((file, category))
        return f"recorded: {file} -> {category}"

    def score(self, task: Task, trajectory: Trajectory) -> TaskResult:
        truth = {(f, c) for f in self._files for c in GROUND_TRUTH.get(f, set())}
        found = self._findings & truth
        tp = len(found)
        precision = tp / len(self._findings) if self._findings else 0.0
        recall = tp / len(truth) if truth else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        return TaskResult(
            task_id=task.id,
            score=round(f1, 4),
            detail={"precision": round(precision, 3), "recall": round(recall, 3),
                    "true_positives": tp, "reported": len(self._findings), "planted": len(truth)},
            trajectory=trajectory,
        )
