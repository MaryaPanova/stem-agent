"""Evaluation harness.

The harness tests the *evolution*, not just the output, exactly as the brief asks:

  1. **baseline**  — a blank genome (gen 0) on held-out test tasks. Expected ~0.
  2. **evolve**    — run K generations over the TRAIN tasks, mutating the genome.
  3. **evolved**   — the resulting genome on the SAME held-out test tasks.

If evolution did nothing real, baseline == evolved. The gap is the signal. Alongside task
scores it reports process metrics (tool calls, errors, recoveries, reflection, sub-agents,
and which genome surfaces evolution touched).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..agent import StemAgent
from ..environment import Environment
from ..evolution import Evolver
from ..models import GenerationRecord, Mutation, Specialization, Task, TaskResult


@dataclass
class PhaseMetrics:
    mean_score: float
    per_task: dict[str, float]
    tool_calls: int
    errors: int
    recoveries: int
    reflections: int
    subagent_runs: int


@dataclass
class DomainReport:
    domain: str
    baseline: PhaseMetrics
    evolved: PhaseMetrics
    generations: list[GenerationRecord]
    mutations_by_surface: dict[str, int]
    final_genome: Specialization
    test_details: dict[str, Any] = field(default_factory=dict)

    @property
    def improvement(self) -> float:
        return self.evolved.mean_score - self.baseline.mean_score

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "baseline_mean": round(self.baseline.mean_score, 4),
            "evolved_mean": round(self.evolved.mean_score, 4),
            "improvement": round(self.improvement, 4),
            "generations": len(self.generations),
            "mutations_by_surface": self.mutations_by_surface,
            "baseline": self.baseline.__dict__,
            "evolved": self.evolved.__dict__,
            "lineage": self.final_genome.lineage,
            "test_details": self.test_details,
        }


class Harness:
    def __init__(self, llm: Any, verbose: bool = False) -> None:
        self.llm = llm
        self.agent = StemAgent(llm, verbose=verbose)
        self.evolver = Evolver(llm)

    # ------------------------------------------------------------------

    def _score_task(self, env: Environment, task: Task, genome: Specialization) -> TaskResult:
        traj = self.agent.rollout(env, task, genome)
        return env.score(task, traj)

    def _phase_metrics(self, results: list[TaskResult]) -> PhaseMetrics:
        scores = [r.score for r in results]
        trajs = [r.trajectory for r in results if r.trajectory]
        return PhaseMetrics(
            mean_score=sum(scores) / len(scores) if scores else 0.0,
            per_task={r.task_id: round(r.score, 4) for r in results},
            tool_calls=sum(t.num_tool_calls for t in trajs),
            errors=sum(t.num_errors for t in trajs),
            recoveries=sum(t.num_recoveries for t in trajs),
            reflections=sum(len(t.notes) for t in trajs),
            subagent_runs=sum(t.subagent_runs for t in trajs),
        )

    def evaluate(self, env: Environment, genome: Specialization) -> tuple[PhaseMetrics, dict]:
        results = [self._score_task(env, t, genome) for t in env.test_tasks()]
        details = {r.task_id: r.detail for r in results}
        return self._phase_metrics(results), details

    def evolve(
        self, env: Environment, genome: Specialization, generations: int
    ) -> tuple[Specialization, list[GenerationRecord]]:
        history: list[GenerationRecord] = []
        train = env.train_tasks()
        for _ in range(generations):
            results = [self._score_task(env, t, genome) for t in train]
            mean = sum(r.score for r in results) / len(results) if results else 0.0
            genome, mutations = self.evolver.evolve(genome, env, results)
            history.append(GenerationRecord(generation=genome.generation, mean_train_score=mean,
                                            mutations=mutations))
        return genome, history

    def run_domain(self, env: Environment, generations: int = 3) -> DomainReport:
        baseline_genome = Specialization()
        baseline, _ = self.evaluate(env, baseline_genome)

        evolved_genome, history = self.evolve(env, baseline_genome, generations)
        evolved, details = self.evaluate(env, evolved_genome)

        surface_counts: Counter[str] = Counter()
        for rec in history:
            for m in rec.mutations:
                surface_counts[m.type.value] += 1

        return DomainReport(
            domain=env.name,
            baseline=baseline,
            evolved=evolved,
            generations=history,
            mutations_by_surface=dict(surface_counts),
            final_genome=evolved_genome,
            test_details=details,
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_markdown(reports: list[DomainReport]) -> str:
    lines = ["| domain | baseline | evolved | Δ | generations | surfaces evolved |",
             "|---|---|---|---|---|---|"]
    for r in reports:
        surfaces = ", ".join(f"{k}×{v}" for k, v in r.mutations_by_surface.items()) or "—"
        lines.append(
            f"| {r.domain} | {r.baseline.mean_score:.2f} | {r.evolved.mean_score:.2f} | "
            f"{r.improvement:+.2f} | {len(r.generations)} | {surfaces} |"
        )
    lines.append("")
    lines.append("Process metrics (held-out test set, evolved agent):")
    lines.append("")
    lines.append("| domain | tool calls | errors | recoveries | reflections | sub-agents |")
    lines.append("|---|---|---|---|---|---|")
    for r in reports:
        e = r.evolved
        lines.append(f"| {r.domain} | {e.tool_calls} | {e.errors} | {e.recoveries} | "
                     f"{e.reflections} | {e.subagent_runs} |")
    return "\n".join(lines)
