"""The evolution engine: turn a generation's results into genome mutations.

This is the part the old project was missing. The old "evolution" rewrote a prompt. Here,
evolution can change *any* surface of the agent — identity, which tools it uses, the skills
it carries, its loop structure, its sub-agents, and its own success criteria — based on
what actually happened when it attempted the tasks.

``apply_mutation`` is pure (returns a new genome) so it is trivially testable without an
LLM. ``Evolver.evolve`` is the LLM-driven step that proposes the mutations.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from .environment import Environment
from .models import (
    AdoptedTool,
    LoopConfig,
    Mutation,
    MutationType,
    Skill,
    Specialization,
    SubagentSpec,
    TaskResult,
)

EVOLVE_SYSTEM = """\
You are the evolution engine for a self-specializing agent. The agent was dropped into an
unknown environment with no idea what it should become. You are shown what happened when it
attempted the tasks, and you decide how it should change.

You may change ANY of these surfaces (each is a mutation `type`):
- rewrite_identity      payload: {"identity": "<new system-prompt persona/instructions>"}
- adopt_tool            payload: {"name": "<tool>", "usage_notes": "<how to call it well>"}
- add_skill             payload: {"name": "...", "when": "...", "body": "<reusable procedure>"}
- set_loop              payload: {"plan": bool, "verify": bool, "reflect": bool}  (any subset)
- define_subagent       payload: {"name": "...", "role": "...", "tools": ["<tool>", ...]}
- update_eval_criteria  payload: {"criteria": ["...", "..."]}

Diagnose what is BLOCKING success (wrong/no tools used? no strategy? repeated errors? no
verification?) and propose the smallest set of mutations that would most improve scores.
Adopt tools the agent has not yet adopted if they are clearly needed. Give the agent a real
identity once the domain is obvious from the tools.

Respond with ONLY a JSON array of mutations, e.g.:
[{"type": "adopt_tool", "rationale": "never priced before trading",
  "payload": {"name": "get_price", "usage_notes": "call before any order"}}]
"""


# ---------------------------------------------------------------------------
# Applying mutations (pure)
# ---------------------------------------------------------------------------


def apply_mutation(genome: Specialization, mut: Mutation) -> Specialization:
    g = deepcopy(genome)
    p = mut.payload

    if mut.type == MutationType.REWRITE_IDENTITY:
        g.identity = str(p.get("identity", g.identity)).strip() or g.identity

    elif mut.type == MutationType.ADOPT_TOOL:
        name = p.get("name")
        if name:
            g.adopted_tools = [t for t in g.adopted_tools if t.name != name]
            g.adopted_tools.append(AdoptedTool(name=name, usage_notes=p.get("usage_notes", "")))

    elif mut.type == MutationType.ADD_SKILL:
        name = p.get("name")
        if name:
            g.skills = [s for s in g.skills if s.name != name]
            g.skills.append(Skill(name=name, when=p.get("when", ""), body=p.get("body", "")))

    elif mut.type == MutationType.SET_LOOP:
        current = g.loop.model_dump()
        for key in ("plan", "verify", "reflect"):
            if key in p:
                current[key] = bool(p[key])
        g.loop = LoopConfig(**current)

    elif mut.type == MutationType.DEFINE_SUBAGENT:
        name = p.get("name")
        if name:
            g.subagents = [s for s in g.subagents if s.name != name]
            g.subagents.append(SubagentSpec(name=name, role=p.get("role", ""), tools=list(p.get("tools", []))))

    elif mut.type == MutationType.UPDATE_EVAL_CRITERIA:
        crit = p.get("criteria")
        if isinstance(crit, list):
            g.eval_criteria = [str(c) for c in crit]

    return g


def apply_all(genome: Specialization, mutations: list[Mutation]) -> Specialization:
    g = genome
    for mut in mutations:
        g = apply_mutation(g, mut)
    return g


def parse_mutations(text: str) -> list[Mutation]:
    """Parse an LLM JSON response into Mutation objects, skipping anything malformed."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []

    out: list[Mutation] = []
    valid = {t.value for t in MutationType}
    for item in raw:
        if not isinstance(item, dict) or item.get("type") not in valid:
            continue
        out.append(Mutation(
            type=MutationType(item["type"]),
            rationale=str(item.get("rationale", "")),
            payload=item.get("payload", {}) if isinstance(item.get("payload"), dict) else {},
        ))
    return out


# ---------------------------------------------------------------------------
# The LLM-driven evolution step
# ---------------------------------------------------------------------------


class Evolver:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    def evolve(
        self, genome: Specialization, env: Environment, results: list[TaskResult]
    ) -> tuple[Specialization, list[Mutation]]:
        context = self._context(genome, env, results)
        text = self.llm.run(EVOLVE_SYSTEM, [{"role": "user", "content": context}], tools=None).text
        mutations = parse_mutations(text)

        new_genome = apply_all(genome, mutations)
        new_genome.generation = genome.generation + 1
        if mutations:
            new_genome.lineage = genome.lineage + [
                f"gen {new_genome.generation}: " + ", ".join(m.type.value for m in mutations)
            ]
        return new_genome, mutations

    def _context(self, genome: Specialization, env: Environment, results: list[TaskResult]) -> str:
        adopted = {t.name for t in genome.adopted_tools}
        available = [t.name for t in env.available_tools()]
        unadopted = [n for n in available if n not in adopted]

        blocks = [
            "## Current agent genome",
            f"identity: {genome.identity[:400]}",
            f"adopted_tools: {sorted(adopted) or 'none'}",
            f"skills: {[s.name for s in genome.skills] or 'none'}",
            f"loop: {genome.loop.model_dump()}",
            f"subagents: {[s.name for s in genome.subagents] or 'none'}",
            f"eval_criteria: {genome.eval_criteria or 'none'}",
            "",
            f"## Tools available but NOT yet adopted: {unadopted or 'none'}",
            "",
            "## What happened this generation",
        ]
        for r in results:
            traj = r.trajectory
            used = sorted({c.name for c in traj.tool_calls}) if traj else []
            errs = traj.num_errors if traj else 0
            blocks.append(
                f"- task {r.task_id}: score={r.score:.2f} | tools_used={used or 'none'} | "
                f"errors={errs} | final={(traj.final_answer[:160] if traj else '')!r}"
            )
            if traj:
                for note in traj.notes:
                    blocks.append(f"    note: {note[:300]}")
        blocks.append("\nPropose mutations (JSON array only).")
        return "\n".join(blocks)
