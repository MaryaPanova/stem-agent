"""The undifferentiated agent.

One class, one ``rollout`` method. Give it an environment, a task, and a genome
(``Specialization``); it attempts the task by calling the environment's tools and returns a
full ``Trajectory``. The *same* code produces wildly different behaviour depending only on
(a) which environment it is pointed at and (b) how the genome has evolved.

The genome controls real things about this loop:
  * ``loop.plan``    -> a dedicated planning turn before acting
  * ``loop.verify``  -> a self-check before a final answer is accepted
  * ``loop.reflect`` -> a post-task reflection (also feeds the evolution engine)
  * ``subagents``    -> a ``spawn_subagent`` tool that delegates to a focused nested rollout
"""

from __future__ import annotations

import json
from typing import Any

from .environment import Environment, StepResult
from .llm import LLMResponse, ToolUse
from .models import Specialization, Task, ToolCall, ToolSpec, Trajectory

SUBAGENT_TOOL = "spawn_subagent"
SUBAGENT_MAX_STEPS = 6


def _obs_to_text(obs: Any) -> str:
    if isinstance(obs, str):
        return obs
    try:
        return json.dumps(obs, default=str)
    except TypeError:
        return str(obs)


class StemAgent:
    def __init__(self, llm: Any, verbose: bool = False) -> None:
        self.llm = llm
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public: one attempt at one task
    # ------------------------------------------------------------------

    def rollout(self, env: Environment, task: Task, genome: Specialization) -> Trajectory:
        traj = Trajectory(task_id=task.id)
        system = genome.render_system_prompt()
        tools = self._tool_schemas(env, genome)
        opening = env.reset(task)

        framing = (
            f"OBJECTIVE: {task.objective}\n\n"
            f"You have at most {task.max_steps} tool calls. Explore the available tools to "
            f"learn what this environment is and how to achieve the objective. When you are "
            f"finished, reply with your final answer and DO NOT call a tool.\n\n"
            f"Initial observation:\n{opening}"
        )

        if genome.loop.plan:
            plan = self._plan(system, task, framing)
            traj.notes.append(f"PLAN: {plan}")
            framing += f"\n\nYour plan:\n{plan}"

        messages: list[dict[str, Any]] = [{"role": "user", "content": framing}]
        verified = False

        for step in range(task.max_steps):
            resp = self.llm.run(system, messages, tools=tools)

            if not resp.wants_tool:
                if genome.loop.verify and not verified:
                    verified = True
                    messages.append({"role": "assistant", "content": resp.text or "(done)"})
                    messages.append({"role": "user", "content": (
                        "Before finalising: check your answer against the objective"
                        + (" and your own criteria" if genome.eval_criteria else "")
                        + ". If it is not yet satisfied, keep using tools. Otherwise restate "
                        "your final answer."
                    )})
                    continue
                traj.final_answer = resp.text
                break

            messages.append({"role": "assistant", "content": self._assistant_blocks(resp)})
            results = []
            for tu in resp.tool_uses:
                result = self._handle_tool(env, genome, tu, traj, step)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": _obs_to_text(result.observation if result.ok else result.error),
                    "is_error": not result.ok,
                })
            messages.append({"role": "user", "content": results})
        else:
            # ran out of steps: force a closing answer
            messages.append({"role": "user", "content": "Step budget exhausted. Give your final answer now."})
            traj.final_answer = self.llm.run(system, messages, tools=None).text

        if genome.loop.reflect:
            traj.notes.append("REFLECT: " + self._reflect(system, task, traj))

        return traj

    # ------------------------------------------------------------------
    # Loop phases
    # ------------------------------------------------------------------

    def _plan(self, system: str, task: Task, framing: str) -> str:
        msg = [{"role": "user", "content": framing + "\n\nWrite a short numbered plan. Do not call tools yet."}]
        return self.llm.run(system, msg, tools=None).text.strip()

    def _reflect(self, system: str, task: Task, traj: Trajectory) -> str:
        summary = self._trajectory_digest(traj)
        msg = [{"role": "user", "content": (
            f"You just attempted: {task.objective}\n\n{summary}\n\n"
            "In 2-3 sentences: what worked, what failed, and what you'd do differently."
        )}]
        return self.llm.run(system, msg, tools=None).text.strip()

    # ------------------------------------------------------------------
    # Tool handling
    # ------------------------------------------------------------------

    def _handle_tool(
        self, env: Environment, genome: Specialization, tu: ToolUse, traj: Trajectory, step: int
    ) -> StepResult:
        if tu.name == SUBAGENT_TOOL:
            result = self._spawn_subagent(env, genome, tu.input, traj)
        else:
            result = env.execute(tu.name, tu.input)
        traj.tool_calls.append(ToolCall(
            name=tu.name,
            arguments=tu.input,
            result=result.observation if result.ok else None,
            error=result.error,
            step=step,
        ))
        if self.verbose:
            tag = "ERR" if not result.ok else "ok"
            print(f"  [{tag}] {tu.name}({tu.input}) -> {result.error or result.observation}")
        return result

    def _spawn_subagent(
        self, env: Environment, genome: Specialization, args: dict[str, Any], traj: Trajectory
    ) -> StepResult:
        name, subtask = args.get("name", ""), args.get("task", "")
        spec = next((s for s in genome.subagents if s.name == name), None)
        if spec is None:
            return StepResult(error=f"no sub-agent named {name!r}")
        traj.subagent_runs += 1

        allowed = {t for t in spec.tools}
        tools = [t.to_anthropic() for t in env.available_tools() if t.name in allowed]
        sub_system = f"You are a focused sub-agent. Role: {spec.role}\nComplete the delegated task and report back concisely."
        messages: list[dict[str, Any]] = [{"role": "user", "content": f"Delegated task: {subtask}"}]

        for step in range(SUBAGENT_MAX_STEPS):
            resp = self.llm.run(sub_system, messages, tools=tools or None)
            if not resp.wants_tool:
                return StepResult(observation=resp.text or "(sub-agent returned nothing)")
            messages.append({"role": "assistant", "content": self._assistant_blocks(resp)})
            results = []
            for tu in resp.tool_uses:
                r = env.execute(tu.name, tu.input) if tu.name in allowed else StepResult(error=f"sub-agent may not use {tu.name!r}")
                traj.tool_calls.append(ToolCall(name=f"{name}:{tu.name}", arguments=tu.input,
                                                result=r.observation if r.ok else None, error=r.error, step=step))
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": _obs_to_text(r.observation if r.ok else r.error), "is_error": not r.ok})
            messages.append({"role": "user", "content": results})
        return StepResult(observation="(sub-agent ran out of steps)")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _tool_schemas(self, env: Environment, genome: Specialization) -> list[dict[str, Any]]:
        schemas = [t.to_anthropic() for t in env.available_tools()]
        if genome.subagents:
            schemas.append(ToolSpec(
                name=SUBAGENT_TOOL,
                description="Delegate a focused subtask to one of your sub-agents.",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "sub-agent name"},
                        "task": {"type": "string", "description": "the subtask to delegate"},
                    },
                    "required": ["name", "task"],
                },
            ).to_anthropic())
        return schemas

    @staticmethod
    def _assistant_blocks(resp: LLMResponse) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        if resp.text:
            blocks.append({"type": "text", "text": resp.text})
        for tu in resp.tool_uses:
            blocks.append({"type": "tool_use", "id": tu.id, "name": tu.name, "input": tu.input})
        return blocks

    @staticmethod
    def _trajectory_digest(traj: Trajectory) -> str:
        lines = [f"- {c.name}({c.arguments}) -> {'ERROR: ' + c.error if c.error else c.result}"
                 for c in traj.tool_calls[-12:]]
        return "Recent tool calls:\n" + ("\n".join(lines) or "(none)") + f"\nFinal answer: {traj.final_answer or '(none)'}"
