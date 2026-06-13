"""The rollout loop: tool calls, loop phases, sub-agents — all with a scripted LLM."""

from stem.agent import StemAgent
from stem.envs.trading.env import TradingEnvironment
from stem.envs.trading.exchange import make_prices
from stem.llm import ScriptedLLM, call, say
from stem.models import LoopConfig, Specialization, SubagentSpec


def _trading_task():
    env = TradingEnvironment()
    task = next(t for t in env.tasks() if t.id == "trade-train-11")
    return env, task


def test_dumb_agent_does_nothing_and_scores_zero():
    env, task = _trading_task()
    llm = ScriptedLLM([say("I don't know what to do, I'll stop.")])
    traj = StemAgent(llm).rollout(env, task, Specialization())
    assert traj.num_tool_calls == 0
    assert env.score(task, traj).score == 0.0


def test_competent_script_trades_and_scores_positive():
    env, task = _trading_task()
    prices, warm = make_prices(seed=11, n_visible=task.metadata["n_visible"])
    seg = prices[warm:]
    lo = min(range(len(seg)), key=lambda i: seg[i])
    hi = max(range(lo + 1, len(seg)), key=lambda i: seg[i])

    script = [
        call("advance", steps=max(1, lo)),
        call("buy", quantity=1.0),
        call("advance", steps=hi - lo),
        call("sell", quantity=1.0),
        say("done trading"),
    ]
    traj = StemAgent(ScriptedLLM(script)).rollout(env, task, Specialization())
    assert traj.final_answer == "done trading"
    assert traj.num_tool_calls == 4
    assert env.score(task, traj).score > 0


def test_tool_error_is_recorded_in_trajectory():
    env, task = _trading_task()
    # selling with no position -> env returns an error the agent can observe
    traj = StemAgent(ScriptedLLM([call("sell", quantity=5.0), say("oops")])).rollout(
        env, task, Specialization())
    assert traj.num_errors == 1
    assert "insufficient position" in traj.tool_calls[0].error


def test_loop_phases_add_plan_and_reflect_notes():
    env, task = _trading_task()
    llm = ScriptedLLM(lambda system, messages, tools: say("ok"))
    genome = Specialization(loop=LoopConfig(plan=True, verify=True, reflect=True))
    traj = StemAgent(llm).rollout(env, task, genome)

    assert any(n.startswith("PLAN:") for n in traj.notes)
    assert any(n.startswith("REFLECT:") for n in traj.notes)
    # verify forces at least one extra model turn before accepting the answer
    assert len(llm.calls) >= 3


def test_subagent_is_spawned_when_genome_defines_one():
    env, task = _trading_task()
    genome = Specialization(subagents=[SubagentSpec(name="helper", role="do a thing", tools=[])])
    script = [
        call("spawn_subagent", name="helper", task="check the price"),
        say("subagent reported back"),  # consumed by the sub-agent rollout
        say("final answer"),
    ]
    traj = StemAgent(ScriptedLLM(script)).rollout(env, task, genome)
    assert traj.subagent_runs == 1
    assert traj.final_answer == "final answer"
