"""Evolution surface: every mutation type changes the right part of the genome."""

from stem.envs.trading.env import TradingEnvironment
from stem.evolution import Evolver, apply_mutation, parse_mutations
from stem.llm import ScriptedLLM, say
from stem.models import Mutation, MutationType, Specialization, TaskResult, Trajectory


def test_each_mutation_type_changes_its_surface():
    g = Specialization()
    assert g.is_blank()

    g = apply_mutation(g, Mutation(type=MutationType.REWRITE_IDENTITY,
                                   payload={"identity": "You are a trader."}))
    assert "trader" in g.identity

    g = apply_mutation(g, Mutation(type=MutationType.ADOPT_TOOL,
                                   payload={"name": "get_price", "usage_notes": "first"}))
    assert g.adopted_tools[0].name == "get_price"

    g = apply_mutation(g, Mutation(type=MutationType.ADD_SKILL,
                                   payload={"name": "cycle", "when": "always", "body": "buy low"}))
    assert g.skills[0].name == "cycle"

    g = apply_mutation(g, Mutation(type=MutationType.SET_LOOP, payload={"verify": True}))
    assert g.loop.verify and not g.loop.plan

    g = apply_mutation(g, Mutation(type=MutationType.DEFINE_SUBAGENT,
                                   payload={"name": "scout", "role": "look", "tools": ["get_price"]}))
    assert g.subagents[0].name == "scout"

    g = apply_mutation(g, Mutation(type=MutationType.UPDATE_EVAL_CRITERIA,
                                   payload={"criteria": ["beat buy-and-hold"]}))
    assert g.eval_criteria == ["beat buy-and-hold"]
    assert not g.is_blank()


def test_adopt_tool_is_idempotent_by_name():
    g = Specialization()
    g = apply_mutation(g, Mutation(type=MutationType.ADOPT_TOOL, payload={"name": "buy", "usage_notes": "v1"}))
    g = apply_mutation(g, Mutation(type=MutationType.ADOPT_TOOL, payload={"name": "buy", "usage_notes": "v2"}))
    assert len(g.adopted_tools) == 1 and g.adopted_tools[0].usage_notes == "v2"


def test_parse_mutations_skips_malformed_and_unwraps_fences():
    text = """```json
    [{"type":"adopt_tool","payload":{"name":"buy"}},
     {"type":"not_a_real_type","payload":{}},
     "garbage"]
    ```"""
    muts = parse_mutations(text)
    assert len(muts) == 1 and muts[0].type == MutationType.ADOPT_TOOL


def test_parse_mutations_handles_non_json():
    assert parse_mutations("sorry, I cannot help") == []


def test_evolver_applies_mutations_and_bumps_generation():
    env = TradingEnvironment()
    genome = Specialization()
    result = TaskResult(task_id="t", score=0.0, trajectory=Trajectory(task_id="t"))
    llm = ScriptedLLM([say('[{"type":"adopt_tool","payload":{"name":"get_price","usage_notes":"x"}}]')])

    new_genome, muts = Evolver(llm).evolve(genome, env, [result])
    assert new_genome.generation == 1
    assert new_genome.adopted_tools[0].name == "get_price"
    assert new_genome.lineage  # change logged
    assert genome.generation == 0  # original untouched
