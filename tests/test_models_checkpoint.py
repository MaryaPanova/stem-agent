"""Genome rendering, blank detection, and checkpoint round-trips."""

from stem.checkpoint import load_genome, load_run, save_genome, save_run
from stem.models import (
    AdoptedTool,
    GenerationRecord,
    LoopConfig,
    RunState,
    Skill,
    Specialization,
    SubagentSpec,
)


def test_blank_genome_render_is_generic():
    g = Specialization()
    assert g.is_blank()
    prompt = g.render_system_prompt()
    assert "undifferentiated" in prompt.lower()
    # nothing domain-specific leaks into a blank genome
    assert "Tools you have learned" not in prompt


def test_evolved_genome_render_includes_all_surfaces():
    g = Specialization(
        identity="You are a trader.",
        adopted_tools=[AdoptedTool(name="buy", usage_notes="at troughs")],
        skills=[Skill(name="cycle", when="always", body="buy low sell high")],
        loop=LoopConfig(plan=True, reflect=True),
        subagents=[SubagentSpec(name="scout", role="watch", tools=["get_price"])],
        eval_criteria=["beat buy-and-hold"],
    )
    p = g.render_system_prompt()
    for needle in ("trader", "buy", "cycle", "scout", "beat buy-and-hold", "plan", "reflect"):
        assert needle in p


def test_genome_round_trip(tmp_path):
    g = Specialization(identity="x", adopted_tools=[AdoptedTool(name="buy")])
    path = save_genome(g, tmp_path / "g.json")
    assert load_genome(path).adopted_tools[0].name == "buy"


def test_run_round_trip(tmp_path):
    state = RunState(run_id="r1", domain="trading",
                     specialization=Specialization(identity="x"),
                     history=[GenerationRecord(generation=1, mean_train_score=0.5)])
    run_path = save_run(state, tmp_path)
    assert (tmp_path / "r1.genome.json").exists()  # sidecar written
    loaded = load_run(run_path)
    assert loaded.domain == "trading" and loaded.history[0].generation == 1
