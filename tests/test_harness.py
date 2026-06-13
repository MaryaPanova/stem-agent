"""End-to-end harness: the whole pipeline, tokenless, proving evolution moves the score."""

from stem.eval.harness import Harness, render_markdown
from stem.eval.mock_demo import ToyEnvironment, make_mock_llm


def test_pipeline_baseline_low_evolved_high():
    report = Harness(make_mock_llm()).run_domain(ToyEnvironment(), generations=1)
    # the brief's core requirement: baseline can't do it, evolution makes it possible
    assert report.baseline.mean_score == 0.0
    assert report.evolved.mean_score == 1.0
    assert report.improvement == 1.0


def test_evolution_touches_multiple_surfaces():
    report = Harness(make_mock_llm()).run_domain(ToyEnvironment(), generations=1)
    surfaces = report.mutations_by_surface
    # not just a prompt rewrite — several distinct surfaces changed
    assert len(surfaces) >= 3
    assert "add_skill" in surfaces and "adopt_tool" in surfaces


def test_process_metrics_are_collected():
    report = Harness(make_mock_llm()).run_domain(ToyEnvironment(), generations=1)
    assert report.evolved.tool_calls > 0
    assert "toy-test" in report.evolved.per_task


def test_markdown_renders():
    report = Harness(make_mock_llm()).run_domain(ToyEnvironment(), generations=1)
    md = render_markdown([report])
    assert "| toy |" in md and "baseline" in md
