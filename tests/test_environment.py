"""Tool dispatch + error capture, across the registry and a real env."""

from stem.environment import ToolRegistry
from stem.envs import get_environment, list_environments
from stem.envs.security.env import SecurityEnvironment
from stem.models import ToolSpec, Trajectory


def test_registry_dispatch_and_errors():
    reg = ToolRegistry()
    reg.register(ToolSpec(name="echo", description="", parameters={}), lambda x: x * 2)

    assert reg.execute("echo", {"x": 3}).observation == 6
    assert reg.execute("nope", {}).error.startswith("unknown tool")
    assert "bad arguments" in reg.execute("echo", {"y": 1}).error  # wrong kwarg


def test_handler_exception_becomes_error_not_crash():
    reg = ToolRegistry()

    def boom():
        raise RuntimeError("kaboom")

    reg.register(ToolSpec(name="boom", description="", parameters={}), boom)
    res = reg.execute("boom", {})
    assert not res.ok and "kaboom" in res.error


def test_all_registered_domains_construct():
    for name in list_environments():
        env = get_environment(name)
        assert env.available_tools()
        assert env.train_tasks() and env.test_tasks()


def test_security_scores_perfect_when_findings_correct():
    env = SecurityEnvironment()
    task = next(t for t in env.tasks() if t.split == "train")
    env.reset(task)
    from stem.envs.security.samples import GROUND_TRUTH
    for f in task.metadata["files"]:
        for cat in GROUND_TRUTH[f]:
            env.execute("report_finding", {"file": f, "category": cat})
    result = env.score(task, Trajectory(task_id=task.id))
    assert result.score == 1.0
    assert result.detail["recall"] == 1.0


def test_security_scores_zero_when_no_findings():
    env = SecurityEnvironment()
    task = next(t for t in env.tasks() if t.split == "test")
    env.reset(task)
    result = env.score(task, Trajectory(task_id=task.id))
    assert result.score == 0.0


def test_research_scorer_counts_fact_coverage():
    env = get_environment("research")
    task = env.train_tasks()[0]
    env.reset(task)
    traj = Trajectory(task_id=task.id,
                      final_answer="HTTP/2 server push was deprecated; preload + cache replaced it.")
    result = env.score(task, traj)
    assert 0 < result.score <= 1.0
