"""Stem Agent CLI.

    python main.py domains                       # list available environments
    python main.py eval --domain trading -g 3    # baseline vs evolved (real model)
    python main.py eval --mock                   # tokenless end-to-end smoke test
    python main.py evolve --domain trading -o genome.json
    python main.py run --domain trading --genome genome.json   # inspect one rollout
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()
app = typer.Typer(help="Stem Agent — a self-specializing agent that starts undifferentiated", add_completion=False)
console = Console()


def _llm(model: str | None):
    from stem.llm import LLMClient

    return LLMClient.from_env(model)


@app.command()
def domains() -> None:
    """List the environments the agent can be pointed at."""
    from stem.envs import list_environments

    for name in list_environments():
        console.print(f"- {name}")


@app.command()
def eval(
    domain: str = typer.Option("all", help="environment name, or 'all'"),
    generations: int = typer.Option(3, "--generations", "-g"),
    out: Path = typer.Option(Path("results/eval.json"), help="where to write results JSON"),
    model: str = typer.Option(None, help="model id override"),
    mock: bool = typer.Option(False, help="run the tokenless toy demo (no API key)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run baseline (gen 0) vs evolved across one or all domains."""
    from stem.eval.harness import Harness, render_markdown
    from stem.eval.run_eval import save_results

    if mock:
        from stem.eval.mock_demo import ToyEnvironment, make_mock_llm

        harness = Harness(make_mock_llm(), verbose=verbose)
        reports = [harness.run_domain(ToyEnvironment(), generations=max(1, generations))]
    else:
        from stem.envs import get_environment, list_environments

        names = list_environments() if domain == "all" else [domain]
        harness = Harness(_llm(model), verbose=verbose)
        reports = [harness.run_domain(get_environment(n), generations=generations) for n in names]

    console.print(render_markdown(reports))
    path = save_results(reports, out)
    console.print(f"\n[green]results -> {path}[/]")


@app.command()
def evolve(
    domain: str = typer.Option(..., help="environment name"),
    generations: int = typer.Option(3, "--generations", "-g"),
    out: Path = typer.Option(Path("results/genome.json"), help="where to write the evolved genome"),
    model: str = typer.Option(None, help="model id override"),
) -> None:
    """Evolve a genome on a domain's train tasks and save it."""
    from stem.checkpoint import save_genome
    from stem.eval.harness import Harness
    from stem.envs import get_environment
    from stem.models import Specialization

    env = get_environment(domain)
    harness = Harness(_llm(model))
    genome, history = harness.evolve(env, Specialization(), generations)
    for rec in history:
        muts = ", ".join(m.type.value for m in rec.mutations) or "none"
        console.print(f"gen {rec.generation}: train={rec.mean_train_score:.2f} | mutations: {muts}")
    path = save_genome(genome, out)
    console.print(f"\n[green]evolved genome -> {path}[/]")
    console.print(f"[dim]lineage: {genome.lineage}[/]")


@app.command()
def run(
    domain: str = typer.Option(..., help="environment name"),
    genome: Path = typer.Option(None, help="genome JSON (omit for a blank, undifferentiated agent)"),
    task_id: str = typer.Option(None, help="task id (default: first test task)"),
    model: str = typer.Option(None, help="model id override"),
) -> None:
    """Run a single rollout and print the trajectory + score (for inspection)."""
    from stem.agent import StemAgent
    from stem.checkpoint import load_genome
    from stem.envs import get_environment
    from stem.models import Specialization

    env = get_environment(domain)
    g = load_genome(genome) if genome else Specialization()
    tasks = env.tasks()
    task = next((t for t in tasks if t.id == task_id), None) or env.test_tasks()[0]

    agent = StemAgent(_llm(model), verbose=True)
    traj = agent.rollout(env, task, g)
    result = env.score(task, traj)
    console.print(f"\n[bold]score:[/] {result.score:.3f}  [dim]{json.dumps(result.detail, default=str)}[/]")


if __name__ == "__main__":
    app()
