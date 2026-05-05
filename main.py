"""CLI entry point."""

import os
from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(help="Stem Agent — self-specializing AI agent")


@app.command()
def run(
    model: str = typer.Option(
        os.getenv("STEM_MODEL", "claude-sonnet-4-6"), help="Anthropic model ID"
    ),
    checkpoint_dir: Path = typer.Option(
        Path(os.getenv("STEM_CHECKPOINT_DIR", "./checkpoints")),
        help="Directory for session checkpoints",
    ),
    playbook_dir: Path = typer.Option(
        Path(os.getenv("STEM_PLAYBOOK_DIR", "./playbooks")),
        help="Directory for emitted playbooks",
    ),
    resume: str | None = typer.Option(None, help="Checkpoint ID to resume from"),
) -> None:
    from stem import StemAgent
    from stem.models import Checkpoint

    agent = StemAgent(model=model, checkpoint_dir=checkpoint_dir, playbook_dir=playbook_dir)

    if resume:
        cp_path = checkpoint_dir / f"{resume}.json"
        if not cp_path.exists():
            typer.echo(f"Checkpoint not found: {cp_path}", err=True)
            raise typer.Exit(1)
        cp = Checkpoint.load(cp_path)
        agent.state = cp.state
        typer.echo(f"Resumed from checkpoint {resume} (phase: {agent.state.phase})")

    agent.run()


if __name__ == "__main__":
    app()
