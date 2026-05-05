"""Main agent loop and phase state machine."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from .models import (
    AgentState,
    ArtifactAnalysis,
    Checkpoint,
    ConversationTurn,
    Phase,
    Playbook,
    SpecialistProfile,
)
from .convergence import DRIFT_THRESHOLD, DRIFT_WINDOW, ConvergenceDetector
from .crystallizer import Crystallizer
from .llm import LLMClient
from .prompts import CONVERGENCE_SYSTEM, EVOLVE_PLAYBOOK, EVOLVE_PROFILE, EXTRACT_SYSTEM, INTERVIEW_SYSTEM, PROFILE_SYSTEM
from .task_archaeologist import TaskArchaeologist

console = Console()

CONVERGENCE_THRESHOLD = 0.85       # specialist_profile.convergence_score to crystallize
INTERVIEW_MIN_TURNS = 3            # minimum user turns before convergence is checked
INTERVIEW_MAX_TURNS = 10           # hard cap; advance regardless of score
INTERVIEW_CONVERGENCE_THRESHOLD = 0.75
POOR_TURN_THRESHOLD = 0.7          # score below this is recorded as a poor turn
EXECUTION_CHECKPOINT_INTERVAL = 5  # checkpoint every N execution turns


class StemAgent:
    """
    Self-specializing agent.

    Phases
    ------
    INTERVIEW       → gather problem class from user
    ARCHAEOLOGY     → reverse-engineer task approach from artifacts / failures
    CRYSTALLIZATION → emit system prompt, playbook, runnable agent code
    EXECUTION       → run as specialist; collect feedback
    EVOLUTION       → update profile/playbook when feedback triggers drift
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        checkpoint_dir: Path = Path("./checkpoints"),
        playbook_dir: Path = Path("./playbooks"),
        llm: LLMClient | None = None,
    ) -> None:
        self.llm = llm if llm is not None else LLMClient.from_env(model)
        self.checkpoint_dir = checkpoint_dir
        self.playbook_dir = playbook_dir
        self.archaeologist = TaskArchaeologist(llm=self.llm)
        self.state = AgentState(session_id=str(uuid.uuid4()))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start an interactive session."""
        console.print(Panel("[bold cyan]Stem Agent[/] — self-specializing AI", expand=False))
        while True:
            try:
                self._tick()
            except KeyboardInterrupt:
                self._checkpoint("user interrupt")
                console.print("\n[yellow]Session paused — checkpoint saved.[/]")
                break

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        phase = self.state.phase
        if phase == Phase.INTERVIEW:
            self._phase_interview()
        elif phase == Phase.ARCHAEOLOGY:
            self._phase_archaeology()
        elif phase == Phase.CRYSTALLIZATION:
            self._phase_crystallization()
        elif phase == Phase.EXECUTION:
            self._phase_execution()
        elif phase == Phase.EVOLUTION:
            self._phase_evolution()

    def _phase_interview(self) -> None:
        """Structured interview loop with LLM-based convergence check."""
        if not self.state.history:
            opening = self._stream_assistant(
                messages=[{"role": "user", "content": "Please start the intake interview."}],
                system=INTERVIEW_SYSTEM,
            )
            self.state.history.append(ConversationTurn(role="assistant", content=opening))
            self._checkpoint("interview opened")

        while True:
            user_text = self._get_user_input()
            self.state.history.append(ConversationTurn(role="user", content=user_text))

            user_turns = sum(1 for t in self.state.history if t.role == "user")

            if user_turns >= INTERVIEW_MIN_TURNS:
                score, gaps = self._check_convergence()
                console.print(
                    f"[dim]convergence: {score:.2f} — gaps: {', '.join(gaps) or 'none'}[/]"
                )
                if score >= INTERVIEW_CONVERGENCE_THRESHOLD or user_turns >= INTERVIEW_MAX_TURNS:
                    console.print("\n[green]Interview complete — proceeding to archaeology[/]")
                    self._advance(Phase.ARCHAEOLOGY)
                    return

            response = self._stream_assistant(
                messages=self._history_to_messages(),
                system=INTERVIEW_SYSTEM,
            )
            self.state.history.append(ConversationTurn(role="assistant", content=response))

    def _phase_archaeology(self) -> None:
        """Delegate to TaskArchaeologist, synthesize specialist profile, then advance."""
        console.print(Panel("[cyan]Task Archaeology[/] — 3-pass analysis", expand=False))

        artifacts, failures = self._extract_artifacts_from_history()
        console.print(
            f"[dim]Extracted {len(artifacts)} artifact(s), {len(failures)} failure example(s)[/]"
        )

        analysis = self.archaeologist.run(artifacts, failures)
        self.state.artifact_analysis = analysis
        console.print(
            f"[dim]Patterns: {len(analysis.patterns)} | "
            f"Bottlenecks: {len(analysis.bottlenecks)} | "
            f"Decision points: {len(analysis.decision_points)}[/]"
        )

        profile = self._synthesize_profile(analysis)
        self.state.specialist_profile = profile
        console.print(
            f"[green]Specialist:[/] {profile.domain} "
            f"[dim](convergence: {profile.convergence_score:.2f})[/]"
        )

        if profile.convergence_score >= CONVERGENCE_THRESHOLD:
            self._advance(Phase.CRYSTALLIZATION)
        else:
            console.print("[yellow]Signal insufficient — returning to interview[/]")
            self._advance(Phase.INTERVIEW)

    def _phase_crystallization(self) -> None:
        """Emit system prompt + playbook + agent code, with pre/post checkpoints."""
        profile = self.state.specialist_profile
        analysis = self.state.artifact_analysis
        if not profile or not analysis:
            raise RuntimeError("Cannot crystallize without a specialist profile and artifact analysis")

        console.print(Panel("[cyan]Crystallization[/] — emitting specialist", expand=False))
        self._checkpoint("pre-crystallization")

        cr = Crystallizer(llm=self.llm, playbook_dir=self.playbook_dir)

        console.print("[dim]Synthesizing system prompt...[/]")
        system_prompt, playbook, agent_code = cr.crystallize(profile, analysis)

        self.state.system_prompt = system_prompt
        self.state.playbook = playbook
        self.state.agent_code = agent_code

        playbook_path, agent_path = cr.save(playbook, agent_code)
        console.print(f"[green]Playbook[/] → {playbook_path}")
        console.print(f"[green]Agent[/]    → {agent_path}")

        self._advance(Phase.EXECUTION)

    def _phase_execution(self) -> None:
        """Run as crystallized specialist; score each turn; switch to EVOLUTION on drift."""
        profile = self.state.specialist_profile
        if not profile:
            raise RuntimeError("Cannot execute without a crystallized specialist profile")

        console.print(Panel(f"[green]{profile.domain}[/] — specialist active", expand=False))

        detector = ConvergenceDetector(
            llm=self.llm,
            drift_threshold=DRIFT_THRESHOLD,
            window=DRIFT_WINDOW,
        )
        execution_messages: list[dict] = []
        turn_count = 0

        while True:
            user_text = self._get_user_input()
            self.state.history.append(ConversationTurn(role="user", content=user_text))
            execution_messages.append({"role": "user", "content": user_text})

            response = self._stream_assistant(
                messages=execution_messages,
                system=self.state.system_prompt,
            )
            self.state.history.append(ConversationTurn(role="assistant", content=response))
            execution_messages.append({"role": "assistant", "content": response})

            score, reason = detector.score_turn(user_text, response, profile)
            self.state.execution_scores.append(score)
            self.state.execution_feedback.append(reason)
            console.print(f"[dim]turn score: {score:.2f} — {reason}[/]")

            turn_count += 1
            if turn_count % EXECUTION_CHECKPOINT_INTERVAL == 0:
                self._checkpoint(f"execution turn {turn_count}")

            if detector.is_drifting(self.state.execution_scores):
                console.print("[yellow]Drift detected — triggering evolution[/]")
                self._advance(Phase.EVOLUTION)
                return

    def _phase_evolution(self) -> None:
        """Update profile + playbook from feedback, re-emit artifacts, return to EXECUTION."""
        profile = self.state.specialist_profile
        playbook = self.state.playbook
        if not profile or not playbook:
            raise RuntimeError("Cannot evolve without a specialist profile and playbook")

        console.print(
            Panel(f"[yellow]Evolution {self.state.evolution_count + 1}[/] — updating specialist", expand=False)
        )
        self._checkpoint("pre-evolution")

        feedback = self._build_feedback_context()

        updated_profile = self._evolve_profile(profile, feedback)
        self.state.specialist_profile = updated_profile

        updated_playbook = self._evolve_playbook(playbook, feedback)
        self.state.playbook = updated_playbook

        cr = Crystallizer(llm=self.llm, playbook_dir=self.playbook_dir)
        system_prompt, agent_code = cr.rebuild_after_evolution(updated_profile)
        self.state.system_prompt = system_prompt
        self.state.agent_code = agent_code

        cr.save(updated_playbook, agent_code)

        self.state.evolution_count += 1
        console.print(f"[green]Evolution {self.state.evolution_count} complete — returning to execution[/]")
        self._advance(Phase.EXECUTION)

    # ------------------------------------------------------------------
    # Evolution helpers
    # ------------------------------------------------------------------

    def _build_feedback_context(self) -> str:
        poor = [
            f"Turn {i + 1} (score {s:.2f}): {r}"
            for i, (s, r) in enumerate(
                zip(self.state.execution_scores, self.state.execution_feedback)
            )
            if s < POOR_TURN_THRESHOLD
        ]
        if not poor:
            return "General performance drift — no specific turn failures identified."
        return "\n".join(poor)

    def _evolve_profile(self, profile: SpecialistProfile, feedback: str) -> SpecialistProfile:
        result = self._json_completion(
            system=EVOLVE_PROFILE,
            user=f"Current profile:\n{json.dumps(profile.model_dump(), indent=2)}\n\nExecution feedback:\n{feedback}",
        )
        return SpecialistProfile(**result)

    def _evolve_playbook(self, playbook: Playbook, feedback: str) -> Playbook:
        result = self._json_completion(
            system=EVOLVE_PLAYBOOK,
            user=f"Current playbook:\n{playbook.model_dump_json(indent=2)}\n\nExecution feedback:\n{feedback}",
        )
        return Playbook(
            domain=playbook.domain,
            version=playbook.version + 1,
            steps=result.get("steps", playbook.steps),
            tools=result.get("tools", playbook.tools),
            guardrails=result.get("guardrails", playbook.guardrails),
        )

    # ------------------------------------------------------------------
    # Interview helpers
    # ------------------------------------------------------------------

    def _get_user_input(self) -> str:
        return Prompt.ask("\n[bold cyan]You[/]").strip()

    def _history_to_messages(self) -> list[dict]:
        msgs = [{"role": t.role, "content": t.content} for t in self.state.history]
        # Anthropic requires the first message to be from the user
        if msgs and msgs[0]["role"] == "assistant":
            msgs = [{"role": "user", "content": "Please start the intake interview."}] + msgs
        return msgs

    def _check_convergence(self) -> tuple[float, list[str]]:
        transcript = "\n".join(f"{t.role.upper()}: {t.content}" for t in self.state.history)
        result = self._json_completion(
            system=CONVERGENCE_SYSTEM,
            user=f"Interview transcript:\n\n{transcript}",
        )
        return float(result.get("score", 0.0)), list(result.get("gaps", []))

    # ------------------------------------------------------------------
    # Archaeology helpers
    # ------------------------------------------------------------------

    def _extract_artifacts_from_history(self) -> tuple[list[str], list[str]]:
        transcript = "\n".join(f"{t.role.upper()}: {t.content}" for t in self.state.history)
        result = self._json_completion(
            system=EXTRACT_SYSTEM,
            user=f"Interview transcript:\n\n{transcript}",
        )
        return list(result.get("artifact_descriptions", [])), list(result.get("failure_examples", []))

    def _synthesize_profile(self, analysis: ArtifactAnalysis) -> SpecialistProfile:
        result = self._json_completion(
            system=PROFILE_SYSTEM,
            user=f"ArtifactAnalysis:\n{json.dumps(analysis.model_dump(), indent=2)}",
        )
        return SpecialistProfile(**result)

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _checkpoint(self, note: str = "") -> Path:
        cp = Checkpoint(
            state=self.state,
            checkpoint_id=f"{self.state.session_id}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}",
            note=note,
        )
        path = cp.save(self.checkpoint_dir)
        console.print(f"[dim]Checkpoint saved → {path}[/]")
        return path

    def _advance(self, phase: Phase) -> None:
        self.state.phase = phase
        self.state.updated_at = datetime.now(timezone.utc)
        self._checkpoint(f"advance to {phase}")

    def _stream_assistant(self, messages: list[dict], system: str = "") -> str:
        """Stream a completion and return the full text."""
        full = []
        for text in self.llm.stream_tokens(messages, system=system):
            console.print(text, end="")
            full.append(text)
        console.print()
        return "".join(full)

    def _json_completion(self, system: str, user: str) -> dict[str, Any]:
        text = self.llm.complete(system=system, user=user)
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
