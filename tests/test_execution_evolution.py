"""Tests for EXECUTION / EVOLUTION phases and ConvergenceDetector."""

from pathlib import Path
from unittest.mock import MagicMock, patch  # patch still used in phase tests

import pytest

from stem.convergence import ConvergenceDetector, DRIFT_THRESHOLD, DRIFT_WINDOW
from stem.models import ArtifactAnalysis, Playbook, Phase, SpecialistProfile
from stem.stem_agent import StemAgent, POOR_TURN_THRESHOLD


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def detector():
    return ConvergenceDetector(
        llm=MagicMock(),
        drift_threshold=DRIFT_THRESHOLD,
        window=DRIFT_WINDOW,
    )


@pytest.fixture
def agent(tmp_path):
    return StemAgent(
        checkpoint_dir=tmp_path / "cp",
        playbook_dir=tmp_path / "pb",
        llm=MagicMock(),
    )


def _make_profile(**kwargs) -> SpecialistProfile:
    defaults = dict(
        domain="data pipeline engineer",
        core_competencies=["pandas"],
        preferred_tools=["pandas"],
        heuristics=["stream large files"],
        known_failure_modes=["OOM"],
        convergence_score=0.9,
    )
    return SpecialistProfile(**{**defaults, **kwargs})


def _ready_agent(agent):
    """Put agent in a state ready for EXECUTION/EVOLUTION."""
    agent.state.specialist_profile = _make_profile()
    agent.state.playbook = Playbook(domain="data pipeline engineer", steps=["step 1"])
    agent.state.system_prompt = "## Identity\nYou are a data pipeline engineer."
    agent.state.agent_code = "# agent code"
    return agent


# ------------------------------------------------------------------
# ConvergenceDetector — is_drifting
# ------------------------------------------------------------------

def test_is_drifting_false_before_window_fills(detector):
    scores = [0.1] * (DRIFT_WINDOW - 1)
    assert not detector.is_drifting(scores)


def test_is_drifting_false_when_scores_high(detector):
    scores = [0.9] * DRIFT_WINDOW
    assert not detector.is_drifting(scores)


def test_is_drifting_true_when_scores_low(detector):
    scores = [DRIFT_THRESHOLD - 0.1] * DRIFT_WINDOW
    assert detector.is_drifting(scores)


def test_is_drifting_uses_only_recent_window(detector):
    # Old scores are high, recent DRIFT_WINDOW scores are low
    scores = [0.95] * 20 + [0.1] * DRIFT_WINDOW
    assert detector.is_drifting(scores)


def test_is_drifting_false_when_recent_scores_recover(detector):
    # Old scores are low, recent DRIFT_WINDOW scores are high
    scores = [0.1] * 20 + [0.9] * DRIFT_WINDOW
    assert not detector.is_drifting(scores)


# ------------------------------------------------------------------
# ConvergenceDetector — score_turn
# ------------------------------------------------------------------

def test_score_turn_returns_float_and_string(detector):
    detector._json_completion = MagicMock(return_value={"score": 0.8, "reason": "good answer"})
    score, reason = detector.score_turn("question", "answer", _make_profile())
    assert isinstance(score, float)
    assert isinstance(reason, str)
    assert score == 0.8
    assert reason == "good answer"


def test_score_turn_defaults_to_one_on_empty_response(detector):
    detector._json_completion = MagicMock(return_value={})
    score, reason = detector.score_turn("q", "a", _make_profile())
    assert score == 1.0
    assert reason == ""


def test_score_turn_includes_domain_in_request(detector):
    detector._json_completion = MagicMock(return_value={"score": 0.9, "reason": "fine"})
    detector.score_turn("q", "a", _make_profile(domain="SQL optimizer"))
    user_arg = detector._json_completion.call_args.kwargs["user"]
    assert "SQL optimizer" in user_arg


# ------------------------------------------------------------------
# _build_feedback_context
# ------------------------------------------------------------------

def test_build_feedback_context_includes_poor_turns(agent):
    agent.state.execution_scores = [0.9, 0.3, 0.8, 0.2]
    agent.state.execution_feedback = ["great", "missed schema", "fine", "wrong tool"]

    ctx = agent._build_feedback_context()

    assert "missed schema" in ctx
    assert "wrong tool" in ctx
    assert "great" not in ctx
    assert "fine" not in ctx


def test_build_feedback_context_fallback_when_no_poor_turns(agent):
    agent.state.execution_scores = [0.9, 0.95]
    agent.state.execution_feedback = ["great", "excellent"]

    ctx = agent._build_feedback_context()

    assert "General performance drift" in ctx


# ------------------------------------------------------------------
# _phase_execution
# ------------------------------------------------------------------

def test_execution_raises_without_profile(agent):
    agent.state.specialist_profile = None
    with pytest.raises(RuntimeError, match="specialist profile"):
        agent._phase_execution()


def test_execution_appends_turns_to_history(agent):
    _ready_agent(agent)

    with patch("stem.stem_agent.ConvergenceDetector") as MockDet:
        mock_det = MagicMock()
        MockDet.return_value = mock_det
        mock_det.score_turn.return_value = (0.9, "good")
        mock_det.is_drifting.side_effect = [False, True]

        agent._stream_assistant = MagicMock(return_value="specialist reply")
        agent._get_user_input = MagicMock(return_value="user question")

        agent._phase_execution()

    user_turns = [t for t in agent.state.history if t.role == "user"]
    assistant_turns = [t for t in agent.state.history if t.role == "assistant"]
    assert len(user_turns) == 2
    assert len(assistant_turns) == 2


def test_execution_scores_appended_to_state(agent):
    _ready_agent(agent)

    with patch("stem.stem_agent.ConvergenceDetector") as MockDet:
        mock_det = MagicMock()
        MockDet.return_value = mock_det
        mock_det.score_turn.side_effect = [(0.85, "ok"), (0.4, "poor")]
        mock_det.is_drifting.side_effect = [False, True]

        agent._stream_assistant = MagicMock(return_value="reply")
        agent._get_user_input = MagicMock(return_value="question")

        agent._phase_execution()

    assert agent.state.execution_scores == [0.85, 0.4]
    assert agent.state.execution_feedback == ["ok", "poor"]


def test_execution_advances_to_evolution_on_drift(agent):
    _ready_agent(agent)

    with patch("stem.stem_agent.ConvergenceDetector") as MockDet:
        mock_det = MagicMock()
        MockDet.return_value = mock_det
        mock_det.score_turn.return_value = (0.3, "poor")
        mock_det.is_drifting.return_value = True

        agent._stream_assistant = MagicMock(return_value="reply")
        agent._get_user_input = MagicMock(return_value="question")

        agent._phase_execution()

    assert agent.state.phase == Phase.EVOLUTION


def test_execution_uses_crystallized_system_prompt(agent):
    _ready_agent(agent)
    agent.state.system_prompt = "## Identity\nYou are a specialist."

    with patch("stem.stem_agent.ConvergenceDetector") as MockDet:
        mock_det = MagicMock()
        MockDet.return_value = mock_det
        mock_det.score_turn.return_value = (0.9, "good")
        mock_det.is_drifting.side_effect = [True]

        agent._stream_assistant = MagicMock(return_value="reply")
        agent._get_user_input = MagicMock(return_value="question")

        agent._phase_execution()

    call_kwargs = agent._stream_assistant.call_args.kwargs
    assert call_kwargs["system"] == "## Identity\nYou are a specialist."


# ------------------------------------------------------------------
# _phase_evolution
# ------------------------------------------------------------------

def _stub_evolution(agent):
    new_profile = _make_profile(domain="evolved domain", convergence_score=0.95)
    new_playbook = Playbook(domain="data pipeline engineer", version=2, steps=["new step"])
    agent._evolve_profile = MagicMock(return_value=new_profile)
    agent._evolve_playbook = MagicMock(return_value=new_playbook)
    agent._build_feedback_context = MagicMock(return_value="Turn 1 (score 0.30): missed schema")

    mock_cr = MagicMock()
    mock_cr.rebuild_after_evolution.return_value = ("new system prompt", "new agent code")
    mock_cr.save.return_value = (Path("/tmp/pb.json"), Path("/tmp/agent.py"))
    return mock_cr


def test_evolution_raises_without_profile(agent):
    agent.state.specialist_profile = None
    agent.state.playbook = Playbook(domain="test")
    with pytest.raises(RuntimeError):
        agent._phase_evolution()


def test_evolution_raises_without_playbook(agent):
    agent.state.specialist_profile = _make_profile()
    agent.state.playbook = None
    with pytest.raises(RuntimeError):
        agent._phase_evolution()


def test_evolution_updates_profile_in_state(agent):
    _ready_agent(agent)
    mock_cr = _stub_evolution(agent)

    with patch("stem.stem_agent.Crystallizer", return_value=mock_cr):
        agent._phase_evolution()

    assert agent.state.specialist_profile.domain == "evolved domain"


def test_evolution_updates_playbook_in_state(agent):
    _ready_agent(agent)
    mock_cr = _stub_evolution(agent)

    with patch("stem.stem_agent.Crystallizer", return_value=mock_cr):
        agent._phase_evolution()

    assert agent.state.playbook.version == 2
    assert agent.state.playbook.steps == ["new step"]


def test_evolution_updates_system_prompt_and_agent_code(agent):
    _ready_agent(agent)
    mock_cr = _stub_evolution(agent)

    with patch("stem.stem_agent.Crystallizer", return_value=mock_cr):
        agent._phase_evolution()

    assert agent.state.system_prompt == "new system prompt"
    assert agent.state.agent_code == "new agent code"


def test_evolution_increments_evolution_count(agent):
    _ready_agent(agent)
    agent.state.evolution_count = 2
    mock_cr = _stub_evolution(agent)

    with patch("stem.stem_agent.Crystallizer", return_value=mock_cr):
        agent._phase_evolution()

    assert agent.state.evolution_count == 3


def test_evolution_advances_to_execution(agent):
    _ready_agent(agent)
    mock_cr = _stub_evolution(agent)

    with patch("stem.stem_agent.Crystallizer", return_value=mock_cr):
        agent._phase_evolution()

    assert agent.state.phase == Phase.EXECUTION


def test_evolution_pre_checkpoints(agent):
    _ready_agent(agent)
    agent._checkpoint = MagicMock()
    mock_cr = _stub_evolution(agent)

    with patch("stem.stem_agent.Crystallizer", return_value=mock_cr):
        agent._phase_evolution()

    notes = [
        (c.args[0] if c.args else c.kwargs.get("note", ""))
        for c in agent._checkpoint.call_args_list
    ]
    assert any("pre-evolution" in n for n in notes)
