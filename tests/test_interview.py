"""Tests for the INTERVIEW phase."""

from unittest.mock import MagicMock

import pytest

from stem.models import ConversationTurn, Phase
from stem.stem_agent import (
    INTERVIEW_CONVERGENCE_THRESHOLD,
    INTERVIEW_MAX_TURNS,
    INTERVIEW_MIN_TURNS,
    StemAgent,
)


@pytest.fixture
def agent(tmp_path):
    return StemAgent(
        checkpoint_dir=tmp_path / "cp",
        playbook_dir=tmp_path / "pb",
        llm=MagicMock(),
    )


def _add_user_turns(agent, n):
    for i in range(n):
        agent.state.history.append(ConversationTurn(role="user", content=f"user turn {i}"))


# ------------------------------------------------------------------
# _check_convergence
# ------------------------------------------------------------------

def test_check_convergence_parses_score(agent):
    agent._json_completion = MagicMock(return_value={"score": 0.9, "gaps": []})
    score, gaps = agent._check_convergence()
    assert score == 0.9
    assert gaps == []


def test_check_convergence_defaults_on_missing_fields(agent):
    agent._json_completion = MagicMock(return_value={})
    score, gaps = agent._check_convergence()
    assert score == 0.0
    assert gaps == []


def test_check_convergence_passes_transcript(agent):
    agent.state.history = [
        ConversationTurn(role="assistant", content="What is your problem domain?"),
        ConversationTurn(role="user", content="I work on data pipelines."),
    ]
    agent._json_completion = MagicMock(return_value={"score": 0.3, "gaps": ["artifacts"]})
    agent._check_convergence()
    call_kwargs = agent._json_completion.call_args
    assert "data pipelines" in call_kwargs.kwargs["user"]


# ------------------------------------------------------------------
# _phase_interview — convergence-driven advance
# ------------------------------------------------------------------

def test_interview_advances_when_convergence_threshold_met(agent):
    _add_user_turns(agent, INTERVIEW_MIN_TURNS - 1)
    agent._check_convergence = MagicMock(return_value=(INTERVIEW_CONVERGENCE_THRESHOLD, []))
    agent._get_user_input = MagicMock(return_value="final answer")

    agent._phase_interview()

    assert agent.state.phase == Phase.ARCHAEOLOGY
    agent._check_convergence.assert_called_once()


def test_interview_advances_at_max_turns_regardless_of_score(agent):
    _add_user_turns(agent, INTERVIEW_MAX_TURNS - 1)
    agent._check_convergence = MagicMock(return_value=(0.0, ["everything missing"]))
    agent._get_user_input = MagicMock(return_value="answer")

    agent._phase_interview()

    assert agent.state.phase == Phase.ARCHAEOLOGY


def test_interview_does_not_advance_before_min_turns(agent):
    _add_user_turns(agent, INTERVIEW_MIN_TURNS - 2)
    agent._check_convergence = MagicMock(return_value=(1.0, []))
    # Supply exactly one more input, which brings us to MIN_TURNS - 1 (still below threshold)
    # then a second input that brings us to MIN_TURNS and triggers convergence check
    agent._get_user_input = MagicMock(side_effect=["penultimate answer", "final answer"])
    agent._stream_assistant = MagicMock(return_value="follow-up question")

    agent._phase_interview()

    # Convergence is only checked once — on the turn that reaches MIN_TURNS
    agent._check_convergence.assert_called_once()
    assert agent.state.phase == Phase.ARCHAEOLOGY


# ------------------------------------------------------------------
# _phase_interview — opening question
# ------------------------------------------------------------------

def test_interview_generates_opening_on_empty_history(agent):
    agent._stream_assistant = MagicMock(return_value="What problem are you solving?")
    agent._check_convergence = MagicMock(return_value=(1.0, []))
    agent._get_user_input = MagicMock(return_value="my answer")

    agent._phase_interview()

    assert agent.state.history[0].role == "assistant"
    assert agent.state.history[0].content == "What problem are you solving?"


def test_interview_skips_opening_when_history_exists(agent):
    _add_user_turns(agent, INTERVIEW_MIN_TURNS - 1)
    agent._stream_assistant = MagicMock(return_value="follow-up question")
    agent._check_convergence = MagicMock(return_value=(1.0, []))
    agent._get_user_input = MagicMock(return_value="answer")

    agent._phase_interview()

    # _stream_assistant should only be called for follow-up questions, not an opening
    for call in agent._stream_assistant.call_args_list:
        msgs = call.kwargs.get("messages") or call.args[0]
        assert not any(
            m["content"] == "Please start the intake interview." and m["role"] == "user"
            for m in msgs[-1:]  # bootstrap only appears as first message
        ), "Opening was regenerated on resume"
