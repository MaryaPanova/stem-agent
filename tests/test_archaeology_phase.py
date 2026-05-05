"""Tests for the ARCHAEOLOGY phase."""

from unittest.mock import MagicMock

import pytest

from stem.models import ArtifactAnalysis, ConversationTurn, Phase, SpecialistProfile
from stem.stem_agent import CONVERGENCE_THRESHOLD, StemAgent


@pytest.fixture
def agent(tmp_path):
    return StemAgent(
        checkpoint_dir=tmp_path / "cp",
        playbook_dir=tmp_path / "pb",
        llm=MagicMock(),
    )


def _stub_archaeology(agent, *, convergence_score: float):
    agent._extract_artifacts_from_history = MagicMock(
        return_value=(["artifact description"], ["failure example"])
    )
    agent.archaeologist.run = MagicMock(
        return_value=ArtifactAnalysis(
            artifacts=["pipeline.py"],
            patterns=["uses pandas"],
            bottlenecks=["slow on large files"],
            decision_points=["chose CSV over parquet"],
            failure_modes=["OOM on 10 GB inputs"],
        )
    )
    agent._synthesize_profile = MagicMock(
        return_value=SpecialistProfile(domain="data pipeline engineer", convergence_score=convergence_score)
    )


# ------------------------------------------------------------------
# _extract_artifacts_from_history
# ------------------------------------------------------------------

def test_extract_artifacts_returns_lists(agent):
    agent.state.history = [
        ConversationTurn(role="assistant", content="Tell me about your work."),
        ConversationTurn(role="user", content="I have a Python ETL pipeline."),
        ConversationTurn(role="assistant", content="What breaks?"),
        ConversationTurn(role="user", content="It OOMs on large CSV files."),
    ]
    agent._json_completion = MagicMock(
        return_value={
            "artifact_descriptions": ["Python ETL pipeline"],
            "failure_examples": ["OOM on large CSV files"],
        }
    )

    artifacts, failures = agent._extract_artifacts_from_history()

    assert artifacts == ["Python ETL pipeline"]
    assert failures == ["OOM on large CSV files"]


def test_extract_artifacts_defaults_on_empty_response(agent):
    agent._json_completion = MagicMock(return_value={})
    artifacts, failures = agent._extract_artifacts_from_history()
    assert artifacts == []
    assert failures == []


# ------------------------------------------------------------------
# _synthesize_profile
# ------------------------------------------------------------------

def test_synthesize_profile_builds_from_analysis(agent):
    analysis = ArtifactAnalysis(
        artifacts=["pipeline.py"],
        patterns=["uses pandas"],
        bottlenecks=["slow on large files"],
    )
    agent._json_completion = MagicMock(
        return_value={
            "domain": "data pipeline engineer",
            "core_competencies": ["pandas", "memory optimization"],
            "preferred_tools": ["pandas", "dask"],
            "heuristics": ["stream large files instead of loading into RAM"],
            "known_failure_modes": ["OOM on large datasets"],
            "convergence_score": 0.87,
        }
    )

    profile = agent._synthesize_profile(analysis)

    assert profile.domain == "data pipeline engineer"
    assert profile.convergence_score == 0.87
    assert "pandas" in profile.preferred_tools


def test_synthesize_profile_passes_analysis_json(agent):
    analysis = ArtifactAnalysis(bottlenecks=["the bottleneck"])
    agent._json_completion = MagicMock(
        return_value={"domain": "test", "convergence_score": 0.5}
    )

    agent._synthesize_profile(analysis)

    call_kwargs = agent._json_completion.call_args.kwargs
    assert "the bottleneck" in call_kwargs["user"]


# ------------------------------------------------------------------
# _phase_archaeology — phase transitions
# ------------------------------------------------------------------

def test_archaeology_stores_analysis_in_state(agent):
    _stub_archaeology(agent, convergence_score=0.9)

    agent._phase_archaeology()

    assert agent.state.artifact_analysis is not None
    assert agent.state.artifact_analysis.bottlenecks == ["slow on large files"]


def test_archaeology_stores_profile_in_state(agent):
    _stub_archaeology(agent, convergence_score=0.9)

    agent._phase_archaeology()

    assert agent.state.specialist_profile is not None
    assert agent.state.specialist_profile.domain == "data pipeline engineer"


def test_archaeology_advances_to_crystallization_when_profile_strong(agent):
    _stub_archaeology(agent, convergence_score=CONVERGENCE_THRESHOLD)

    agent._phase_archaeology()

    assert agent.state.phase == Phase.CRYSTALLIZATION


def test_archaeology_returns_to_interview_when_profile_weak(agent):
    _stub_archaeology(agent, convergence_score=CONVERGENCE_THRESHOLD - 0.01)

    agent._phase_archaeology()

    assert agent.state.phase == Phase.INTERVIEW


def test_archaeology_calls_archaeologist_with_extracted_data(agent):
    agent._extract_artifacts_from_history = MagicMock(
        return_value=(["my artifact"], ["my failure"])
    )
    agent.archaeologist.run = MagicMock(return_value=ArtifactAnalysis())
    agent._synthesize_profile = MagicMock(
        return_value=SpecialistProfile(domain="x", convergence_score=0.9)
    )

    agent._phase_archaeology()

    agent.archaeologist.run.assert_called_once_with(["my artifact"], ["my failure"])
