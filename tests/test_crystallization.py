"""Tests for the CRYSTALLIZATION phase and Crystallizer."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from stem.crystallizer import Crystallizer
from stem.models import ArtifactAnalysis, Playbook, Phase, SpecialistProfile
from stem.stem_agent import StemAgent


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def cr(tmp_path):
    return Crystallizer(llm=MagicMock(), playbook_dir=tmp_path)


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
        core_competencies=["pandas", "memory optimization"],
        preferred_tools=["pandas", "dask"],
        heuristics=["stream large files"],
        known_failure_modes=["OOM on large inputs"],
        convergence_score=0.9,
    )
    return SpecialistProfile(**{**defaults, **kwargs})


# ------------------------------------------------------------------
# _build_system_prompt
# ------------------------------------------------------------------

def test_build_system_prompt_returns_llm_output(cr):
    expected = "## Identity\nYou are a data pipeline engineer."
    cr._completion = MagicMock(return_value=expected)

    result = cr._build_system_prompt(_make_profile())

    assert result == expected


def test_build_system_prompt_includes_domain_in_request(cr):
    cr._completion = MagicMock(return_value="## Identity\nYou are...")

    cr._build_system_prompt(_make_profile(domain="SQL query optimizer"))

    user_arg = cr._completion.call_args.kwargs["user"]
    assert "SQL query optimizer" in user_arg


# ------------------------------------------------------------------
# _build_playbook
# ------------------------------------------------------------------

def test_build_playbook_returns_playbook_model(cr):
    cr._json_completion = MagicMock(return_value={
        "steps": ["ingest data", "validate schema", "transform"],
        "tools": ["pandas", "great_expectations"],
        "guardrails": ["reject inputs > 10 GB without streaming"],
    })

    playbook = cr._build_playbook(_make_profile(), ArtifactAnalysis())

    assert isinstance(playbook, Playbook)
    assert playbook.domain == "data pipeline engineer"
    assert playbook.steps == ["ingest data", "validate schema", "transform"]
    assert "pandas" in playbook.tools


def test_build_playbook_uses_profile_domain(cr):
    cr._json_completion = MagicMock(return_value={"steps": [], "tools": [], "guardrails": []})

    playbook = cr._build_playbook(_make_profile(domain="SQL optimizer"), ArtifactAnalysis())

    assert playbook.domain == "SQL optimizer"


def test_build_playbook_tolerates_missing_keys(cr):
    cr._json_completion = MagicMock(return_value={})

    playbook = cr._build_playbook(_make_profile(), ArtifactAnalysis())

    assert playbook.steps == []
    assert playbook.tools == []
    assert playbook.guardrails == []


# ------------------------------------------------------------------
# _build_agent_code
# ------------------------------------------------------------------

def test_build_agent_code_embeds_system_prompt(cr):
    profile = _make_profile(domain="test specialist")
    system_prompt = "You are a test specialist.\n\n## Identity\n..."

    code = cr._build_agent_code(profile, system_prompt)

    assert repr(system_prompt) in code


def test_build_agent_code_includes_domain(cr):
    code = cr._build_agent_code(_make_profile(domain="SQL optimizer"), "sys prompt")
    assert "SQL optimizer" in code


def test_build_agent_code_is_valid_python(cr):
    code = cr._build_agent_code(_make_profile(), "You are a specialist.")
    compile(code, "<generated>", "exec")


def test_build_agent_code_contains_anthropic_call(cr):
    code = cr._build_agent_code(_make_profile(), "sys")
    assert "Anthropic()" in code
    assert "client.messages.create" in code


# ------------------------------------------------------------------
# _atomic_write
# ------------------------------------------------------------------

def test_atomic_write_creates_file(cr, tmp_path):
    path = tmp_path / "out.json"
    cr._atomic_write(path, '{"ok": true}')
    assert path.exists()
    assert path.read_text() == '{"ok": true}'


def test_atomic_write_leaves_no_tmp_file(cr, tmp_path):
    path = tmp_path / "out.json"
    cr._atomic_write(path, "content")
    assert not path.with_suffix(".json.tmp").exists()


def test_atomic_write_creates_parent_dirs(cr, tmp_path):
    path = tmp_path / "nested" / "deep" / "out.txt"
    cr._atomic_write(path, "hi")
    assert path.exists()


# ------------------------------------------------------------------
# save
# ------------------------------------------------------------------

def test_save_writes_both_files(cr, tmp_path):
    cr.playbook_dir = tmp_path
    playbook = Playbook(domain="test specialist", steps=["step 1"])

    pb_path, agent_path = cr.save(playbook, "# agent code")

    assert pb_path.exists()
    assert agent_path.exists()


def test_save_slugifies_domain(cr, tmp_path):
    cr.playbook_dir = tmp_path
    playbook = Playbook(domain="my cool domain")

    pb_path, agent_path = cr.save(playbook, "# code")

    assert "my_cool_domain" in pb_path.name
    assert "my_cool_domain" in agent_path.name


# ------------------------------------------------------------------
# crystallize (integration of builders)
# ------------------------------------------------------------------

def test_crystallize_returns_all_three_artifacts(cr):
    cr._build_system_prompt = MagicMock(return_value="## Identity\nYou are...")
    cr._build_playbook = MagicMock(return_value=Playbook(domain="test", steps=["step"]))
    cr._build_agent_code = MagicMock(return_value="# agent code")

    system_prompt, playbook, agent_code = cr.crystallize(_make_profile(), ArtifactAnalysis())

    assert system_prompt == "## Identity\nYou are..."
    assert isinstance(playbook, Playbook)
    assert agent_code == "# agent code"


# ------------------------------------------------------------------
# _phase_crystallization
# ------------------------------------------------------------------

def _stub_crystallizer(system_prompt="sys", domain="test", agent_code="# code"):
    mock_cr = MagicMock()
    mock_cr.crystallize.return_value = (
        system_prompt,
        Playbook(domain=domain, steps=["step"]),
        agent_code,
    )
    mock_cr.save.return_value = (Path("/tmp/pb.json"), Path("/tmp/agent.py"))
    return mock_cr


def test_phase_crystallization_updates_state(agent):
    agent.state.specialist_profile = _make_profile()
    agent.state.artifact_analysis = ArtifactAnalysis()
    mock_cr = _stub_crystallizer(system_prompt="## Identity\nYou are a specialist.")

    with patch("stem.stem_agent.Crystallizer", return_value=mock_cr):
        agent._phase_crystallization()

    assert agent.state.system_prompt == "## Identity\nYou are a specialist."
    assert agent.state.playbook is not None
    assert agent.state.agent_code == "# code"


def test_phase_crystallization_advances_to_execution(agent):
    agent.state.specialist_profile = _make_profile()
    agent.state.artifact_analysis = ArtifactAnalysis()

    with patch("stem.stem_agent.Crystallizer", return_value=_stub_crystallizer()):
        agent._phase_crystallization()

    assert agent.state.phase == Phase.EXECUTION


def test_phase_crystallization_pre_checkpoints(agent):
    agent.state.specialist_profile = _make_profile()
    agent.state.artifact_analysis = ArtifactAnalysis()
    agent._checkpoint = MagicMock()

    with patch("stem.stem_agent.Crystallizer", return_value=_stub_crystallizer()):
        agent._phase_crystallization()

    notes = [
        (c.args[0] if c.args else c.kwargs.get("note", ""))
        for c in agent._checkpoint.call_args_list
    ]
    assert any("pre-crystallization" in n for n in notes)


def test_phase_crystallization_raises_without_profile(agent):
    agent.state.specialist_profile = None
    agent.state.artifact_analysis = ArtifactAnalysis()

    with pytest.raises(RuntimeError, match="specialist profile"):
        agent._phase_crystallization()


def test_phase_crystallization_raises_without_analysis(agent):
    agent.state.specialist_profile = _make_profile()
    agent.state.artifact_analysis = None

    with pytest.raises(RuntimeError, match="specialist profile"):
        agent._phase_crystallization()
