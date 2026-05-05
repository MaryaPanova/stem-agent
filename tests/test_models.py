"""Smoke tests for data models."""

from stem.models import AgentState, Checkpoint, Phase, Playbook, SpecialistProfile


def test_agent_state_defaults():
    state = AgentState()
    assert state.phase == Phase.INTERVIEW
    assert state.history == []


def test_playbook_save(tmp_path):
    pb = Playbook(domain="test", steps=["step 1"], tools=["tool_a"])
    path = pb.save(tmp_path)
    assert path.exists()


def test_checkpoint_round_trip(tmp_path):
    state = AgentState(session_id="abc123")
    cp = Checkpoint(state=state, checkpoint_id="cp_001")
    path = cp.save(tmp_path)
    loaded = Checkpoint.load(path)
    assert loaded.state.session_id == "abc123"
