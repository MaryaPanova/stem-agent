"""Stem Agent — a self-specializing agent that starts undifferentiated.

The same code, pointed at different environments, becomes fundamentally different agents.
Public surface:

    from stem import StemAgent, Evolver, Harness, Specialization
    from stem.envs import get_environment
"""

from .agent import StemAgent
from .evolution import Evolver, apply_mutation
from .llm import LLMClient, ScriptedLLM
from .models import Mutation, MutationType, Specialization, Task
from .eval import Harness

__all__ = [
    "StemAgent",
    "Evolver",
    "apply_mutation",
    "LLMClient",
    "ScriptedLLM",
    "Mutation",
    "MutationType",
    "Specialization",
    "Task",
    "Harness",
]
