# Stem Agent

A self-specializing AI agent that starts undifferentiated, interviews you about a problem class, reverse-engineers how that class of task is approached (task archaeology), then crystallizes into a domain specialist — emitting a reusable system prompt, structured playbook, and standalone runnable agent.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the agent](#running-the-agent)
7. [What to expect — phase by phase](#what-to-expect--phase-by-phase)
8. [Pausing and resuming](#pausing-and-resuming)
9. [The emitted artifacts](#the-emitted-artifacts)
10. [Running the crystallized specialist](#running-the-crystallized-specialist)
11. [Project layout](#project-layout)
12. [Running tests](#running-tests)
13. [Switching LLM providers](#switching-llm-providers)
14. [Troubleshooting](#troubleshooting)

---

## How it works

```
INTERVIEW → ARCHAEOLOGY → CRYSTALLIZATION → EXECUTION ↔ EVOLUTION
```

| Phase | What happens | When it ends |
|---|---|---|
| **INTERVIEW** | The agent asks you focused questions one at a time: domain, artifacts, failure examples, constraints | Convergence score ≥ 0.75 (or 10 turns max) |
| **ARCHAEOLOGY** | 3-pass LLM analysis: artifact patterns → decision points → failure triangulation | Always completes; advances or loops back to INTERVIEW if signal is too weak |
| **CRYSTALLIZATION** | Emits a layered system prompt, a playbook JSON, and a standalone Python agent file | Always completes; advances to EXECUTION |
| **EXECUTION** | Runs as the crystallized specialist; every response is silently scored | Rolling score drops below 0.6 over 5 turns → triggers EVOLUTION |
| **EVOLUTION** | Updates the specialist profile and playbook from failure feedback; re-emits artifacts | Always completes; returns to EXECUTION |

The session state is checkpointed to disk after every phase transition and every 5 execution turns, so it can always be resumed.

---

## Architecture

### Component diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          StemAgent                              │
│                      (stem_agent.py)                            │
│                                                                 │
│   ┌──────────┐   ┌────────────┐   ┌─────────────┐               │
│   │INTERVIEW │──▶│ARCHAEOLOGY │──▶│CRYSTALLIZE  │               │
│   └──────────┘   └────────────┘   └──────┬──────┘               │
│        ▲                │                │                      │
│        │          TaskArchaeologist       ▼                     │
│        │          (3-pass LLM)    ┌─────────────┐               │
│        │                          │  EXECUTION  │◀─┐            │
│        │                          └──────┬──────┘  │            │
│        │                                 │ drift    │           │
│        │                          ┌──────▼──────┐  │            │
│        │                          │  EVOLUTION  │──┘            │
│        │                          └─────────────┘               │
│        │                                                        │
│   AgentState (Pydantic) ──▶ Checkpoint (JSON on disk)           │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                          LLMClient
                        (stem/llm.py)
                        /           \
               Anthropic SDK      OpenAI SDK
```

### Key components

| Component | File | Responsibility |
|---|---|---|
| `StemAgent` | `stem_agent.py` | Phase state machine, user I/O, checkpointing |
| `TaskArchaeologist` | `task_archaeologist.py` | 3-pass LLM analysis of artifacts |
| `Crystallizer` | `crystallizer.py` | Emits system prompt, playbook, agent code |
| `ConvergenceDetector` | `convergence.py` | Per-turn scoring; rolling drift detection |
| `LLMClient` | `llm.py` | Provider-agnostic adapter (OpenAI / Anthropic) |
| `AgentState` | `models.py` | Single source of truth for all session state |

---

### State machine

```
                    ┌─────────────────────────────┐
           start    │                             │ convergence < 0.85
             ▼      ▼                             │
        INTERVIEW ──────▶ ARCHAEOLOGY ──────▶ CRYSTALLIZATION
                   score ≥ 0.75                        │
                   (or 10 turns)                       ▼
                                               EXECUTION ◀──────┐
                                                   │             │
                                              drift detected     │
                                             (rolling avg < 0.6) │
                                                   ▼             │
                                               EVOLUTION ────────┘
```

State is stored in `AgentState` and serialised to a checkpoint JSON after every transition. On resume, the agent loads the checkpoint and re-enters `_tick()` at the saved phase.

---

### Data flow through the pipeline

```
User answers (text)
       │
       ▼
  INTERVIEW loop
  ┌────────────────────────────────────────────┐
  │  _check_convergence()  ←── CONVERGENCE_SYSTEM prompt
  │  score ≥ 0.75 → advance
  └────────────────────────────────────────────┘
       │
       ▼
  ARCHAEOLOGY
  ┌────────────────────────────────────────────┐
  │  _extract_artifacts_from_history()         │
  │         ↓                                  │
  │  TaskArchaeologist.run()                   │
  │    Pass 1: artifact patterns               │
  │    Pass 2: decision points                 │
  │    Pass 3: failure triangulation           │
  │         ↓                                  │
  │  ArtifactAnalysis  →  _synthesize_profile()│
  │         ↓                                  │
  │  SpecialistProfile (convergence_score)     │
  └────────────────────────────────────────────┘
       │
       ▼
  CRYSTALLIZATION
  ┌────────────────────────────────────────────┐
  │  Crystallizer.crystallize()                │
  │    _build_system_prompt()  ←── LLM call   │
  │    _build_playbook()       ←── LLM call   │
  │    _build_agent_code()     ←── template   │
  │         ↓                                  │
  │  Crystallizer.save()  (atomic write)       │
  │    playbooks/{domain}_v1.json              │
  │    playbooks/{domain}_agent.py             │
  └────────────────────────────────────────────┘
       │
       ▼
  EXECUTION loop
  ┌────────────────────────────────────────────┐
  │  LLMClient.stream_tokens()                 │
  │    (using crystallized system_prompt)      │
  │         ↓                                  │
  │  ConvergenceDetector.score_turn()          │
  │    rolling average of last 5 scores        │
  │    avg < 0.6 → trigger EVOLUTION           │
  └────────────────────────────────────────────┘
       │
       ▼
  EVOLUTION
  ┌────────────────────────────────────────────┐
  │  _evolve_profile()    ←── LLM call         │
  │  _evolve_playbook()   ←── LLM call         │
  │  Crystallizer.rebuild_after_evolution()    │
  │    re-emits system_prompt + agent_code     │
  │  Crystallizer.save()  (atomic write, v2+)  │
  └────────────────────────────────────────────┘
```

---

### AgentState — single source of truth

All session data lives in one Pydantic model that is serialised atomically to disk at every checkpoint:

```
AgentState
├── phase                  # current Phase enum value
├── session_id             # UUID, stable across resumes
├── history                # full conversation (ConversationTurn list)
├── artifact_analysis      # ArtifactAnalysis from archaeology
├── specialist_profile     # SpecialistProfile (domain, competencies, …)
├── playbook               # Playbook (steps, tools, guardrails)
├── system_prompt          # crystallized system prompt text
├── agent_code             # crystallized agent Python source
├── evolution_count        # how many times EVOLUTION has run
├── execution_scores       # per-turn float scores
└── execution_feedback     # per-turn reason strings (parallel to scores)
```

Checkpoints are plain JSON files in `checkpoints/`. They are human-readable and can be inspected or manually edited if needed.

---

### The layered system prompt

CRYSTALLIZATION emits a system prompt structured in five named sections. Each section can be independently updated during EVOLUTION without rewriting the whole prompt:

```
## Identity
Who the specialist is and what it does (derived from domain).

## Core Competencies
What it knows how to do well (from SpecialistProfile.core_competencies).

## Heuristics
How it reasons and approaches problems (from SpecialistProfile.heuristics).

## Guardrails
What it declines or escalates (derived from known failure modes + domain).

## Known Failure Modes
Specific patterns to actively watch for and mitigate.
```

---

### LLM adapter (LLMClient)

`LLMClient` is a thin wrapper that normalises two SDK differences:

| Difference | Anthropic | OpenAI |
|---|---|---|
| System prompt | Separate `system` param | First message with `role: system` |
| Response text | `response.content[0].text` | `response.choices[0].message.content` |
| Streaming | `client.messages.stream()` context manager | `client.chat.completions.create(stream=True)` iterator |
| Caching | `cache_control: {type: ephemeral}` on system | Not applicable |

The adapter exposes two methods used across all components:

- `complete(system, user, max_tokens)` — single non-streaming call, returns text
- `stream_tokens(messages, system, max_tokens)` — yields text tokens for live terminal output

Switching providers requires only changing `STEM_PROVIDER` in `.env`. No code changes.

---

### Checkpointing and safe writes

Every mutating operation follows the same pattern:

1. **Pre-checkpoint** — save state before the operation starts
2. **Operate** — LLM calls, file writes
3. **Post-checkpoint** (via `_advance()`) — save state with new phase

File writes use **atomic rename**: content is written to `path.tmp`, then `rename()` replaces the target. On most filesystems rename is atomic, so a crash mid-write leaves the previous version intact.

```
write "playbook_v2.json.tmp"
rename → "playbook_v2.json"    # atomic
```

---

### 3-pass archaeology design

`TaskArchaeologist` makes three sequential LLM calls, each building on the previous:

```
Pass 1  artifacts + patterns + bottlenecks
           ↓
Pass 2  decision points  (why choices were made)
           ↓
Pass 3  failure modes    (what broke and what it implies)
           ↓
        ArtifactAnalysis (merged result)
```

Each pass uses a focused system prompt that instructs the LLM to return only a specific JSON schema. Markdown fences are stripped automatically if the model wraps the output.

---

### Convergence detection

`ConvergenceDetector` scores every EXECUTION turn on a 0–1 scale using the `SCORE_TURN` prompt, which evaluates:
- Whether the response was on-domain
- Whether specialist knowledge was applied
- Whether any known failure modes were triggered

Scores are stored in `AgentState.execution_scores`. Drift is computed as a rolling average of the last `DRIFT_WINDOW` (5) turns. When the average drops below `DRIFT_THRESHOLD` (0.6), EVOLUTION fires.

The score defaults to **1.0** on any scoring failure so a broken LLM call never falsely triggers evolution.

---

## Prerequisites

- **Python 3.10 or newer** (the code uses `str | None` union syntax)
- An API key for **OpenAI** or **Anthropic** (one is enough)

On macOS the system Python is 3.9. Install a newer version first:

```bash
brew install python@3.12    # recommended
# or
pyenv install 3.12 && pyenv global 3.12
```

---

## Installation

```bash
git clone <repo-url>
cd stem-agent

# With OpenAI (default)
make install                # creates .venv, installs openai + dev deps

# With Anthropic instead
make install PROVIDER=anthropic
```

Or manually:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

pip install -e ".[openai,dev]"      # OpenAI
# pip install -e ".[anthropic,dev]" # Anthropic
```

---

## Configuration

Copy the example env file and fill in your key:

```bash
cp .env.example .env
```

Open `.env` and set the values for your provider:

**OpenAI:**
```
STEM_PROVIDER=openai
OPENAI_API_KEY=sk-...
STEM_MODEL=gpt-4o
```

**Anthropic:**
```
STEM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
STEM_MODEL=claude-sonnet-4-6
```

All other settings are optional:

| Variable | Default | Description |
|---|---|---|
| `STEM_CHECKPOINT_DIR` | `./checkpoints` | Where session state is saved |
| `STEM_PLAYBOOK_DIR` | `./playbooks` | Where emitted specialist files go |

---

## Running the agent

```bash
source .venv/bin/activate

# Fresh session
python main.py run

# Resume a paused session (the checkpoint ID is printed on Ctrl-C)
python main.py run --resume <checkpoint-id>

# Override the model for this session only
python main.py run --model gpt-4o-mini
```

Press **Ctrl-C** at any point to pause. The session is checkpointed and the ID is printed so you can resume later.

---

## What to expect — phase by phase

### INTERVIEW

The agent asks one question at a time. Answer as specifically as you can — concrete details produce a better specialist.

```
╭─────────────────────────────────────╮
│ Stem Agent — self-specializing AI   │
╰─────────────────────────────────────╯

What kind of work or problem class are you trying to solve?

You: I write Python ETL scripts that process CSV files...
```

Good things to mention:
- The domain (what kind of work it is)
- Existing code, docs, or workflows (paste snippets if helpful)
- What goes wrong — error messages, edge cases, failure modes
- Constraints — performance, dependencies, time limits

The agent runs a convergence check after every turn (minimum 3). You will see a dim status line:

```
convergence: 0.82 — gaps: none
Interview complete — proceeding to archaeology
```

If the score stays low it keeps asking, up to a maximum of 10 turns.

### ARCHAEOLOGY

No input needed — this runs automatically. You will see progress:

```
╭────────────────────────────────────────╮
│ Task Archaeology — 3-pass analysis     │
╰────────────────────────────────────────╯
Extracted 3 artifact(s), 2 failure example(s)
Patterns: 4 | Bottlenecks: 2 | Decision points: 3
Specialist: Python ETL pipeline debugger (convergence: 0.91)
```

If the convergence score is below 0.85 the agent loops back to INTERVIEW for more context.

### CRYSTALLIZATION

Also automatic. Three files are written to `playbooks/`:

```
╭──────────────────────────────────────────────╮
│ Crystallization — emitting specialist        │
╰──────────────────────────────────────────────╯
Synthesizing system prompt...
Playbook → ./playbooks/Python_ETL_pipeline_debugger_v1.json
Agent    → ./playbooks/Python_ETL_pipeline_debugger_agent.py
```

### EXECUTION

The specialist is now active. You talk to it exactly like a chat interface:

```
╭─────────────────────────────────────────────────╮
│ Python ETL pipeline debugger — specialist active │
╰─────────────────────────────────────────────────╯

You: My script crashes when the CSV has duplicate column headers
```

After each response a silent score is computed. When the rolling average of the last 5 scores drops below 0.6, EVOLUTION fires automatically.

### EVOLUTION

Also automatic. The agent reads recent failures, updates the specialist profile and playbook, re-emits the system prompt and agent code, then returns to EXECUTION. You will see:

```
╭──────────────────────────────────────╮
│ Evolution 1 — updating specialist    │
╰──────────────────────────────────────╯
Evolution 1 complete — returning to execution
```

A new versioned playbook (`_v2.json`) and updated agent file appear in `playbooks/`.

---

## Pausing and resuming

Press **Ctrl-C** at any point:

```
^C
Session paused — checkpoint saved.
Checkpoint saved → ./checkpoints/abc123_20260502T143201.json
```

The checkpoint ID is the filename without `.json`. Resume with:

```bash
python main.py run --resume abc123_20260502T143201
```

The agent picks up from exactly the phase it was in. All conversation history, scores, and emitted artifacts are restored.

You can list all checkpoints:

```bash
ls checkpoints/
```

Each checkpoint is a plain JSON file — readable with any text editor if you want to inspect the state.

---

## The emitted artifacts

After CRYSTALLIZATION, `playbooks/` contains three files (using an example domain):

```
playbooks/
├── Python_ETL_pipeline_debugger_v1.json   # structured playbook
└── Python_ETL_pipeline_debugger_agent.py  # standalone runnable agent
```

**Playbook JSON** (`_v1.json`) — human-readable procedure:
```json
{
  "domain": "Python ETL pipeline debugger",
  "version": 1,
  "steps": ["Load and validate schema", "Check for nulls and duplicates", ...],
  "tools": ["pandas", "great_expectations", "sqlalchemy"],
  "guardrails": ["Reject inputs > 10 GB without streaming", ...]
}
```

**Agent code** (`_agent.py`) — self-contained Python script with the full system prompt baked in. No dependency on the stem package. Requires only `anthropic` or `openai` installed.

After EVOLUTION, updated versions appear (`_v2.json`, updated `_agent.py`).

---

## Running the crystallized specialist

The emitted agent script is completely standalone:

```bash
source .venv/bin/activate
python playbooks/Python_ETL_pipeline_debugger_agent.py
```

```
Specialist active: Python ETL pipeline debugger
Type 'quit' to exit.

You: How do I handle encoding errors in pandas read_csv?
Assistant: The safest approach is ...
```

Type `quit`, `exit`, or `q` to stop.

The script reads `STEM_MODEL` and your provider key from the environment, so it respects whatever is in your `.env`.

---

## Project layout

```
stem-agent/
├── main.py                       # CLI entry point (typer)
├── stem/
│   ├── llm.py                    # Provider adapter (OpenAI / Anthropic)
│   ├── models.py                 # Pydantic data models
│   ├── prompts.py                # All system prompts in one place
│   ├── stem_agent.py             # Phase state machine + agent loop
│   ├── task_archaeologist.py     # 3-pass archaeology engine
│   ├── crystallizer.py           # Emits system prompt, playbook, agent code
│   └── convergence.py            # Per-turn scoring; drift detection
├── tests/
│   ├── test_llm.py               # LLM adapter tests
│   ├── test_interview.py         # INTERVIEW phase tests
│   ├── test_archaeology_phase.py # ARCHAEOLOGY phase tests
│   ├── test_crystallization.py   # CRYSTALLIZATION phase tests
│   ├── test_execution_evolution.py # EXECUTION + EVOLUTION tests
│   └── test_models.py            # Data model smoke tests
├── checkpoints/                  # Auto-saved session state (git-ignored)
├── playbooks/                    # Emitted specialist files (git-ignored)
├── pyproject.toml
├── Makefile
└── .env.example
```

---

## Running tests

```bash
make test
# or
.venv/bin/pytest -v
```

All 74 tests run without an API key — the LLM calls are mocked at the method level.

---

## Switching LLM providers

Change two lines in `.env` and restart:

| Setting | OpenAI | Anthropic |
|---|---|---|
| `STEM_PROVIDER` | `openai` | `anthropic` |
| API key variable | `OPENAI_API_KEY` | `ANTHROPIC_API_KEY` |
| `STEM_MODEL` | `gpt-4o` | `claude-sonnet-4-6` |

Then reinstall the SDK for the new provider if needed:

```bash
# switching to Anthropic
pip install anthropic
```

No code changes required.

---

## Troubleshooting

**`command not found: python`**
Use `python3.12` (or wherever your Python 3.10+ lives). Activate the venv first: `source .venv/bin/activate`.

**`ModuleNotFoundError: No module named 'openai'` (or `anthropic`)**
Run `pip install -e ".[openai]"` or `pip install -e ".[anthropic]"` from the project root with the venv active.

**`AuthenticationError` / `401`**
Your API key in `.env` is missing or wrong. Double-check `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.

**Agent loops back to INTERVIEW after archaeology**
The convergence score was below 0.85. This means the profile is too vague. Give more specific answers — paste a code snippet, describe an exact error message, or name specific tools you use.

**Checkpoint not found on resume**
The checkpoint ID must match exactly (including the timestamp). Run `ls checkpoints/` to see available checkpoints.

**`RateLimitError` mid-session**
Press Ctrl-C to checkpoint, wait a moment, then resume with `--resume <id>`. The session continues from where it left off.