# Stem Agent

A self-specializing AI agent that starts knowing nothing and grows into whatever a class
of tasks requires.

You don't tell it what to be. You point it at a *world* — some tasks, some tools, a way of
scoring attempts — and the same code discovers what it needs to become by trying things and
watching what works.

---

## The idea

A "stem agent" is undifferentiated on purpose. At generation zero its system prompt says,
in effect, *"you've been dropped somewhere unfamiliar with some tools and an objective; you
don't yet know what kind of agent you should be."* It is given an environment, not an
identity. It pokes at the tools, attempts the tasks, fails, and an evolution step rewrites
the agent based on what actually happened. Run that loop and the **same Python** turns into
a trader in one environment and a security auditor in another.

This is the second version. The first version (preserved in git history) was a learning
loop, but it cheated: its very first question was *"what domain do you want?"* — so the
starting point was already specialized, and the only thing that ever changed was the
wording of a prompt. That's the critique from JetBrains, and it was right. A learning loop
that refines a prompt around a domain you handed it is not a stem agent. A stem agent has to
*find* the domain in the environment, and it has to be able to change more than its prose.

So the rebuild changes two things:

1. **Truly undifferentiated start.** There is no domain input anywhere. The domain is
   implicit in the tools and the task objective, and is never named to the agent. A task
   says *"grow `portfolio_value` before the final step"* — it never says *"you are a
   trader."* That word only ever shows up later, in the genome, if evolution puts it there.

2. **A real evolution surface.** Evolution can change identity, **and** which tools the
   agent uses, **and** the reusable skills it carries, **and** its loop structure (whether
   it plans / verifies / reflects), **and** the sub-agents it can spawn, **and** its own
   success criteria. Not just the prompt.

---

## What evolves

The mutable self is a single object — the `Specialization` genome. Every field is a
distinct surface the evolution engine can rewrite, each with its own mutation type:

| Surface | What it controls | Mutation |
|---|---|---|
| `identity` | the system-prompt persona / instructions | `rewrite_identity` |
| `adopted_tools` | which discovered tools it actually uses, + learned usage notes | `adopt_tool` |
| `skills` | named, reusable procedures it accumulates instead of re-deriving | `add_skill` |
| `loop` | which phases run: plan / act / verify / reflect | `set_loop` |
| `subagents` | specs for focused sub-agents it can delegate to | `define_subagent` |
| `eval_criteria` | the rubric it judges its own work against | `update_eval_criteria` |

A blank genome has none of these (`Specialization().is_blank() == True`). Turning on
`loop.verify` literally inserts a self-check turn before a final answer is accepted;
defining a sub-agent literally adds a `spawn_subagent` tool to the agent's toolset. These
are behavioral changes, not decoration.

---

## How to run

Docker-first, because a self-modifying agent with web access and code execution should be
contained.

```bash
# 1. Prove the whole pipeline with no API key (toy domain, tokenless):
make docker-eval-mock          # or: make eval-mock

# 2. Real evaluation across all domains (spends tokens):
cp .env.example .env           # add your ANTHROPIC_API_KEY
make docker-eval               # or, on the host: make eval
```

Without Docker:

```bash
make install                   # venv + deps
make test                      # 34 tests, no API key needed
make eval-mock                 # tokenless end-to-end demo

# real runs:
python main.py eval --domain trading -g 3      # baseline vs evolved
python main.py evolve --domain trading -o results/genome.json
python main.py run --domain trading            # inspect one rollout (verbose)
python main.py run --domain trading --genome results/genome.json   # with the evolved genome
```

Tool use is implemented for Anthropic; that is the supported provider.

---

## Evaluation

The harness is built to test the **evolution**, not just the output:

1. **baseline** — a blank genome on the held-out *test* tasks. Expected near zero.
2. **evolve** — run K generations over the *train* tasks, mutating the genome.
3. **evolved** — the resulting genome on the *same* held-out tasks.

If evolution did nothing real, baseline and evolved are equal. The gap is the signal. It
also reports process metrics: tool calls, errors, error recoveries, reflection depth,
sub-agent runs, and which genome surfaces evolution touched.

### What's actually proven here

**The pipeline (deterministic, no tokens).** `make eval-mock` runs the full
rollout → evolve → re-evaluate loop on a toy environment with a scripted brain. Baseline
scores **0.00**, evolved scores **1.00**, and evolution touches four different surfaces
(`rewrite_identity`, `adopt_tool`, `add_skill`, `update_eval_criteria`) — not just the
prompt. This is asserted in `tests/test_harness.py`.

**The trading domain's economics (deterministic).** The simulated exchange is a seeded,
noisy sinusoid that completes whole cycles, so the score has a genuine ~0 floor and real
headroom:

| strategy | return | resulting score |
|---|---|---|
| do nothing | 0% | **0.00** |
| buy-and-hold | ≈ −5% to −8% | **0.00** (floored) |
| clairvoyant cycle trading | ≈ +44% to +46% | **~1.00** |

A fresh agent that never figures out the buy-low-sell-high pattern genuinely makes no
money; an agent that discovers it is rewarded in proportion to a clairvoyant optimum. Tests
in `tests/test_exchange.py` pin all three rows.

**Cross-domain, with a real model — you run this.** I did not burn tokens auto-running the
live eval (and Docker isn't available in the build sandbox). Run it yourself:

```bash
make eval          # writes results/eval.json + prints a table like:
```

```
| domain   | baseline | evolved |  Δ   | generations | surfaces evolved |
|----------|----------|---------|------|-------------|------------------|
| trading  |   ...    |   ...   | ...  |      3      |       ...        |
| security |   ...    |   ...   | ...  |      3      |       ...        |
| research |   ...    |   ...   | ...  |      3      |       ...        |
```

### Honest about the domains

- **Trading** is the proven one: objective P&L scoring, a real ~0 baseline, multi-step
  agentic tasks (you have to discover the API, advance time, and trade).
- **Security** is functional but scaffolded: virtual files with planted vulnerabilities,
  an objective precision/recall (F1) score, tools to read and grep. Less tuned than trading.
- **Research** is the fuzziest: it uses real web search/fetch tools, but scores by keyword
  coverage of ground-truth facts — a crude proxy, and a strong base model can sometimes
  answer from memory, so its baseline is the least clean of the three. It's here to prove
  the *same agent code* runs on a web-using, non-objective domain, not as a polished
  benchmark.

---

## Architecture

The agent, the evolution engine, and the eval harness are all completely domain-agnostic.
Adding a domain is one line in `stem/envs/registry.py`.

```
   ENVIRONMENT (the world)            UNDIFFERENTIATED AGENT              EVOLUTION ENGINE
 ┌───────────────────────────┐     ┌───────────────────────────┐     ┌──────────────────┐
 │ tasks  (objectives only)  │     │  rollout(genome):         │     │ reflect over     │
 │ tools  (names + schemas)  │────▶│   plan? act verify? reflect│────▶│ trajectories +   │
 │ score  (objective signal) │     │   (calls env tools)       │     │ scores           │
 └───────────────────────────┘     └───────────────────────────┘     │      │           │
        ▲                                    │ Trajectory             │ propose Mutations │
        │                                    ▼                        │      │           │
        │                         held-out eval: gen0 (~0)            │ apply to genome   │
        └──────────────────────────  vs evolved  ◀───────────────────┴──────────────────┘
```

### The rollout loop

```
genome ── render ──▶ system prompt + tool schemas
                          │
   task objective ───▶ [ plan? ] ─▶ act loop ──────────────┐
                                     │  LLM turn            │
                                     │   ├─ tool_use ─▶ env.execute() ─▶ observation ─┐
                                     │   └─ final text ─▶ [ verify? ] ─▶ accept       │
                                     └◀────────────────────────────────────────────────┘
                          │
                     [ reflect? ] ─▶ Trajectory (tool calls, errors, recoveries, notes)
```

`plan` / `verify` / `reflect` are off in a blank genome and switched on by `set_loop`
mutations. A `spawn_subagent` tool appears only once the genome defines a sub-agent; the
agent intercepts that call and runs a focused nested rollout with a restricted toolset.

### The genome (single source of the agent's self)

```
Specialization
├── identity          # generic at gen 0; a real persona once the domain is obvious
├── adopted_tools     # [] at gen 0
├── skills            # [] at gen 0
├── loop              # plan/verify/reflect all off at gen 0
├── subagents         # [] at gen 0
├── eval_criteria     # [] at gen 0
├── generation        # bumped each evolution step
└── lineage           # human-readable change log per generation
```

Checkpointed atomically (write-temp-then-rename) to `results/` as both a full run and a
genome sidecar, so evolved agents can be reloaded and re-run.

### Layout

```
stem/
├── models.py          # Specialization genome, Task, ToolSpec, Trajectory, Mutation, RunState
├── llm.py             # Anthropic tool-use client + ScriptedLLM (tokenless test double)
├── environment.py     # Environment ABC + ToolRegistry (tool dispatch + error capture)
├── agent.py           # the rollout loop (plan/act/verify/reflect) + sub-agent spawning
├── evolution.py       # reflect over trajectories -> Mutations -> apply (pure) to genome
├── checkpoint.py      # atomic save/load of run state + genome
├── tools/builtin.py   # discoverable generic tools: web_search, web_fetch, run_python
├── envs/
│   ├── trading/       # FULLY IMPLEMENTED: deterministic exchange + objective P&L score
│   ├── security/      # SCAFFOLDED: planted-vuln files + F1 audit score
│   └── research/      # SCAFFOLDED: real web tools + fact-coverage score
└── eval/
    ├── harness.py     # baseline vs evolved + process metrics
    └── mock_demo.py   # toy env + scripted brain for the tokenless end-to-end demo
main.py                # CLI: domains / eval / evolve / run
tests/                 # 34 tests, no API key required
Dockerfile, docker-compose.yml, Makefile
```

---

## What I learned

**The differentiation was hiding in the first prompt, and it was easy to miss.** The
original design felt legitimately adaptive — it had convergence detection, drift scoring,
re-crystallization. But all of that machinery operated *after* a human had already said
"build me a Python ETL debugger." The agent never had to discover anything; it refined
wording around a target it was handed. The lesson that stuck: if you want to know whether an
agent is really undifferentiated, look at the very first thing it requires from you. If it's
the domain, you've already lost.

**"What can change" matters more than "how well it changes."** It's tempting to spend all
your effort on a clever evolution algorithm. But a clever algorithm that can only edit a
prompt will always top out at prompt-shaped improvements. Widening the surface — letting
evolution adopt tools, toggle a verification step, add a sub-agent — does more for genuine
specialization than a smarter optimizer over a narrow surface would. Most of the work in the
rebuild went into making each surface a real, behavioral lever and keeping `apply_mutation`
pure so it's trivially testable.

**A benchmark has to be able to score zero.** The hardest part wasn't the agent, it was
designing an environment where a competent base model genuinely *can't* succeed cold. LLMs
are good enough that most "tasks" get a decent score with no evolution at all, which hides
whether evolution did anything. The trading exchange was deliberately shaped so that the
obvious passive strategy (buy-and-hold) makes nothing and doing nothing makes nothing — only
discovering and exploiting the cycle pays. That ~0 floor is what makes the improvement
legible.

**What I'd do differently with more time.** (1) Score research with an LLM judge against a
rubric instead of keyword coverage — the current proxy is too blunt and leaks base-model
knowledge. (2) Bring security up to the trading domain's level: more files, partial-credit
on line numbers, and a baseline I've measured rather than assumed. (3) Add a guard against
evolution *regressing* a genome (keep the best-scoring ancestor), which matters once you run
many generations on a noisy signal. (4) Run the real cross-domain eval and put the actual
numbers in this README — right now the honest statement is "the pipeline and the trading
economics are proven; the live multi-domain numbers are one `make eval` away."
