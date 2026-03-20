# AGENTS.md — Multi-Agent Collaboration Protocol for FPPE

## Philosophy: Teamwork Over Territory

This project is built by a **team of AI agents working together**, not a collection of individuals working in parallel. Every agent's work depends on and feeds into the work of others. The quality of the final system is determined by the quality of collaboration, not individual brilliance.

**Core principles:**
1. **No agent works in isolation.** Every implementation is reviewed by a different agent. Every review is synthesized by a coordinator. Every decision traces back to a shared plan.
2. **Disagreement is productive.** When agents disagree on an approach, that's signal — it means the design space has unexplored corners. Surface disagreements early; resolve them with evidence.
3. **Trust but verify.** Trust the agent who wrote the code to have followed the spec. Verify by reading their work against the design doc, not by rewriting it.
4. **The human (Sleep) is the tiebreaker.** When agents can't resolve a design conflict, escalate to the human with both sides clearly stated. Never silently override another agent's decision.
5. **Linear is the single source of truth.** All task state, assignments, and handoffs live in Linear. If it's not in Linear, it didn't happen.

---

## Agent Profiles & Routing Configuration

To ensure seamless automation through Linear, each agent must be assigned using its exact Cursor API string in the issue description (for example: `[model=claude-3-opus]`).

1. Claude 3 Opus
Cursor API String: claude-3-opus

Role: Architect / Lead

Best For: Complex multi-file changes, merge coordination, final plan execution, and critical path synthesis. Owns the hardest architectural integration tasks for the quantitative engine.

2. Claude 3.5 Sonnet
Cursor API String: claude-3.5-sonnet

Role: Fast Implementer

Best For: Scoped tasks with clear specs, writing test coverage, and updating project documentation to accurately reflect the current repository state.

3. Composer 2
Cursor API String: N/A (Native Cursor Interface)

Role: Plan Review / Synthesizer

Best For: Reviewing plans before execution, synthesizing cross-agent reviews, and acting as the human-in-the-loop gateway to approve milestones.

4. GPT-4o
Cursor API String: gpt-4o

Role: Parallel Implementation

Best For: Independent feature branches, isolated Python module work, and drafting highly specific data processing scripts.

5. o1 (OpenAI Reasoning Model)
Cursor API String: o1

Role: Deep Analysis

Best For: Research-heavy tasks, deep algorithmic validation (e.g., verifying drawdown brake mechanics and position sizing math), and extreme edge-case detection.

6. Gemini 3.1 Pro
Cursor API String: gemini-3.1-pro-preview

Role: Cross-Validation / Red Team

Best For: Second-opinion reviews, proposing alternative architectural approaches, and identifying logic blind spots missed by the Anthropic or OpenAI models.
---

## Collaboration Patterns

### Pattern 1: Write → Cross-Review
```
Agent A writes code → Agent B (different family) reviews it
```
- Opus writes risk_engine.py → Gemini reviews
- GPT-4o writes risk_state.py → o1 reviews
- **Rule:** The writer NEVER reviews their own code. Cross-family reviews catch more bugs.

### Pattern 2: Parallel Implementation → Integration
```
Agent A implements Module X  ─┐
                               ├── Agent C integrates both
Agent B implements Module Y  ─┘
```
- GPT-4o implements risk_state.py (pure dataclasses, no deps)
- Opus implements risk_engine.py (imports risk_state)
- Opus integrates both into backtest_engine.py
- **Rule:** Parallel agents agree on interfaces BEFORE implementation starts.

### Pattern 3: Multi-Agent Review → Synthesis
```
Agent A reviews aspect 1  ─┐
Agent B reviews aspect 2   ├── Composer synthesizes into final plan
Agent C reviews aspect 3  ─┘
```
- o1 reviews math, Opus reviews integration, Gemini proposes alternatives
- Composer reads all reviews, finds consensus, escalates disagreements
- **Rule:** Reviews are posted as Linear comments. Composer reads ALL comments before synthesizing.

### Pattern 4: Escalation
```
Agent disagrees with design → Posts concern on Linear issue → Composer evaluates → Human decides
```
- **Rule:** Never silently deviate from the approved plan. If you think the plan is wrong, say so on the issue.

---

## Rules of Engagement

### For ALL agents:

1. **Read the issue description completely** before starting work. Every issue has: Objective, Scope, Deliverable, Agent Assignment, Acceptance Criteria.

2. **Post progress as Linear comments.** When you start an issue, comment "Starting work." When blocked, comment with the blocker. When done, comment with deliverable summary.

3. **Never modify files outside your issue scope.** If you discover a bug in another agent's module, file a new issue — don't fix it yourself.

4. **Run the full test suite before claiming "done."** `python -m pytest tests/ -v` must show 0 failures.

5. **Follow the project's code style** (see CLAUDE.md). Frozen dataclasses, type hints, docstrings with Args/Returns.

6. **Respect blocking relationships.** If your issue is blocked by SLE-9, do NOT start until SLE-9 is Done.

7. **One branch per agent per issue.** Use the Linear-generated branch name (e.g., `isaiahms267/sle-11-implement-risk_enginepy...`).

### For reviewers specifically:

8. **Rate findings by severity:** Critical (blocks merge), Important (should fix), Suggestion (nice to have).

9. **Include code references.** "Line 47 of risk_engine.py has an off-by-one" — not "the sizing looks wrong somewhere."

10. **Verify against the design doc**, not your personal preferences. If the design says "linear brake," don't suggest "exponential" in a code review — that's a design issue, not a code issue.

---

## Task Assignment by Milestone

### M1: Design Review & Plan Approval
| Issue | Agent | Type | Priority |
|-------|-------|------|----------|
| SLE-5: Risk model math review | o1 | Review | High |
| SLE-6: Integration feasibility | Opus | Review | High |
| SLE-7: PROJECT_GUIDE accuracy | Sonnet | Review | Medium |
| SLE-8: Alternative approaches | Gemini | Review | Medium |
| SLE-9: Synthesize into plan | Composer | Review | Urgent |

### M2: Core Implementation
| Issue | Agent | Type | Priority |
|-------|-------|------|----------|
| SLE-10: risk_state.py | GPT-4o | Implementation | High |
| SLE-11: risk_engine.py | Opus | Implementation | Urgent |
| SLE-12: Stress tests | Sonnet | Implementation | Medium |

### M3: Integration & Backtest
| Issue | Agent | Type | Priority |
|-------|-------|------|----------|
| SLE-13: backtest_engine integration | Opus | Integration | Urgent |
| SLE-14: Integration tests | GPT-4o | Implementation | High |
| SLE-15: run_phase2.py | Sonnet | Implementation | Medium |
| SLE-16: Cross-agent code review | Gemini + o1 | Review | High |

### M4: Validation & Documentation
| Issue | Agent | Type | Priority |
|-------|-------|------|----------|
| SLE-17: Backtest validation | Opus | Integration | Urgent |
| SLE-18: Doc updates | Sonnet | Implementation | High |
| SLE-19: Final sign-off | Composer | Review | Urgent |

---

## Communication Protocol

### Where to communicate:
- **Task-specific discussion:** Comment on the Linear issue
- **Cross-task concerns:** Create a new Linear issue tagged with affected agents
- **Design disagreements:** Comment on SLE-9 (synthesis issue) or the relevant review issue
- **Emergencies (broken tests, blocked pipeline):** Tag as Urgent priority in Linear

### Handoff format:
When completing an issue that another agent depends on, post a comment:
```
HANDOFF to [Agent Name]:
- Files created/modified: [list]
- Tests passing: [count]
- Any caveats: [notes]
- Ready for: [next issue ID]
```

---

## Linear Project Structure

- **Project:** FPPE Phase 3: Portfolio Manager
- **Team:** Sleepern
- **Labels:** `Agent: *` for assignment, `Phase 3` for scope, `Review`/`Implementation`/`Integration` for type
- **Milestones:** M1 → M2 → M3 → M4 → M5 (sequential gates, parallel work within each)

---

## Phase 3 Milestone Assignments

### M1: Data Structures & Unit Tests
| Issue | Agent | Type | Status |
|-------|-------|------|--------|
| SLE-30: portfolio_state.py dataclasses | Codex | Implementation | ✅ Done |
| SLE-31: test_portfolio_state.py | Sonnet | Implementation | ✅ Done |

### M2: Core Logic & Coverage
| Issue | Agent | Type | Status |
|-------|-------|------|--------|
| SLE-32: portfolio_manager.py | Opus | Implementation | ✅ Done |
| SLE-33: test_portfolio_manager.py | Sonnet | Implementation | ✅ Done |

### M3: Integration & Backtest
| Issue | Agent | Type | Status |
|-------|-------|------|--------|
| SLE-34: backtest_engine.py integration | Opus | Integration | ✅ Done |
| SLE-35: test_phase3_integration.py | Sonnet | Implementation | ✅ Done |

### M4: Exports & Regression
| Issue | Agent | Type | Status |
|-------|-------|------|--------|
| SLE-36: trading_system/__init__.py | Sonnet | Implementation | ✅ Done |
| SLE-37: Full regression suite + tag | Opus | Integration | ✅ Done |

### M5: Validation & Code Review
| Issue | Agent | Type | Status |
|-------|-------|------|--------|
| SLE-38: Backtest comparison P1/P2/P3 | Opus | Analysis | ✅ Done |
| SLE-39: Cross-agent code review | Sonnet + Gemini | Review | ✅ Done |
| SLE-40: Final cleanup + git push prep | Sonnet | Implementation | ✅ Done |

**Phase 3 result:** 556 tests passing, 0 failures. Phase 3 matches Phase 2 performance metrics exactly (Sharpe 1.93, Ann. 15.1%, MaxDD 3.9%). Portfolio layer adds explicit rejection tracking (37 PM rejections in 2024 backtest).

---

*AGENTS.md v2.0 — March 20, 2026 — Updated for Phase 3 completion*
*This file is read by ALL agents at the start of each session. Keep it current.*

---

## Cursor Cloud specific instructions

### Environment

- **Python 3.12** is required. The VM ships with Python 3.12.3 at `/usr/bin/python3`.
- The update script creates a venv at `/workspace/venv` and installs both `requirements.txt` and `requirements-dev.txt`. Activate with `source venv/bin/activate`.
- `python3.12-venv` must be installed via apt if missing (the update script handles venv creation).

### Running tests

- `source venv/bin/activate && python -m pytest tests/ -v` — runs full suite (~458 tests, ~2.5 min on cloud VM).
- All tests use synthetic data from `tests/conftest.py` fixtures; no network access or data files needed.
- CI targets Windows (`windows-latest`), but the full suite also passes on Linux.

### Running the engine

- Entry points are run as Python modules: `python -m pattern_engine.live`, `python -m pattern_engine.overnight`, etc.
- The live/overnight runners require market data (downloaded via yfinance); for dev/testing, use synthetic data as the test fixtures demonstrate.

### Lint

- No linter (flake8/ruff/mypy/pylint) is currently configured. Only `py_compile` checks are feasible.

### Gotchas

- `predict()` returns a `PredictionResult` object (not a DataFrame). Access `.signals`, `.calibrated_probabilities`, `.n_matches`, `.buy_count`, `.sell_count`, `.hold_count`, `.avg_matches`.
- `EngineConfig` uses `top_k` (not `k`) and `projection_horizon` (not `forward_horizon`).
- `EngineConfig` is a frozen dataclass containing a dict field (`feature_weights`) — use `repr()` instead of `hash()`.
- The `data/` directory is gitignored and empty in fresh clones; tests do not need it.
