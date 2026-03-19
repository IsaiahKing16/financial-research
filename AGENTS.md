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

## Agent Roster

### Claude Opus 4.6 — Lead Architect
**Label:** `Agent: Opus`
**Strengths:** Deep codebase understanding, complex multi-file changes, merge coordination, architectural decisions
**Weaknesses:** Slower output, higher cost per token
**Assigned to:** Core algorithm implementation, integration work, backtest engine modifications, final validation
**How to use:** Give Opus the hardest problems — the ones where understanding 5 files simultaneously matters. Don't waste Opus on simple doc updates.

### Claude Sonnet 4.6 — Fast Implementer
**Label:** `Agent: Sonnet`
**Strengths:** Speed, well-scoped tasks, test writing, documentation updates
**Weaknesses:** May miss cross-module implications on very complex tasks
**Assigned to:** Entry-point scripts, stress tests, documentation updates, fast verification tasks
**How to use:** Give Sonnet tasks with clear boundaries and explicit acceptance criteria. Sonnet excels when the "what" is defined and the "how" needs execution.

### Cursor Composer 2 — Coordinator
**Label:** `Agent: Composer`
**Strengths:** Multi-agent synthesis, conflict resolution, plan management, big-picture thinking
**Weaknesses:** Not the best for line-by-line implementation
**Assigned to:** Review synthesis, implementation plan approval, final sign-off, cross-agent coordination
**How to use:** Composer is the glue. Use it to read all reviews, find consensus, resolve conflicts, and produce the plan that everyone follows. Composer never writes production code.

### GPT-5.3 Codex Extra High — Parallel Implementer
**Label:** `Agent: Codex`
**Strengths:** Isolated module implementation, test generation, parallel execution
**Weaknesses:** Needs explicit context — doesn't carry conversation history
**Assigned to:** Self-contained modules (risk_state.py), integration test writing
**How to use:** Give Codex modules with zero or minimal dependencies on other new code. Include the full design doc section and all type signatures in the issue description.

### GPT-5.4 Extra High — Deep Analyst
**Label:** `Agent: GPT-5.4`
**Strengths:** Mathematical analysis, algorithm verification, academic literature awareness
**Weaknesses:** May over-engineer or suggest unnecessary complexity
**Assigned to:** Design review (risk model math), code review of data structures, edge case analysis
**How to use:** Point GPT-5.4 at algorithms and ask "is this correct?" Include the mathematical spec and expected invariants.

### Gemini 3.1 Pro — Independent Reviewer
**Label:** `Agent: Gemini`
**Strengths:** Fresh perspective, alternative approaches, catches assumptions others miss
**Weaknesses:** Less context on FPPE history — needs full design docs provided
**Assigned to:** Design alternative proposals, cross-agent code review, assumption stress-testing
**How to use:** Gemini is the "red team." Give it the design and ask it to find problems. Give it code written by Claude/GPT and ask it to find bugs. Its value is in being different, not better.

---

## Collaboration Patterns

### Pattern 1: Write → Cross-Review
```
Agent A writes code → Agent B (different family) reviews it
```
- Opus writes risk_engine.py → Gemini reviews
- Codex writes risk_state.py → GPT-5.4 reviews
- **Rule:** The writer NEVER reviews their own code. Cross-family reviews catch more bugs.

### Pattern 2: Parallel Implementation → Integration
```
Agent A implements Module X  ─┐
                               ├── Agent C integrates both
Agent B implements Module Y  ─┘
```
- Codex implements risk_state.py (pure dataclasses, no deps)
- Opus implements risk_engine.py (imports risk_state)
- Opus integrates both into backtest_engine.py
- **Rule:** Parallel agents agree on interfaces BEFORE implementation starts.

### Pattern 3: Multi-Agent Review → Synthesis
```
Agent A reviews aspect 1  ─┐
Agent B reviews aspect 2   ├── Composer synthesizes into final plan
Agent C reviews aspect 3  ─┘
```
- GPT-5.4 reviews math, Opus reviews integration, Gemini proposes alternatives
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
| SLE-5: Risk model math review | GPT-5.4 | Review | High |
| SLE-6: Integration feasibility | Opus | Review | High |
| SLE-7: PROJECT_GUIDE accuracy | Sonnet | Review | Medium |
| SLE-8: Alternative approaches | Gemini | Review | Medium |
| SLE-9: Synthesize into plan | Composer | Review | Urgent |

### M2: Core Implementation
| Issue | Agent | Type | Priority |
|-------|-------|------|----------|
| SLE-10: risk_state.py | Codex | Implementation | High |
| SLE-11: risk_engine.py | Opus | Implementation | Urgent |
| SLE-12: Stress tests | Sonnet | Implementation | Medium |

### M3: Integration & Backtest
| Issue | Agent | Type | Priority |
|-------|-------|------|----------|
| SLE-13: backtest_engine integration | Opus | Integration | Urgent |
| SLE-14: Integration tests | Codex | Implementation | High |
| SLE-15: run_phase2.py | Sonnet | Implementation | Medium |
| SLE-16: Cross-agent code review | Gemini + GPT-5.4 | Review | High |

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

- **Project:** FPPE Phase 2: Risk Engine
- **Team:** Sleepern
- **Labels:** `Agent: *` for assignment, `Phase 2` for scope, `Review`/`Implementation`/`Integration` for type
- **Milestones:** M1 → M2 → M3 → M4 (sequential gates, parallel work within each)

---

*AGENTS.md v1.0 — March 19, 2026*
*This file is read by ALL agents at the start of each session. Keep it current.*
