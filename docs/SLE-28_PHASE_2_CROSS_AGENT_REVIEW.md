# SLE-28 — Phase 2 cross-agent review (Gemini / independent)

**Review date:** 2026-03-20  
**Scope:** Incoming work on remote branches (merged locally for validation):  
`cursor/SLE-13-risk-engine-backtest-integration-7507`, `cursor/SLE-14-phase-2-integration-tests-87c6`, `cursor/SLE-15-phase-2-comparison-runner-4866`, `cursor/SLE-25-phase-2-documentation-updates-9966`, `cursor/SLE-9-approved-implementation-plan-0bf1`.

**Test run (merged tip):** `402 passed` (~2.3 min), `python -m pytest tests/ -v`.

**Not verified in-repo:** GitHub PR numbers **#16, #17, #18** (no GitHub API access from this environment). Treat branch names above as the merge units to verify in the PR UI.

**Missing from fetched branches (issue checklist):**

- `docs/SLE-26_REMAKE_ALTERNATIVE_APPROACHES_REVIEW.md` — not present on listed remotes.
- `docs/PHASE2_RISK_ENGINE.md` — not present; `docs/PHASE2_SYSTEM_DESIGN.md` is the live spec.

---

## 1. Bugs, logic risks, and design deviations

### Critical — none found

Core sizing identity `raw_weight = max_loss_per_trade_pct / stop_distance_pct` with clamping, stop at `entry × (1 − stop_distance_pct)`, and drawdown linear brake + hard halt are implemented consistently between `compute_drawdown_scalar` and `RiskState.update`.

### Important

1. **ATR implementation vs approved plan (`docs/SLE-9_APPROVED_IMPLEMENTATION_PLAN.md`)**  
   The approved plan states ATR should use the **`ta` library** (`ta.volatility.AverageTrueRange`). **`risk_engine.compute_atr_pct`** implements True Range + **pandas `ewm`** instead. This is a deliberate, testable implementation, but it is a **spec deviation**: numerics may differ slightly from Wilder ATR as implemented in `ta`.  
   **Suggestion:** Either switch to `ta` for parity with the written gate, or amend SLE-9 / `PHASE2_SYSTEM_DESIGN.md` to “pandas EWM with α = 1/lookback” so reviews and future refactors do not treat it as a regression.

2. **Portfolio drawdown halt uses prior-day equity in the exit loop**  
   Forced `drawdown_halt` exits use the loop’s `equity` / `peak_equity` **before** that day’s mark-to-market (MTM runs after entries/exits). This matches the existing Phase 1 pattern and is conservative for same-day stress (you do not react to same-day MTM until the daily record step). Document clearly if portfolio halt is intended to be **T−1 aware** only.

3. **`RiskState` fields `daily_atr_cache` / `active_stops`**  
   `register_stop` / `remove_stop` maintain `active_stops`, but sizing does not read from it; `daily_atr_cache` is unused. Low risk, but either wire them for diagnostics or remove to avoid “dead state” confusion.

### Suggestions (non-blocking)

- **`sector_map` in `size_position`:** Parameter is explicitly unused (`del sector_map`). Fine for Phase 2; track for Phase 3 defense-in-depth.
- **Exit priority (Phase 2 branch):** Order is `max_hold` → then **stop overwrites** → then signal if not stop. Net effect: **stop-loss wins over max-hold** when both fire the same day — aligned with the comment intent. No change required unless product wants max-hold to cap duration even when stop would also fire.

---

## 2. Dataclasses (`risk_state.py`)

- **`PositionDecision`:** `__post_init__` validates approved rows; good.
- **`StopLossEvent`:** `gap_through` uses `trigger_low < stop_price` — sensible for gap-through detection.
- **`RiskState`:** Mutable tracker; `initial` / `update` / `register_stop` / `remove_stop` are coherent with backtest integration.

---

## 3. Backtest integration (`backtest_engine.py`)

- **`use_risk_engine`:** Constructor + per-`run()` override behave as documented.
- **History window:** `_get_ticker_history` uses `Date <= as_of_date` with `tail(n_rows)`; paired with `volatility_lookback + 1` rows, this matches ATR’s need for prior close.
- **Stop convention:** Trigger on **today’s low** vs stop; fill at **next open** — matches design.
- **Phase 1 parity:** Tests assert `use_risk_engine=False` preserves legacy outputs (see `tests/test_backtest_engine.py`).

---

## 4. `run_phase2.py`

- Clear Phase 1 vs Phase 2 comparison runner; guards when `run()` lacks `use_risk_engine`.
- Requires local `data/` price file unless overridden — expected for validation workflows.

---

## 5. Test coverage gaps

| Gap | Severity | Note |
|-----|----------|------|
| No dedicated `tests/test_risk_engine.py` | **Important** | SLE-9 explicitly called for ~30 unit tests on pure helpers (`compute_atr_pct`, `compute_drawdown_scalar`, rejection paths). Much of this is **indirectly** covered by `test_phase2_integration.py` and `test_backtest_engine.py`, but **direct** unit tests reduce regression risk when refactoring ATR. |
| No dedicated `tests/test_risk_state.py` | **Suggestion** | `RiskState.update` edge cases (e.g. `peak_equity` refresh, brake band boundaries) are better tested in isolation. |
| Success criteria “≥ 40 new risk tests” (`PHASE2_SYSTEM_DESIGN`) | **Process** | Count tests in PR before declaring M2 complete; integration file is large but confirm **net new** count vs baseline. |

---

## 6. Documentation consistency

- **`docs/SLE-9_APPROVED_IMPLEMENTATION_PLAN.md`:** Matches architecture; **ATR/`ta` bullet** conflicts with code (see §1 Important #1).
- **`docs/PHASE2_SYSTEM_DESIGN.md`:** Updated to “IMPLEMENTED” — good; still lists success metrics (Sharpe, DD) that require **full 2024 run**, not CI alone.

---

## 7. Merge recommendation

**Recommendation: Ready to merge** the integrated Phase 2 branches **with conditions**:

1. **Resolve or document** the **`ta` vs manual ATR** deviation (code change or doc/plan update).
2. **Add or explicitly defer** `test_risk_engine.py` / `test_risk_state.py` per SLE-9 — if deferred, record in Linear why integration coverage is sufficient.
3. **Confirm PR ↔ branch mapping** for #16–#18 in GitHub (not verifiable here).
4. Publish or link **`SLE-26` remake doc** and **`PHASE2_RISK_ENGINE.md`** if they remain part of the compliance checklist, or **drop them from the issue scope** if superseded by `PHASE2_SYSTEM_DESIGN.md`.

Once (1)–(2) are addressed or accepted by the coordinator, merge risk is **low**: full suite green on merged tip, integration tests exercise the risk path end-to-end.

---

*Reviewer role: independent cross-check (Gemini). Findings are based on static review + full pytest on a local merge of the remote branches named above.*

---

## SLE-29 — Resolution (2026-03-20)

| SLE-28 item | Resolution |
|-------------|------------|
| **ATR `ta` vs pandas** | `docs/SLE-9_APPROVED_IMPLEMENTATION_PLAN.md` amended: pandas EWM Wilder-style TR/ATR is the approved implementation. |
| **Missing unit test files** | `tests/test_risk_engine.py`, `tests/test_risk_state.py`, and `tests/test_phase2_integration.py` are on `main` (merged via Phase 2 completion branch). |
| **Full suite count** | **`458 passed`** on `python -m pytest tests/ -v` (post-merge workspace); earlier **402**/**485** figures were pre-merge estimates — use CI/local count as source of truth. |
| **PRs #16–#18** | Merge to `main` performed from integrated remote branches where GitHub PR mapping could not be verified from CI; confirm closure in GitHub UI. |
| **Phase 2 status** | **Complete** — see `PROJECT_GUIDE.md` roadmap and `docs/PHASE2_RESULTS.md`. |
