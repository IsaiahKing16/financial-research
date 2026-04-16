# HANDOFF: Track A — Dynamic Universe Sizing Sweep
**Created:** 2026-04-16  
**Campaign:** P8 Recovery (docs/campaigns/P8_RECOVERY_CAMPAIGN.md §2)  
**Task type:** SR (senior — new module + sweep execution)  
**Predecessor:** P8-PRE-1 FAIL closeout (commit d1db9ea)

---

## §0 CONTEXT (Read Before Starting)

P8-PRE-1 FAILED: 585T full-stack walk-forward produced 0/6 positive BSS folds (Sharpe=0.04, WR=47.5%). Root cause confirmed by Murphy B3 decomposition: **Resolution ≈ 0 at 585T from pool dilution**. KNN discriminative power collapses when analogues are drawn from 585 tickers rather than 52.

Track A tests the hypothesis: **KNN signal capacity exists somewhere between N=52 and N=585**, and a criterion-driven selection mechanism can identify the right tickers for a given N.

Full campaign context: `docs/campaigns/P8_RECOVERY_CAMPAIGN.md`

---

## §1 PRE-FLIGHT CHECKLIST

Before writing any code:

1. Read `docs/campaigns/P8_RECOVERY_CAMPAIGN.md` §2 (Track A spec)
2. Confirm 945 tests pass: `PYTHONUTF8=1 py -3.12 -m pytest tests/ -q -m "not slow"`
3. Confirm ruff baseline: `py -3.12 -m ruff check pattern_engine/ trading_system/` — must stay at 275 (no new violations)
4. Check git log — should see commit `d1db9ea` (p8-pre1 closeout) as most recent
5. Use `superpowers:using-git-worktrees` to set up an isolated worktree for this track

---

## §2 GOAL

Build a **dynamic ticker selector** (6 equal-weight criteria, sum=1.0) and run a **universe-size sweep** across N ∈ {52, 100, 150, 200, 300, 400, 585} using the full 585T master pool. For each N, run a full 6-fold walk-forward with Platt calibration, returns_candle(23) features, and H7 regime hold.

---

## §3 LOCKED SETTINGS FOR ALL TRACK A EXPERIMENTS

Do NOT change these without a new experiment + ADR:

```
Features          = returns_candle(23)
Calibration       = Platt (PatternMatcher native — NOT walkforward.run_fold beta_abm path)
max_distance      = 2.5
top_k             = 50
confidence_threshold = 0.65
regime            = hold_spy_threshold+0.05 (mode=hold, H7)
horizon           = fwd_7d_up
```

---

## §4 MANDATORY FIRST STEP — CRITERION DIAGNOSTICS

**Before running any universe sweep**, produce criterion diagnostics for a fixed reference date (2019-01-01 train boundary):

Output: `results/campaign_p8_recovery/track_a/criterion_diagnostics.tsv`  
Columns: `(ticker, liquidity_score, data_completeness_score, vol_regime_score, sector_repr_score, signal_contrib_score, survival_adj_score, pairwise_correlations_note)`

Check:
1. Each criterion produces a non-degenerate distribution (std > 0)
2. No two criteria are >0.95 correlated (if they are, drop one and document in ADR-014)
3. No criterion produces NaN/Inf for any ticker

If any criterion is >0.95 correlated with another: drop the lower-priority one (prefer signal_contribution > vol_regime > sector_repr > data_completeness > survival_adj > liquidity), document drop in ADR-014, adjust remaining weights to sum=1.0.

---

## §5 IMPLEMENTATION STEPS (Ordered)

### Step 1: TDD — Write tests FIRST

```
tests/unit/test_dynamic_selector_no_leakage.py  ← WRITE FIRST (critical)
tests/unit/test_universe_selector.py
```

**Leakage test must pass before any selector code is written.** Assert:
- `UniverseSelector.score(ticker, as_of_date)` raises `RuntimeError` if any data accessed has timestamp >= as_of_date
- `UniverseSelector.select_top_n(n, as_of_date)` raises `RuntimeError` if as_of_date is in the future

### Step 2: Pydantic schema

```
pattern_engine/contracts/selector.py  (NEW)
```

```python
class SelectionCriteria(BaseModel):
    liquidity_weight: FiniteFloat = 0.1667
    data_completeness_weight: FiniteFloat = 0.1667
    vol_regime_weight: FiniteFloat = 0.1667
    sector_repr_weight: FiniteFloat = 0.1667
    signal_contribution_weight: FiniteFloat = 0.1667
    survival_adj_weight: FiniteFloat = 0.1667

    @validator('*')  # or model_validator
    def weights_sum_to_one(cls, v, values):
        # Validate sum is 1.0 ± 1e-6
        ...
```

All weight fields use `FiniteFloat` (from `pattern_engine/contracts/finite_types.py`). Use `icontract.require` on the class or `__init__`.

### Step 3: UniverseSelector class

```
pattern_engine/universe_selector.py  (NEW)
```

Public interface (follow R4: ≤60 lines per function, ≤50 statements):

```python
class UniverseSelector:
    def __init__(self, master_tickers: list[str], criteria: SelectionCriteria): ...
    def score(self, ticker: str, as_of_date: date) -> float: ...
    def select_top_n(self, n: int, as_of_date: date) -> list[str]: ...
    def criterion_diagnostics(self, as_of_date: date) -> pd.DataFrame: ...
```

Each of the 6 criteria must be a private method `_score_<criterion>(ticker, as_of_date) -> float`. This makes each criterion unit-testable in isolation.

**Leakage guard:** Each `_score_*` method must check that no data it accesses has `timestamp >= as_of_date`. The RuntimeError message must name the offending criterion and timestamp.

icontract guards:
```python
@icontract.require(lambda n: 1 <= n <= 585)
@icontract.require(lambda as_of_date: as_of_date < date.today())
def select_top_n(self, n: int, as_of_date: date) -> list[str]: ...
```

### Step 4: Criterion diagnostics script (run BEFORE sweep)

```
scripts/run_track_a_criterion_diagnostics.py  (NEW)
```

- Fixed reference date: `as_of_date = date(2019, 1, 1)`
- Outputs `criterion_diagnostics.tsv`
- Prints pairwise correlations
- Exits with non-zero if any criterion produces NaN/Inf

### Step 5: Sweep driver

```
scripts/run_track_a_universe_sweep.py  (NEW)
```

```
Universe sizes to test: [52, 100, 150, 200, 300, 400, 585]

For each N:
  For each fold in 6-fold walk-forward:
    1. as_of_date = fold train start - 1 day
    2. selected = selector.select_top_n(N, as_of_date)
    3. Build feature matrix restricted to selected tickers
    4. Run walk-forward fold (Platt, returns_candle(23), max_distance=2.5)
    5. Compute BSS + Murphy B3 (Resolution, Reliability, Uncertainty)
    6. Compute trading metrics (Sharpe, MaxDD, n_trades, win_rate)
  Write fold results to results/campaign_p8_recovery/track_a/N{N}/walkforward.tsv
  Write Murphy B3 to results/campaign_p8_recovery/track_a/N{N}/murphy_b3.tsv
  Write gate_check.txt for this N
```

Output summary TSV: `results/campaign_p8_recovery/track_a/summary.tsv`  
Columns: `N, mean_bss, pos_folds, mean_sharpe, mean_maxdd, mean_resolution, mean_reliability`

### Step 6: Gate evaluation

`results/campaign_p8_recovery/track_a/gate_check.txt` — identifies winning N (if any):
```
Track A Gate Check — Universe Sizing Sweep
Date: <date>

N=52:  mean_bss=-0.00xxx, pos_folds=X/6 — FAIL
N=100: mean_bss=+0.00xxx, pos_folds=X/6 — PASS/FAIL
...

WINNING N: <N> (mean_bss=+X.XXXXX, pos_folds=X/6)
OVERALL: PASS/FAIL
```

### Step 7: ADR-014

```
docs/adr/ADR-014-dynamic-universe-selection.md  (NEW)
```

Document:
- Selector design and rationale for each criterion
- Criterion diagnostics results (were any dropped?)
- Winning N (or null result if all fail)
- Deferred weight sweep result (if Track A passed)

---

## §6 DEFERRED WEIGHT SWEEP (Only If Track A Passes)

If a passing N* is found (BSS > 0 on >= 3/6 folds):

Run Optuna TPESampler sweep over weights:
- 50 trials at N* only
- Objective: mean BSS across 6 folds
- Parametrize as 5 free weights (Dirichlet-like), 6th = 1 − sum
- Gate: winning vector must improve mean BSS by ≥ +0.002 vs equal weights

Script: `scripts/run_track_a_weight_sweep.py` (NEW, only if needed)  
Output: `results/campaign_p8_recovery/track_a/weight_sweep_summary.tsv`

If no vector improves by ≥ +0.002: equal weights stay default. Document null result in ADR-014.

---

## §7 ADVERSARIAL SELF-REVIEW (Required Before Claiming PASS)

Before any Track A PASS claim, produce in writing:

**3 ways the result could be wrong:**
1. Data leakage: selector uses post-fold data (test must prove otherwise)
2. Survivorship bias: master pool only contains current tickers, not historical universe
3. Overfitting to fold structure: criteria tuned (even implicitly) on specific folds

**3 ways the test setup could favor passing:**
1. N=52 is included — it already passes on its own criteria, so any score ≥ 52T should pass
2. Signal contribution criterion uses prior-fold BSS — this is forward-looking within the sweep (fold 2 uses fold 1 BSS)
3. Small N at top of sweep may capitalize on outlier fold results

**1 alternative explanation for any positive result:**
- If N=100-200 passes: this may simply be the 52T result reproduced with slightly different ticker composition, not a new finding

---

## §8 SUCCESS CRITERIA (Track A Gate)

| Metric | Threshold | Gate |
|--------|-----------|------|
| BSS positive folds | ≥ 3/6 at winning N | G1 |
| Sharpe | ≥ 1.0 at winning N | G2 |
| MaxDD | ≤ 15% at winning N | G3 |
| Trades | ≥ 50 at winning N | G4 |
| Win rate | ≥ 50% at winning N | G5 |
| Resolution (Murphy) | > 0.002 at winning N | G6 |
| No leakage | Leakage test passes | G7 |

OVERALL PASS = all 7 gates.

---

## §9 OUTPUT STRUCTURE

```
results/campaign_p8_recovery/track_a/
├── criterion_diagnostics.tsv          ← Step 4 output (run first)
├── summary.tsv                         ← All N results
├── gate_check.txt                      ← Overall Track A verdict
├── N52/
│   ├── config.json
│   ├── walkforward.tsv
│   ├── murphy_b3.tsv
│   └── gate_check.txt
├── N100/ ...
├── N150/ ...
├── N200/ ...
├── N300/ ...
├── N400/ ...
└── N585/
    ├── config.json
    ├── walkforward.tsv
    ├── murphy_b3.tsv
    └── gate_check.txt
```

---

## §10 VERIFICATION COMMANDS

```bash
# After implementing selector + tests:
PYTHONUTF8=1 py -3.12 -m pytest tests/unit/test_universe_selector.py tests/unit/test_dynamic_selector_no_leakage.py -v

# Run criterion diagnostics FIRST (before sweep):
PYTHONUTF8=1 py -3.12 scripts/run_track_a_criterion_diagnostics.py

# Run full sweep:
PYTHONUTF8=1 py -3.12 scripts/run_track_a_universe_sweep.py

# Check results:
cat results/campaign_p8_recovery/track_a/gate_check.txt
cat results/campaign_p8_recovery/track_a/summary.tsv
```

---

## §11 FILES TO CREATE / MODIFY

| File | Action | Priority |
|------|--------|----------|
| `tests/unit/test_dynamic_selector_no_leakage.py` | NEW | FIRST |
| `tests/unit/test_universe_selector.py` | NEW | Step 1 |
| `pattern_engine/contracts/selector.py` | NEW | Step 2 |
| `pattern_engine/universe_selector.py` | NEW | Step 3 |
| `scripts/run_track_a_criterion_diagnostics.py` | NEW | Step 4 |
| `scripts/run_track_a_universe_sweep.py` | NEW | Step 5 |
| `docs/adr/ADR-014-dynamic-universe-selection.md` | NEW | After sweep |
| `scripts/run_track_a_weight_sweep.py` | NEW (conditional) | Only if A passes |
| `docs/campaigns/P8_RECOVERY_CAMPAIGN.md` | MODIFY §7 track status | After gate |

**Do NOT modify:** `prepare.py`, `CLAUDE.md` (except via explicit instruction), `walkforward.py` (Platt path must be used directly via PatternMatcher, not walkforward.run_fold).

---

## §12 CODE STANDARDS REMINDER

All new code in `pattern_engine/` must follow NASA P10 rules (ADR-012):

- **R4:** Functions ≤60 lines / ≤50 statements
- **R5/R10:** `@icontract.require` / `@icontract.ensure` on every new public API
- **R7:** No `except ...: pass`. Re-raise or log+re-raise.
- **R9:** Zero new ruff/mypy/bandit findings (baseline: 275 ruff)
- **R1:** No recursion in production paths

Use `FiniteFloat` (not `float`) for all financial quantities in Pydantic models.

---

## §13 NEXT TRACK (After A Completes)

If Track A PASSES: re-run P8-PRE-1 gates with winning config, then decide on Track B.  
If Track A FAILS: proceed to Track B — see `docs/campaigns/P8_RECOVERY_CAMPAIGN.md` §3.

---

*Handoff prepared by Claude Code (Sonnet) — 2026-04-16*  
*Campaign advisor: Claude Opus — 2026-04-16*
