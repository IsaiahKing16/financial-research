# PROJECT_GUIDE.md — Multi-AI Collaboration Reference
# Last Updated: 2026-03-16 (Evening)
# Owner: Sleep (Isaia)
# Primary AI: Claude (Anthropic) | Supporting: Gemini, ChatGPT

---

## QUICK CONTEXT FOR ANY AI READING THIS

You are helping build an autonomous financial prediction system. The core engine
(System A) uses historical analogue matching. The project follows Karpathy's
"autoresearch" pattern: an AI agent iterates on strategy code, runs experiments,
checks if metrics improved, keeps or discards changes, and repeats.

**Do NOT modify prepare.py or this file unless explicitly asked.**

**When starting any AI session, provide:**
1. This file (PROJECT_GUIDE.md)
2. strategy.py (current version)
3. results_analogue.tsv (experiment history)
4. Terminal output or screenshots from the most recent run

**NUMBERS REQUIRE TSV PROVENANCE.** Any claimed metric must trace to a specific
row in results_analogue.tsv. If it cannot be traced, it is fabricated.
This applies to all AIs without exception. See Section 12.

---

## 1. PRODUCTION VISION

The final system operates as a fully automated overnight pipeline:

1. **4:00 PM ET** — US markets close
2. **4:05 PM ET** — Pipeline fetches end-of-day OHLCV data for all tickers via yfinance
3. **4:10 PM ET** — Compute 8-feature return fingerprint for each ticker
4. **4:15 PM ET** — Run analogue search: find K nearest historical twins across
   25-year database, filtered to same macro regime (bull/bear)
5. **4:30 PM ET** — Multi-model consensus: only pass tickers where 2+/3 configs agree
6. **4:35 PM ET** — Apply calibrated probabilities, generate BUY/SELL/HOLD signals
7. **4:40 PM ET** — Save overnight report (JSON + text) and Telegram alert
8. **9:15 AM ET** — Predictions delivered before market open

**Ticker universe:** Starting at 52, expanding 10x–100x (520–5,200 tickers).
The pipeline is being designed for arbitrary scale from the start.
Wider universe = rarer but higher-quality analogue matches.
The three-filter signal gate keeps trade volume manageable at any universe size.

**Key architectural principle:** The system scans 5,200 tickers to identify the
10–20 highest-conviction signals, not to trade all of them.

---

## 2. SYSTEM ARCHITECTURE

### Files

| File | Who Modifies | Purpose |
|------|-------------|---------|
| `prepare.py` | Human only | Downloads OHLCV, computes features, builds analogue DB |
| `strategy.py` | AI agent | Full engine: matching, calibration, walk-forward, live signals |
| `strategy_overnight.py` | AI agent | 6-hour continuous walk-forward runner |
| `test_strategy.py` | AI agent | Regression suite — **52 tests** — run before every experiment |
| `diagnose_distances.py` | AI agent | Feature space geometry diagnostic |
| `pipeline.py` | AI agent | Production orchestrator — TO BE BUILT |
| `PROJECT_GUIDE.md` | Human only | This file. Ground truth for all AI collaborators. |

### Hardware
- CPU: AMD Ryzen 9 5900X (12-core, 24 threads, 3.70 GHz)
- RAM: 32GB DDR4, OS: Windows 10, Python 3.12
- venv: `C:\Users\Isaia\.claude\financial-research\venv`
- At 12-core parallelism: 5,200 tickers completes in ~58 min — fits overnight window

---

## 3. DATASET

### Current Tickers (52 across 6 sectors + Index)
**Index (2):** SPY, QQQ
**Tech (19):** AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AVGO, ORCL, ADBE,
CRM, AMD, NFLX, INTC, CSCO, QCOM, TXN, MU, PYPL
**Finance (9):** JPM, BAC, WFC, GS, MS, V, MA, AXP, BRK-B
**Health (10):** LLY, UNH, JNJ, ABBV, MRK, PFE, TMO, ISRG, AMGN, GILD
**Consumer (6):** WMT, COST, PG, KO, PEP, HD
**Indust. (4):** DIS, CAT, BA, GE
**Energy (2):** XOM, CVX

### Data Split (Temporal — NO leakage)
- **Training:** 2010-01-01 → 2023-12-31 (~175,605 rows after latest prepare.py)
- **Validation:** 2024-01-01 → 2024-12-31 (~13,104 rows)
- **Test:** 2025-01-01 → 2026-01-28 (13,936 rows — held out, do not touch)

### Active Feature Set: Return-Only (8 features) — LOCKED
Full 16-feature configs produced BSS ≈ −0.135, 3.5× worse. Only these 8 used:
```
ret_1d, ret_3d, ret_7d, ret_14d, ret_30d, ret_45d, ret_60d, ret_90d
```
Feature weights locked at 1.0 (flat) — overnight ablation confirmed hand-tuned
weights have zero marginal value for return-only features.

### Candlestick Features (available, not yet in FEATURE_COLS)
6 continuous ratio features added to prepare.py and saved to parquet:
`candle_body_to_range`, `candle_upper_wick_ratio`, `candle_lower_wick_ratio`,
`candle_body_upper_ratio`, `candle_body_lower_ratio`, `candle_direction`
Available via `feature_cols_override=RETURN_COLS + CANDLE_COLS`. Not yet tested.
Source: Gemini deep research — continuous proportional scaling for KNN.

### Scaler
When `feature_cols_override` is active (return-only), a local StandardScaler is
refitted on `train_db` inside `_run_matching_loop`. The saved scaler
(`models/analogue_scaler.pkl`) was fitted on 16 features and cannot be used on the
8-column matrix — shape mismatch crash. This is handled automatically.

---

## 4. THE ANALOGUE ENGINE PIPELINE

### 5-Step Process
1. **Profiling:** Compute 8-feature return fingerprint for target stock/date
2. **Analogue Search:** Euclidean K-NN (ball_tree) finds nearest historical twins
3. **Regime Filter:** Restrict candidates to same macro regime (SPY ret_90d bull/bear)
4. **Calibration:** Platt or isotonic regression maps raw P(up) → calibrated P(up)
5. **Signal:** Three-filter rule (MIN_MATCHES + AGREEMENT_SPREAD + THRESHOLD)

### Signal Generation — Three Required Filters (all must pass)
```python
if n_matches < MIN_MATCHES:        → HOLD
if agreement < AGREEMENT_SPREAD:   → HOLD
if prob >= CONFIDENCE_THRESHOLD:   → BUY
if prob <= 1-CONFIDENCE_THRESHOLD: → SELL
else:                              → HOLD
```

### Macro-Regime Conditioning (`regime_filter=True`)
Source: Gemini deep research — macro-regime filtration (2026-03-16).

Each date is classified as bull (SPY ret_90d > 0) or bear (≤ 0).
Analogue candidates are filtered to match the query date's regime.
This prevents 2010–2021 bull-market analogues from being used in bear-market queries.

**Critical implementation detail (v4 fix):**
- Train analogue labels → use `train_db` SPY rows (no leakage, correct)
- Val query labels → use `val_db` SPY rows (accurate current-year regime)

v3 bug: both used `train_db` SPY. When training ended in a bear year (2022-12-31),
searchsorted projected bear labels onto all 2023 val queries — but 2023 was +24%
bull. All 13,000 queries mislabeled, near-zero signals. v4 fixes this.

### Calibration Methods
Two calibrators available:
- `fit_platt_scaling()` — LogisticRegression(C=1.0). Parametric sigmoid.
- `fit_isotonic_scaling()` — IsotonicRegression. Non-parametric monotone step function.
- `calibrate_probabilities(calibrator, probs)` — dispatch helper (works for both)

Both are fitted on **training set probabilities** (not val set) so the calibrator
sees all 14 years of market regimes and generalises across folds.

### Locked Hyperparameters
```python
TOP_K = 50                       # Neighbours retrieved per query
MAX_DISTANCE = 1.1019            # Euclidean ceiling (AvgK ~42)
DISTANCE_WEIGHTING = "uniform"   # Beats inverse — locked sweep 1
PROJECTION_HORIZON = "fwd_7d_up" # Best BSS across horizons
CONFIDENCE_THRESHOLD = 0.65      # Locked sweep 1
AGREEMENT_SPREAD = 0.10          # Inert at AvgK ~42
MIN_MATCHES = 10
SAME_SECTOR_ONLY = False         # Sector-only worst BSS — locked off
EXCLUDE_SAME_TICKER = True
NN_JOBS = 1                      # Prevents Windows/Python 3.12 joblib deadlock
BATCH_SIZE = 256                 # 10-50x speedup vs per-row queries
```

---

## 5. EVALUATION METRICS

### Primary: Brier Skill Score (BSS)
```
BSS = 1 - (Brier_model / Brier_climatology)
BSS > 0    = beats base rate  ← ACHIEVED on validation (2026-03-16)
BSS > 0.05 = materially useful ← next target
BSS < 0    = worse than base rate
```

### Secondary
- **CRPS**: Continuous Ranked Probability Score. Lower = better.
- **accuracy_confident**: % correct on BUY/SELL trades only. Orthogonal to BSS.
- **avg_matches (AvgK)**: Average analogues after all filters. Target ~42.

### Walk-Forward: 6 Expanding Folds
```
Fold 1: Train → 2018, Val 2019
Fold 2: Train → 2019, Val 2020  (COVID crash)
Fold 3: Train → 2020, Val 2021
Fold 4: Train → 2021, Val 2022  (Bear market — historically hardest fold)
Fold 5: Train → 2022, Val 2023
Fold 6: Train → 2023, Val 2024  (Standard val split)
```

---

## 6. COMPLETE EXPERIMENT HISTORY

### All-Time Best Metrics
| Metric | Value | Config | TSV provenance |
|--------|-------|--------|---------------|
| Best raw BSS | −0.029 | sweep_uniform_d0.3 | sweep 1 |
| Best Platt BSS | **+0.00100** | regime_wf_v4 fold 2024 Platt | overnight run |
| Best Isotonic BSS | **+0.00069** | regime_wf_v4 fold 2024 Isotonic | overnight run |
| Best binary Acc | 58.6% | euc_r_d1.2457 | sweep 3 |
| Positive BSS | **YES** | 2026-03-16 ★ | |

### Sweep 1 — Cosine Baseline (9 configs)
All negative BSS. Best: `sweep_uniform_d0.3` BSS=−0.029, AvgK=50.0.
Finding: 93.3% of training vectors within cosine 0.20 — filter saturated.

### Distance Diagnostic (diagnose_distances.py)
- Cosine: 93.3% within 0.20 — permanently ruled out
- Euclidean: 15.4% within 0.20 — correct behaviour
- Cross-sectional encoding: 92.4% ≈ 93.3% — ruled out

### Sweep 2 — Euclidean + Return-Only (10 configs)
Best: `euc_d1.0115_retonly` BSS=−0.050. Euclidean + return-only locked in.

### Sweep 3 — Fine Grid (10 configs)
Best: `euc_r_d1.2457` BSS=−0.038, Acc=58.6%. BSS monotone with distance — no peak.

### Overnight Phase 1 — Platt Discovery
Best: `platt_euc_r_d1.0115` post_bss=−0.00192 at cal_frac=0.50. Gap to 0: 0.002.
Platt lifts BSS +0.031–0.036 consistently.

### Platt Sweep 1 — cal_frac 0.50–0.80
Best at cal_frac=0.75: post_bss=−0.00048. Gap: 0.00048.

### Platt Sweep 2 — Fine Scan 0.73–0.79 ★ FIRST POSITIVE BSS
| Experiment | cal_frac | post_BSS |
|-----------|---------|---------|
| platt_euc_r_d1.1019_cf76 | 0.76 | **+0.00033** |
| platt_euc_r_d1.2457_cf76 | 0.76 | **+0.00003** |
Zero-crossing at cal_frac=0.76.

### Walk-Forward v1 (val-set Platt calibration)
1/6 positive. 2022 Bear: 0 trades — Platt calibrator overfit to bear regime.
Root cause: cal within each fold's val year is regime-homogeneous.

### Walk-Forward v2 (train-set Platt calibration)
1/6 positive. 2022 Bear: 11,962 trades but Acc=48.7%, BSS=−0.031.
Fixed zero-trades. Root cause: signal itself is wrong — bull analogues in bear year.

### Walk-Forward v3 (regime_filter=True) — BUG IN REGIME LABELING
1/6 positive. All folds showed 0 trades except 2024.
Bug: val queries labeled using last train_db SPY row.
Fold 2023: all 13,000 labeled bear (wrong — 2023 was +24% bull).
Fold 2024: correct by coincidence (2023 and 2024 both bull).

### Walk-Forward v4 (regime_filter=True + val_db SPY labeling) ★ CURRENT
v4 fixes the regime labeling bug. Results from overnight run:

**[PLATT]**
| Fold | BSS | AvgK |
|------|-----|------|
| 2019 | −0.01243 | 16.0 |
| 2020 (COVID) | −0.00270 | 27.0 |
| 2021 | −0.00131 | 42.6 |
| 2022 (Bear) | −0.02935 | 31.1 |
| 2023 | −0.00305 | 19.9 |
| **2024** | **+0.00100** | 42.0 |

**[ISOTONIC]**
| Fold | BSS |
|------|-----|
| 2024 | **+0.00069** |
| others | negative |

**v4 diagnosis — still 1/6 positive:**
The regime labeling bug is fixed but AvgK values reveal a new issue:
- 2019: AvgK=16.0 (very thin pool)
- 2023: AvgK=19.9, regime shows val=0 bull/13000 bear

The val_db SPY fix resolves the mislabeling direction, but AvgK being far below
normal (~42) in non-2024 folds suggests the regime filter is excluding too many
analogues in early folds where the training set is smaller. The 2022 bear filter
is working (AvgK=31.1 vs 50.0 without it) but signal quality remains poor.

**OVERNIGHT RUN IN PROGRESS:** Testing distances 1.0115, 1.1019, 1.2457, 1.5000
across 6 hours with regime_filter=True + both calibration methods.

---

## 7. LOCKED SETTINGS (do not re-test without strong evidence)

| Setting | Value | Evidence |
|---------|-------|---------|
| Distance metric | Euclidean | Sweep 2 — cosine saturated |
| DISTANCE_WEIGHTING | "uniform" | Sweep 1 — beats inverse |
| SAME_SECTOR_ONLY | False | Sweep 1 — sector-only worst BSS |
| CONFIDENCE_THRESHOLD | 0.65 | Sweep 1 — best accuracy |
| Feature set | Return-only 8 | Sweeps 2/3 — full 16 is 3.5× worse |
| Feature weights | Flat (1.0) | Overnight ablation |
| Calibrator source | Training set | v2/v3 confirmed val-set calibration fails |
| NN_JOBS | 1 | Prevents Windows/Python 3.12 deadlock |
| nn_algorithm | ball_tree | Avoids pairwise distance hang |
| Horizon | fwd_7d_up | Overnight — best BSS across horizons |
| Cross-sectional encoding | Do not use | Diagnostic — marginal only |
| Candlestick features | Not yet in FEATURE_COLS | Available but untested |

---

## 8. BUGS FIXED (52-test suite, all passing)

1. Evaluator/signal mismatch — confident trades counted by threshold only
2. Python default-arg binding — generate_signal called bare
3. Global mutation in sweep — swept wrong module object
4. BSS falsiness — 0.0 or -99 = -99
5. Calibration edge case — probability=1.0 outside all buckets
6. Euclidean threshold arithmetic — wrong values in early sweeps
7. Scaler mismatch (FATAL) — 16-feature saved scaler on 8-column matrix
8. exclude_same_ticker global — now explicit kwarg
9. TSV column misalignment — new fields inserted mid-dict
10. Walk-forward continue trap — BSS/CRPS skipped for zero-trade folds
11. joblib deadlock — n_jobs=-1 + brute + Euclidean on Windows
12. NaN poisoning — np.nan passes `is not None`
13. Winner printout incomplete — missing distance_metric and feature_set_name
14. Warning watchdog — sklearn flood caused overnight hang
15. Overnight identical cycles — fixed with schedule rotation
16. **Regime labeling bug v3→v4 (2026-03-16):** val queries labeled with train_db
    SPY rows — when training ended in bear year, all next-year val queries mislabeled.
    Fixed: val_db SPY rows used for val query regime labels. No leakage (ret_90d
    is a lagging indicator). Platt calibrator still trains on train_db only.

---

## 9. WHAT'S BUILT AND WHAT'S NEXT

### Built (strategy.py)
- `run_live_signals()` — production EOD scan, all 52 tickers, ranked BUY/SELL
- `run_regime_walkforward()` — v4 walk-forward with regime filter + Platt/isotonic
- `run_platt_sweep()` — calibration fraction sweep
- `run_platt_walkforward()` — train-set calibrated walk-forward
- `fit_isotonic_scaling()` + `calibrate_probabilities()` — isotonic calibration
- `fit_platt_scaling()` + `evaluate_with_calibration()` — Platt calibration

### Immediate research priority
The overnight run (in progress) tests 4 distance values across 6 hours.
Bring back terminal output + TSV to diagnose:
- Does loosening MAX_DISTANCE (1.5) improve AvgK and BSS in early folds?
- Does isotonic outperform Platt in non-2024 folds?
- Is the 2023 fold AvgK=19.9 explained by an insufficient bear analogue pool?

### After overnight results
1. Build `pipeline.py` — the production orchestrator (see Section 10)
2. Expand ticker universe to S&P 500 (503 tickers) — re-run prepare.py
3. System B personas (Institutional Quant, Retail Momentum, Macro Economist)

---

## 10. PRODUCTION PIPELINE (TO BUILD: pipeline.py)

The overnight pipeline must be designed for 10x–100x ticker scale from day one.
All functions use config files, not hardcoded values.

### Architecture
```python
TICKER_UNIVERSE_PATH = 'config/tickers.csv'  # swap to expand universe
BATCH_SIZE_FETCH     = 100                   # yfinance rate-limit safe
N_WORKERS            = 12                    # Ryzen 9 cores
MAX_REPORT_SIGNALS   = 20                    # top N regardless of universe size

run_pipeline():
  1. load_ticker_universe()           # reads tickers.csv, any size
  2. fetch_eod_data_batched()         # 100-ticker batches, rate-limit safe
  3. compute_fingerprints()           # vectorised across all tickers
  4. run_analogue_search_parallel()   # multiprocessing.Pool, N_WORKERS chunks
  5. run_consensus_filter()           # A1 + A2 + A3 vote (2/3 must agree)
  6. rank_and_filter()                # top MAX_REPORT_SIGNALS by calibrated prob
  7. save_report()                    # JSON + text + TSV ledger entry
  8. deliver_telegram()               # top 5-10 signals only
  9. log_outcomes()                   # next-day: actual vs predicted
```

### Scaling timeline at current hardware
| Universe | Tickers | ~Train rows | 12-core runtime |
|----------|---------|------------|-----------------|
| Current | 52 | 175K | ~1 min |
| S&P 500 | 503 | 1.7M | ~6 min |
| 50× | 2,600 | 8.8M | ~29 min |
| 100× | 5,200 | 17.6M | ~58 min |
All fit within the 17.5-hour overnight window.

### Functions remaining to build for pipeline.py
1. `run_consensus_signals()` — multi-config agreement filter (~2 days)
2. `save_overnight_report()` — JSON + dated text file (~1 day)
3. `deliver_telegram_alert()` — python-telegram-bot integration (~1 day)
4. `log_outcomes()` — next-day actual vs predicted ledger (~1 day)
5. `run_pipeline()` — scheduler + error recovery (~1 day)

---

## 11. RUNNING EXPERIMENTS

### Pre-flight (always run first)
```cmd
cd C:\Users\Isaia\.claude\financial-research
venv\Scripts\activate
python test_strategy.py
```
All **52 tests** must pass.

### Modes (strategy.py __main__ block)

| Mode | Function | Status |
|------|----------|--------|
| 0 | `run_live_signals()` | Production — run after market close |
| 1 | `run_experiment()` | Single baseline |
| 2 | `run_walkforward()` | Standard walk-forward (default config) |
| 3 | `run_analogue_sweep()` | Distance/feature parameter sweep |
| 4 | `run_platt_sweep()` | Platt cal_frac sweep |
| 5 | `run_platt_walkforward()` | Train-set Platt walk-forward |
| 6 | `run_regime_walkforward()` | Regime-conditioned v4 ← CURRENT |

### Overnight Runner
```cmd
python strategy_overnight.py
```
6-hour wall clock. Cycles through distance schedule [1.0115, 1.1019, 1.2457, 1.5000].
Each cycle = one full regime walk-forward v4 (~90 min).
Both Platt and isotonic calibration per cycle.
Results append to TSV after every fold. Safe to Ctrl+C.
Distance comparison summary printed at end.

---

## 12. HALLUCINATION PREVENTION

### Documented incidents
- **ChatGPT ×3:** BSS=+0.0483 (fabricated before sweep 2 ran); walk-forward
  positive BSS (no such run); cross-sectional encoding 16.8% (actual: 92.4%)
- **Gemini ×1:** Walk-forward Mean BSS=+0.0303, std=0.007 — session primed with
  description of intended next steps rather than raw TSV data

### Rules
1. Numbers require TSV provenance — no row = not real
2. Anchor sessions with raw TSV rows, not prose summaries
3. "Positive BSS on walk-forward" requires ≥3/6 folds positive in TSV
4. "System A complete" requires stable walk-forward
5. "Production-ready" requires BSS > 0.05 and validated pipeline.py
6. AvgK=50.0 in any fold = regime filter not working — investigate before reporting

---

## 13. ROADMAP

### Phase 1: System A — Calibrated Analogue Matching (CURRENT)
- [x] 52-ticker data pipeline, 8-feature return vectors
- [x] Euclidean K-NN, ball_tree, batched queries, 12-core ready
- [x] BSS + CRPS probabilistic evaluation
- [x] 52-test regression suite
- [x] Platt + isotonic calibration (train-set fitted)
- [x] Macro-regime conditioning with v4 labeling fix
- [x] `run_live_signals()` — production EOD mode built
- [x] Candlestick features in prepare.py (available, untested)
- [x] **Positive BSS achieved: +0.00100 (regime_wf_v4 fold 2024)**
- [ ] Overnight run results — does regime filter fix non-2024 folds?
- [ ] ≥3/6 folds positive on walk-forward
- [ ] BSS > 0.05 (materially useful)

### Phase 2: pipeline.py + System B
- [ ] pipeline.py orchestrator (scalable, 10x–100x tickers)
- [ ] Telegram delivery
- [ ] JSON handshake (System A → System B)
- [ ] System B LLM personas (Institutional, Retail, Macro)
- [ ] Cross-model consensus verification

### Phase 3: Paper Trading
- [ ] EOD pipeline live on 52-ticker test set (2025–2026 data)
- [ ] Hypothetical P&L, Sharpe ratio, max drawdown
- [ ] Expand to S&P 500 (503 tickers) during paper trading phase

### Phase 4: Live Deployment
- [ ] $2K starting capital
- [ ] Kill switches + drawdown limits
- [ ] Automated Telegram alerts

---

## 14. GLOSSARY

| Term | Definition |
|------|-----------|
| AvgK | Average analogues per query after all filters. Target: ~42. AvgK=16 = thin pool. |
| BSS | Brier Skill Score. Primary metric. Best walk-forward: +0.00100 (fold 2024, v4 Platt) |
| CANDLE_COLS | 6 candlestick ratio features in parquet. Not yet in FEATURE_COLS. |
| CRPS | Continuous Ranked Probability Score. Lower = better. |
| Euclidean | Current distance metric. ball_tree algorithm. Not comparable to cosine thresholds. |
| isotonic | IsotonicRegression calibration. Non-parametric. Alternative to Platt. |
| pipeline.py | The production orchestrator — to be built for scalable nightly operation. |
| Platt scaling | LogisticRegression calibration. Maps raw P(up) → calibrated P(up). |
| regime_filter | Restricts analogue candidates to same SPY bull/bear regime as query date. |
| Return-only | 8-feature config (ret_1d–ret_90d). 3.5× better BSS than full 16. |
| TSV provenance | A row in results_analogue.tsv confirming a number is real. |
| v4 fix | Val query regime labels now use val_db SPY rows, not train_db SPY rows. |
| Walk-forward | 6-fold expanding validation 2019–2024. ≥3/6 positive BSS = stable signal. |
