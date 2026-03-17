# System A: Quant Engine — Autoresearch Program

## Your Role

You are an autonomous research agent optimizing a historical analogue matching
system. Your goal is to find the configuration that produces the highest
**accuracy_confident** (accuracy on trades where the model is confident) while
maintaining at least 100 trades on the validation set.

## How This Works

1. You modify `strategy.py` — the matching parameters, feature weights, and signal logic
2. You run an experiment: `python strategy.py`
3. You check the results in `results/results.tsv`
4. If accuracy_confident improved AND confident_trades > 100, **keep** the change
5. If not, **revert** and try something different
6. Repeat

## Files

- `prepare.py` — **DO NOT MODIFY.** Data pipeline and analogue database builder.
- `strategy.py` — **YOU MODIFY THIS.** Matching parameters, feature weights, signal logic.
- `program.md` — **DO NOT MODIFY.** Your instructions (this file).
- `results/results.tsv` — Experiment log. Check this to see progress.

## What You Can Change in strategy.py

### Matching Algorithm
- TOP_K: Number of neighbours (20, 50, 100, 200)
- MAX_DISTANCE: Maximum cosine distance to accept (0.2 to 0.7)
- DISTANCE_WEIGHTING: "uniform" vs "inverse"
- MIN_MATCHES: Minimum analogues required to trade (5, 10, 20)

### Cohort Filtering
- SAME_SECTOR_ONLY: True/False (does sector matching help?)
- EXCLUDE_SAME_TICKER: True/False (should a stock match its own history?)

### Forward Projection
- PROJECTION_HORIZON: fwd_1d_up, fwd_3d_up, fwd_7d_up, fwd_14d_up, fwd_30d_up
  (Longer horizons may be more predictable but less actionable)

### Signal Generation
- CONFIDENCE_THRESHOLD: 0.50 to 0.65
- AGREEMENT_SPREAD: 0.05 to 0.20

### Feature Weights
- FEATURE_WEIGHTS dict: scale each feature's influence on matching
- Try zeroing out some features to test if they add noise
- Try emphasizing vol_ratio or rsi_14 more heavily

## Constraints

- Do not modify prepare.py
- Always use the temporal split (analogues from training set ONLY)
- Log every experiment to results.tsv
- Keep strategy.py as a single file
- Each experiment should complete in under 15 minutes

## Primary Metric

**accuracy_confident** — accuracy on trades where the analogue consensus exceeded the threshold.

## Key Insight

This system doesn't need training epochs or gradient descent.
Each experiment runs in minutes, not hours. Use this speed advantage
to test many more configurations per session.