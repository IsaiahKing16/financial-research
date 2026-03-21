# Phase C Research Roadmap

Deferred from Phase B. Each domain below is a structured stub — enough detail to
resume work without re-reading the source papers. Implement only after Phase B
modules (EMD, BMA, SlipDeficit) are validated by a walk-forward experiment showing
BSS or Sharpe improvement.

---

## Promotion Gate (applies to all Phase C modules)

A Phase C module is ready for production promotion when:

1. Walk-forward BSS or Sharpe improvement confirmed on a held-out fold
2. All 574 existing tests (556 production + 6 EMD + 5 BMA + 7 slip-deficit) still
   pass with the module wired in
3. Locked settings in `CLAUDE.md` updated with experiment evidence citation
4. A spec document following the same review-loop process as Phase B lives in
   `docs/superpowers/specs/`

---

## Domain 1: FAISS / HNSW Approximate Nearest-Neighbour Index

**What it does:**
Replaces or augments the sklearn ball_tree in `pattern_engine/matching.py` with an
approximate nearest-neighbour index (Hierarchical Navigable Small World graph).
Trades a small accuracy loss for sub-linear query time, enabling larger training
universes (currently ~10 k fingerprints).

**Integration point:**
`pattern_engine/matching.py → Matcher._build_index()`.  HNSW would subclass
`BaseDistanceMetric` — `fit()` builds the index, `compute()` queries it.  Because
HNSW returns approximate neighbours, a post-filter exact-distance pass may be needed
to maintain precision parity.

**Key dependencies:**
`faiss-cpu` (Facebook Research) or `hnswlib` (nmslib).  Both are pip-installable.

**Success metric:**
Query latency < 10 ms on 50 k fingerprints with recall@50 > 0.95 vs exact ball_tree.

**Estimated complexity:** Medium (3–5 days including validation harness)

---

## Domain 2: Hawkes Process + Multiplex Financial Contagion

**What it does:**
Models self-exciting event cascades (e.g., signal clusters, drawdown contagion
across correlated tickers).  A Hawkes process estimates the conditional intensity
of signal events given the history of prior events; the multiplex layer adds
cross-asset contagion (a large drawdown in one sector elevates risk in correlated
sectors).

**Integration point:**
`trading_system/risk_engine.py` — as an additive `BaseRiskOverlay`.  The overlay
would raise `position_risk_score` when contagion intensity exceeds a threshold,
feeding into the existing risk-weight calculation.

**Key dependencies:**
`tick` (Hawkes process MLE) or manual log-likelihood maximisation with `scipy.optimize`.
No mandatory new library — scipy is sufficient for a basic self-exciting Poisson.

**Success metric:**
Hawkes process fits training signal history without degenerate parameters (mu > 0,
alpha < beta).  Contagion overlay reduces max-drawdown in backtests vs baseline.

**Estimated complexity:** High (1–2 weeks; Hawkes MLE is numerically sensitive)

---

## Domain 3: OODA Loop + CPOD / EILOF Anomaly Detection

**What it does:**
Frames the signal pipeline as an OODA (Observe–Orient–Decide–Act) loop with an
explicit anomaly detector at the Orient stage.  CPOD (Change-Point Online Detection)
flags structural breaks in the return distribution; EILOF (Extended Isolation
Local Outlier Factor) identifies fingerprints that are outliers within the current
regime — these should receive lower signal confidence or be held out of matching.

**Integration point:**
`pattern_engine/matching.py` — as a pre-filter applied to the candidate set before
KNN scoring.  Outlier fingerprints get a `regime_outlier=True` flag; the signal
adapter can discard or down-weight them.

**Key dependencies:**
`ruptures` (change-point detection) or manual CUSUM.  `pyod` for isolation-based
LOF; or manual Extended-IF implementation.

**Success metric:**
CPOD correctly flags known regime breaks in held-out test data (2008, 2020).
EILOF outlier rate < 5 % on normal market data, > 20 % during known stress periods.

**Estimated complexity:** High (2 weeks; two algorithms, two integration points)

---

## Domain 4: Case-Based Reasoning + OWA Dynamic Feature Weighting

**What it does:**
Replaces the static 8-feature `returns_only` fingerprint with a dynamically weighted
version.  OWA (Ordered Weighted Averaging) operators apply varying weights to features
based on their magnitude and sorted order under the current macro regime state
(using `pattern_engine/regime.py`, which is an existing module).  CBR (Case-Based
Reasoning) retrieves the most structurally similar historical episode, not just the
closest fingerprint in Euclidean space.

**Integration point:**
`pattern_engine/features.py → build_fingerprint()`.  A new `OWAFeatureWeighter`
subclasses `BaseDistanceMetric` (or is applied as a pre-transform before the metric).
The current locked setting `Features=returns_only(8)` would be changed only after
walk-forward evidence; the locked setting in `CLAUDE.md` must be updated with a
citations to the experiment log.

**Key dependencies:**
No new libraries — numpy is sufficient.  CBR similarity function is a weighted
cosine distance, implementable in pure numpy.

**Success metric:**
OWA-weighted fingerprints improve BSS by ≥ 0.02 on held-out fold vs static weights.
Locked settings in `CLAUDE.md` updated with experiment evidence.

**Estimated complexity:** Medium–High (1 week; touching locked production settings
requires extra caution and experiment evidence)
