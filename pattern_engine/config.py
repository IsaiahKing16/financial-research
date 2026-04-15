"""
pattern_engine/config.py — EngineConfig dataclass + WALKFORWARD_FOLDS.

All default values match the locked settings in CLAUDE.md.
Any change to a default requires new walk-forward evidence and a provenance
comment citing the TSV file + date.

nn_jobs=1 is a hard constraint on Windows/Python 3.12 (joblib deadlock).
Do not override this in any sweep — the dataclass default enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EngineConfig:
    """Mutable configuration for PatternMatcher.

    Mutable (not frozen) so sweep scripts can do ``cfg.max_distance = x``
    without the .replace() ceremony.  Production usage should treat it as
    effectively frozen — only experiment scripts override fields.

    All public API guards use RuntimeError (not assert) per CLAUDE.md rule 4.
    """

    # ── Matching algorithm ────────────────────────────────────────────────────
    top_k: int = 50
    max_distance: float = 2.5              # Phase 6 locked; see CLAUDE.md
    distance_weighting: str = "uniform"   # Locked — beats inverse on 52T sweep
    distance_metric: str = "euclidean"
    nn_jobs: int = 1                      # MUST stay 1 — Windows/Py3.12 deadlock
    batch_size: int = 256
    use_hnsw: bool = True                 # HNSWMatcher (faster; parity confirmed)

    # ── Features ─────────────────────────────────────────────────────────────
    feature_set: str = "returns_candle"
    """Feature set name resolved via FeatureRegistry. Default is locked per CLAUDE.md."""
    feature_weights: dict = field(default_factory=dict)
    standardize_features: bool = True
    """If True (default), StandardScaler normalizes all features to unit variance
    before L2 distance computation.  Set to False only in controlled experiments.

    NOTE: StandardScaler is ALREADY applied in _prepare_features (default=True
    preserves current production behavior).  False is the experimental condition.
    ADR-007 documents this decision with provenance from the experiment results."""

    # ── Forward projection ────────────────────────────────────────────────────
    projection_horizon: str = "fwd_7d_up"

    # ── Signal generation ────────────────────────────────────────────────────
    confidence_threshold: float = 0.65
    agreement_spread: float = 0.05
    min_matches: int = 5

    # ── Cohort filtering ─────────────────────────────────────────────────────
    same_sector_only: bool = False
    exclude_same_ticker: bool = True

    # ── Sector soft-prior boost (H2 experiment only; default 1.0 = no-op) ───
    # When > 1.0, same-sector neighbours are up-weighted in the inverse-distance
    # kernel.  matcher.py _package_results() reads this via getattr() guard.
    same_sector_boost_factor: float = 1.0

    # ── Regime ───────────────────────────────────────────────────────────────
    regime_filter: bool = False           # Off until validated (CLAUDE.md)
    regime_fallback: bool = False
    calibration_method: str = "beta_abm"
    cal_max_samples: int = 100_000

    # ── Research pilots (all off in production) ───────────────────────────────
    use_sax_filter: bool = False
    use_wfa_rerank: bool = False
    use_ib_compression: bool = False
    journal_top_n: int = 0

    # ── Signal intelligence filters (M9) ─────────────────────────────────────
    use_sector_conviction: bool = False
    use_momentum_filter: bool = False
    use_sentiment_veto: bool = False
    sector_conviction_lift: float = 0.005
    momentum_min_outperformance: float = 0.015

    # ── Phase 7 enhancements (all False in production) ───────────────────────────
    use_bma: bool = False               # E1: Bayesian Model Averaging calibrator
    use_owa: bool = False               # E2: OWA feature weighting
    use_dtw_reranker: bool = False      # E3: DTW post-retrieval reranker
    use_conformal: bool = False         # E4: Adaptive Conformal Prediction
    use_anomaly_filter: bool = False    # E5: LOF anomaly filter
    use_stumpy: bool = False            # E6: STUMPY matrix profile signal

    # Phase 7 enhancement parameters
    owa_alpha: float = 1.0              # E2: concentration exponent
    dtw_rerank_k: int = 20             # E3: neighbours to return after reranking
    conformal_alpha: float = 0.10      # E4: nominal miscoverage rate
    conformal_gamma: float = 0.05      # E4: ACI online learning rate
    anomaly_contamination: float = 0.05  # E5: expected outlier fraction
    anomaly_penalty: float = 0.50      # E5: confidence multiplier for outliers
    stumpy_weight: float = 0.20        # E6: blend weight for STUMPY signal
    stumpy_subsequence_length: int = 50  # E6: matrix profile subsequence length


# ── Power of 10: Loop bound constants (Rule 2) ────────────────────────────────
# All loops in production code must have an explicit upper bound.
# Exhaustion raises RuntimeError, never silently continues.
# Reference: PRD §7A.1 Rule R2, CLAUDE.md Critical Rules.

MAX_CALIBRATION_ITER: int = 10_000
"""Maximum iterations for calibration fitting (beta_abm, Platt scaling)."""

MAX_ORDER_RETRIES: int = 5
"""Maximum retry attempts for broker API order submission."""

MAX_WALK_FORWARD_FOLDS: int = 20
"""Maximum number of walk-forward folds in any single experiment."""

MAX_HNSW_ELEMENTS: int = 500_000
"""Maximum elements in hnswlib index before index rotation required."""

MAX_BATCH_ITER: int = 100_000
"""Maximum batches in any single pipeline run (prevents runaway loops)."""

MAX_POLYGON_PAGES: int = 1_000
"""Maximum pages to fetch from Polygon.io API in a single request."""

MAX_DOWNLOAD_RETRIES: int = 3
"""Maximum download retry attempts per ticker in data pipeline."""


# ── Walk-forward fold definitions ─────────────────────────────────────────────
# 6 expanding-window folds matching scripts/run_walkforward.py FOLDS.
# Do not redefine these in individual scripts — always import from here.

WALKFORWARD_FOLDS: list[dict] = [
    {
        "label":      "2019",
        "train_end":  "2018-12-31",
        "val_start":  "2019-01-01",
        "val_end":    "2019-12-31",
    },
    {
        "label":      "2020-COVID",
        "train_end":  "2019-12-31",
        "val_start":  "2020-01-01",
        "val_end":    "2020-12-31",
    },
    {
        "label":      "2021",
        "train_end":  "2020-12-31",
        "val_start":  "2021-01-01",
        "val_end":    "2021-12-31",
    },
    {
        "label":      "2022-Bear",
        "train_end":  "2021-12-31",
        "val_start":  "2022-01-01",
        "val_end":    "2022-12-31",
    },
    {
        "label":      "2023",
        "train_end":  "2022-12-31",
        "val_start":  "2023-01-01",
        "val_end":    "2023-12-31",
    },
    {
        "label":      "2024-Val",
        "train_end":  "2023-12-31",
        "val_start":  "2024-01-01",
        "val_end":    "2024-12-31",
    },
]
