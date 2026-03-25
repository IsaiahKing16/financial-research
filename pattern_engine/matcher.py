"""
matcher.py — 5-stage PatternMatcher with config-driven backend selection.

Refactors the production Matcher (pattern_engine/matching.py, 377 LOC)
into five explicit stages with typed inputs/outputs:

  Stage 1 — _prepare_features():  StandardScaler + feature weighting
  Stage 2 — _build_index():       Instantiate + fit BaseMatcher backend
  Stage 3 — _query_batch():       kneighbors() call for one batch
  Stage 4 — _post_filter():       Distance, ticker, sector, regime masks
  Stage 5 — _package_results():   prob_up, mean_ret, signal generation

Parity guarantee:
  When config.use_hnsw=False, PatternMatcher.query() returns bit-identical
  results to the production Matcher on the same input data. This is enforced
  by the parity test suite (tests/parity/test_matcher_parity_staged.py).

Backend selection:
  config.use_hnsw=False → BallTreeMatcher (exact, sklearn ball_tree)
  config.use_hnsw=True  → HNSWMatcher  (approximate, hnswlib)

HNSW promotion gate:
  See docs/rebuild/HNSW_PROMOTION_GATE.md and
  scripts/check_hnsw_promotion_gate.py for the criteria that must pass
  before HNSW can become the default backend.

Linear: SLE-60 (staged architecture), SLE-61 (HNSW backend wiring)
"""

from __future__ import annotations

import time
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pattern_engine.contracts.matcher import BaseMatcher
from pattern_engine.contracts.matchers.balltree_matcher import BallTreeMatcher
from pattern_engine.contracts.signals import NeighborResult
from pattern_engine.features import apply_feature_weights


# ─── SECTOR_MAP import (production source of truth) ───────────────────────────

try:
    from pattern_engine.sector import SECTOR_MAP as _SECTOR_MAP
except ImportError:
    _SECTOR_MAP = {}  # Empty fallback for isolated tests — sector filter disabled


# ─── Platt calibrator (SLE-89 — closes M8 calibration gap) ───────────────────

class _PlattCalibrator:
    """Thin sklearn LogisticRegression wrapper for Platt scaling.

    Matches the interface of ``pattern_engine.calibration.PlattCalibrator``
    without introducing a cross-package import dependency.

    The calibrator is fitted during PatternMatcher.fit() via the calibration
    double-pass (train queries itself) and applied in query() after the batch
    loop.  When calibration_method == 'none', no _PlattCalibrator is created
    and PatternMatcher returns raw K-NN frequencies unchanged.
    """

    def __init__(self) -> None:
        self._lr = None

    def fit(self, raw: np.ndarray, y: np.ndarray) -> "_PlattCalibrator":
        from sklearn.linear_model import LogisticRegression
        self._lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        self._lr.fit(raw.reshape(-1, 1), y)
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(raw.reshape(-1, 1))[:, 1]


# ─── PatternMatcher ───────────────────────────────────────────────────────────

class PatternMatcher:
    """5-stage nearest-neighbour analogue matcher.

    Replaces the monolithic production Matcher with a staged pipeline
    that separates feature preparation, index building, batch querying,
    post-filtering, and result packaging into distinct, testable methods.

    The public interface (fit / query / scaler / fitted) is identical to
    the production Matcher for backward compatibility.

    Usage:
        matcher = PatternMatcher(config)
        matcher.fit(train_db, feature_cols)
        probs, signals, reasons, n_matches, returns, ensembles = (
            matcher.query(val_db)
        )
    """

    def __init__(self, config) -> None:
        """
        Args:
            config: EngineConfig instance (production pattern_engine.config.EngineConfig).
                    PatternMatcher reads: use_hnsw, top_k, max_distance,
                    distance_weighting, feature_weights, batch_size,
                    confidence_threshold, agreement_spread, min_matches,
                    exclude_same_ticker, same_sector_only, regime_filter,
                    regime_fallback, projection_horizon.
        """
        self.config = config
        self._scaler: Optional[StandardScaler] = None
        self._backend: Optional[BaseMatcher] = None
        self._train_db: Optional[pd.DataFrame] = None
        self._feature_cols: Optional[list[str]] = None
        self._regime_labels_train = None
        self._fitted: bool = False

        # Pre-cached numpy arrays (set by _rebuild_caches after fit)
        self._train_tickers_arr: Optional[np.ndarray] = None
        self._train_sector_arr: Optional[np.ndarray] = None
        self._train_target_arr: Optional[np.ndarray] = None
        self._train_ret_arr: Optional[np.ndarray] = None
        self._train_dates_arr: Optional[np.ndarray] = None

        # Platt calibrator (SLE-89): None until fit() runs the double-pass.
        # query() returns raw probs if None; applies transform() otherwise.
        self._calibrator: Optional[_PlattCalibrator] = None

        # Research pilot modules (SLE-72–78, all None until flag activates in fit())
        self._sax_filter = None         # SAXFilter | None (use_sax_filter flag)
        self._wfa_reranker = None       # WFAReranker | None (use_wfa_rerank flag)
        self._ib_compressor = None      # IBCompressor | None (use_ib_compression flag)
        self._X_train_weighted: Optional[np.ndarray] = None   # stored for SAX
        self._active_overlays: list = []  # list[BaseRiskOverlay] — caller-managed

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 1 — Feature preparation
    # ──────────────────────────────────────────────────────────────────────────

    def _prepare_features(
        self,
        X_raw: np.ndarray,
        fit_scaler: bool = False,
    ) -> np.ndarray:
        """Scale and weight a raw feature matrix.

        Stage 1 of the 5-stage pipeline.

        Args:
            X_raw: (N, D) float64 array of raw feature values.
            fit_scaler: If True, fit the StandardScaler on X_raw and transform.
                        If False, apply the already-fitted scaler transform only.

        Returns:
            X_weighted: (N, D) float64 array — scaled and weighted.
        """
        cfg = self.config
        if fit_scaler:
            X_scaled = self._scaler.fit_transform(X_raw)
        else:
            # Guard: sklearn raises AttributeError if transform() is called before fit(),
            # but the message is cryptic.  Provide a clear RuntimeError instead.
            if not hasattr(self._scaler, "mean_"):
                raise RuntimeError(
                    "_prepare_features(fit_scaler=False) called before the scaler has "
                    "been fitted.  Call PatternMatcher.fit() first."
                )
            X_scaled = self._scaler.transform(X_raw)

        return apply_feature_weights(X_scaled, self._feature_cols, cfg.feature_weights)

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 2 — Index build
    # ──────────────────────────────────────────────────────────────────────────

    def _build_index(self, X_weighted: np.ndarray) -> None:
        """Instantiate and fit the backend matcher.

        Stage 2 of the 5-stage pipeline. Backend is chosen by config.use_hnsw.

        n_probe = min(top_k * 3, N): matches production exactly. We over-fetch
        3× candidates to give the post-filter room to apply distance / ticker /
        sector / regime constraints without falling below top_k survivors.

        Args:
            X_weighted: (N, D) scaled+weighted training feature matrix.
        """
        cfg = self.config
        N = X_weighted.shape[0]
        n_probe = min(cfg.top_k * 3, N)

        if cfg.use_hnsw:
            from pattern_engine.contracts.matchers.hnsw_matcher import HNSWMatcher
            # M=16 (default). M=32 tested 2026-03-24: +39% build time, zero BSS
            # improvement (D: -0.000099 unchanged). Recall gap vs BallTree is
            # statistical noise, not missing neighbours. Revert to M=16.
            _hnsw_M = getattr(cfg, 'hnsw_M', 16)
            self._backend = HNSWMatcher(n_neighbors=n_probe, M=_hnsw_M)
        else:
            self._backend = BallTreeMatcher(n_neighbors=n_probe)

        self._backend.fit(X_weighted)

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 3 — Batch query
    # ──────────────────────────────────────────────────────────────────────────

    def _query_batch(
        self,
        X_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Query the backend index for one batch of query points.

        Stage 3 of the 5-stage pipeline.

        Args:
            X_batch: (B, D) scaled+weighted query feature matrix for this batch.

        Returns:
            (distances, indices): both (B, n_probe) arrays.
                distances: Euclidean distances (sqrt applied, matching sklearn).
                indices:   Training row indices.
        """
        cfg = self.config
        n_probe = min(cfg.top_k * 3, len(self._train_db))
        return self._backend.kneighbors(X_batch, n_neighbors=n_probe)

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 4 — Post-filter
    # ──────────────────────────────────────────────────────────────────────────

    def _post_filter(
        self,
        distances_b: np.ndarray,   # (B, n_probe)
        indices_b: np.ndarray,     # (B, n_probe)
        val_tickers_b: np.ndarray, # (B,) object array of ticker strings
        val_sectors_b: np.ndarray, # (B,) object array of sector strings
        val_regime_b,              # (B,) regime labels or None
        X_batch: Optional[np.ndarray] = None,  # (B, D) — for SAX filter
    ) -> np.ndarray:
        """Build the combined boolean mask selecting which candidates survive.

        Stage 4 of the 5-stage pipeline. Filter order is identical to the
        production Matcher to guarantee bit-identical parity.

        Filters applied in order (must match production matching.py exactly):
          1. Distance filter:       candidate distance <= max_distance
          2. Ticker exclusion:      exclude same ticker as query (if configured)
          3. Sector filter:         only same sector (if configured)
          4. Regime filter:         only matching regime label (if configured)
          5. top_k head:            keep only first top_k survivors per row

        Args:
            distances_b:  (B, n_probe) Euclidean distances from Stage 3.
            indices_b:    (B, n_probe) training row indices from Stage 3.
            val_tickers_b: (B,) query ticker strings.
            val_sectors_b: (B,) query sector strings.
            val_regime_b:  (B,) query regime labels, or None if no regime filter.

        Returns:
            top_mask: (B, n_probe) boolean array — True = this candidate
                      survived all filters and is in the final top_k set.
        """
        cfg = self.config

        # Filter 1: distance threshold
        mask = distances_b <= cfg.max_distance

        # Filter 2: exclude same ticker
        if cfg.exclude_same_ticker:
            mask &= (
                self._train_tickers_arr[indices_b] != val_tickers_b[:, np.newaxis]
            )

        # Filter 3: same sector only
        if cfg.same_sector_only:
            mask &= (
                self._train_sector_arr[indices_b] == val_sectors_b[:, np.newaxis]
            )

        # Filter 4: regime matching
        if (self._regime_labels_train is not None
                and val_regime_b is not None
                and cfg.regime_filter):
            train_r = np.asarray(self._regime_labels_train)[indices_b]
            mask &= (train_r == np.asarray(val_regime_b)[:, np.newaxis])

        # Filter 4.5 (optional): SAX symbolic filter (SLE-72).
        # Applied after regime filter so only regime-compatible candidates
        # are checked for shape similarity — avoids wasting SAX distance
        # computation on candidates already excluded by domain filters.
        if X_batch is not None and self._sax_filter is not None:
            from pattern_engine.sax_filter import apply_sax_filter
            mask = apply_sax_filter(
                self._sax_filter,
                self._X_train_weighted,
                X_batch,
                indices_b,
                mask,
            )

        # Filter 5: head(top_k) — keep first top_k True entries per row
        top_mask = mask & (np.cumsum(mask, axis=1) <= cfg.top_k)

        return top_mask

    # ──────────────────────────────────────────────────────────────────────────
    # Stage 5 — Package results
    # ──────────────────────────────────────────────────────────────────────────

    def _package_results(
        self,
        top_mask: np.ndarray,       # (B, n_probe) bool
        distances_b: np.ndarray,    # (B, n_probe)
        indices_b: np.ndarray,      # (B, n_probe)
        val_tickers_b: np.ndarray,  # (B,) for NeighborResult
        val_dates_b,                # (B,) dates for NeighborResult
    ) -> tuple:
        """Compute signal outputs from the post-filtered candidate set.

        Stage 5 of the 5-stage pipeline. Vectorized computation identical
        to production Matcher._query_vectorized_batch() for bit-parity.

        Args:
            top_mask:     (B, n_probe) boolean — which candidates survived.
            distances_b:  (B, n_probe) Euclidean distances.
            indices_b:    (B, n_probe) training row indices.
            val_tickers_b: (B,) query tickers for NeighborResult.
            val_dates_b:   (B,) query dates for NeighborResult.

        Returns:
            (prob_up, mean_ret, signals, reasons, n_matches, ensemble_list, neighbor_results)
            prob_up:         (B,) float64 — calibrated probability of up-move
            mean_ret:        (B,) float64 — mean 7d return of analogues
            signals:         list[str] of B signal strings (BUY/SELL/HOLD)
            reasons:         list[str] of B human-readable signal reasons
            n_matches:       list[int] — count of accepted analogues per query
            ensemble_list:   list of B float64 arrays (ensemble returns)
            neighbor_results: list of B NeighborResult objects
        """
        cfg = self.config
        B = top_mask.shape[0]
        top_f = top_mask.astype(np.float64)     # (B, n_probe)
        n_matches = top_mask.sum(axis=1)         # (B,) int
        n_safe = np.maximum(n_matches, 1).astype(np.float64)

        targets = self._train_target_arr[indices_b]   # (B, n_probe)
        returns = self._train_ret_arr[indices_b]      # (B, n_probe)

        # Probability and return estimation
        if cfg.distance_weighting == "inverse":
            inv_w = np.where(top_mask, 1.0 / (distances_b + 0.01), 0.0)
            inv_w_sum = inv_w.sum(axis=1, keepdims=True)
            inv_w_norm = inv_w / np.maximum(inv_w_sum, 1e-9)
            prob_up = (targets * inv_w_norm).sum(axis=1)
            mean_ret = (returns * inv_w_norm).sum(axis=1)
        else:  # uniform (locked setting)
            prob_up = (targets * top_f).sum(axis=1) / n_safe
            mean_ret = (returns * top_f).sum(axis=1) / n_safe

        # Neutral probability when no matches found
        prob_up = np.where(n_matches == 0, 0.5, prob_up)

        # Research pilot: risk overlay multipliers (SLE-74, SLE-75).
        # Callers attach overlays via add_overlay() and are responsible for
        # calling overlay.update(date, **market_data) before each query().
        # Each overlay's get_signal_multiplier() returns a scalar in [0, 1]
        # that scales prob_up toward 0.5 — reducing effective signal confidence.
        if self._active_overlays:
            _combined = 1.0
            for _ov in self._active_overlays:
                _combined *= _ov.get_signal_multiplier()
            if _combined != 1.0:
                # Attenuate toward the base rate (0.5), not toward zero.
                # prob_up * combined would collapse a 70% signal to 0% when
                # combined=0, destroying calibration.  The correct form scales
                # the deviation from 0.5 by the combined multiplier.
                prob_up = 0.5 + (prob_up - 0.5) * _combined

        # Signal generation (vectorized masks, O(B) string loop)
        agreement = np.abs(prob_up - 0.5) * 2
        insuf = n_matches < cfg.min_matches
        low_agree = agreement < cfg.agreement_spread
        buy = ~insuf & ~low_agree & (prob_up >= cfg.confidence_threshold)
        sell = ~insuf & ~low_agree & (prob_up <= 1.0 - cfg.confidence_threshold)

        signals: list[str] = []
        reasons: list[str] = []
        for i in range(B):
            if insuf[i]:
                signals.append("HOLD"); reasons.append("insufficient_matches")
            elif low_agree[i]:
                signals.append("HOLD"); reasons.append("low_agreement")
            elif buy[i]:
                signals.append("BUY")
                reasons.append(f"prob={prob_up[i]:.3f}_agree={agreement[i]:.3f}")
            elif sell[i]:
                signals.append("SELL")
                reasons.append(f"prob={prob_up[i]:.3f}_agree={agreement[i]:.3f}")
            else:
                signals.append("HOLD")
                reasons.append(f"prob={prob_up[i]:.3f}_below_threshold")

        ensemble_list = [returns[i][top_mask[i]] for i in range(B)]

        # Build NeighborResult objects for each query (Stage 5 contract output)
        neighbor_results: list[NeighborResult] = []
        for i in range(B):
            accepted_mask_i = top_mask[i]       # (n_probe,) bool
            accepted_indices = indices_b[i][accepted_mask_i].tolist()
            accepted_distances = distances_b[i][accepted_mask_i].tolist()
            accepted_labels = [
                int(self._train_target_arr[idx]) for idx in accepted_indices
            ]
            # Normalise query_date: np.datetime64 / pd.Timestamp → datetime.date
            raw_date = val_dates_b[i]
            if hasattr(raw_date, "date"):
                # pd.Timestamp
                query_date_val: date = raw_date.date()
            elif hasattr(raw_date, "astype"):
                # np.datetime64 → convert via pd.Timestamp
                import pandas as _pd
                query_date_val = _pd.Timestamp(raw_date).date()
            elif raw_date is None or (isinstance(raw_date, float) and np.isnan(raw_date)):
                query_date_val = date.today()
            else:
                query_date_val = raw_date  # already datetime.date or string
            neighbor_results.append(NeighborResult(
                query_ticker=str(val_tickers_b[i]),
                query_date=query_date_val,
                neighbor_indices=accepted_indices,
                neighbor_distances=accepted_distances,
                neighbor_labels=accepted_labels,
                n_neighbors_requested=cfg.top_k,
                n_neighbors_found=int(n_matches[i]),
            ))

        return prob_up, mean_ret, signals, reasons, n_matches.tolist(), ensemble_list, neighbor_results

    # ──────────────────────────────────────────────────────────────────────────
    # Public API — fit / query
    # ──────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_db: pd.DataFrame,
        feature_cols: list[str],
        regime_labeler=None,
    ) -> "PatternMatcher":
        """Fit scaler and build the NN index on training data.

        Calls Stage 1 (prepare_features with fit_scaler=True) then
        Stage 2 (build_index). Builds numpy cache arrays for fast querying.

        Args:
            train_db: Training DataFrame with feature columns and targets.
            feature_cols: Ordered list of feature column names.
            regime_labeler: Optional fitted RegimeLabeler for pre-computing
                            training set regime labels (passed through for
                            parity with production Matcher signature).

        Returns:
            self (for method chaining).
        """
        self._train_db = train_db
        self._feature_cols = feature_cols
        self._scaler = StandardScaler()

        cfg = self.config

        # Stage 1: scale + weight
        X_raw = train_db[feature_cols].values
        X_weighted = self._prepare_features(X_raw, fit_scaler=True)

        # Research pilot: IB compression (SLE-78) — applied *before* building
        # the index so the index lives in compressed d_out-dimensional space.
        # When flag is False (default), X_weighted passes through unchanged.
        if getattr(cfg, 'use_ib_compression', False):
            from pattern_engine.ib_compression import IBCompressor
            _horizon = getattr(cfg, 'projection_horizon', 'fwd_7d_up')
            _y_train = (
                train_db[_horizon].values.astype(np.float64)
                if _horizon in train_db.columns
                else np.zeros(len(train_db), dtype=np.float64)
            )
            _d_out = getattr(cfg, 'ib_d_out', 4)
            self._ib_compressor = IBCompressor(d_out=_d_out)
            X_weighted = self._ib_compressor.fit_transform(X_weighted, _y_train)

        # Stage 2: build index
        self._build_index(X_weighted)

        # Store training features in the index space for SAX lookup.
        self._X_train_weighted = X_weighted

        # Research pilot: SAX symbolic filter (SLE-72) — fit on training
        # features in the same space as the index (post-IB if active).
        if getattr(cfg, 'use_sax_filter', False):
            from pattern_engine.sax_filter import SAXFilter
            self._sax_filter = SAXFilter()
            self._sax_filter.fit(X_weighted)

        # Research pilot: WFA DTW reranker (SLE-73) — fit probe lookup on
        # training features.
        if getattr(cfg, 'use_wfa_rerank', False):
            from pattern_engine.wfa_reranker import WFAReranker
            self._wfa_reranker = WFAReranker()
            self._wfa_reranker.fit(X_weighted)

        # Pre-compute regime labels for training data
        if regime_labeler is not None and getattr(regime_labeler, "fitted", False):
            self._regime_labels_train = regime_labeler.label(
                train_db, reference_db=train_db
            )

        self._fitted = True
        self._rebuild_caches()

        # Calibration double-pass (SLE-89 — closes I8 gap from M2-M5 review).
        # Matches production PatternEngine.fit() step 3:
        #   1. self._calibrator = None so self.query() returns raw probs
        #   2. Query train_db against itself (no look-ahead: same regime/distance)
        #   3. Fit PlattCalibrator on (raw_probs, y_true_binary)
        # Subsequent query() calls see self._calibrator and apply transform().
        _cal_method = getattr(cfg, 'calibration_method', 'platt')
        self._calibrator = None  # sentinel: self.query() returns raw probs below
        if _cal_method not in ('none', None):
            # cal_frac (default 0.76): sample a fraction of training rows for the
            # calibration double-pass.  Platt scaling (logistic regression) reaches
            # stable estimates with ~10k–50k samples; querying the full training set
            # costs proportionally more time with negligible calibration benefit.
            # cal_max_samples (default 100_000): absolute cap so the calibration
            # cost stays constant regardless of universe size.  cal_frac was locked
            # at 0.76 for a 52-ticker universe (~110k training rows → ~83k cal rows);
            # at 585 tickers the same ratio produces 1.9M cal rows — 23× more than
            # needed for stable Platt estimates.
            _cal_frac = getattr(cfg, 'cal_frac', 0.76)
            _cal_max = getattr(cfg, 'cal_max_samples', 100_000)
            _n_train = len(train_db)
            _n_cal = max(1000, min(int(_cal_frac * _n_train), _cal_max))
            if _n_cal < _n_train:
                _rng = np.random.RandomState(42)
                _cal_idx = _rng.choice(_n_train, size=_n_cal, replace=False)
                _cal_db = train_db.iloc[_cal_idx]
            else:
                _cal_db = train_db
            _cal_raw_probs, _, _, _, _, _ = self.query(
                _cal_db, regime_labeler=regime_labeler, verbose=0
            )
            _y_true = (
                _cal_db[cfg.projection_horizon].values.astype(np.float64)
                if cfg.projection_horizon in _cal_db.columns
                else np.zeros(len(_cal_db), dtype=np.float64)
            )
            self._calibrator = _PlattCalibrator().fit(_cal_raw_probs, _y_true)

        return self

    def _rebuild_caches(self) -> None:
        """Pre-cache numpy arrays from _train_db for vectorized batch queries.

        Called after fit() and can be called again if _train_db is manually
        replaced (e.g., after deserialization). Mirrors the production
        Matcher._rebuild_caches() exactly for parity.
        """
        train_db = self._train_db
        cfg = self.config
        horizon = cfg.projection_horizon          # "fwd_7d_up"
        ret_col = horizon.replace("_up", "")     # "fwd_7d"

        # np.asarray(..., dtype=object): forces regular numpy array even when
        # pandas uses Arrow-backed strings (ArrowExtensionArray rejects 2D fancy
        # indexing used in _post_filter and _package_results).
        self._train_tickers_arr = np.asarray(train_db["Ticker"], dtype=object)
        self._train_sector_arr = np.array(
            [_SECTOR_MAP.get(t, "") for t in self._train_tickers_arr]
        )
        self._train_target_arr = (
            train_db[horizon].values.astype(np.float64)
            if horizon in train_db.columns
            else np.zeros(len(train_db), dtype=np.float64)
        )
        self._train_ret_arr = (
            train_db[ret_col].values.astype(np.float64)
            if ret_col in train_db.columns
            else np.zeros(len(train_db), dtype=np.float64)
        )
        self._train_dates_arr = (
            train_db["Date"].values
            if "Date" in train_db.columns
            else np.array([None] * len(train_db))
        )

    def query(
        self,
        val_db: pd.DataFrame,
        regime_labeler=None,
        verbose: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, list, list[int], list, list]:
        """Run batched analogue matching on query rows.

        Executes the full 5-stage pipeline in batches of config.batch_size.
        Return signature is identical to the production Matcher.query() for
        backward compatibility and parity testing.

        Args:
            val_db: Query DataFrame (must have feature columns + Ticker).
            regime_labeler: Optional RegimeLabeler for regime filtering.
            verbose: 0=silent, 1=progress every ~2000 queries.

        Returns:
            (probabilities, signals, reasons, n_matches, mean_returns, ensemble_list)
            probabilities: (N,) float64 array — prob_up per query row
            signals:       (N,) object array — "BUY"/"SELL"/"HOLD"
            reasons:       list[str] — human-readable signal reasons
            n_matches:     list[int] — accepted analogue count per row
            mean_returns:  list[float] — mean forward return of analogues
            ensemble_list: list[np.ndarray] — full ensemble for each row

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before query().")

        cfg = self.config
        n_val = len(val_db)

        # Scale + weight the query features (Stage 1, fit_scaler=False)
        X_val_raw = val_db[self._feature_cols].values
        X_val_weighted = self._prepare_features(X_val_raw, fit_scaler=False)

        # Research pilot: IB compression of query features (SLE-78).
        # The index was built on compressed features in fit(); query features
        # must be projected into the same compressed space.
        if self._ib_compressor is not None:
            X_val_weighted = self._ib_compressor.transform(X_val_weighted)

        # Pre-compute regime labels for all query rows
        regime_labels_val = None
        if regime_labeler is not None and getattr(regime_labeler, "fitted", False) and cfg.regime_filter:
            regime_labels_val = regime_labeler.label(val_db, reference_db=self._train_db)
            if verbose:
                bull_q = (regime_labels_val == 1).sum() if getattr(regime_labeler, "mode", None) == "binary" else "N/A"
                bear_q = (regime_labels_val == 0).sum() if getattr(regime_labeler, "mode", None) == "binary" else "N/A"
                print(f"    Regime filter ({regime_labeler.mode}): "
                      f"val={bull_q} bull / {bear_q} bear queries")

        # Pre-extract val arrays once (avoids iloc per row in batch loop)
        val_tickers_arr = np.asarray(val_db["Ticker"], dtype=object)
        val_sectors_arr = np.array([_SECTOR_MAP.get(t, "") for t in val_tickers_arr])
        val_dates_arr = val_db.get("Date", pd.Series([None] * n_val)).values
        regime_labels_arr = (
            np.asarray(regime_labels_val) if regime_labels_val is not None else None
        )

        # Accumulators
        all_probs: list[float] = []
        all_signals: list[str] = []
        all_reasons: list[str] = []
        all_n_matches: list[int] = []
        all_mean_returns: list[float] = []
        all_ensembles: list = []

        start_time = time.time()

        for batch_start in range(0, n_val, cfg.batch_size):
            batch_end = min(batch_start + cfg.batch_size, n_val)
            q_batch = X_val_weighted[batch_start:batch_end]
            B = batch_end - batch_start

            # Stage 3: batch query
            distances_b, indices_b = self._query_batch(q_batch)

            # Stage 4: post-filter
            val_regime_b = (
                regime_labels_arr[batch_start:batch_end]
                if regime_labels_arr is not None else None
            )
            top_mask = self._post_filter(
                distances_b, indices_b,
                val_tickers_arr[batch_start:batch_end],
                val_sectors_arr[batch_start:batch_end],
                val_regime_b,
                X_batch=q_batch if self._sax_filter is not None else None,
            )

            # Research pilot: WFA DTW reranker (SLE-73).
            # Reorders top_k survivors by constrained DTW distance so Stage 5
            # sees the most temporally-aligned analogues first.  Does not
            # change which candidates are included — only their ordering.
            if self._wfa_reranker is not None:
                top_mask = self._wfa_reranker.rerank_mask(
                    q_batch, indices_b, top_mask, cfg.top_k
                )

            # Stage 5: package results
            (prob_b, ret_b, sigs_b, rsns_b, nm_b, ens_b, _nr_b) = self._package_results(
                top_mask, distances_b, indices_b,
                val_tickers_arr[batch_start:batch_end],
                val_dates_arr[batch_start:batch_end],
            )

            all_probs.extend(prob_b.tolist())
            all_signals.extend(sigs_b)
            all_reasons.extend(rsns_b)
            all_n_matches.extend(nm_b)
            all_mean_returns.extend(ret_b.tolist())
            all_ensembles.extend(ens_b)

            # Progress reporting (matches production Matcher output format)
            if verbose and batch_end % 2000 < cfg.batch_size:
                elapsed = time.time() - start_time
                rate = batch_end / elapsed if elapsed > 0 else 0
                remaining = (n_val - batch_end) / rate if rate > 0 else 0
                print(f"    {batch_end:,}/{n_val:,} "
                      f"({batch_end / n_val * 100:.0f}%) | "
                      f"{rate:.0f} queries/sec | "
                      f"ETA: {remaining / 60:.1f} min")

        raw_probs = np.array(all_probs)

        # Apply Platt calibration (SLE-89).  The calibrator is None during the
        # double-pass self-query inside fit(); it is set after that pass.
        # When calibration is applied, signals must be regenerated from the
        # calibrated probabilities (same threshold logic as _package_results).
        if self._calibrator is not None:
            cal_probs = self._calibrator.transform(raw_probs)
            # Regenerate signals from calibrated probabilities
            n_m = np.array(all_n_matches)
            cal_agree = np.abs(cal_probs - 0.5) * 2
            _insuf = n_m < cfg.min_matches
            _low_agree = cal_agree < cfg.agreement_spread
            _buy = ~_insuf & ~_low_agree & (cal_probs >= cfg.confidence_threshold)
            _sell = ~_insuf & ~_low_agree & (cal_probs <= 1.0 - cfg.confidence_threshold)
            all_signals = []
            all_reasons = []
            for _i in range(n_val):
                if _insuf[_i]:
                    all_signals.append("HOLD"); all_reasons.append("insufficient_matches")
                elif _low_agree[_i]:
                    all_signals.append("HOLD"); all_reasons.append("low_agreement")
                elif _buy[_i]:
                    all_signals.append("BUY")
                    all_reasons.append(f"prob={cal_probs[_i]:.3f}_agree={cal_agree[_i]:.3f}")
                elif _sell[_i]:
                    all_signals.append("SELL")
                    all_reasons.append(f"prob={cal_probs[_i]:.3f}_agree={cal_agree[_i]:.3f}")
                else:
                    all_signals.append("HOLD")
                    all_reasons.append(f"prob={cal_probs[_i]:.3f}_below_threshold")
            out_probs = cal_probs
        else:
            out_probs = raw_probs

        return (
            out_probs,
            np.array(all_signals),
            all_reasons,
            all_n_matches,
            all_mean_returns,
            all_ensembles,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Properties (match production Matcher interface)
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def scaler(self) -> Optional[StandardScaler]:
        """The fitted StandardScaler (None if not yet fitted)."""
        return self._scaler

    @property
    def fitted(self) -> bool:
        """True if fit() has been called successfully."""
        return self._fitted

    @property
    def backend(self) -> Optional[BaseMatcher]:
        """The active matcher backend (BallTreeMatcher or HNSWMatcher)."""
        return self._backend

    @property
    def backend_name(self) -> str:
        """Human-readable backend name."""
        if self._backend is None:
            return "unfitted"
        return self._backend.get_params().get("backend", "unknown")

    # ──────────────────────────────────────────────────────────────────────────
    # Research pilot helpers (SLE-74, SLE-75)
    # ──────────────────────────────────────────────────────────────────────────

    def add_overlay(self, overlay) -> None:
        """Attach a risk overlay that scales prob_up at Stage 5 (SLE-74/75).

        The overlay's get_signal_multiplier() is called on every _package_results()
        pass.  Callers MUST call overlay.update(date, **market_data) before each
        query() to keep the overlay's internal state current.

        Args:
            overlay: BaseRiskOverlay instance (LiquidityCongestionGate,
                     FatigueAccumulationOverlay, or any custom subclass).
        """
        self._active_overlays.append(overlay)

    def clear_overlays(self) -> None:
        """Remove all active risk overlays."""
        self._active_overlays.clear()
