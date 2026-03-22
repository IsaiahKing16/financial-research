"""
matching.py — Core analogue matching engine.

Wraps sklearn NearestNeighbors with batched queries, feature weighting,
distance filtering, ticker exclusion, sector filtering, and regime
conditioning. This is the computational heart of the pattern engine.

Performance notes:
  - BATCH_SIZE=256: queries kneighbors in chunks, reducing Python overhead
  - nn_jobs=1: prevents Windows/Python 3.12 joblib deadlock
  - ball_tree algorithm: used for Euclidean distance (avoids threading path)
  - Progress reporting every ~2000 queries for long runs
"""

import time
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from pattern_engine.projection import project_forward, generate_signal
from pattern_engine.regime import RegimeLabeler, apply_regime_filter, fallback_regime_mode
from pattern_engine.sector import SECTOR_MAP


def apply_feature_weights(X: np.ndarray, feature_cols: list[str],
                          weights: dict) -> np.ndarray:
    """Apply feature weights before distance calculation.

    Higher weight = that feature influences matching more.
    Weights are applied BEFORE NN index build AND query transform.
    """
    X_weighted = X.copy()
    for i, col in enumerate(feature_cols):
        w = weights.get(col, 1.0)
        X_weighted[:, i] *= w
    return X_weighted


class Matcher:
    """Nearest-neighbor analogue matcher with batched queries and filtering.

    Encapsulates the core matching loop from strategy.py _run_matching_loop().
    Handles scaler fitting, NN index building, feature weighting, and all
    post-search filters (distance, ticker, sector, regime).
    """

    def __init__(self, config):
        """
        Args:
            config: EngineConfig instance with matching parameters
        """
        self.config = config
        self._scaler = None
        self._nn_index = None
        self._train_db = None
        self._feature_cols = None
        self._regime_labels_train = None
        self._fitted = False

    def fit(self, train_db: pd.DataFrame, feature_cols: list[str],
            regime_labeler: RegimeLabeler = None) -> "Matcher":
        """Fit scaler and NN index on training data.

        Args:
            train_db: training DataFrame with feature columns
            feature_cols: list of feature column names to use
            regime_labeler: optional fitted RegimeLabeler for pre-computing train labels
        """
        self._train_db = train_db
        self._feature_cols = feature_cols
        cfg = self.config

        # Fit scaler on training features
        self._scaler = StandardScaler()
        X_train = self._scaler.fit_transform(train_db[feature_cols].values)

        # Apply feature weights
        X_train_weighted = apply_feature_weights(X_train, feature_cols, cfg.feature_weights)

        # Build NN index — HNSW or sklearn ball_tree
        n_probe = min(cfg.top_k * 3, len(train_db))
        if cfg.use_hnsw:
            from research.hnsw_distance import HNSWIndex
            self._nn_index = HNSWIndex(
                n_neighbors=n_probe,
                dim=X_train_weighted.shape[1],
            )
        else:
            self._nn_index = NearestNeighbors(
                n_neighbors=n_probe,
                metric=cfg.distance_metric,
                algorithm=cfg.nn_algorithm,
                n_jobs=cfg.nn_jobs,
            )
        self._nn_index.fit(X_train_weighted)

        # Pre-compute regime labels for training data
        if regime_labeler is not None and regime_labeler.fitted:
            self._regime_labels_train = regime_labeler.label(
                train_db, reference_db=train_db
            )

        self._fitted = True

        # Pre-cache numpy arrays for vectorized query (avoids iloc per row)
        horizon = cfg.projection_horizon                      # e.g. "fwd_7d_up"
        ret_col = horizon.replace("_up", "")                 # e.g. "fwd_7d"
        self._rebuild_caches()
        return self

    def _rebuild_caches(self) -> None:
        """Rebuild numpy cache arrays from _train_db.

        Called by fit() and also by engine.load() after manual Matcher
        reconstruction — ensures the vectorized query path always has its
        pre-cached arrays regardless of how the Matcher was assembled.
        """
        train_db = self._train_db
        cfg = self.config
        horizon = cfg.projection_horizon          # e.g. "fwd_7d_up"
        ret_col = horizon.replace("_up", "")     # e.g. "fwd_7d"

        # np.asarray(..., dtype=object) forces a regular numpy array even when
        # pandas uses Arrow-backed strings (.values → ArrowExtensionArray which
        # rejects 2D fancy indexing used in _query_vectorized_batch).
        self._train_tickers_arr = np.asarray(train_db["Ticker"], dtype=object)
        self._train_sector_arr = np.array(
            [SECTOR_MAP.get(t, "") for t in self._train_tickers_arr]
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

    def _query_vectorized_batch(
        self,
        distances_b: np.ndarray,   # (B, n_probe)
        indices_b: np.ndarray,     # (B, n_probe)
        val_tickers_b: np.ndarray, # (B,) string
        val_sectors_b: np.ndarray, # (B,) string
        val_regime_b,              # (B,) labels or None
    ) -> tuple:
        """Vectorized projection for one batch — no DataFrame ops inside.

        Replaces the per-row iloc+filter+project_forward inner loop with
        pure numpy operations on pre-cached arrays from fit().

        Returns:
            (prob_up, mean_ret, n_matches, signals, reasons, ensembles)
            prob_up, mean_ret: (B,) float64 ndarray
            n_matches: (B,) int ndarray
            signals, reasons: lists of B strings
            ensembles: list of B variable-length float64 arrays
        """
        cfg = self.config
        B = distances_b.shape[0]

        # --- Combined filter mask (B, n_probe) ---
        mask = distances_b <= cfg.max_distance

        if cfg.exclude_same_ticker:
            mask &= self._train_tickers_arr[indices_b] != val_tickers_b[:, np.newaxis]

        if cfg.same_sector_only:
            mask &= self._train_sector_arr[indices_b] == val_sectors_b[:, np.newaxis]

        if (self._regime_labels_train is not None
                and val_regime_b is not None and cfg.regime_filter):
            train_r = np.asarray(self._regime_labels_train)[indices_b]
            mask &= train_r == np.asarray(val_regime_b)[:, np.newaxis]

        # --- head(top_k): first top_k valid candidates per row ---
        top_mask = mask & (np.cumsum(mask, axis=1) <= cfg.top_k)  # (B, n_probe)

        # --- Projection ---
        n_matches = top_mask.sum(axis=1)                            # (B,)
        n_safe = np.maximum(n_matches, 1).astype(np.float64)

        targets = self._train_target_arr[indices_b]                 # (B, n_probe)
        returns = self._train_ret_arr[indices_b]                    # (B, n_probe)
        top_f = top_mask.astype(np.float64)                        # (B, n_probe)

        if cfg.distance_weighting == "inverse":
            inv_w = np.where(top_mask, 1.0 / (distances_b + 0.01), 0.0)
            inv_w_sum = inv_w.sum(axis=1, keepdims=True)
            inv_w_norm = inv_w / np.maximum(inv_w_sum, 1e-9)
            prob_up = (targets * inv_w_norm).sum(axis=1)
            mean_ret = (returns * inv_w_norm).sum(axis=1)
        else:
            prob_up = (targets * top_f).sum(axis=1) / n_safe
            mean_ret = (returns * top_f).sum(axis=1) / n_safe

        prob_up = np.where(n_matches == 0, 0.5, prob_up)           # (B,)

        # --- Signal generation (vectorized masks, O(B) string loop) ---
        agreement = np.abs(prob_up - 0.5) * 2
        insuf = n_matches < cfg.min_matches
        low_agree = agreement < cfg.agreement_spread
        buy = ~insuf & ~low_agree & (prob_up >= cfg.confidence_threshold)
        sell = ~insuf & ~low_agree & (prob_up <= 1.0 - cfg.confidence_threshold)

        signals, reasons = [], []
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

        ensembles = [returns[i][top_mask[i]] for i in range(B)]

        return prob_up, mean_ret, n_matches, signals, reasons, ensembles

    def query(self, val_db: pd.DataFrame, regime_labeler: RegimeLabeler = None,
              verbose: int = 1) -> tuple:
        """Run batched analogue matching on query rows.

        Args:
            val_db: query DataFrame
            regime_labeler: optional RegimeLabeler for regime filtering
            verbose: 0=silent, 1=progress

        Returns:
            (probabilities, signals, reasons, n_matches, mean_returns, ensemble_list)
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before query().")
        cfg = self.config
        train_db = self._train_db
        fcols = self._feature_cols

        # Scale and weight query features
        X_val_scaled = self._scaler.transform(val_db[fcols].values)
        X_val_weighted = apply_feature_weights(X_val_scaled, fcols, cfg.feature_weights)

        # Pre-compute regime labels for query rows
        regime_labels_val = None
        if regime_labeler is not None and regime_labeler.fitted and cfg.regime_filter:
            regime_labels_val = regime_labeler.label(val_db, reference_db=train_db)
            if verbose:
                bull_q = (regime_labels_val == 1).sum() if regime_labeler.mode == "binary" else "N/A"
                bear_q = (regime_labels_val == 0).sum() if regime_labeler.mode == "binary" else "N/A"
                print(f"    Regime filter ({regime_labeler.mode}): "
                      f"val={bull_q} bull / {bear_q} bear queries")

        # Results accumulators
        probabilities = []
        all_signals = []
        all_reasons = []
        all_n_matches = []
        all_mean_returns = []
        all_ensembles = []

        n_val = len(val_db)
        start_time = time.time()

        # Pre-extract val arrays (avoids iloc per row inside batch loop)
        val_tickers_arr = np.asarray(val_db["Ticker"], dtype=object)
        val_sectors_arr = np.array([SECTOR_MAP.get(t, "") for t in val_tickers_arr])
        regime_labels_arr = (
            np.asarray(regime_labels_val) if regime_labels_val is not None else None
        )

        # Regime fallback state — computed lazily once per query() call if needed
        _coarse_labeler = None
        _coarse_train = None
        _coarse_val = None

        for batch_start in range(0, n_val, cfg.batch_size):
            batch_end = min(batch_start + cfg.batch_size, n_val)
            q_batch = X_val_weighted[batch_start:batch_end]

            distances_batch, indices_batch = self._nn_index.kneighbors(q_batch)

            # --- Vectorized batch projection (no DataFrame ops inside) ---
            val_regime_b = (
                regime_labels_arr[batch_start:batch_end]
                if regime_labels_arr is not None else None
            )
            (prob_b, ret_b, n_matches_b,
             sigs_b, rsns_b, ens_b) = self._query_vectorized_batch(
                distances_batch,
                indices_batch,
                val_tickers_arr[batch_start:batch_end],
                val_sectors_arr[batch_start:batch_end],
                val_regime_b,
            )

            # --- Regime fallback: per-row only for the rare under-matched rows ---
            if (cfg.regime_fallback and regime_labeler is not None
                    and regime_labels_arr is not None):
                fallback_rows = np.where(n_matches_b < cfg.min_matches)[0]
                if len(fallback_rows):
                    coarser = fallback_regime_mode(regime_labeler.mode)
                    if coarser is not None:
                        # Lazy-init coarse labeler (once per query() call)
                        if _coarse_labeler is None:
                            _coarse_labeler = RegimeLabeler(mode=coarser)
                            _coarse_labeler.fit(train_db)
                            _coarse_train = _coarse_labeler.label(
                                train_db, reference_db=train_db
                            )
                            _coarse_val = _coarse_labeler.label(
                                val_db, reference_db=train_db
                            )
                        for local_i in fallback_rows:
                            gidx = batch_start + local_i
                            indices = indices_batch[local_i]
                            distances = distances_batch[local_i]
                            ticker = val_tickers_arr[gidx]

                            matches = train_db.iloc[indices].copy()
                            matches["distance"] = distances
                            matches = matches[matches["distance"] <= cfg.max_distance]
                            if cfg.exclude_same_ticker:
                                matches = matches[matches["Ticker"] != ticker]
                            matches = apply_regime_filter(
                                matches, _coarse_train, _coarse_val[gidx], indices
                            )
                            matches = matches.head(cfg.top_k)

                            proj = project_forward(
                                matches, cfg.projection_horizon, cfg.distance_weighting
                            )
                            sig, rsn = generate_signal(
                                proj, cfg.confidence_threshold,
                                cfg.agreement_spread, cfg.min_matches,
                            )
                            prob_b[local_i] = proj["probability_up"]
                            ret_b[local_i] = proj["mean_return"]
                            n_matches_b[local_i] = proj["n_matches"]
                            sigs_b[local_i] = sig
                            rsns_b[local_i] = rsn
                            ens_b[local_i] = proj["ensemble_returns"]

            probabilities.extend(prob_b.tolist())
            all_signals.extend(sigs_b)
            all_reasons.extend(rsns_b)
            all_n_matches.extend(n_matches_b.tolist())
            all_mean_returns.extend(ret_b.tolist())
            all_ensembles.extend(ens_b)

            # Progress reporting
            if verbose and batch_end % 2000 < cfg.batch_size:
                elapsed = time.time() - start_time
                rate = batch_end / elapsed if elapsed > 0 else 0
                remaining = (n_val - batch_end) / rate if rate > 0 else 0
                print(f"    {batch_end:,}/{n_val:,} "
                      f"({batch_end / n_val * 100:.0f}%) | "
                      f"{rate:.0f} queries/sec | "
                      f"ETA: {remaining / 60:.1f} min")

        return (np.array(probabilities), np.array(all_signals), all_reasons,
                all_n_matches, all_mean_returns, all_ensembles)

    @property
    def scaler(self) -> StandardScaler:
        return self._scaler

    @property
    def fitted(self) -> bool:
        return self._fitted
