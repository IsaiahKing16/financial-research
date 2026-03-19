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


# Sector map for cohort filtering
SECTOR_MAP = {
    "SPY": "Index", "QQQ": "Index",
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AMZN": "Tech",
    "GOOGL": "Tech", "META": "Tech", "TSLA": "Tech", "AVGO": "Tech",
    "ORCL": "Tech", "ADBE": "Tech", "CRM": "Tech", "AMD": "Tech",
    "NFLX": "Tech", "INTC": "Tech", "CSCO": "Tech", "QCOM": "Tech",
    "TXN": "Tech", "MU": "Tech", "PYPL": "Tech",
    "JPM": "Finance", "BAC": "Finance", "WFC": "Finance", "GS": "Finance",
    "MS": "Finance", "V": "Finance", "MA": "Finance", "AXP": "Finance",
    "BRK-B": "Finance",
    "LLY": "Health", "UNH": "Health", "JNJ": "Health", "ABBV": "Health",
    "MRK": "Health", "PFE": "Health", "TMO": "Health", "ISRG": "Health",
    "AMGN": "Health", "GILD": "Health",
    "WMT": "Consumer", "COST": "Consumer", "PG": "Consumer", "KO": "Consumer",
    "PEP": "Consumer", "HD": "Consumer",
    "DIS": "Industrial", "CAT": "Industrial", "BA": "Industrial", "GE": "Industrial",
    "XOM": "Energy", "CVX": "Energy",
}


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

        # Build NN index
        self._nn_index = NearestNeighbors(
            n_neighbors=min(cfg.top_k * 3, len(train_db)),
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
        return self

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
            raise RuntimeError("Matcher is not fitted. Call fit() first.")
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

        for batch_start in range(0, n_val, cfg.batch_size):
            batch_end = min(batch_start + cfg.batch_size, n_val)
            q_batch = X_val_weighted[batch_start:batch_end]

            distances_batch, indices_batch = self._nn_index.kneighbors(q_batch)

            for local_i in range(batch_end - batch_start):
                idx = batch_start + local_i
                row = val_db.iloc[idx]
                ticker = row["Ticker"]
                sector = SECTOR_MAP.get(ticker, None)

                distances = distances_batch[local_i]
                indices = indices_batch[local_i]

                matches = train_db.iloc[indices].copy()
                matches["distance"] = distances

                # Apply filters
                matches = matches[matches["distance"] <= cfg.max_distance]
                if cfg.exclude_same_ticker:
                    matches = matches[matches["Ticker"] != ticker]
                if cfg.same_sector_only and sector:
                    matches = matches[matches["Ticker"].map(SECTOR_MAP) == sector]

                # Regime filter with fallback
                if (self._regime_labels_train is not None and
                        regime_labels_val is not None and cfg.regime_filter):
                    query_regime = regime_labels_val[idx]
                    matches = apply_regime_filter(
                        matches, self._regime_labels_train, query_regime, indices
                    )

                    # Graceful fallback if too few matches
                    if cfg.regime_fallback and len(matches) < cfg.min_matches:
                        # Widen to coarser regime (only for multi/octet)
                        coarser = fallback_regime_mode(regime_labeler.mode)
                        if coarser is not None:
                            # Re-label with coarser mode
                            coarse_labeler = RegimeLabeler(mode=coarser)
                            coarse_labeler.fit(train_db)
                            coarse_train = coarse_labeler.label(train_db, reference_db=train_db)
                            coarse_val = coarse_labeler.label(val_db, reference_db=train_db)
                            # Re-fetch matches from original distance results
                            matches = train_db.iloc[indices].copy()
                            matches["distance"] = distances
                            matches = matches[matches["distance"] <= cfg.max_distance]
                            if cfg.exclude_same_ticker:
                                matches = matches[matches["Ticker"] != ticker]
                            matches = apply_regime_filter(
                                matches, coarse_train, coarse_val[idx], indices
                            )

                matches = matches.head(cfg.top_k)

                projection = project_forward(
                    matches,
                    horizon=cfg.projection_horizon,
                    weighting=cfg.distance_weighting,
                )

                signal, reason = generate_signal(
                    projection,
                    threshold=cfg.confidence_threshold,
                    min_agreement=cfg.agreement_spread,
                    min_matches=cfg.min_matches,
                )

                probabilities.append(projection["probability_up"])
                all_signals.append(signal)
                all_reasons.append(reason)
                all_n_matches.append(projection["n_matches"])
                all_mean_returns.append(projection["mean_return"])
                all_ensembles.append(projection["ensemble_returns"])

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
