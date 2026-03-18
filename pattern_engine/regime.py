"""
regime.py — Macro-regime detection and filtering.

Classifies market conditions into regimes (bull/bear, volatility,
trend strength) and filters analogues to match only within the
same regime as the query date.

Supports three modes:
  - binary (2 regimes): Bull/Bear via SPY ret_90d
  - multi  (4 regimes): Bull/Bear × Low/High volatility
  - octet  (8 regimes): Bull/Bear × Low/High vol × Trending/Range

Includes graceful fallback: if a regime bin has too few analogues,
automatically widens to a coarser regime classification.

Critical: v4 SPY fallback branch — when val_db == train_db (the
calibration pass), SPY rows come from the same DataFrame. The
RegimeLabeler handles this case correctly.
"""

import numpy as np
import pandas as pd


class RegimeLabeler:
    """Labels market regime for each row based on SPY indicators.

    Args:
        mode: "binary" (2 regimes), "multi" (4), or "octet" (8)
        adx_threshold: ADX value above which market is "trending" (octet only)
    """

    MODES = {"binary": 2, "multi": 4, "octet": 8}

    def __init__(self, mode: str = "binary", adx_threshold: float = 25.0):
        if mode not in self.MODES:
            raise ValueError(f"Unknown regime mode: {mode!r}. Choose from: {list(self.MODES.keys())}")
        self.mode = mode
        self.adx_threshold = adx_threshold
        self._vol_median = None
        self._fitted = False

    def fit(self, train_db: pd.DataFrame) -> "RegimeLabeler":
        """Compute regime thresholds from training data.

        For multi/octet modes, computes the median vol_30d from training
        SPY rows to establish the Low/High volatility boundary.
        """
        if self.mode in ("multi", "octet"):
            spy_train = train_db[train_db["Ticker"] == "SPY"]
            if "vol_30d" in spy_train.columns and len(spy_train) > 0:
                self._vol_median = float(spy_train["vol_30d"].median())
            else:
                self._vol_median = 0.01  # safe fallback
        self._fitted = True
        return self

    def label(self, db: pd.DataFrame, reference_db: pd.DataFrame = None) -> np.ndarray:
        """Assign regime labels to each row.

        Uses SPY rows from `db` itself for regime classification.
        This implements the v4 fix: val queries use val_db SPY rows,
        not train_db SPY rows.

        When val_db == train_db (calibration pass), SPY rows come from
        the same DataFrame — this is the correct behavior (fallback branch).

        Args:
            db: DataFrame to label (must contain Ticker, Date, ret_90d columns)
            reference_db: Optional fallback DataFrame for SPY rows if db has none

        Returns:
            np.ndarray of integer regime labels (0-indexed)
        """
        # Extract SPY rows from db (v4 fix: use db's own SPY rows)
        spy = db[db["Ticker"] == "SPY"][["Date", "ret_90d"]].copy()

        # Fallback branch: if db has no SPY rows, use reference_db
        if len(spy) == 0 and reference_db is not None:
            spy = reference_db[reference_db["Ticker"] == "SPY"][["Date", "ret_90d"]].copy()

        if len(spy) < 10:
            # Insufficient SPY data — return all zeros (single regime, no filtering)
            return np.zeros(len(db), dtype=np.int8)

        spy = spy.sort_values("Date").reset_index(drop=True)
        spy_dates = np.array(pd.to_datetime(spy["Date"].values), dtype="datetime64[ns]")
        spy_ret90 = spy["ret_90d"].values

        db_dates = np.array(pd.to_datetime(db["Date"].values), dtype="datetime64[ns]")
        idx = np.searchsorted(spy_dates, db_dates, side="right") - 1
        idx = np.clip(idx, 0, len(spy_ret90) - 1)

        # Dimension 1: Direction (bull=1, bear=0)
        direction = (spy_ret90[idx] > 0).astype(np.int8)

        if self.mode == "binary":
            return direction

        # Dimension 2: Volatility (need vol_30d in SPY rows)
        spy_vol = db[db["Ticker"] == "SPY"][["Date", "vol_30d"]].copy()
        if len(spy_vol) == 0 and reference_db is not None:
            spy_vol = reference_db[reference_db["Ticker"] == "SPY"][["Date", "vol_30d"]].copy()

        if len(spy_vol) > 0 and "vol_30d" in spy_vol.columns:
            spy_vol = spy_vol.sort_values("Date").reset_index(drop=True)
            spy_vol_dates = np.array(pd.to_datetime(spy_vol["Date"].values), dtype="datetime64[ns]")
            spy_vol_vals = spy_vol["vol_30d"].values
            v_idx = np.searchsorted(spy_vol_dates, db_dates, side="right") - 1
            v_idx = np.clip(v_idx, 0, len(spy_vol_vals) - 1)
            vol_high = (spy_vol_vals[v_idx] >= self._vol_median).astype(np.int8)
        else:
            vol_high = np.zeros(len(db), dtype=np.int8)

        if self.mode == "multi":
            # 4 regimes: direction * 2 + vol_high
            return (direction * 2 + vol_high).astype(np.int8)

        # Dimension 3: Trend strength via ADX (octet mode)
        if "adx_14" in db.columns:
            trending = (db["adx_14"].values > self.adx_threshold).astype(np.int8)
        else:
            # If ADX not available, compute from SPY or default to 0
            trending = np.zeros(len(db), dtype=np.int8)

        # 8 regimes: direction * 4 + vol_high * 2 + trending
        return (direction * 4 + vol_high * 2 + trending).astype(np.int8)

    @property
    def n_regimes(self) -> int:
        return self.MODES[self.mode]

    @property
    def fitted(self) -> bool:
        return self._fitted


def apply_regime_filter(matches: pd.DataFrame, match_labels: np.ndarray,
                        query_label: int, match_indices: np.ndarray) -> pd.DataFrame:
    """Filter matches to keep only those in the same regime as the query.

    Args:
        matches: DataFrame of candidate matches
        match_labels: regime labels for ALL training rows
        query_label: regime label for the current query
        match_indices: indices into training DB for current matches

    Returns:
        Filtered DataFrame containing only same-regime matches
    """
    regime_mask = match_labels[match_indices] == query_label
    return matches[regime_mask[:len(matches)]]


def fallback_regime_mode(current_mode: str) -> str | None:
    """Get the next coarser regime mode for graceful fallback.

    Returns None if already at the coarsest level (binary).
    """
    fallback_chain = {"octet": "multi", "multi": "binary", "binary": None}
    return fallback_chain.get(current_mode)
