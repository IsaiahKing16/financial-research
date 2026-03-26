"""
test_regime.py — Unit tests for RegimeLabeler.

Tests cover:
    - mode attribute ("binary" vs "multi_factor")
    - Bull/Bear labeling from SPY ret_90d
    - NaN/missing date fallback (Bull)
    - RuntimeError before fit()
    - VIX spike z-score Bear override
    - Inverted yield curve Bear override
    - Output shape contract

Linear: M9 (Signal Intelligence Layer)
"""

import numpy as np
import pandas as pd
import pytest

from pattern_engine.regime import RegimeLabeler


# ─── Helpers ────────────────────────────────────────────────────────────────

def _make_spy_db(dates, spy_ret90_values):
    """Build minimal reference DataFrame with SPY rows."""
    return pd.DataFrame({
        "Ticker": "SPY",
        "Date": pd.to_datetime(dates),
        "ret_90d": spy_ret90_values,
    })


def _make_query_db(dates, tickers=None):
    """Build minimal query DataFrame with Date column."""
    if tickers is None:
        tickers = ["AAPL"] * len(dates)
    return pd.DataFrame({
        "Ticker": tickers,
        "Date": pd.to_datetime(dates),
    })


# ─── TestRegimeLabeler ───────────────────────────────────────────────────────

class TestRegimeLabeler:

    # ── mode ──────────────────────────────────────────────────────────────

    def test_binary_mode_when_no_extra_series(self):
        """mode == 'binary' when vix_series=None and yield_spread_series=None."""
        ref_db = _make_spy_db(["2023-01-03"], [0.05])
        labeler = RegimeLabeler()
        labeler.fit(ref_db)
        assert labeler.mode == "binary"

    def test_multi_factor_mode_with_vix(self):
        """mode == 'multi_factor' when vix_series is provided."""
        dates = pd.to_datetime(["2023-01-03", "2023-01-04", "2023-01-05"])
        vix = pd.Series([20.0, 21.0, 19.5], index=dates)
        ref_db = _make_spy_db(["2023-01-03"], [0.05])
        labeler = RegimeLabeler(vix_series=vix)
        labeler.fit(ref_db)
        assert labeler.mode == "multi_factor"

    def test_multi_factor_mode_with_yield_spread(self):
        """mode == 'multi_factor' when yield_spread_series is provided."""
        dates = pd.to_datetime(["2023-01-03"])
        spread = pd.Series([-0.1], index=dates)
        ref_db = _make_spy_db(["2023-01-03"], [0.05])
        labeler = RegimeLabeler(yield_spread_series=spread)
        labeler.fit(ref_db)
        assert labeler.mode == "multi_factor"

    # ── fitted flag ───────────────────────────────────────────────────────

    def test_fitted_false_before_fit(self):
        """labeler.fitted == False before fit() is called."""
        labeler = RegimeLabeler()
        assert labeler.fitted is False

    def test_fitted_true_after_fit(self):
        """labeler.fitted == True after fit() is called."""
        ref_db = _make_spy_db(["2023-01-03"], [0.05])
        labeler = RegimeLabeler()
        labeler.fit(ref_db)
        assert labeler.fitted is True

    # ── RuntimeError before fit ───────────────────────────────────────────

    def test_label_before_fit_raises(self):
        """label() raises RuntimeError when called before fit()."""
        labeler = RegimeLabeler()
        query_db = _make_query_db(["2023-01-03"])
        with pytest.raises(RuntimeError):
            labeler.label(query_db)

    # ── basic Bull / Bear labeling ────────────────────────────────────────

    def test_bull_when_spy_positive(self):
        """SPY ret_90d > 0 → label = 1 (Bull)."""
        ref_db = _make_spy_db(["2023-01-03"], [0.08])
        query_db = _make_query_db(["2023-01-03"])
        labeler = RegimeLabeler()
        labeler.fit(ref_db)
        labels = labeler.label(query_db)
        assert labels[0] == 1

    def test_bull_when_spy_zero(self):
        """SPY ret_90d == 0 → label = 1 (Bull, boundary: ≥0 → Bull)."""
        ref_db = _make_spy_db(["2023-01-03"], [0.0])
        query_db = _make_query_db(["2023-01-03"])
        labeler = RegimeLabeler()
        labeler.fit(ref_db)
        labels = labeler.label(query_db)
        assert labels[0] == 1

    def test_bear_when_spy_negative(self):
        """SPY ret_90d < 0 → label = 0 (Bear)."""
        ref_db = _make_spy_db(["2023-01-03"], [-0.05])
        query_db = _make_query_db(["2023-01-03"])
        labeler = RegimeLabeler()
        labeler.fit(ref_db)
        labels = labeler.label(query_db)
        assert labels[0] == 0

    def test_multiple_dates_bull_bear(self):
        """Correct Bull/Bear assignment across multiple dates."""
        ref_db = _make_spy_db(
            ["2023-01-03", "2023-01-04", "2023-01-05"],
            [0.05, -0.03, 0.02],
        )
        query_db = _make_query_db(["2023-01-03", "2023-01-04", "2023-01-05"])
        labeler = RegimeLabeler()
        labeler.fit(ref_db)
        labels = labeler.label(query_db)
        np.testing.assert_array_equal(labels, [1, 0, 1])

    def test_bull_fallback_when_spy_missing(self):
        """Date not in SPY rows → label = 1 (fallback Bull)."""
        ref_db = _make_spy_db(["2023-01-03"], [0.05])
        # Query on a date far from any SPY data
        query_db = _make_query_db(["2030-01-01"])
        labeler = RegimeLabeler()
        labeler.fit(ref_db)
        labels = labeler.label(query_db)
        # Fallback is Bull (1)
        assert labels[0] == 1

    # ── output shape ──────────────────────────────────────────────────────

    def test_returns_correct_length(self):
        """label() returns np.ndarray with len == len(db)."""
        ref_db = _make_spy_db(["2023-01-03", "2023-01-04"], [0.05, -0.02])
        query_db = _make_query_db(
            ["2023-01-03", "2023-01-04", "2023-01-03"],
            ["AAPL", "AAPL", "MSFT"],
        )
        labeler = RegimeLabeler()
        labeler.fit(ref_db)
        labels = labeler.label(query_db)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == 3

    def test_returns_int_dtype(self):
        """label() returns integer dtype array."""
        ref_db = _make_spy_db(["2023-01-03"], [0.05])
        query_db = _make_query_db(["2023-01-03"])
        labeler = RegimeLabeler()
        labeler.fit(ref_db)
        labels = labeler.label(query_db)
        assert np.issubdtype(labels.dtype, np.integer)

    # ── VIX spike override ─────────────────────────────────────────────

    def test_vix_spike_overrides_bull_to_bear(self):
        """VIX z-score spike > threshold forces Bear even when SPY positive."""
        # Build a VIX series where the last day has a huge spike
        # Window=5 for predictability in tests
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        # Stable VIX with a sudden spike on the 8th day
        vix_values = [20.0, 20.1, 19.9, 20.2, 20.0, 20.1, 19.8, 20.0, 35.0, 20.1]
        vix = pd.Series(vix_values, index=dates)

        spike_date = dates[8]  # The huge-spike day (35.0 from ~20.0)

        # SPY is positive on all dates
        spy_date = spike_date.strftime("%Y-%m-%d")
        ref_db = _make_spy_db([spy_date], [0.10])

        labeler = RegimeLabeler(
            vix_series=vix,
            vix_spike_zscore=1.0,
            vix_spike_window=5,
        )
        labeler.fit(ref_db)

        # Query on the spike date — should be Bear despite positive SPY
        query_db = _make_query_db([spy_date])
        labels = labeler.label(query_db)
        assert labels[0] == 0, (
            f"Expected Bear (0) on VIX spike date, got {labels[0]}"
        )

    def test_no_vix_spike_keeps_bull(self):
        """Normal VIX variation does not override Bull label."""
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        # Flat VIX — no spike
        vix_values = [20.0] * 10
        vix = pd.Series(vix_values, index=dates)

        query_date = dates[9].strftime("%Y-%m-%d")
        ref_db = _make_spy_db([query_date], [0.05])

        labeler = RegimeLabeler(
            vix_series=vix,
            vix_spike_zscore=1.0,
            vix_spike_window=5,
        )
        labeler.fit(ref_db)

        query_db = _make_query_db([query_date])
        labels = labeler.label(query_db)
        assert labels[0] == 1

    # ── Inverted yield curve override ─────────────────────────────────

    def test_inverted_yield_curve_forces_bear(self):
        """yield_spread < 0 → Bear even if SPY positive."""
        date = "2023-06-01"
        ref_db = _make_spy_db([date], [0.12])  # SPY strongly positive

        dates_idx = pd.to_datetime([date])
        spread = pd.Series([-0.25], index=dates_idx)  # Inverted

        labeler = RegimeLabeler(yield_spread_series=spread)
        labeler.fit(ref_db)

        query_db = _make_query_db([date])
        labels = labeler.label(query_db)
        assert labels[0] == 0, (
            f"Expected Bear (0) for inverted yield curve, got {labels[0]}"
        )

    def test_positive_yield_spread_keeps_bull(self):
        """Positive yield spread does not override Bull label."""
        date = "2023-06-01"
        ref_db = _make_spy_db([date], [0.12])

        dates_idx = pd.to_datetime([date])
        spread = pd.Series([0.50], index=dates_idx)  # Normal / steep

        labeler = RegimeLabeler(yield_spread_series=spread)
        labeler.fit(ref_db)

        query_db = _make_query_db([date])
        labels = labeler.label(query_db)
        assert labels[0] == 1

    def test_both_overrides_active(self):
        """Both VIX spike and inverted yield curve → Bear."""
        dates = pd.date_range("2023-01-02", periods=10, freq="B")
        vix_values = [20.0, 20.1, 19.9, 20.2, 20.0, 20.1, 19.8, 20.0, 35.0, 20.1]
        vix = pd.Series(vix_values, index=dates)

        spike_date = dates[8]
        spy_date_str = spike_date.strftime("%Y-%m-%d")
        ref_db = _make_spy_db([spy_date_str], [0.10])

        spread_idx = pd.to_datetime([spy_date_str])
        spread = pd.Series([-0.5], index=spread_idx)

        labeler = RegimeLabeler(
            vix_series=vix,
            vix_spike_zscore=1.0,
            vix_spike_window=5,
            yield_spread_series=spread,
        )
        labeler.fit(ref_db)

        query_db = _make_query_db([spy_date_str])
        labels = labeler.label(query_db)
        assert labels[0] == 0

    # ── fit returns self ──────────────────────────────────────────────

    def test_fit_returns_self(self):
        """fit() returns self for method chaining."""
        ref_db = _make_spy_db(["2023-01-03"], [0.05])
        labeler = RegimeLabeler()
        result = labeler.fit(ref_db)
        assert result is labeler

    # ── reference_db parameter passthrough ───────────────────────────

    def test_label_accepts_reference_db_kwarg(self):
        """label() accepts optional reference_db= kwarg without error."""
        ref_db = _make_spy_db(["2023-01-03"], [0.05])
        labeler = RegimeLabeler()
        labeler.fit(ref_db)
        query_db = _make_query_db(["2023-01-03"])
        # Should not raise
        labels = labeler.label(query_db, reference_db=ref_db)
        assert len(labels) == 1
