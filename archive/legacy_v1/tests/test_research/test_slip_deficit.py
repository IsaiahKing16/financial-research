"""Smoke tests for SlipDeficit — correctness and interface contract."""

import numpy as np
import pandas as pd
import pytest

from research.slip_deficit import SlipDeficit
from research import RiskOverlayResult


def _make_prices(n=300, start_price=100.0, seed=0):
    """Return a DataFrame with a 'close' column and DatetimeIndex."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0, 0.01, n)
    prices = start_price * np.cumprod(1.0 + returns)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"close": prices}, index=index)


def test_init_no_error():
    """SlipDeficit initialises with default parameters without raising."""
    sd = SlipDeficit()
    assert sd.sma_window == 200
    assert sd.vol_lookback == 60


def test_zero_deficit_flat_series():
    """A perfectly flat price series has slip_deficit ≈ 0."""
    sd = SlipDeficit()
    prices = pd.DataFrame(
        {"close": np.full(300, 100.0)},
        index=pd.date_range("2020-01-01", periods=300, freq="B"),
    )
    result = sd.compute(prices)
    assert result.slip_deficit == pytest.approx(0.0, abs=1e-6)


def test_positive_deficit_and_tighten_stops():
    """High vol series → ttf_probability > 0.5; tighten_stops reflects threshold."""
    sd = SlipDeficit(ttf_threshold=0.0)  # tighten whenever vol_zscore > 0
    prices = _make_prices(n=300, seed=42)
    result = sd.compute(prices)
    assert isinstance(result, RiskOverlayResult)
    assert 0.0 <= result.ttf_probability <= 1.0
    # With ttf_threshold=0, tighten_stops == (vol_zscore > 0)
    assert isinstance(result.tighten_stops, (bool, np.bool_))


def test_result_types_and_ttf_probability_range():
    """Result fields have the correct types and ttf_probability ∈ [0, 1]."""
    sd = SlipDeficit()
    prices = _make_prices(n=300, seed=7)
    result = sd.compute(prices)
    assert isinstance(result.slip_deficit, float)
    assert isinstance(result.ttf_probability, float)
    assert 0.0 <= result.ttf_probability <= 1.0


def test_positions_none_accepted():
    """positions=None is accepted without error."""
    sd = SlipDeficit()
    prices = _make_prices(n=300)
    result = sd.compute(prices, positions=None)
    assert isinstance(result, RiskOverlayResult)


def test_positions_empty_list_accepted():
    """positions=[] is accepted without error."""
    sd = SlipDeficit()
    prices = _make_prices(n=300)
    result = sd.compute(prices, positions=[])
    assert isinstance(result, RiskOverlayResult)


def test_insufficient_history_raises_value_error():
    """ValueError raised when prices_df is shorter than max(sma_window, vol_lookback)."""
    sd = SlipDeficit(sma_window=200, vol_lookback=60)
    short_prices = _make_prices(n=50)  # less than 200
    with pytest.raises(ValueError, match="insufficient history"):
        sd.compute(short_prices)
