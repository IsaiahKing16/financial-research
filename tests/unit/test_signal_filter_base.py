"""Tests for SignalFilterBase ABC and SignalPipeline."""
import numpy as np
import pandas as pd
import pytest


def test_signal_filter_base_is_abstract():
    """SignalFilterBase cannot be instantiated directly."""
    from pattern_engine.signal_filter_base import SignalFilterBase
    with pytest.raises(TypeError):
        SignalFilterBase()  # type: ignore


def test_concrete_filter_without_apply_raises():
    """Subclass that does not implement apply() raises TypeError on instantiation."""
    from pattern_engine.signal_filter_base import SignalFilterBase

    class IncompleteFilter(SignalFilterBase):
        pass

    with pytest.raises(TypeError):
        IncompleteFilter()


def test_signal_pipeline_applies_filters_in_order():
    """SignalPipeline threads signals through each filter sequentially."""
    from pattern_engine.signal_filter_base import SignalFilterBase
    from pattern_engine.signal_pipeline import SignalPipeline

    class NoOpFilter(SignalFilterBase):
        def apply(self, probs, signals, val_db, **kwargs):
            return list(signals), np.zeros(len(signals), dtype=bool)

    class HoldAllFilter(SignalFilterBase):
        def apply(self, probs, signals, val_db, **kwargs):
            mask = np.array([s != "HOLD" for s in signals], dtype=bool)
            return ["HOLD"] * len(signals), mask

    pipeline = SignalPipeline(filters=[NoOpFilter(), HoldAllFilter()])
    probs = np.array([0.70, 0.30])
    signals = ["BUY", "SELL"]
    val_db = pd.DataFrame({"Ticker": ["AAPL", "JPM"]})
    filtered, mask = pipeline.run(probs, signals, val_db)
    assert filtered == ["HOLD", "HOLD"]
    assert mask.all()


def test_signal_pipeline_combined_mask_is_union():
    """combined_mask is the union of all per-filter masks."""
    from pattern_engine.signal_filter_base import SignalFilterBase
    from pattern_engine.signal_pipeline import SignalPipeline

    class FilterFirst(SignalFilterBase):
        def apply(self, probs, signals, val_db, **kwargs):
            filtered = list(signals)
            mask = np.zeros(len(signals), dtype=bool)
            filtered[0] = "HOLD"
            mask[0] = True
            return filtered, mask

    class FilterSecond(SignalFilterBase):
        def apply(self, probs, signals, val_db, **kwargs):
            filtered = list(signals)
            mask = np.zeros(len(signals), dtype=bool)
            filtered[1] = "HOLD"
            mask[1] = True
            return filtered, mask

    pipeline = SignalPipeline(filters=[FilterFirst(), FilterSecond()])
    probs = np.array([0.70, 0.68])
    signals = ["BUY", "BUY"]
    val_db = pd.DataFrame({"Ticker": ["AAPL", "JPM"]})
    filtered, mask = pipeline.run(probs, signals, val_db)
    assert filtered == ["HOLD", "HOLD"]
    assert mask[0] and mask[1]
