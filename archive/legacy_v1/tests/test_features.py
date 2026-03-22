"""Tests for pattern_engine.features."""

import pytest
from pattern_engine.features import (
    RETURN_WINDOWS, RETURN_COLS, SUPPLEMENT_COLS, CANDLE_COLS,
    CANDLE_1D_COLS, CANDLE_3D_COLS, CANDLE_5D_COLS,
    SECTOR_COLS, VOL_COLS,
    FORWARD_WINDOWS, FORWARD_RETURN_COLS, FORWARD_BINARY_COLS,
    FeatureSet, FeatureRegistry,
)


class TestColumnDefinitions:
    """Test column name constants."""

    def test_return_windows(self):
        assert RETURN_WINDOWS == [1, 3, 7, 14, 30, 45, 60, 90]

    def test_return_cols_count(self):
        assert len(RETURN_COLS) == 8
        assert RETURN_COLS[0] == "ret_1d"
        assert RETURN_COLS[-1] == "ret_90d"

    def test_supplement_cols_count(self):
        assert len(SUPPLEMENT_COLS) == 8

    def test_candle_cols_count(self):
        assert len(CANDLE_1D_COLS) == 5
        assert len(CANDLE_3D_COLS) == 5
        assert len(CANDLE_5D_COLS) == 5
        assert len(CANDLE_COLS) == 15

    def test_sector_cols(self):
        assert len(SECTOR_COLS) == 3
        assert "sector_relative_return_7d" in SECTOR_COLS

    def test_forward_cols(self):
        assert len(FORWARD_WINDOWS) == 5
        assert len(FORWARD_RETURN_COLS) == 5
        assert len(FORWARD_BINARY_COLS) == 5
        assert "fwd_7d" in FORWARD_RETURN_COLS
        assert "fwd_7d_up" in FORWARD_BINARY_COLS


class TestFeatureRegistry:
    """Test FeatureRegistry class."""

    def test_builtin_sets(self):
        names = FeatureRegistry.list_sets()
        assert "returns_only" in names
        assert "returns_candle" in names
        assert "returns_vol" in names
        assert "returns_sector" in names
        assert "full" in names
        assert "returns_hybrid" in names

    def test_returns_only_columns(self):
        fs = FeatureRegistry.get("returns_only")
        assert fs.columns == RETURN_COLS

    def test_returns_candle_columns(self):
        fs = FeatureRegistry.get("returns_candle")
        assert len(fs.columns) == 8 + 15  # returns + candles

    def test_full_columns(self):
        fs = FeatureRegistry.get("full")
        assert len(fs.columns) == 8 + 8 + 15 + 3  # returns + suppl + candles + sector

    def test_unknown_set_raises(self):
        with pytest.raises(KeyError, match="Unknown feature set"):
            FeatureRegistry.get("nonexistent_set")

    def test_register_custom(self):
        FeatureRegistry.register("test_custom", ["a", "b", "c"], "test set")
        fs = FeatureRegistry.get("test_custom")
        assert fs.columns == ["a", "b", "c"]
        assert fs.description == "test set"
        # Clean up
        del FeatureRegistry._sets["test_custom"]

    def test_feature_set_dataclass(self):
        fs = FeatureSet(name="test", columns=["x", "y"])
        assert fs.name == "test"
        assert fs.columns == ["x", "y"]
        assert fs.description == ""

    def test_returns_hybrid_raises_not_implemented(self):
        """C2 guard: returns_hybrid requires a trained CONV_LSTM encoder."""
        with pytest.raises(NotImplementedError, match="neural network"):
            FeatureRegistry.get("returns_hybrid")

    def test_returns_hybrid_is_registered(self):
        """returns_hybrid should appear in list_sets even though it can't be used."""
        assert "returns_hybrid" in FeatureRegistry.list_sets()
