"""Tests for pattern_engine.sector."""

import numpy as np
import pandas as pd
import pytest
from pattern_engine.sector import (
    SECTOR_MAP, SECTOR_PROXIES, TICKERS, get_sector, compute_sector_features,
)


class TestSectorConstants:
    def test_52_tickers(self):
        assert len(TICKERS) == 52

    def test_sector_proxies_cover_all_sectors(self):
        sectors = set(SECTOR_MAP.values())
        assert sectors == set(SECTOR_PROXIES.keys())

    def test_proxies_are_in_ticker_list(self):
        for proxy in SECTOR_PROXIES.values():
            assert proxy in TICKERS

    def test_dis_is_consumer_not_industrial(self):
        """Regression test for C1 fix: DIS is Consumer (Entertainment/Media).

        DIS was incorrectly classified as Industrial in v0.1-0.3.
        This test ensures the fix is never reverted in the canonical
        SECTOR_MAP (pattern_engine.sector), which all other modules import.
        """
        assert SECTOR_MAP["DIS"] == "Consumer"

    def test_sector_counts(self):
        """Verify expected sector distribution: 7 Consumer, 3 Industrial."""
        from collections import Counter
        counts = Counter(SECTOR_MAP.values())
        assert counts["Consumer"] == 7, f"Expected 7 Consumer, got {counts['Consumer']}"
        assert counts["Industrial"] == 3, f"Expected 3 Industrial, got {counts['Industrial']}"

    def test_all_importers_share_canonical_map(self):
        """Ensure matching.py and trading_system.config use the same SECTOR_MAP object."""
        from pattern_engine.matching import SECTOR_MAP as matching_map
        from trading_system.config import SECTOR_MAP as trading_map
        # They should be the exact same dict (imported, not copied)
        assert matching_map is SECTOR_MAP
        # trading_system uses SECTOR_MAP.copy() in the dataclass default,
        # but the module-level SECTOR_MAP should be the same reference
        assert trading_map is SECTOR_MAP


class TestGetSector:
    def test_known_ticker(self):
        assert get_sector("AAPL") == "Tech"
        assert get_sector("JPM") == "Finance"
        assert get_sector("SPY") == "Index"

    def test_unknown_ticker(self):
        assert get_sector("UNKNOWN_TICKER") is None


class TestComputeSectorFeatures:
    @pytest.fixture
    def sector_db(self, rng):
        """Synthetic DB with multiple sector tickers for sector feature testing."""
        n = 50
        dates = pd.bdate_range("2023-01-01", periods=n)
        dfs = []
        for ticker in ["SPY", "QQQ", "AAPL", "MSFT", "JPM", "WMT"]:
            df = pd.DataFrame({
                "Date": dates,
                "Ticker": ticker,
                "Close": 100 + np.cumsum(rng.randn(n) * 2),
                "ret_7d": rng.randn(n) * 0.02,
                "ret_30d": rng.randn(n) * 0.05,
            })
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def test_adds_sector_columns(self, sector_db):
        result = compute_sector_features(sector_db)
        assert "sector_relative_return_7d" in result.columns
        assert "spy_correlation_30d" in result.columns
        assert "sector_rank_30d" in result.columns

    def test_does_not_mutate_input(self, sector_db):
        original_len = len(sector_db)
        compute_sector_features(sector_db)
        assert len(sector_db) == original_len

    def test_sector_rank_bounded(self, sector_db):
        result = compute_sector_features(sector_db)
        assert result["sector_rank_30d"].min() >= 0
        assert result["sector_rank_30d"].max() <= 1
