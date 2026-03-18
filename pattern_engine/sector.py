"""
sector.py — Sector mapping and cross-asset feature computation.

Provides sector classification for the 52-ticker universe and
computes cross-asset features: sector-relative returns, SPY
correlation, and within-sector percentile rankings.
"""

import numpy as np
import pandas as pd

# 52 tickers across 7 sectors
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

# Sector ETF proxies (first ticker of each sector used as proxy)
SECTOR_PROXIES = {
    "Index": "SPY",
    "Tech": "QQQ",
    "Finance": "JPM",
    "Health": "UNH",
    "Consumer": "WMT",
    "Industrial": "CAT",
    "Energy": "XOM",
}

TICKERS = list(SECTOR_MAP.keys())


def get_sector(ticker: str) -> str | None:
    """Get sector for a ticker."""
    return SECTOR_MAP.get(ticker)


def compute_sector_features(db: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-asset sector features for each row.

    Features computed:
      - sector_relative_return_7d: stock ret_7d minus sector proxy ret_7d
      - spy_correlation_30d: rolling 30-day correlation with SPY
      - sector_rank_30d: percentile rank of ret_30d within sector peers on same date

    Args:
        db: full analogue database with Ticker, Date, ret_7d, ret_30d columns

    Returns:
        DataFrame with sector feature columns added
    """
    db = db.copy()

    # Initialize columns
    db["sector_relative_return_7d"] = 0.0
    db["spy_correlation_30d"] = 0.0
    db["sector_rank_30d"] = 0.5

    # Sector-relative return: stock ret_7d - sector proxy ret_7d
    for sector, proxy in SECTOR_PROXIES.items():
        sector_tickers = [t for t, s in SECTOR_MAP.items() if s == sector and t != proxy]
        if not sector_tickers:
            continue

        proxy_data = db[db["Ticker"] == proxy][["Date", "ret_7d"]].copy()
        proxy_data = proxy_data.rename(columns={"ret_7d": "proxy_ret_7d"})

        for ticker in sector_tickers:
            mask = db["Ticker"] == ticker
            if mask.sum() == 0:
                continue
            ticker_data = db[mask][["Date", "ret_7d"]].copy()
            merged = ticker_data.merge(proxy_data, on="Date", how="left")
            relative = (merged["ret_7d"] - merged["proxy_ret_7d"]).values
            db.loc[mask, "sector_relative_return_7d"] = relative

    # Sector rank: percentile rank of ret_30d within sector peers on same date
    for date in db["Date"].unique():
        date_mask = db["Date"] == date
        date_rows = db[date_mask]

        for sector in SECTOR_MAP.values():
            sector_mask = date_mask & (db["Ticker"].map(SECTOR_MAP) == sector)
            sector_rows = db[sector_mask]
            if len(sector_rows) < 2:
                continue
            ranks = sector_rows["ret_30d"].rank(pct=True)
            db.loc[sector_mask, "sector_rank_30d"] = ranks.values

    return db
