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
    # NOTE: DIS is Consumer (Entertainment/Media), not Industrial (C1 fix, 2026-03-19)
    "DIS": "Consumer",
    "CAT": "Industrial", "BA": "Industrial", "GE": "Industrial",
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
    # Build proxy lookup keyed by the sector each proxy REPRESENTS
    # (not the proxy ticker's own sector — QQQ is "Index" but represents "Tech")
    db["_sector"] = db["Ticker"].map(SECTOR_MAP)
    proxy_parts = []
    for sector_name, proxy_ticker in SECTOR_PROXIES.items():
        rows = db.loc[db["Ticker"] == proxy_ticker, ["Date", "ret_7d"]].copy()
        rows["_sector"] = sector_name
        proxy_parts.append(rows)
    proxy_lookup = pd.concat(proxy_parts, ignore_index=True).rename(
        columns={"ret_7d": "_proxy_ret_7d"}
    )
    db = db.merge(proxy_lookup, on=["Date", "_sector"], how="left")
    db["sector_relative_return_7d"] = db["ret_7d"] - db["_proxy_ret_7d"].fillna(0)
    db = db.drop(columns=["_proxy_ret_7d"])

    # SPY correlation: rolling 30-day correlation of each ticker's daily returns with SPY
    if "Close" in db.columns:
        spy_close = db.loc[db["Ticker"] == "SPY", ["Date", "Close"]].set_index("Date").sort_index()
        if len(spy_close) > 0:
            spy_ret = spy_close["Close"].pct_change().rename("_spy_ret")
            corr_parts = []
            for ticker, grp in db.groupby("Ticker"):
                grp_sorted = grp[["Date", "Close"]].set_index("Date").sort_index()
                ticker_ret = grp_sorted["Close"].pct_change()
                aligned_spy = spy_ret.reindex(ticker_ret.index)
                corr = ticker_ret.rolling(30, min_periods=15).corr(aligned_spy)
                corr_df = pd.DataFrame({"Date": corr.index, "_corr": corr.values})
                corr_df["Ticker"] = ticker
                corr_parts.append(corr_df)
            corr_all = pd.concat(corr_parts, ignore_index=True)
            db = db.merge(corr_all, on=["Date", "Ticker"], how="left")
            db["spy_correlation_30d"] = db["_corr"].fillna(0.0)
            db = db.drop(columns=["_corr"])

    # Sector rank: percentile rank of ret_30d within sector peers on same date
    # _sector column already set above by vectorized relative return code
    ranks = db.groupby(["Date", "_sector"])["ret_30d"].rank(pct=True)
    # Only apply where group size >= 2 (rank is meaningful)
    group_sizes = db.groupby(["Date", "_sector"])["ret_30d"].transform("count")
    db["sector_rank_30d"] = np.where(group_sizes >= 2, ranks, 0.5)
    db = db.drop(columns=["_sector"])

    return db
