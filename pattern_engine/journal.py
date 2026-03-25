"""
journal.py — Decision Journal for PatternMatcher signal inspection.

Records the top-N historical analogues for each BUY/SELL signal, enabling
human-readable audit trails: which past patterns drove a prediction, when
they occurred, how close they were, and what actually happened.

Linear: M9 (Signal Intelligence Layer)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class AnalogueRecord:
    """One historical analogue that contributed to a signal."""
    rank: int
    ticker: str
    date: date
    distance: float
    label: int
    fwd_return: float


@dataclass
class JournalEntry:
    """Complete decision record for one BUY or SELL signal."""
    query_date: date
    query_ticker: str
    raw_prob: float
    calibrated_prob: float
    signal: str
    n_matches: int
    top_analogues: list[AnalogueRecord] = field(default_factory=list)


def _normalize_date(val) -> date:
    """Normalize any date-like value to datetime.date."""
    if hasattr(val, "date"):
        return val.date()
    if isinstance(val, np.datetime64):
        return pd.Timestamp(val).date()
    if val is None:
        return date(1900, 1, 1)
    return val


def build_journal_entries(
    *,
    top_masks: np.ndarray,
    distances: np.ndarray,
    indices: np.ndarray,
    val_tickers: np.ndarray,
    val_dates: np.ndarray,
    raw_probs: np.ndarray,
    cal_probs: np.ndarray,
    signals: list[str],
    n_matches: list[int],
    train_tickers: np.ndarray,
    train_dates: np.ndarray,
    train_targets: np.ndarray,
    train_returns: np.ndarray,
    top_n: int = 25,
) -> list[JournalEntry]:
    """Build journal entries for all BUY/SELL rows in a batch.

    HOLD rows are skipped — they have no actionable signal to explain.
    Only the top_n closest accepted analogues are stored per entry.
    """
    entries: list[JournalEntry] = []
    B = top_masks.shape[0]

    for i in range(B):
        sig = signals[i]
        if sig not in ("BUY", "SELL"):
            continue

        accepted_pos = np.where(top_masks[i])[0]
        if len(accepted_pos) == 0:
            continue

        sorted_pos = accepted_pos[np.argsort(distances[i][accepted_pos])]
        cap = min(top_n, len(sorted_pos))
        sorted_pos = sorted_pos[:cap]

        analogues: list[AnalogueRecord] = []
        for rank, pos in enumerate(sorted_pos, start=1):
            train_idx = int(indices[i][pos])
            analogues.append(AnalogueRecord(
                rank=rank,
                ticker=str(train_tickers[train_idx]),
                date=_normalize_date(train_dates[train_idx]),
                distance=float(distances[i][pos]),
                label=int(train_targets[train_idx]),
                fwd_return=float(train_returns[train_idx]),
            ))

        entries.append(JournalEntry(
            query_date=_normalize_date(val_dates[i]),
            query_ticker=str(val_tickers[i]),
            raw_prob=float(raw_probs[i]),
            calibrated_prob=float(cal_probs[i]),
            signal=sig,
            n_matches=int(n_matches[i]),
            top_analogues=analogues,
        ))

    return entries


def write_journal_parquet(entries: list[JournalEntry], path: Path | str) -> None:
    """Write journal entries to a flat Parquet file (one row per analogue)."""
    rows = []
    for e in entries:
        for a in e.top_analogues:
            rows.append({
                "query_date":          e.query_date,
                "query_ticker":        e.query_ticker,
                "raw_prob":            e.raw_prob,
                "calibrated_prob":     e.calibrated_prob,
                "signal":              e.signal,
                "n_matches":           e.n_matches,
                "analogue_rank":       a.rank,
                "analogue_ticker":     a.ticker,
                "analogue_date":       a.date,
                "analogue_distance":   a.distance,
                "analogue_label":      a.label,
                "analogue_fwd_return": a.fwd_return,
            })

    if not rows:
        df = pd.DataFrame(columns=[
            "query_date", "query_ticker", "raw_prob", "calibrated_prob",
            "signal", "n_matches", "analogue_rank", "analogue_ticker",
            "analogue_date", "analogue_distance", "analogue_label",
            "analogue_fwd_return",
        ])
    else:
        df = pd.DataFrame(rows)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(path), index=False)


def read_journal_parquet(path: Path | str) -> pd.DataFrame:
    """Read a journal Parquet file back into a DataFrame."""
    return pd.read_parquet(str(path))


def top_n_view(journal_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Filter journal to top-N analogues per signal."""
    if n < 1:
        raise ValueError(f"n must be >= 1 (got {n})")
    return journal_df[journal_df["analogue_rank"] <= n].copy()
