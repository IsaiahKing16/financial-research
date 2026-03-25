"""
query_journal.py — CLI inspection tool for FPPE decision journals.

Reads the Parquet journal files written by run_walkforward.py and displays
the top-N analogues for each BUY/SELL signal, optionally filtered by ticker,
date range, or signal type.

Usage:
    # Show all BUY signals for AAPL in fold 6, top-5 analogues each:
    py -3.12 scripts/query_journal.py --fold 2024-Val --ticker AAPL --top 5

    # Show top-10 for all SELL signals in all folds:
    py -3.12 scripts/query_journal.py --signal SELL --top 10

    # Show signals from a specific query date range:
    py -3.12 scripts/query_journal.py --from 2024-03-01 --to 2024-03-31 --top 25

Options:
    --fold      Fold label (e.g. 2024-Val). Omit for all folds.
    --ticker    Query ticker to filter (e.g. AAPL).
    --signal    BUY or SELL. Omit for both.
    --top       Analogues to show per signal: 5, 10, or 25 (default: 5).
    --from      Start date filter (YYYY-MM-DD).
    --to        End date filter (YYYY-MM-DD).
    --out       Optional CSV output path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
JOURNAL_DIR = REPO_ROOT / "results" / "journals"


def load_all_journals(fold: str | None = None) -> pd.DataFrame:
    """Load all (or one) journal Parquet files from results/journals/."""
    if not JOURNAL_DIR.exists():
        print(f"No journal directory found at {JOURNAL_DIR}")
        print("Run scripts/run_walkforward.py with journal_top_n > 0 first.")
        sys.exit(1)

    files = sorted(JOURNAL_DIR.glob("journal_fold_*.parquet"))
    if not files:
        print(f"No journal files found in {JOURNAL_DIR}")
        sys.exit(1)

    if fold:
        # Match fold label (normalize spaces to underscores)
        target = fold.replace(" ", "_")
        files = [f for f in files if target in f.stem]
        if not files:
            print(f"No journal file matching fold '{fold}'")
            print(f"Available: {[f.stem for f in sorted(JOURNAL_DIR.glob('*.parquet'))]}")
            sys.exit(1)

    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        df["fold"] = f.stem.replace("journal_fold_", "")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect FPPE decision journal")
    parser.add_argument("--fold",   default=None, help="Fold label (e.g. 2024-Val)")
    parser.add_argument("--ticker", default=None, help="Query ticker (e.g. AAPL)")
    parser.add_argument("--signal", default=None, choices=["BUY", "SELL"], help="Signal type")
    parser.add_argument("--top",    default=5, type=int, choices=[5, 10, 25],
                        help="Analogues per signal (default: 5)")
    parser.add_argument("--from",   dest="date_from", default=None,
                        help="Start date YYYY-MM-DD")
    parser.add_argument("--to",     dest="date_to", default=None,
                        help="End date YYYY-MM-DD")
    parser.add_argument("--out",    default=None, help="Optional CSV output path")
    args = parser.parse_args()

    df = load_all_journals(args.fold)

    # Apply filters
    if args.ticker:
        df = df[df["query_ticker"] == args.ticker.upper()]
    if args.signal:
        df = df[df["signal"] == args.signal]
    if args.date_from:
        df = df[pd.to_datetime(df["query_date"]) >= pd.Timestamp(args.date_from)]
    if args.date_to:
        df = df[pd.to_datetime(df["query_date"]) <= pd.Timestamp(args.date_to)]

    # Apply top-N filter
    df = df[df["analogue_rank"] <= args.top]

    if df.empty:
        print("No results matching filters.")
        sys.exit(0)

    # Summary
    n_signals = df.groupby(["query_date", "query_ticker"]).ngroups
    print(f"\nFound {n_signals} signals ({args.top} analogues each)")
    print(f"Folds:   {df['fold'].unique().tolist()}")
    print(f"Signals: {df['signal'].value_counts().to_dict()}")
    print(f"Tickers: {df['query_ticker'].nunique()} unique\n")

    # Display
    display_cols = [
        "fold", "query_date", "query_ticker", "calibrated_prob", "signal",
        "n_matches", "analogue_rank", "analogue_ticker", "analogue_date",
        "analogue_distance", "analogue_label", "analogue_fwd_return",
    ]
    print(df[display_cols].to_string(index=False, max_rows=50))

    if args.out:
        df[display_cols].to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
