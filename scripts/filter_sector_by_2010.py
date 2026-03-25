"""
filter_sector_by_2010.py — Remove tickers without sufficient 2010 historical data.

Fixes the ATR(window=14) IndexError in prepare.py step 2/4:
  "IndexError: index 13 is out of bounds for axis 0 with size 1"

Post-2010 IPOs (ABBV, META, HPE, PYPL, etc.) have zero rows in the training
window, causing the rolling ATR to fail. This script gates on >= 14 trading
days present in 2010 H1.

Run once, then re-run prepare.py:
    python scripts/filter_sector_by_2010.py
    py -3.12 prepare.py
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pattern_engine.sector import SECTOR_MAP, SECTOR_PROXIES  # noqa: E402

OUTPUT_PATH = REPO_ROOT / "pattern_engine" / "sector.py"

# Minimum rows in 2010-H1 required for ATR(window=14) to work
MIN_ROWS = 14
BATCH_SIZE = 100
CHECK_START = "2010-01-01"
CHECK_END = "2010-06-30"

# Always keep these regardless of data depth
ALWAYS_KEEP = {"SPY", "QQQ"}


# ── 2010 depth check ──────────────────────────────────────────────────────────

def check_2010_depth(tickers: list[str]) -> set[str]:
    """Return set of tickers with >= MIN_ROWS trading days in CHECK range."""
    print(f"\nChecking 2010 historical depth for {len(tickers)} tickers "
          f"({CHECK_START} → {CHECK_END}, need >= {MIN_ROWS} rows)...")
    valid: set[str] = set()
    n_batches = (len(tickers) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        try:
            data = yf.download(
                batch, start=CHECK_START, end=CHECK_END,
                auto_adjust=True, progress=False, threads=True,
            )
            if data.empty:
                good: list[str] = []
            elif len(batch) == 1:
                good = batch if len(data) >= MIN_ROWS else []
            elif "Close" in data.columns:
                close = data["Close"]
                good = close.columns[close.count() >= MIN_ROWS].tolist()
            else:
                good = []
            valid.update(good)
            print(f"  Batch {batch_num}/{n_batches}: "
                  f"{len(good)}/{len(batch)} pass depth gate")
        except Exception as exc:
            print(f"  Batch {batch_num}/{n_batches} error: {exc}")

    return valid


# ── sector.py writer (mirrors build_sector_map.write_sector_py) ───────────────

def write_sector_py(sector_map: dict[str, str]) -> None:
    tickers = sorted(sector_map.keys())
    dist = Counter(sector_map.values())

    by_sector: dict[str, list[str]] = {}
    for t in tickers:
        by_sector.setdefault(sector_map[t], []).append(t)

    lines = [
        '"""',
        'sector.py — FPPE ticker universe and sector mapping.',
        '',
        f'Universe: {len(tickers)} tickers (Russell 1000 scope)',
        'Sectors: Index, Tech, Finance, Health, Consumer, Industrial, Energy',
        '',
        'Regenerate: python scripts/build_sector_map.py',
        '"""',
        'from __future__ import annotations',
        '',
        'SECTOR_MAP: dict[str, str] = {',
    ]

    for sector in ["Index", "Tech", "Finance", "Health", "Consumer", "Industrial", "Energy"]:
        grp = by_sector.get(sector, [])
        if not grp:
            continue
        lines.append(f"    # {sector} ({len(grp)})")
        for t in grp:
            lines.append(f'    "{t}": "{sector}",')

    lines += [
        '}',
        '',
        'TICKERS: list[str] = list(SECTOR_MAP.keys())',
        '',
        'SECTOR_PROXIES: dict[str, str] = {',
    ]
    for s, proxy in SECTOR_PROXIES.items():
        lines.append(f'    "{s}": "{proxy}",')
    lines += [
        '}',
        '',
        '',
        'def compute_sector_features(db):  # type: ignore[override]',
        '    """Placeholder -- sector relative features deferred to M9 data pipeline.',
        '    Returns db unchanged until sector feature engineering is implemented.',
        '    """',
        '    return db',
    ]

    content = "\n".join(lines) + "\n"
    OUTPUT_PATH.write_text(content, encoding="utf-8")
    print(f"\nWrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Total tickers: {len(tickers)}")
    for s in ["Index", "Tech", "Finance", "Health", "Consumer", "Industrial", "Energy"]:
        print(f"  {s}: {dist.get(s, 0)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    all_tickers = list(SECTOR_MAP.keys())
    print(f"Current universe: {len(all_tickers)} tickers")

    # Don't depth-check index proxies — they predate 2010 trivially
    to_check = [t for t in all_tickers if t not in ALWAYS_KEEP]

    valid = check_2010_depth(to_check)
    valid |= ALWAYS_KEEP

    removed = sorted(set(all_tickers) - valid)
    kept = {t: SECTOR_MAP[t] for t in all_tickers if t in valid}

    print(f"\n{'─' * 60}")
    print(f"Removing {len(removed)} tickers (no 2010 data):")
    by_sector_removed: dict[str, list[str]] = {}
    for t in removed:
        by_sector_removed.setdefault(SECTOR_MAP[t], []).append(t)
    for s in ["Tech", "Finance", "Health", "Consumer", "Industrial", "Energy"]:
        grp = by_sector_removed.get(s, [])
        if grp:
            print(f"  {s}: {', '.join(sorted(grp))}")

    print(f"\nFiltered: {len(all_tickers)} → {len(kept)} tickers")
    write_sector_py(kept)
    print("\nDone. Now run: py -3.12 prepare.py")


if __name__ == "__main__":
    main()
