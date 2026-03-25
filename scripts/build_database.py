"""
build_database.py — One-time setup: build data/processed/ parquet files.

Uses cached CSVs from data/raw/ (no network download needed if already present).
Produces:
    data/processed/full_db.parquet
    data/processed/train_db.parquet
    data/processed/val_db.parquet
    data/processed/test_db.parquet

Usage:
    python scripts/build_database.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pattern_engine.data import DataLoader

print("Building FPPE database from cached CSVs in data/raw/ ...")
print("(Download step is skipped — existing CSVs will be reused)\n")

loader = DataLoader(data_dir=str(REPO_ROOT / "data"))
full_db, train_db, val_db, test_db = loader.build_database()

print(f"\nDone. Parquets saved to: {REPO_ROOT / 'data' / 'processed'}/")
print(f"  full_db:  {len(full_db):>8,} rows")
print(f"  train_db: {len(train_db):>8,} rows")
print(f"  val_db:   {len(val_db):>8,} rows")
print(f"  test_db:  {len(test_db):>8,} rows")
