# FPPE — Financial Pattern Prediction Engine

K-NN historical analogue matching for probabilistic equity prediction, with walk-forward validation and Bayesian hyperparameter optimization.

## Architecture

FPPE uses a modular `pattern_engine/` package that replaces the earlier Conv1D/LSTM approach. The core idea: find historically similar market conditions via K-nearest neighbors, then aggregate their forward returns into calibrated probability forecasts.

```
pattern_engine/
├── config.py          # EngineConfig frozen dataclass (proven defaults)
├── engine.py          # PatternEngine: fit → predict → evaluate
├── matching.py        # K-NN analogue matching with regime filtering
├── calibration.py     # Platt scaling / isotonic calibration
├── evaluation.py      # Brier Score, BSS, CRPS, calibration curves
├── features.py        # FeatureRegistry with pluggable feature sets
├── data.py            # DataLoader: download → features → temporal split
├── walkforward.py     # Expanding-window walk-forward validation
├── sweep.py           # Grid + Bayesian (Optuna TPE) parameter sweeps
├── overnight.py       # Multi-phase overnight runner with crash isolation
├── reliability.py     # Atomic writes, lock files, progress logging
├── experiment_logging.py  # TSV experiment logger with provenance tracking
├── regime.py          # Market regime labeling (binary/multi/octet)
├── projection.py      # Signal generation (BUY/SELL/HOLD)
├── scoring.py         # Proper scoring rules (Brier, CRPS)
├── schema.py          # DataFrame schema validation
├── sector.py          # Sector features and ticker universe
├── candlestick.py     # Candlestick pattern encoding
├── cross_validation.py # Multi-model cross-validation
└── live.py            # Live prediction runner
```

## Quick Start

```bash
# Setup
python -m venv venv && venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Build database (downloads data, computes features)
python -c "from pattern_engine.data import DataLoader; DataLoader().build_database()"

# Run walk-forward validation with default config
python -c "
from pattern_engine.config import EngineConfig
from pattern_engine.walkforward import WalkForwardRunner
import pandas as pd

db = pd.read_parquet('data/processed/full_db.parquet')
runner = WalkForwardRunner(EngineConfig())
runner.run(db)
"

# Run overnight Bayesian optimization
python -c "
from pattern_engine.overnight import OvernightRunner
import pandas as pd

db = pd.read_parquet('data/processed/full_db.parquet')
runner = OvernightRunner(bayesian_mode=True, n_trials=50, max_hours=6)
runner.run(db)
"
```

## Key Metrics

| Metric | Purpose |
|--------|---------|
| **Brier Skill Score (BSS)** | Primary metric — positive = beats base rate |
| **CRPS** | Proper scoring rule for full probability distribution |
| **Accuracy (confident)** | Classification accuracy on BUY/SELL signals only |

## Experiment Tracking

All results are logged to `data/results/experiments.tsv` with full config provenance. Each row includes:
- Walk-forward fold label and config hash
- All evaluation metrics (BSS, accuracy, F1, CRPS)
- Complete EngineConfig field values for reproducibility

## Walk-Forward Validation

6 expanding-window folds (2019-2024), each training on all prior data:
- 2019, 2020 (COVID), 2021, 2022 (Bear), 2023, 2024 (Standard Val)

## Project Structure

```
financial-research/
├── pattern_engine/    # Active modular package (v2.1)
├── tests/             # 242+ pytest tests
├── data/
│   ├── raw/           # Cached OHLCV data per ticker
│   ├── processed/     # Parquet databases (full, train, val, test)
│   └── results/       # Experiment TSVs, checkpoints, logs
├── archive/           # Legacy scripts (superseded by pattern_engine/)
├── PROJECT_GUIDE.md   # Detailed research documentation
├── CLAUDE.md          # Agent instructions and conventions
└── requirements.txt   # Python dependencies
```

## Legacy Files

The root-level `strategy*.py`, `prepare.py`, and related scripts are from the original Phase 1 Conv1D/LSTM approach. They have been moved to `archive/` and are superseded by the `pattern_engine/` package.

## Key References

- Fed NY Staff Report 917: Overnight drift phenomenon (~3.7% annualized)
- Karpathy autoresearch: Three-file architecture pattern
- 24/7 Equities (Hyperliquid): Session decomposition features
