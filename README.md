# Financial Research Workflow — Phase 1

Autonomous AI-driven financial prediction system using Conv1D + LSTM with Monte Carlo Dropout.

## Quick Setup (Windows)

### 1. Install Python 3.10+
If you don't have Python installed:
- Download from https://www.python.org/downloads/
- **IMPORTANT**: Check "Add Python to PATH" during installation
- Verify: open Command Prompt and run `python --version`

### 2. Create project environment
```cmd
cd financial-research
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Download data and compute features
```cmd
python prepare.py
```
This will:
- Download 10+ years of stock data for 10 tickers (no API key needed)
- Compute 38 technical indicators per ticker
- Create temporal train/test split (train ≤ 2023, val = 2024, test = 2025+)
- Save prepared data to `data/prepared_data.npz`

### 4. Run the baseline model
```cmd
python strategy.py
```
This will:
- Build the Conv1D + LSTM + MCDropout model
- Train on historical data with EarlyStopping
- Run 50 Monte Carlo forward passes on validation set
- Generate BUY/HOLD/SELL signals with 70% confidence threshold
- Save results to `results/results.tsv`
- Save model to `models/baseline_v1.keras`

## Project Structure

```
financial-research/
├── prepare.py         # Data pipeline & evaluation (DO NOT MODIFY)
├── strategy.py        # Model architecture (AGENT MODIFIES THIS)
├── program.md         # Research directives (HUMAN MODIFIES THIS)
├── requirements.txt   # Python dependencies
├── README.md          # This file
├── data/              # Downloaded data & prepared datasets
│   ├── SPY.csv        # Cached stock data
│   ├── ...
│   └── prepared_data.npz  # Windowed features ready for training
├── models/            # Saved models & configs
│   ├── baseline_v1.keras
│   ├── baseline_v1_config.json
│   └── scaler_*.pkl   # Per-ticker feature scalers
└── results/
    └── results.tsv    # Experiment log for autoresearch tracking
```

## Three-File Architecture (from Karpathy's autoresearch)

| File | Who modifies | Purpose |
|------|-------------|---------|
| `prepare.py` | Human only | Data pipeline, feature engineering, evaluation metrics |
| `strategy.py` | AI agent | Model architecture, hyperparameters, signal logic |
| `program.md` | Human only | Research directives and constraints for the agent |

## What's Next

After running the baseline:
1. Review `results/results.tsv` to see baseline metrics
2. Use Claude Code or Cowork to iterate on `strategy.py`
3. Each change → run experiment → check if metrics improved → keep or revert
4. Phase 2 will add the autoresearch loop to automate this overnight

## Data Sources

- **yfinance**: US stock data (no API key)
- **CoinGecko**: Crypto data (no API key, add in Phase 2)
- **Polymarket CLOB API**: Prediction markets (add in Phase 2)

## Key References

- Karpathy autoresearch: github.com/karpathy/autoresearch
- Noisy prediction model: Conv1D + LSTM + MCDropout + 38 indicators
- Nunchi agent-cli: github.com/Nunchi-trade/agent-cli
