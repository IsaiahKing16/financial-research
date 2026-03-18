# Financial Research — Autoresearch Program

## Your Role

You are an autonomous research agent optimizing a financial prediction model. Your goal is to achieve the highest possible **accuracy on confident trades** (predictions above the confidence threshold) on the validation set, while maintaining a meaningful number of trades (not just cherry-picking 3 easy ones).

## How This Works

1. You modify `strategy.py` — the model architecture, hyperparameters, and signal generation logic
2. You run an experiment: `python strategy.py`
3. You check the results in `results/results.tsv`
4. If accuracy_confident improved AND confident_trades > 50, **keep** the change
5. If not, **revert** and try something different
6. Repeat

## Files

- `prepare.py` — **DO NOT MODIFY.** Fixed data pipeline and evaluation.
- `strategy.py` — **YOU MODIFY THIS.** Model architecture, hyperparameters, training config.
- `program.md` — **DO NOT MODIFY.** Your instructions (this file).
- `results/results.tsv` — Experiment log. Check this to see progress.

## What You Can Change in strategy.py

### Architecture
- Number of Conv1D filters, kernel sizes
- LSTM units, add/remove layers
- Dense layer sizes
- Dropout rates
- Try adding attention mechanisms
- Try replacing LSTM with GRU or Transformer blocks
- Try bidirectional LSTM

### Hyperparameters
- Learning rate (try 1e-4 to 1e-2)
- Batch size (32, 64, 128, 256)
- Epochs / patience settings
- MC Dropout sample count

### Signal Generation
- Confidence threshold (try 0.60 to 0.85)
- Experiment with using std (uncertainty) as a filter
- Try requiring both high confidence AND low uncertainty

### Training
- Loss function (binary_crossentropy, focal loss)
- Optimizer (Adam, AdamW, SGD with momentum)
- Learning rate schedules
- Class weights if imbalanced

## Constraints

- Each experiment must complete in under 15 minutes
- Do not modify prepare.py
- Always use the temporal train/test split (no data leakage)
- Log every experiment to results.tsv via the save_results function
- Keep strategy.py as a single file (no imports from custom modules)

## Primary Metric

**accuracy_confident** — accuracy on trades where the model's confidence exceeds the threshold.

## Secondary Metrics (track but don't optimize solely for these)

- confident_trades — number of trades the model would make (target: 50+ per validation period)
- f1_confident — balance of precision and recall on confident trades
- final_val_loss — should generally decrease

## Current Baseline

Run `python strategy.py` to establish the baseline. Then start experimenting.

## Research Tips

- Documentation is high importance

- CPU used is a Ryzen 9 5900X 12 Core 3.70 GHz CPU Processor
- There is 32GB of DDR4 RAM
- If a change helps, keep it and build on it
- If stuck, try a completely different architecture (e.g., replace Conv1D+LSTM with pure Transformer)
- The confidence threshold has massive impact — a few percentage points can swing results significantly
- More MC samples (100 instead of 50) gives more stable uncertainty estimates but takes longer
