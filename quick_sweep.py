"""
quick_sweep.py — Time-Budgeted Hyperparameter Sweep
Designed for 15-30 minute research sessions on Ryzen 9 5900X.

HOW IT WORKS:
  1. Defines a grid of experiments targeting the most impactful parameters
  2. Runs each experiment with reduced epochs/MC samples to fit the time budget
  3. Logs everything to results/results.tsv (same format as strategy.py)
  4. Prints a ranked leaderboard at the end
  5. Tells you exactly which config won and what to set in strategy.py

USAGE:
  python quick_sweep.py              # Run the default 30-minute sweep
  python quick_sweep.py --budget 15  # Run a 15-minute sweep (fewer experiments)

WHAT IT SWEEPS:
  Based on your results so far, the highest-impact parameters are:
  - Architecture (TRANSFORMER gets 0 trades, CONV_LSTM/DENSE_STATS work)
  - Confidence threshold (0.53 vs 0.55 vs 0.58 — huge impact on trade count)
  - Learning rate (1e-3 vs 5e-4 vs 3e-4)
  - Dropout rate (0.25 vs 0.30 vs 0.35)
  
  Lower-impact (saved for longer sessions):
  - Conv filters, LSTM units, dense units, kernel size
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from pathlib import Path
from datetime import datetime, timedelta
import json

# Force CPU, suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Optimize for Ryzen 9
tf.config.threading.set_intra_op_parallelism_threads(12)
tf.config.threading.set_inter_op_parallelism_threads(12)


# ============================================================
# SWEEP CONFIGURATION
# ============================================================

# Time budget in minutes (override with --budget flag)
TIME_BUDGET_MINUTES = 30

# Quick-run settings (reduced from strategy.py for speed)
QUICK_EPOCHS = 40              # Enough to converge, not enough to overfit
QUICK_PATIENCE = 12            # Stop early if stuck
QUICK_MC_SAMPLES = 30          # Fewer MC passes (still meaningful uncertainty)
QUICK_BATCH_SIZE = 256         # Large batch for CPU throughput

# Paths
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
MODEL_DIR = Path("models")


# ============================================================
# SHARED COMPONENTS (copied from strategy.py to stay self-contained)
# ============================================================

class MCDropout(layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


def build_transformer(ws, nf, filters=128, heads=4, blocks=2, dense=64, dropout=0.3):
    """Lighter transformer for quick sweeps — fewer blocks by default."""
    inputs = keras.Input(shape=(ws, nf))
    x = layers.Conv1D(filters, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    for _ in range(blocks):
        attn = layers.MultiHeadAttention(num_heads=heads, key_dim=filters // heads)(x, x)
        attn = MCDropout(dropout)(attn)
        x = layers.Add()([x, attn])
        x = layers.LayerNormalization()(x)

        ff = layers.Dense(filters * 2, activation="relu")(x)
        ff = layers.Dense(filters)(ff)
        ff = MCDropout(dropout)(ff)
        x = layers.Add()([x, ff])
        x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(dense, activation="relu")(x)
    x = MCDropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return keras.Model(inputs, outputs)


def build_conv_lstm(ws, nf, filters=128, lstm_units=64, dense=64, dropout=0.3):
    return keras.Sequential([
        layers.Conv1D(filters, kernel_size=3, activation="relu", padding="same", input_shape=(ws, nf)),
        layers.BatchNormalization(),
        MCDropout(dropout),
        layers.LSTM(lstm_units, return_sequences=False),
        MCDropout(dropout),
        layers.Dense(dense, activation="relu"),
        MCDropout(dropout),
        layers.Dense(1, activation="sigmoid", dtype="float32"),
    ])


def build_dense_stats(ws, nf, wide1=1024, wide2=512, dropout=0.4):
    inputs = keras.Input(shape=(ws, nf))
    x = layers.Flatten()(inputs)
    x = layers.Dense(wide1, activation="relu")(x)
    x = MCDropout(dropout)(x)
    x = layers.Dense(wide2, activation="relu")(x)
    x = MCDropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)


def mc_predict(model, X, n_samples):
    preds = np.array([model(X, training=True).numpy().flatten() for _ in range(n_samples)])
    return preds.mean(axis=0), preds.std(axis=0)


def count_signals(mean, std, threshold, max_std=0.20):
    buys, sells, holds = 0, 0, 0
    for m, s in zip(mean, std):
        if s > max_std:
            holds += 1
        elif m >= threshold:
            buys += 1
        elif m <= (1 - threshold):
            sells += 1
        else:
            holds += 1
    return buys, sells, holds


# ============================================================
# SINGLE QUICK EXPERIMENT
# ============================================================

def run_quick(config, X_train, y_train, X_val, y_val):
    """
    Run one experiment with time-optimized settings.
    Returns metrics dict with all relevant info.
    """
    name = config["name"]
    arch = config.get("architecture", "CONV_LSTM")
    lr = config.get("lr", 0.001)
    dropout = config.get("dropout", 0.30)
    threshold = config.get("threshold", 0.55)
    max_std = config.get("max_std", 0.20)
    filters = config.get("filters", 128)
    use_focal = config.get("focal_loss", True)

    ws, nf = X_train.shape[1], X_train.shape[2]

    start = time.time()

    # Build model
    if arch == "TRANSFORMER":
        blocks = config.get("blocks", 2)
        model = build_transformer(ws, nf, filters=filters, blocks=blocks, dropout=dropout)
    elif arch == "CONV_LSTM":
        lstm = config.get("lstm_units", 64)
        model = build_conv_lstm(ws, nf, filters=filters, lstm_units=lstm, dropout=dropout)
    elif arch == "DENSE_STATS":
        model = build_dense_stats(ws, nf, dropout=dropout)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    # Compile
    if use_focal:
        loss = keras.losses.BinaryFocalCrossentropy(gamma=2.0, alpha=0.25)
    else:
        loss = "binary_crossentropy"
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss, metrics=["accuracy"])

    # Train (quick settings)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=QUICK_EPOCHS,
        batch_size=QUICK_BATCH_SIZE,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_loss", patience=QUICK_PATIENCE, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7),
        ],
        verbose=0,
    )

    # MC predict
    val_mean, val_std = mc_predict(model, X_val, n_samples=QUICK_MC_SAMPLES)
    buys, sells, holds = count_signals(val_mean, val_std, threshold, max_std)

    # Evaluate
    from prepare import evaluate_predictions, save_results
    metrics = evaluate_predictions(y_val, val_mean, threshold=threshold)

    train_time = time.time() - start

    # Enrich
    metrics.update({
        "experiment_name": name,
        "architecture": arch,
        "learning_rate": lr,
        "dropout_rate": dropout,
        "confidence_threshold": threshold,
        "max_uncertainty": max_std,
        "conv_filters": filters,
        "use_focal_loss": use_focal,
        "epochs_trained": len(history.history["loss"]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
        "buy_signals": buys,
        "sell_signals": sells,
        "hold_signals": holds,
        "train_time_sec": round(train_time, 1),
    })

    save_results(metrics, name)
    return metrics


# ============================================================
# SWEEP DEFINITIONS
# ============================================================

def get_sweep_configs(budget_minutes=30):
    """
    Generate experiment configs based on time budget.
    
    30 min budget: ~8-10 experiments (3 min each)
    15 min budget: ~4-5 experiments (3 min each)
    
    Prioritizes the highest-impact parameters first:
    1. Architecture (what's fundamentally broken vs working)
    2. Threshold (biggest impact on trade count)
    3. Learning rate (training stability)
    4. Dropout (overfitting control)
    """
    configs = []

    # --- TIER 1: Architecture + threshold sweep (always run) ---
    # CONV_LSTM was the best performer — test threshold sensitivity
    configs.append({"name": "sweep_lstm_t50", "architecture": "CONV_LSTM", "threshold": 0.50, "lr": 5e-4, "dropout": 0.30})
    configs.append({"name": "sweep_lstm_t53", "architecture": "CONV_LSTM", "threshold": 0.53, "lr": 5e-4, "dropout": 0.30})
    configs.append({"name": "sweep_lstm_t55", "architecture": "CONV_LSTM", "threshold": 0.55, "lr": 5e-4, "dropout": 0.30})
    configs.append({"name": "sweep_lstm_t58", "architecture": "CONV_LSTM", "threshold": 0.58, "lr": 5e-4, "dropout": 0.30})

    if budget_minutes < 20:
        return configs  # 15-min budget: just these 4

    # --- TIER 2: Learning rate comparison on best arch ---
    configs.append({"name": "sweep_lstm_lr1e3", "architecture": "CONV_LSTM", "threshold": 0.53, "lr": 1e-3, "dropout": 0.30})
    configs.append({"name": "sweep_lstm_lr3e4", "architecture": "CONV_LSTM", "threshold": 0.53, "lr": 3e-4, "dropout": 0.30})

    if budget_minutes < 25:
        return configs  # 20-min budget: 6 experiments

    # --- TIER 3: TRANSFORMER fix attempt (lower threshold + fewer blocks) ---
    # Your TRANSFORMER never produces trades — try much lower threshold and simpler arch
    configs.append({"name": "sweep_tfr_t50_b1", "architecture": "TRANSFORMER", "threshold": 0.50, "lr": 5e-4, "blocks": 1, "dropout": 0.20})
    configs.append({"name": "sweep_tfr_t50_b2", "architecture": "TRANSFORMER", "threshold": 0.50, "lr": 5e-4, "blocks": 2, "dropout": 0.25})

    # --- TIER 4: DENSE_STATS with tighter threshold ---
    configs.append({"name": "sweep_dense_t55", "architecture": "DENSE_STATS", "threshold": 0.55, "lr": 5e-4, "dropout": 0.35})
    configs.append({"name": "sweep_dense_t58", "architecture": "DENSE_STATS", "threshold": 0.58, "lr": 5e-4, "dropout": 0.35})

    return configs


# ============================================================
# MAIN SWEEP RUNNER
# ============================================================

def run_sweep(budget_minutes=TIME_BUDGET_MINUTES):
    """
    Run a time-budgeted hyperparameter sweep.
    Stops adding experiments if the time budget is exceeded.
    Prints a ranked leaderboard at the end.
    """
    configs = get_sweep_configs(budget_minutes)

    print(f"\n{'='*65}")
    print(f"  QUICK SWEEP — {len(configs)} experiments, {budget_minutes} min budget")
    print(f"  Settings: {QUICK_EPOCHS} epochs, {QUICK_MC_SAMPLES} MC samples, batch {QUICK_BATCH_SIZE}")
    print(f"{'='*65}")

    # Load data once
    print("\n  Loading data...")
    data = np.load(DATA_DIR / "prepared_data.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    print(f"  Train: {X_train.shape}, Val: {X_val.shape}")

    session_start = time.time()
    deadline = session_start + (budget_minutes * 60)
    all_results = []

    for i, config in enumerate(configs):
        elapsed = time.time() - session_start
        remaining = deadline - time.time()

        if remaining < 120:  # Less than 2 min left, stop
            print(f"\n  Time budget reached ({elapsed/60:.1f} min). Stopping after {i} experiments.")
            break

        print(f"\n  [{i+1}/{len(configs)}] {config['name']} | "
              f"{config['architecture']} | threshold={config.get('threshold', 0.55)} | "
              f"lr={config.get('lr', 0.001)} | "
              f"Remaining: {remaining/60:.0f} min")

        try:
            metrics = run_quick(config, X_train, y_train, X_val, y_val)
            all_results.append(metrics)

            # Quick inline result
            print(f"    -> Acc(all): {metrics['accuracy_all']:.1%} | "
                  f"Trades: {metrics['confident_trades']} | "
                  f"Acc(conf): {metrics['accuracy_confident']:.1%} | "
                  f"Time: {metrics['train_time_sec']:.0f}s")

        except Exception as e:
            print(f"    -> FAILED: {e}")

        # Clear memory between runs
        keras.backend.clear_session()

    total_time = time.time() - session_start

    # ============================================================
    # LEADERBOARD
    # ============================================================
    print(f"\n{'='*65}")
    print(f"  LEADERBOARD — {len(all_results)} experiments in {total_time/60:.1f} min")
    print(f"{'='*65}\n")

    if not all_results:
        print("  No results to show.")
        return

    # Sort by accuracy_confident (primary), then by confident_trades (tiebreaker)
    ranked = sorted(all_results, key=lambda r: (
        r["confident_trades"] >= 20,     # Must have meaningful trade count
        r["accuracy_confident"],          # Primary metric
        r["confident_trades"],            # More trades = better tiebreak
    ), reverse=True)

    print(f"  {'Rank':<5} {'Name':<25} {'Arch':<13} {'Thr':>5} {'Trades':>7} {'Acc(conf)':>10} {'Acc(all)':>9} {'Time':>6}")
    print(f"  {'~'*80}")

    for rank, r in enumerate(ranked, 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"  {rank:<5} {r['experiment_name']:<25} {r['architecture']:<13} "
              f"{r['confidence_threshold']:>5.2f} {r['confident_trades']:>7} "
              f"{r['accuracy_confident']:>9.1%} {r['accuracy_all']:>9.1%} "
              f"{r['train_time_sec']:>5.0f}s{marker}")

    # Winner summary
    best = ranked[0]
    print(f"\n  {'='*65}")
    print(f"  WINNER: {best['experiment_name']}")
    print(f"  Architecture: {best['architecture']}")
    print(f"  Accuracy (confident): {best['accuracy_confident']:.1%} on {best['confident_trades']} trades")
    print(f"  Accuracy (all): {best['accuracy_all']:.1%}")
    print(f"  {'='*65}")

    print(f"\n  To apply this config to strategy.py, set:")
    print(f"    ARCHITECTURE = \"{best['architecture']}\"")
    print(f"    CONFIDENCE_THRESHOLD = {best['confidence_threshold']}")
    print(f"    LEARNING_RATE = {best['learning_rate']}")
    print(f"    DROPOUT_RATE = {best['dropout_rate']}")
    if best.get("conv_filters"):
        print(f"    CONV_FILTERS = {best['conv_filters']}")

    return ranked


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    # Parse --budget flag
    budget = TIME_BUDGET_MINUTES
    if "--budget" in sys.argv:
        idx = sys.argv.index("--budget")
        if idx + 1 < len(sys.argv):
            budget = int(sys.argv[idx + 1])

    run_sweep(budget_minutes=budget)
