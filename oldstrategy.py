"""
strategy.py — Model architecture, training, and signal generation.
THIS FILE IS MODIFIED BY THE AUTORESEARCH AGENT.

Architecture Options:
  1. CONV_LSTM  — Conv1D + LSTM + MCDropout (original baseline from Noisy)
  2. TRANSFORMER — Conv1D + MultiHeadAttention + MCDropout (from Gemini iteration)

Features:
  - Mixed precision (float16) for GPU acceleration (RTX 3060 etc.)
  - Binary Focal Loss to push confidence toward edges
  - Monte Carlo Dropout for uncertainty estimation
  - Parallel experiment runner for batch hyperparameter search
  - Results logged to results/results.tsv for autoresearch tracking

The autoresearch agent iterates on:
  - Architecture choice and layer configuration
  - Hyperparameters (learning rate, dropout, batch size, filters)
  - Confidence threshold and MC sample count
  - Loss function (focal loss gamma/alpha)
  - Feature selection
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras import mixed_precision
from pathlib import Path
import json
import multiprocessing as mp
from datetime import datetime

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ============================================================
# GPU & HARDWARE CONFIGURATION
# ============================================================

def configure_gpu(memory_limit=None):
    """
    Configure GPU for optimal performance.
    
    Args:
        memory_limit: Max GPU memory in MB per model (e.g., 3000 for 3GB).
                      None = use memory growth (expand as needed).
                      Set a limit when running parallel experiments.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            if memory_limit:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                )
            else:
                tf.config.experimental.set_memory_growth(gpu, True)
        # Mixed precision: float16 compute, float32 storage — ~2x speedup on RTX cards
        mixed_precision.set_global_policy("mixed_float16")
        print(f"  GPU enabled: {gpus[0].name} | Mixed precision: float16")
    else:
        print("  No GPU detected — running on CPU")


# ============================================================
# HYPERPARAMETERS — autoresearch modifies these
# ============================================================

# Architecture selection
ARCHITECTURE = "TRANSFORMER"   # Options: "CONV_LSTM" or "TRANSFORMER"

# Shared architecture params
CONV_FILTERS = 64              # Conv1D filter count
CONV_KERNEL = 3                # Conv1D kernel size
DENSE_UNITS = 32               # Final dense layer units
DROPOUT_RATE = 0.30            # MCDropout rate (higher = more regularization)

# CONV_LSTM-specific
LSTM_UNITS = 64                # LSTM hidden units

# TRANSFORMER-specific
NUM_HEADS = 4                  # Multi-head attention heads
                               # key_dim is auto-computed: CONV_FILTERS // NUM_HEADS

# Training
LEARNING_RATE = 0.001
BATCH_SIZE = 64               # Larger batch for GPU efficiency (reduce to 64 for CPU)
EPOCHS = 250                   # Max epochs (early stopping will cut this short)
PATIENCE_EARLY_STOP = 50       # Stop if no improvement for N epochs
PATIENCE_LR_REDUCE = 25        # Reduce LR if no improvement for N epochs
MIN_LR = 1e-7

# Loss function
USE_FOCAL_LOSS = True          # True = BinaryFocalCrossentropy, False = binary_crossentropy
FOCAL_GAMMA = 2.0              # Focal loss focusing parameter (higher = more focus on hard examples)
FOCAL_ALPHA = 0.25             # Focal loss class balance weight

# Signal generation
CONFIDENCE_THRESHOLD = 0.55    # Only trade when confidence >= threshold
MAX_UNCERTAINTY = 0.08
MC_SAMPLES = 150               # Monte Carlo Dropout forward passes

# Paths
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")


# ============================================================
# MONTE CARLO DROPOUT — keeps dropout active during inference
# ============================================================

class MCDropout(layers.Dropout):
    """
    Dropout that stays active during inference for uncertainty estimation.
    Standard Dropout turns off during prediction — MCDropout stays on,
    allowing us to run the model N times and measure how much predictions vary.
    """
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

def build_conv_lstm(window_size, n_features):
    """
    Original baseline: Conv1D + LSTM + MCDropout.
    From Noisy's prediction market model.
    
    Conv1D:  finds local patterns in recent price action
    LSTM:    remembers long-term dependencies across the 60-day window
    Sigmoid: outputs probability 0-1 (price goes up or down)
    """
    model = keras.Sequential([
        layers.Conv1D(
            CONV_FILTERS, kernel_size=CONV_KERNEL,
            activation="relu", padding="same",
            input_shape=(window_size, n_features),
        ),
        layers.BatchNormalization(),
        MCDropout(DROPOUT_RATE),

        layers.LSTM(LSTM_UNITS, return_sequences=False),
        MCDropout(DROPOUT_RATE),

        layers.Dense(DENSE_UNITS, activation="relu"),
        MCDropout(DROPOUT_RATE),

        # Output in float32 for numerical stability with mixed precision
        layers.Dense(1, activation="sigmoid", dtype="float32"),
    ])
    return model


def build_transformer(window_size, n_features):
    """
    Transformer variant: Conv1D + MultiHeadAttention + MCDropout.
    From Gemini iteration.
    
    Conv1D:              extracts local features from each timestep
    MultiHeadAttention:  lets the model focus on specific high-impact days
                         in the 60-day window (e.g., earnings day, Fed meeting)
    Residual + LayerNorm: stabilizes training with attention
    GlobalAvgPooling:    reduces sequence to single vector for classification
    """
    inputs = keras.Input(shape=(window_size, n_features))

    # Local feature extraction
    x = layers.Conv1D(
        CONV_FILTERS, kernel_size=CONV_KERNEL,
        padding="same", activation="relu"
    )(inputs)
    x = layers.BatchNormalization()(x)

    # Transformer block — self-attention over the time window
    # Each "head" can attend to different patterns (momentum, volume spikes, etc.)
    attention_out = layers.MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=CONV_FILTERS // NUM_HEADS,
    )(x, x)
    x = layers.Add()([x, attention_out])   # Residual connection (prevents degradation)
    x = layers.LayerNormalization()(x)

    # Reduce sequence → single vector
    x = layers.GlobalAveragePooling1D()(x)

    # Dense head with uncertainty estimation
    x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = MCDropout(DROPOUT_RATE)(x)

    # Output in float32 for stability with mixed precision
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)

    return keras.Model(inputs, outputs)


def build_model(window_size, n_features):
    """
    Build the selected architecture and compile with chosen loss function.
    
    Architecture is selected by ARCHITECTURE constant.
    Loss is selected by USE_FOCAL_LOSS constant.
    """
    if ARCHITECTURE == "TRANSFORMER":
        model = build_transformer(window_size, n_features)
    elif ARCHITECTURE == "CONV_LSTM":
        model = build_conv_lstm(window_size, n_features)
    else:
        raise ValueError(f"Unknown architecture: {ARCHITECTURE}. Use 'CONV_LSTM' or 'TRANSFORMER'.")

    # Loss function selection
    if USE_FOCAL_LOSS:
        # Focal loss pushes predictions toward 0 or 1 (more confident outputs)
        # gamma=2.0: down-weights easy examples, focuses on hard-to-classify days
        # alpha=0.25: slight class rebalancing
        loss = keras.losses.BinaryFocalCrossentropy(
            gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA
        )
    else:
        loss = "binary_crossentropy"

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=loss,
        metrics=["accuracy"],
    )

    return model


# ============================================================
# TRAINING
# ============================================================

def train_model(model, X_train, y_train, X_val, y_val, verbose=1):
    """
    Train with EarlyStopping and ReduceLROnPlateau.
    
    EarlyStopping:      stops training when val_loss stops improving
    ReduceLROnPlateau:  halves learning rate when stuck on a plateau
    restore_best_weights: reverts to the best epoch after stopping
    """
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=1e-5,
            patience=PATIENCE_EARLY_STOP,
            restore_best_weights=True,
            verbose=verbose,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            min_lr=MIN_LR,
            patience=PATIENCE_LR_REDUCE,
            verbose=verbose,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=verbose,
    )

    return history


# ============================================================
# MONTE CARLO PREDICTION — the key innovation
# ============================================================

def mc_predict(model, X, n_samples=MC_SAMPLES):
    """
    Run model N times with dropout active.
    Returns mean prediction (confidence) and std (uncertainty).
    
    "Think of it like asking 100 analysts at once.
     If all 100 say BUY, we enter.
     If half say BUY and half say HOLD — we skip.
     We only trade when there's consensus."
    """
    predictions = np.array([
        model(X, training=True).numpy().flatten()
        for _ in range(n_samples)
    ])

    mean = predictions.mean(axis=0)   # Confidence (0 to 1)
    std = predictions.std(axis=0)     # Uncertainty (lower = more agreement)

    return mean, std


def generate_signals(mean, std, threshold=CONFIDENCE_THRESHOLD):
    """
    Generate BUY/SELL/HOLD signals based on confidence threshold.
    
    BUY:  model confidence >= threshold  (predicts UP with high certainty)
    SELL: model confidence <= (1 - threshold)  (predicts DOWN with high certainty)
    HOLD: everything else  (model is uncertain, skip this trade)
    
    "We don't trade every day on every market.
     We wait for strong signals only.
     3 confident trades a week beats 20 uncertain ones."
    """
    signals = []
    for m, s in zip(mean, std):

        # Skip predictions with high uncertainty
        if s > MAX_UNCERTAINTY:
            signals.append("HOLD")

        elif m >= threshold:
            signals.append("BUY")

        elif m <= (1 - threshold):
            signals.append("SELL")

        else:
            signals.append("HOLD")

    return np.array(signals)


# ============================================================
# SINGLE EXPERIMENT — full training + evaluation pipeline
# ============================================================

def run_experiment(experiment_name=None, verbose=1):
    """
    Run a complete experiment: load data → build → train → MC predict → evaluate → save.
    This is what the autoresearch loop calls repeatedly.
    
    Args:
        experiment_name: label for this experiment in results.tsv
        verbose: 1 = show training progress, 0 = silent (for parallel runs)
    """
    if experiment_name is None:
        experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if verbose:
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {experiment_name}")
        print(f"  Architecture: {ARCHITECTURE} | Focal Loss: {USE_FOCAL_LOSS}")
        print(f"  Filters: {CONV_FILTERS} | Dense: {DENSE_UNITS} | Dropout: {DROPOUT_RATE}")
        print(f"  LR: {LEARNING_RATE} | Batch: {BATCH_SIZE} | Threshold: {CONFIDENCE_THRESHOLD}")
        print(f"{'='*60}")

    # --- Load prepared data ---
    if verbose: print("\n[1/5] Loading prepared data...")
    data = np.load(DATA_DIR / "prepared_data.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    window_size = X_train.shape[1]
    n_features = X_train.shape[2]

    if verbose: print(f"  Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # --- Build model ---
    if verbose: print("\n[2/5] Building model...")
    model = build_model(window_size, n_features)
    if verbose: model.summary()

    # --- Train ---
    if verbose: print("\n[3/5] Training...")
    history = train_model(model, X_train, y_train, X_val, y_val, verbose=verbose)

    # --- Monte Carlo prediction on validation set ---
    if verbose: print("\n[4/5] Running Monte Carlo predictions (validation)...")
    val_mean, val_std = mc_predict(model, X_val)
    val_signals = generate_signals(val_mean, val_std)

    buy_count = (val_signals == "BUY").sum()
    sell_count = (val_signals == "SELL").sum()
    hold_count = (val_signals == "HOLD").sum()
    if verbose: print(f"  Signals: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}")

    # --- Evaluate ---
    if verbose: print("\n[5/5] Evaluating...")

    from prepare import evaluate_predictions, print_metrics, save_results

    metrics = evaluate_predictions(y_val, val_mean, threshold=CONFIDENCE_THRESHOLD)
    if verbose: print_metrics(metrics, label=f"Validation | {ARCHITECTURE} | threshold={CONFIDENCE_THRESHOLD}")

    # Add experiment metadata for autoresearch tracking
    metrics["experiment_name"] = experiment_name
    metrics["architecture"] = ARCHITECTURE
    metrics["conv_filters"] = CONV_FILTERS
    metrics["conv_kernel"] = CONV_KERNEL
    metrics["dense_units"] = DENSE_UNITS
    metrics["dropout_rate"] = DROPOUT_RATE
    metrics["learning_rate"] = LEARNING_RATE
    metrics["batch_size"] = BATCH_SIZE
    metrics["confidence_threshold"] = CONFIDENCE_THRESHOLD
    metrics["mc_samples"] = MC_SAMPLES
    metrics["use_focal_loss"] = USE_FOCAL_LOSS
    metrics["focal_gamma"] = FOCAL_GAMMA if USE_FOCAL_LOSS else None
    metrics["focal_alpha"] = FOCAL_ALPHA if USE_FOCAL_LOSS else None
    metrics["epochs_trained"] = len(history.history["loss"])
    metrics["final_val_loss"] = float(history.history["val_loss"][-1])
    metrics["final_val_accuracy"] = float(history.history["val_accuracy"][-1])

    if ARCHITECTURE == "TRANSFORMER":
        metrics["num_heads"] = NUM_HEADS
    if ARCHITECTURE == "CONV_LSTM":
        metrics["lstm_units"] = LSTM_UNITS

    # Save results
    save_results(metrics, experiment_name)

    # Save model
    MODEL_DIR.mkdir(exist_ok=True)
    model_path = MODEL_DIR / f"{experiment_name}.keras"
    model.save(model_path)
    if verbose: print(f"  Model saved to {model_path}")

    # Save hyperparameters
    config = {
        "architecture": ARCHITECTURE,
        "conv_filters": CONV_FILTERS,
        "conv_kernel": CONV_KERNEL,
        "dense_units": DENSE_UNITS,
        "dropout_rate": DROPOUT_RATE,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "mc_samples": MC_SAMPLES,
        "use_focal_loss": USE_FOCAL_LOSS,
        "focal_gamma": FOCAL_GAMMA,
        "focal_alpha": FOCAL_ALPHA,
        "window_size": window_size,
        "n_features": n_features,
    }
    if ARCHITECTURE == "TRANSFORMER":
        config["num_heads"] = NUM_HEADS
    if ARCHITECTURE == "CONV_LSTM":
        config["lstm_units"] = LSTM_UNITS

    with open(MODEL_DIR / f"{experiment_name}_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return metrics


# ============================================================
# PARALLEL EXPERIMENT RUNNER — batch hyperparameter search
# ============================================================

def run_parallel_experiment(config):
    """
    Run a single experiment in an isolated process.
    Used by run_parallel() for batch hyperparameter search.
    
    Args:
        config: dict with keys matching hyperparameter names + 'name' for experiment label
    
    Each process gets its own GPU memory allocation to prevent OOM errors.
    """
    # Configure GPU with memory limit in subprocess
    configure_gpu(memory_limit=config.get("gpu_memory_mb", 3000))

    # Load data
    data = np.load(DATA_DIR / "prepared_data.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    # Override globals with config values
    # (In a subprocess, module-level constants can be overridden locally)
    import strategy as s
    s.ARCHITECTURE = config.get("architecture", ARCHITECTURE)
    s.CONV_FILTERS = config.get("filters", CONV_FILTERS)
    s.DENSE_UNITS = config.get("dense_units", DENSE_UNITS)
    s.DROPOUT_RATE = config.get("dropout", DROPOUT_RATE)
    s.LEARNING_RATE = config.get("lr", LEARNING_RATE)
    s.BATCH_SIZE = config.get("batch_size", BATCH_SIZE)
    s.CONFIDENCE_THRESHOLD = config.get("threshold", CONFIDENCE_THRESHOLD)
    s.MC_SAMPLES = config.get("mc_samples", MC_SAMPLES)
    s.USE_FOCAL_LOSS = config.get("focal_loss", USE_FOCAL_LOSS)
    s.FOCAL_GAMMA = config.get("focal_gamma", FOCAL_GAMMA)
    s.FOCAL_ALPHA = config.get("focal_alpha", FOCAL_ALPHA)
    s.NUM_HEADS = config.get("num_heads", NUM_HEADS)
    s.LSTM_UNITS = config.get("lstm_units", LSTM_UNITS)

    # Build and train
    model = s.build_model(X_train.shape[1], X_train.shape[2])
    history = s.train_model(model, X_train, y_train, X_val, y_val, verbose=0)

    # MC predict
    val_mean, val_std = s.mc_predict(model, X_val, n_samples=s.MC_SAMPLES)

    # Evaluate
    from prepare import evaluate_predictions, save_results
    metrics = evaluate_predictions(y_val, val_mean, threshold=s.CONFIDENCE_THRESHOLD)

    # Log metadata
    metrics.update({
        "experiment_name": config["name"],
        "architecture": s.ARCHITECTURE,
        "conv_filters": s.CONV_FILTERS,
        "dense_units": s.DENSE_UNITS,
        "dropout_rate": s.DROPOUT_RATE,
        "learning_rate": s.LEARNING_RATE,
        "batch_size": s.BATCH_SIZE,
        "confidence_threshold": s.CONFIDENCE_THRESHOLD,
        "use_focal_loss": s.USE_FOCAL_LOSS,
        "epochs_trained": len(history.history["loss"]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    })

    save_results(metrics, config["name"])
    print(f"  DONE: {config['name']} | "
          f"Acc(conf): {metrics['accuracy_confident']:.1%} | "
          f"Trades: {metrics['confident_trades']}")


def run_parallel(experiments, max_workers=2):
    """
    Run multiple experiments in parallel using separate processes.
    
    Args:
        experiments: list of config dicts, each with at least a 'name' key
        max_workers: number of parallel processes (limit by GPU memory)
    
    Example:
        experiments = [
            {"name": "v5_high_lr", "lr": 1e-3, "filters": 64, "threshold": 0.55},
            {"name": "v5_low_lr",  "lr": 3e-4, "filters": 64, "threshold": 0.55},
            {"name": "v5_big_net", "lr": 5e-4, "filters": 128, "threshold": 0.60},
        ]
        run_parallel(experiments, max_workers=2)
    """
    print(f"\n{'='*60}")
    print(f"  PARALLEL RUN: {len(experiments)} experiments, {max_workers} workers")
    print(f"{'='*60}\n")

    processes = []
    for exp in experiments:
        # Wait if we're at max capacity
        while len(processes) >= max_workers:
            for p in processes:
                if not p.is_alive():
                    p.join()
                    processes.remove(p)
                    break

        p = mp.Process(target=run_parallel_experiment, args=(exp,))
        p.start()
        processes.append(p)

    # Wait for remaining
    for p in processes:
        p.join()

    print(f"\n  All {len(experiments)} experiments complete. Check results/results.tsv")


# ============================================================
# AUTO-COMPARE — runs both architectures, picks the winner
# ============================================================

def auto_compare(experiment_prefix=None, metric="accuracy_confident", min_trades=30):
    """
    Automatically run both TRANSFORMER and CONV_LSTM, compare results,
    and declare a winner. No human intervention needed.
    
    This is the seed of the autoresearch loop — the same keep/discard
    logic that Karpathy uses for neural network research and Nunchi uses
    for trading strategy optimization.
    
    Args:
        experiment_prefix: base name for experiments (timestamp added auto)
        metric: which metric to compare ("accuracy_confident" or "f1_confident")
        min_trades: minimum confident trades required to be a valid result.
                    If a model makes fewer trades than this, it loses automatically
                    (prevents the model from cheating by only trading 1 easy case)
    
    Returns:
        dict with winner name, both metrics, and the winning model's config
    """
    global ARCHITECTURE

    if experiment_prefix is None:
        experiment_prefix = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"  AUTO-COMPARE: TRANSFORMER vs CONV_LSTM")
    print(f"  Metric: {metric} | Min trades: {min_trades}")
    print(f"{'='*60}")

    results = {}

    # --- Run TRANSFORMER ---
    print(f"\n{'—'*60}")
    print(f"  Running TRANSFORMER...")
    print(f"{'—'*60}")
    ARCHITECTURE = "TRANSFORMER"
    name_tfr = f"{experiment_prefix}_TRANSFORMER"
    metrics_tfr = run_experiment(name_tfr)
    results["TRANSFORMER"] = metrics_tfr

    # Clear GPU memory between runs
    keras.backend.clear_session()

    # --- Run CONV_LSTM ---
    print(f"\n{'—'*60}")
    print(f"  Running CONV_LSTM...")
    print(f"{'—'*60}")
    ARCHITECTURE = "CONV_LSTM"
    name_lstm = f"{experiment_prefix}_CONV_LSTM"
    metrics_lstm = run_experiment(name_lstm)
    metrics["max_uncertainty"] = MAX_UNCERTAINTY
    results["CONV_LSTM"] = metrics_lstm

    # --- Compare ---
    print(f"\n{'='*60}")
    print(f"  COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"")
    print(f"  {'Metric':<25} {'TRANSFORMER':>15} {'CONV_LSTM':>15}")
    print(f"  {'—'*55}")
    print(f"  {'Accuracy (all)':<25} {metrics_tfr['accuracy_all']:>14.1%} {metrics_lstm['accuracy_all']:>14.1%}")
    print(f"  {'Confident trades':<25} {metrics_tfr['confident_trades']:>15} {metrics_lstm['confident_trades']:>15}")
    print(f"  {'Accuracy (confident)':<25} {metrics_tfr['accuracy_confident']:>14.1%} {metrics_lstm['accuracy_confident']:>14.1%}")
    print(f"  {'Precision (confident)':<25} {metrics_tfr['precision_confident']:>14.1%} {metrics_lstm['precision_confident']:>14.1%}")
    print(f"  {'F1 (confident)':<25} {metrics_tfr['f1_confident']:>14.1%} {metrics_lstm['f1_confident']:>14.1%}")
    print(f"  {'Val loss':<25} {metrics_tfr['final_val_loss']:>15.4f} {metrics_lstm['final_val_loss']:>15.4f}")
    print(f"  {'Epochs trained':<25} {metrics_tfr['epochs_trained']:>15} {metrics_lstm['epochs_trained']:>15}")

    # --- Decide winner ---
    # Disqualify models with too few trades
    tfr_valid = metrics_tfr["confident_trades"] >= min_trades
    lstm_valid = metrics_lstm["confident_trades"] >= min_trades

    if tfr_valid and lstm_valid:
        # Both valid — compare on target metric
        if metrics_tfr[metric] >= metrics_lstm[metric]:
            winner = "TRANSFORMER"
        else:
            winner = "CONV_LSTM"
    elif tfr_valid:
        winner = "TRANSFORMER"
    elif lstm_valid:
        winner = "CONV_LSTM"
    else:
        # Neither made enough trades — pick whichever got more
        if metrics_tfr["confident_trades"] >= metrics_lstm["confident_trades"]:
            winner = "TRANSFORMER"
        else:
            winner = "CONV_LSTM"
        print(f"\n  WARNING: Neither model met the {min_trades} trade minimum.")

    loser = "CONV_LSTM" if winner == "TRANSFORMER" else "TRANSFORMER"

    print(f"\n  {'='*55}")
    print(f"  WINNER: {winner}")
    print(f"  {metric}: {results[winner][metric]:.1%} "
          f"({results[winner]['confident_trades']} trades)")
    print(f"  Loser:  {loser} — {results[loser][metric]:.1%} "
          f"({results[loser]['confident_trades']} trades)")
    print(f"  {'='*55}")

    # Set the architecture to the winner for subsequent runs
    ARCHITECTURE = winner
    print(f"\n  ARCHITECTURE set to {winner} for future experiments.")

    return {
        "winner": winner,
        "results": results,
    }


# ============================================================
# MAIN — run mode selection
# ============================================================

if __name__ == "__main__":
    # Configure GPU
    configure_gpu()

    # ----------------------------------------------------------
    # MODE 1: Single experiment (default)
    # Change the name for each run to track in results.tsv
    # ----------------------------------------------------------
    # metrics = run_experiment("v4_transformer_focal")

    # ----------------------------------------------------------
    # MODE 2: Parallel batch search
    # Define experiments, then run in parallel.
    # Adjust gpu_memory_mb based on your GPU (3000 = 3GB per model)
    # ----------------------------------------------------------
    # experiments = [
    #     {"name": "v5_tfr_lr1e3",   "architecture": "TRANSFORMER", "lr": 1e-3, "filters": 64,  "threshold": 0.55, "gpu_memory_mb": 3000},
    #     {"name": "v5_tfr_lr5e4",   "architecture": "TRANSFORMER", "lr": 5e-4, "filters": 128, "threshold": 0.55, "gpu_memory_mb": 3000},
    #     {"name": "v5_tfr_highThr", "architecture": "TRANSFORMER", "lr": 1e-3, "filters": 64,  "threshold": 0.65, "gpu_memory_mb": 3000},
    #     {"name": "v5_lstm_focal",  "architecture": "CONV_LSTM",   "lr": 5e-4, "filters": 64,  "threshold": 0.55, "gpu_memory_mb": 3000},
    # ]
    # run_parallel(experiments, max_workers=2)

    # ----------------------------------------------------------
    # MODE 3: Auto-compare (runs both architectures, picks winner)
    # This is the recommended default — let the data decide.
    # ----------------------------------------------------------
    comparison = auto_compare(
    experiment_prefix="baseline_v1",
    metric="accuracy_confident",    # What to optimize for
    min_trades=50,                  # Minimum trades to be valid
) #Version v4
