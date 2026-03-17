"""
strategy.py — High-Performance Deep Research Engine (Master Version)
THIS FILE IS MODIFIED BY THE AUTORESEARCH AGENT.

Hardware Profile:
  - CPU: Ryzen 9 5900X (12-Core, 3.70 GHz) - Primary execution engine
  - RAM: 32GB DDR4
  - GPU: DISABLED (Unsupported/Incompatible with Python 3.12 environment)

Architecture Options:
  1. TRANSFORMER — Deep Multi-Head Attention blocks with Feed-Forward stacks
  2. CONV_LSTM   — Conv1D + LSTM + MCDropout (Sequential baseline)
  3. DENSE_STATS — Wide Dense network for statistical correlations (Fast CPU baseline)

Features:
  - Deep Transformer: Stackable encoder blocks with residual connections and internal MC Dropout.
  - Pure CPU Parallelism: Multi-process batch search optimized for 12-core execution.
  - Monte Carlo Dropout: Active across all layers for robust uncertainty estimation (STD calculation).
  - Uncertainty Filtering: Rejects high-variance predictions via the MAX_UNCERTAINTY threshold.
  - Binary Focal Loss: Forces the model to focus on harder-to-classify edge cases.
  - Auto-compare Mode: Head-to-head tournament to find the highest 'accuracy_confident'.
  - Detailed Logging: Records every experiment to results/results.tsv for historical tracking.

The autoresearch agent iterates on:
  - Transformer depth (NUM_TRANSFORMER_BLOCKS) and Attention Heads.
  - Learning rates, Dropout rates, and Kernel sizes.
  - Signal thresholds: CONFIDENCE_THRESHOLD and MAX_UNCERTAINTY (STD).
  - Training dynamics: Batch size and Focal Loss parameters (Gamma/Alpha).
"""

import os
import time
from datetime import timedelta
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from pathlib import Path
import json
import multiprocessing as mp
from datetime import datetime

# Suppress TF warnings and explicitly disable GPU for Python 3.12 compatibility
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

def configure_cpu_only():
    """
    Optimizes TensorFlow for the Ryzen 9 5900X.
    Sets thread affinity to ensure maximum throughput on 12 physical cores.
    """
    tf.config.set_visible_devices([], 'GPU')
    # Intra-op: Parallelism within a single operation
    tf.config.threading.set_intra_op_parallelism_threads(12)
    # Inter-op: Parallelism across different operations
    tf.config.threading.set_inter_op_parallelism_threads(12)
    print(f"  [HARDWARE] Target: Ryzen 9 5900X (12-Core) | GPU: Disabled")


# ============================================================
# HYPERPARAMETERS — autoresearch modifies these
# ============================================================

# --- Global selection ---
ARCHITECTURE = "TRANSFORMER"   

# --- Shared architecture params ---
CONV_FILTERS = 128             # Feature depth (d_model)
CONV_KERNEL = 3                # Temporal receptive field
DENSE_UNITS = 64               # Pre-output bottleneck
DROPOUT_RATE = 0.15            # Standard dropout rate

# --- TRANSFORMER-specific ---
NUM_TRANSFORMER_BLOCKS = 2     # Depth of attention stack
NUM_HEADS = 4                  # Attention heads per block
FF_DIM_MULTIPLIER = 4          # Feed-forward expansion factor

# --- CONV_LSTM-specific ---
LSTM_UNITS = 64                

# --- DENSE_STATS-specific ---
DENSE_WIDE_1 = 1024            
DENSE_WIDE_2 = 512             
DENSE_STATS_DROPOUT = 0.2     

# --- Training Configuration ---
LEARNING_RATE = 0.0005         
BATCH_SIZE = 256                
EPOCHS = 60                      
PATIENCE_EARLY_STOP = 20           
PATIENCE_LR_REDUCE = 5        
MIN_LR = 1e-7                  

# --- Loss Function (Focal Loss) ---
USE_FOCAL_LOSS = False          
FOCAL_GAMMA = 2.0              
FOCAL_ALPHA = 0.25             

# --- Signal & Uncertainty Management ---
CONFIDENCE_THRESHOLD = 0.53    # Probability requirement (0.0 to 1.0)
MAX_UNCERTAINTY = 0.15         # MAX ALLOWED STD: If variance is higher, force HOLD 
MC_SAMPLES = 50               # Samples used for Mean/STD calculation

# --- Paths ---
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")


# ============================================================
# CUSTOM COMPONENTS
# ============================================================

class MCDropout(layers.Dropout):
    """
    Dropout layer that remains active during inference.
    Essential for Monte Carlo sampling to estimate prediction uncertainty.
    """
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)
class ResearchMonitor(callbacks.Callback):
    def __init__(self, val_data, threshold, max_std, total_epochs):
        super().__init__()
        self.X_val, self.y_val = val_data
        self.threshold = threshold
        self.max_std = max_std
        self.total_epochs = total_epochs
        self.train_start_time = None
        self.epoch_start_time = None

    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()
        print(f"\n[SYSTEM] Training Started for {self.total_epochs} Epochs...")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        # 1. Timing Logic
        current_time = time.time()
        epoch_dur = current_time - self.epoch_start_time
        total_elapsed = current_time - self.train_start_time
        
        avg_time = total_elapsed / (epoch + 1)
        remaining = avg_time * (self.total_epochs - (epoch + 1))
        eta_str = str(timedelta(seconds=int(remaining)))

        # 2. High-Speed Efficacy (2,000 random samples)
        indices = np.random.choice(len(self.X_val), min(2000, len(self.X_val)), replace=False)
        x_sample = self.X_val[indices]
        
        # Use vectorized predict for speed
        preds = []
        for _ in range(20):
            preds.append(self.model.predict(x_sample, batch_size=512, verbose=0))
        
        preds = np.array(preds)
        mean, std = preds.mean(axis=0).flatten(), preds.std(axis=0).flatten()
        
        blocked = np.sum(std > self.max_std)
        valid_mean = mean[std <= self.max_std]
        trades = np.sum((valid_mean >= self.threshold) | (valid_mean <= (1 - self.threshold)))

        # 3. Clean Printout
        print(f"\n--- Epoch {epoch+1}/{self.total_epochs} Complete ---")
        print(f"  > Epoch Time: {int(epoch_dur)}s | ETA: {eta_str}")
        print(f"  > Market Stats: {blocked}/2000 Blocked | {trades} Active Signals")
        print("-" * 50)
def transformer_block(x):
    """
    A single Transformer Encoder block (Attention + Feed Forward).
    Integrated with residual connections and Layer Normalization.
    """
    # 1. Multi-Head Self-Attention
    attn_output = layers.MultiHeadAttention(
        num_heads=NUM_HEADS,
        key_dim=CONV_FILTERS // NUM_HEADS,
    )(x, x)
    attn_output = MCDropout(DROPOUT_RATE)(attn_output)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)

    # 2. Position-wise Feed-Forward Network
    ff_hidden = layers.Dense(CONV_FILTERS * FF_DIM_MULTIPLIER, activation="relu")(x)
    ff_output = layers.Dense(CONV_FILTERS)(ff_hidden)
    ff_output = MCDropout(DROPOUT_RATE)(ff_output)
    x = layers.Add()([x, ff_output])
    x = layers.LayerNormalization()(x)
    return x


# ============================================================
# MODEL ARCHITECTURES
# ============================================================

def build_transformer(window_size, n_features):
    """Builds a Deep Stacked Transformer model."""
    inputs = keras.Input(shape=(window_size, n_features))
    
    # Feature Projection
    x = layers.Conv1D(CONV_FILTERS, kernel_size=CONV_KERNEL, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    
    # Layer Stacking
    for _ in range(NUM_TRANSFORMER_BLOCKS):
        x = transformer_block(x)
        
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = MCDropout(DROPOUT_RATE)(x)
    
    outputs = layers.Dense(1, activation="sigmoid", dtype="float32")(x)
    return keras.Model(inputs, outputs)

def build_conv_lstm(window_size, n_features):
    """Builds a sequential Conv1D + LSTM baseline."""
    model = keras.Sequential([
        layers.Conv1D(CONV_FILTERS, kernel_size=CONV_KERNEL, activation="relu", padding="same", input_shape=(window_size, n_features)),
        layers.BatchNormalization(),
        MCDropout(DROPOUT_RATE),
        layers.LSTM(LSTM_UNITS, return_sequences=False),
        MCDropout(DROPOUT_RATE),
        layers.Dense(DENSE_UNITS, activation="relu"),
        MCDropout(DROPOUT_RATE),
        layers.Dense(1, activation="sigmoid", dtype="float32"),
    ])
    return model

def build_dense_stats(window_size, n_features):
    """Builds a wide, statistical-focus Dense model."""
    inputs = keras.Input(shape=(window_size, n_features))
    x = layers.Flatten()(inputs)
    x = layers.Dense(DENSE_WIDE_1, activation="relu")(x)
    x = MCDropout(DENSE_STATS_DROPOUT)(x)
    x = layers.Dense(DENSE_WIDE_2, activation="relu")(x)
    x = MCDropout(DENSE_STATS_DROPOUT)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs)

def build_model(window_size, n_features):
    """Global router for model creation and compilation."""
    if ARCHITECTURE == "TRANSFORMER":
        model = build_transformer(window_size, n_features)
    elif ARCHITECTURE == "CONV_LSTM":
        model = build_conv_lstm(window_size, n_features)
    elif ARCHITECTURE == "DENSE_STATS":
        model = build_dense_stats(window_size, n_features)
    else:
        raise ValueError(f"Unknown architecture: {ARCHITECTURE}")

    loss = keras.losses.BinaryFocalCrossentropy(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA) if USE_FOCAL_LOSS else "binary_crossentropy"
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=loss, metrics=["accuracy"])
    return model


# ============================================================
# EVALUATION & SIGNAL GENERATION
# ============================================================

def mc_predict(model, X, n_samples=MC_SAMPLES):
    """
    Executes Monte Carlo forward passes to calculate prediction uncertainty.
    Returns: (Mean Probability, Standard Deviation)
    """
    predictions = np.array([model(X, training=True).numpy().flatten() for _ in range(n_samples)])
    return predictions.mean(axis=0), predictions.std(axis=0)

def generate_signals(mean, std, threshold=CONFIDENCE_THRESHOLD, max_std=MAX_UNCERTAINTY):
    """
    Converts model outputs to BUY/SELL/HOLD.
    Enforces a strict uncertainty filter: if STD > MAX_UNCERTAINTY, signal is HOLD.
    """
    signals = []
    for m, s in zip(mean, std):
        if s > max_std: 
            signals.append("HOLD")
        elif m >= threshold: 
            signals.append("BUY")
        elif m <= (1 - threshold): 
            signals.append("SELL")
        else: 
            signals.append("HOLD")
    return np.array(signals)


# ============================================================
# EXPERIMENT EXECUTION
# ============================================================

def run_experiment(experiment_name=None, verbose=1):
    """Full end-to-end research pipeline for a single model configuration."""
    if experiment_name is None:
        experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if verbose:
        print(f"\n--- STARTING EXPERIMENT: {experiment_name} | ARCH: {ARCHITECTURE} ---")

    # Load prepared data
    data = np.load(DATA_DIR / "prepared_data.npz")
    X_train, y_train, X_val, y_val = data["X_train"], data["y_train"], data["X_val"], data["y_val"]
    
    model = build_model(X_train.shape[1], X_train.shape[2])
    
    # --- UPDATED CALLBACK SECTION ---
    # 1. Initialize the new ResearchMonitor instead of the old SignalEfficacyCallback
    monitor = ResearchMonitor(
        val_data=(X_val, y_val),
        threshold=CONFIDENCE_THRESHOLD, 
        max_std=MAX_UNCERTAINTY,
        total_epochs=EPOCHS
    )

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE_EARLY_STOP, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=PATIENCE_LR_REDUCE, verbose=1),
        monitor  # Pass the new ResearchMonitor here
    ]
    
    # 2. Run the training (Cleaned up to avoid double-calling fit)
    history = model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        callbacks=cb, 
        verbose=1 # Ensures you see the progress bar
    )

    # Evaluate with MC Inference
    val_mean, val_std = mc_predict(model, X_val)
    
    from prepare import evaluate_predictions, save_results, print_metrics
    metrics = evaluate_predictions(y_val, val_mean, threshold=CONFIDENCE_THRESHOLD)
    
    # Enrichment for results.tsv tracking
    metrics.update({
        "experiment_name": experiment_name,
        "architecture": ARCHITECTURE,
        "trans_blocks": NUM_TRANSFORMER_BLOCKS if ARCHITECTURE == "TRANSFORMER" else 0,
        "lr": LEARNING_RATE,
        "max_uncertainty": MAX_UNCERTAINTY,
        "mc_samples": MC_SAMPLES,
        "val_loss": float(history.history["val_loss"][-1])
    })
    
    if verbose: print_metrics(metrics, label=f"Summary: {ARCHITECTURE}")
    
    # Persist results and model
    save_results(metrics, experiment_name)
    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / f"{experiment_name}.keras")
    return metrics


# ============================================================
# MULTI-PROCESS MANAGEMENT
# ============================================================

def auto_compare(architectures=None, metric="accuracy_confident"):
    """Runs different architectures sequentially to establish a winner."""
    if architectures is None: architectures = ["TRANSFORMER", "CONV_LSTM", "DENSE_STATS"]
    prefix = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}
    
    for arch in architectures:
        global ARCHITECTURE
        ARCHITECTURE = arch
        results[arch] = run_experiment(f"compare_{prefix}_{arch}", verbose=1)
        keras.backend.clear_session()
    
    winner = max(results, key=lambda a: results[a][metric])
    print(f"\n{'='*40}\n [AUTO-COMPARE] WINNER: {winner}\n [METRIC] {metric}: {results[winner][metric]:.1%}\n{'='*40}")
    return winner

def run_parallel_worker(config):
    """Isolated worker for multiprocessing hyperparameter sweeps."""
    configure_cpu_only()
    import strategy as s
    # Apply experiment dictionary to global settings
    for key, value in config.items():
        attr = key.upper()
        if hasattr(s, attr) and key != "name":
            setattr(s, attr, value)
    
    s.run_experiment(config.get("name"), verbose=1)
    print(f"  [PARALLEL] Finished: {config.get('name')}")

def run_parallel(experiments, max_workers=3):
    """Orchestrates multiple experiments across the 12-core CPU."""
    print(f"\n{'='*60}\n  STARTING PARALLEL BATCH: {len(experiments)} configs\n{'='*60}")
    processes = []
    for exp in experiments:
        while len([p for p in processes if p.is_alive()]) >= max_workers:
            pass
        p = mp.Process(target=run_parallel_worker, args=(exp,))
        p.start()
        processes.append(p)
    for p in processes: p.join()


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    configure_cpu_only()
    
    # ----------------------------------------------------------
    # DEFAULT MODE: Auto-Compare All Baseline Architectures
    # Establishing a baseline for the new Deep Transformer
    # ----------------------------------------------------------
    auto_compare(architectures=["TRANSFORMER", "CONV_LSTM", "DENSE_STATS"])

    # ----------------------------------------------------------
    # MODE 2: Single Run (Targeted Research)
    # ----------------------------------------------------------
    # run_experiment("deep_transformer_test_v1")

    # ----------------------------------------------------------
    # MODE 3: Parallel Batch Search (Hyperparameter Sweep)
    # ----------------------------------------------------------
    # sweep_configs = [
    #     {"name": "tfr_depth_3", "num_transformer_blocks": 3, "max_uncertainty": 0.08},
    #     {"name": "tfr_depth_5", "num_transformer_blocks": 5, "max_uncertainty": 0.08},
    #     {"name": "tfr_strict",  "num_transformer_blocks": 3, "max_uncertainty": 0.04},
    # ]
    # run_parallel(sweep_configs, max_workers=3)
