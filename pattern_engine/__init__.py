"""
pattern_engine — Financial Prediction Pattern Engine v2.0

Historical analogue matching engine for stock prediction.
Finds historical "twins" of current market conditions and projects
forward returns to generate probabilistic trading signals.

Usage:
    from pattern_engine import PatternEngine, EngineConfig

    engine = PatternEngine()              # Proven research defaults
    engine.fit(train_db)                  # Fit scaler, NN index, calibrator
    result = engine.predict(query_db)     # Generate predictions
    metrics = engine.evaluate(val_db)     # Predict + score

    engine.save("engine_state.pkl")
    engine = PatternEngine.load("engine_state.pkl")
"""

from pattern_engine.config import EngineConfig
from pattern_engine.engine import PatternEngine

__version__ = "2.0.0"
__all__ = ["PatternEngine", "EngineConfig"]
