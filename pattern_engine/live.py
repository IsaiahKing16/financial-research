"""
live.py — Live signal generation for production EOD scans.

Downloads today's market data, computes features, and runs the
fitted engine to generate BUY/SELL/HOLD signals for all tickers.
"""

import pandas as pd
import numpy as np
from datetime import datetime

from pattern_engine.config import EngineConfig
from pattern_engine.engine import PatternEngine
from pattern_engine.sector import SECTOR_MAP


class LiveSignalRunner:
    """Production end-of-day signal generator.

    Args:
        config: EngineConfig (or None for defaults)
        engine: pre-fitted PatternEngine (or None to fit from data)
        consensus_configs: optional list of additional EngineConfigs for
            cross-model consensus filtering. When provided, only emits
            signals where the majority of models agree.
    """

    def __init__(self, config: EngineConfig = None,
                 engine: PatternEngine = None,
                 consensus_configs: list[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.engine = engine
        self.consensus_configs = consensus_configs

    def run(self, train_db: pd.DataFrame = None,
            query_db: pd.DataFrame = None,
            verbose: int = 1) -> pd.DataFrame:
        """Generate live signals.

        If engine is not pre-fitted, fits on train_db first.
        Then predicts on query_db (today's data).

        Args:
            train_db: training data (only needed if engine not pre-fitted)
            query_db: query data (today's rows for all tickers)
            verbose: 0=silent, 1=progress

        Returns:
            DataFrame with columns: Ticker, Signal, Probability, Matches, Reason
            sorted by signal strength
        """
        # Fit engine if needed
        if self.engine is None:
            if train_db is None:
                raise ValueError("Provide train_db or a pre-fitted engine.")
            self.engine = PatternEngine(self.config)
            self.engine.fit(train_db)

        if query_db is None:
            raise ValueError("Provide query_db with today's data.")

        # Run predictions
        result = self.engine.predict(query_db, verbose=verbose)

        # Build signal table
        rows = []
        for i in range(len(query_db)):
            ticker = query_db.iloc[i]["Ticker"]
            rows.append({
                "Ticker": ticker,
                "Sector": SECTOR_MAP.get(ticker, "Unknown"),
                "Signal": result.signals[i],
                "Probability": round(float(result.calibrated_probabilities[i]), 4),
                "Matches": result.n_matches[i],
                "MeanReturn": round(float(result.mean_returns[i]), 5),
                "Reason": result.reasons[i],
            })

        signals_df = pd.DataFrame(rows)

        # Cross-model consensus filtering (optional)
        if self.consensus_configs and train_db is not None:
            from pattern_engine.cross_validation import CrossValidator
            all_configs = [self.config] + self.consensus_configs
            xval = CrossValidator(all_configs)
            xval_result = xval.run(train_db, query_db, verbose=0)
            consensus = xval.consensus_signals()

            # Override signals with consensus (aligned by position, not ticker dict)
            # consensus has same row order as query_db, so use positional alignment
            signals_df["Signal"] = consensus["ConsensusSignal"].values
            signals_df["ConsensusAgreement"] = consensus["AgreementRatio"].values

        # Sort: BUY/SELL first (by probability strength), HOLD last
        signals_df["_sort_key"] = signals_df["Signal"].map(
            {"BUY": 0, "SELL": 1, "HOLD": 2}
        )
        signals_df = signals_df.sort_values(
            ["_sort_key", "Probability"],
            ascending=[True, False]
        ).drop(columns=["_sort_key"]).reset_index(drop=True)

        if verbose:
            buys = signals_df[signals_df["Signal"] == "BUY"]
            sells = signals_df[signals_df["Signal"] == "SELL"]
            print(f"\n{'=' * 60}")
            print(f"  LIVE SIGNALS — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            print(f"{'=' * 60}")
            print(f"  BUY signals:  {len(buys)}")
            print(f"  SELL signals: {len(sells)}")
            print(f"  HOLD:         {len(signals_df) - len(buys) - len(sells)}")

            if len(buys) > 0:
                print(f"\n  TOP BUY SIGNALS:")
                for _, r in buys.head(10).iterrows():
                    print(f"    {r['Ticker']:<6s} P={r['Probability']:.3f} "
                          f"K={r['Matches']} {r['Sector']}")

            if len(sells) > 0:
                print(f"\n  TOP SELL SIGNALS:")
                for _, r in sells.head(10).iterrows():
                    print(f"    {r['Ticker']:<6s} P={r['Probability']:.3f} "
                          f"K={r['Matches']} {r['Sector']}")

            print(f"{'=' * 60}\n")

        return signals_df
