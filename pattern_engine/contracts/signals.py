"""
signals.py — Pydantic contracts for pattern engine signal pipeline.

Three contract families:
  1. NeighborResult     — raw KNN retrieval output (indices + distances + metadata)
  2. CalibratedProbability — post-Platt calibration result
  3. SignalRecord       — final signal at the pattern_engine → trading_system boundary

NeighborResult and CalibratedProbability live within the pattern engine (internal
pipeline). SignalRecord crosses the module boundary into trading_system.

Columns in production (cached_signals_2024.csv):
    date, ticker, signal, confidence, signal_source, sector, n_matches, raw_prob, mean_7d_return

Linear: SLE-57
"""

from datetime import date as Date
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pattern_engine.contracts.finite_types import FiniteFloat


class SignalDirection(str, Enum):
    """Valid signal directions. Maps to the 'signal' column."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class SignalSource(str, Enum):
    """Signal generation method. Maps to the 'signal_source' column."""
    KNN = "KNN"
    DL = "DL"
    ENSEMBLE = "ENSEMBLE"


class SignalRecord(BaseModel):
    """A single signal emitted by the pattern engine.

    Frozen (immutable) to prevent downstream mutation.
    All fields match the columns in cached_signals_2024.csv.

    Usage:
        signal = SignalRecord(
            date=date(2024, 1, 2),
            ticker="AAPL",
            signal=SignalDirection.BUY,
            confidence=0.72,
            signal_source=SignalSource.KNN,
            sector="Technology",
            n_matches=50,
            raw_prob=0.68,
            mean_7d_return=1.23,
        )
    """
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    date: Date = Field(description="Signal generation date (trading day)")
    ticker: str = Field(min_length=1, max_length=10, description="Stock ticker symbol")
    signal: SignalDirection = Field(description="BUY, SELL, or HOLD")
    confidence: FiniteFloat = Field(ge=0.0, le=1.0, description="Calibrated probability [0, 1]")
    signal_source: SignalSource = Field(default=SignalSource.KNN, description="Model that generated the signal")
    sector: str = Field(min_length=1, description="Sector classification from SECTOR_MAP")
    n_matches: int = Field(ge=0, description="Number of K-NN analogues found")
    raw_prob: FiniteFloat = Field(description="Raw (uncalibrated) probability from K-NN vote")
    mean_7d_return: float = Field(description="Mean 7-day forward return of analogues (%)")

    @field_validator("ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        """Tickers must be uppercase (e.g., 'AAPL', not 'aapl')."""
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase: got '{v}'")
        return v

    @field_validator("n_matches")
    @classmethod
    def matches_reasonable(cls, v: int) -> int:
        """Warn-level: fewer than 5 matches makes probabilities unreliable."""
        # We don't reject — that's the config's min_matches job — but we validate range
        if v > 10_000:
            raise ValueError(f"n_matches={v} is unreasonably high; check data pipeline")
        return v


# ─── NeighborResult ────────────────────────────────────────────────────────────

class NeighborResult(BaseModel):
    """
    Raw output from KNN retrieval for a single query point.

    This is the internal pipeline contract between the Matcher and the
    probability estimation layer. It carries the raw indices and distances
    returned by BallTreeMatcher or HNSWMatcher, plus the target labels
    of the retrieved neighbors (for vote counting).

    Args:
        query_ticker: Ticker being queried.
        query_date: Date of the query fingerprint.
        neighbor_indices: Array of training DB row indices (shape: [k]).
        neighbor_distances: Euclidean distances to each neighbor (shape: [k]).
        neighbor_labels: Binary target labels for each neighbor (shape: [k]).
                         Used to compute raw_prob = mean(labels).
        n_neighbors_requested: k used at query time.
        n_neighbors_found: Actual count returned (may be < k if DB is small).
    """
    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    query_ticker: str = Field(description="Ticker symbol being queried")
    query_date: Date = Field(description="Date of the query fingerprint")
    neighbor_indices: list[int] = Field(description="Training DB row indices of KNN results")
    neighbor_distances: list[float] = Field(description="Euclidean distances to each neighbor")
    neighbor_labels: list[int] = Field(description="Binary target labels of each neighbor (0 or 1)")
    n_neighbors_requested: int = Field(ge=1, description="k requested at query time")
    n_neighbors_found: int = Field(ge=0, description="Actual neighbors returned")

    @model_validator(mode="after")
    def validate_array_lengths(self) -> "NeighborResult":
        """All neighbor arrays must have the same length = n_neighbors_found."""
        n = self.n_neighbors_found
        if len(self.neighbor_indices) != n:
            raise ValueError(
                f"neighbor_indices length {len(self.neighbor_indices)} != n_neighbors_found {n}"
            )
        if len(self.neighbor_distances) != n:
            raise ValueError(
                f"neighbor_distances length {len(self.neighbor_distances)} != n_neighbors_found {n}"
            )
        if len(self.neighbor_labels) != n:
            raise ValueError(
                f"neighbor_labels length {len(self.neighbor_labels)} != n_neighbors_found {n}"
            )
        return self

    @model_validator(mode="after")
    def validate_found_le_requested(self) -> "NeighborResult":
        """Cannot return more neighbors than were requested."""
        if self.n_neighbors_found > self.n_neighbors_requested:
            raise ValueError(
                f"n_neighbors_found ({self.n_neighbors_found}) > "
                f"n_neighbors_requested ({self.n_neighbors_requested})"
            )
        return self

    @field_validator("neighbor_distances")
    @classmethod
    def distances_non_negative(cls, v: list[float]) -> list[float]:
        """All Euclidean distances must be >= 0."""
        if any(d < 0.0 for d in v):
            raise ValueError("neighbor_distances contains negative values; sqrt() may be missing")
        return v

    @field_validator("neighbor_labels")
    @classmethod
    def labels_binary(cls, v: list[int]) -> list[int]:
        """Neighbor labels must be 0 or 1 (binary classification only)."""
        if any(label not in (0, 1) for label in v):
            raise ValueError("neighbor_labels must contain only 0 and 1")
        return v

    @property
    def raw_prob(self) -> float:
        """Raw (uncalibrated) probability = fraction of neighbors with label=1."""
        if self.n_neighbors_found == 0:
            return 0.5  # No information; return neutral
        return sum(self.neighbor_labels) / self.n_neighbors_found

    @property
    def mean_distance(self) -> float:
        """Mean Euclidean distance to all returned neighbors."""
        if self.n_neighbors_found == 0:
            return 0.0  # no analogues found; mean_distance is undefined, use 0.0 sentinel
        return sum(self.neighbor_distances) / self.n_neighbors_found


# ─── CalibratedProbability ─────────────────────────────────────────────────────

class CalibratedProbability(BaseModel):
    """
    Output of Platt scaling applied to a NeighborResult.

    This contract captures everything needed to generate a SignalRecord:
    the calibrated probability, the raw vote, and the decision metadata.

    Args:
        query_ticker: Ticker symbol.
        query_date: Date of the query.
        raw_prob: Uncalibrated KNN vote (fraction of neighbors with label=1).
        calibrated_prob: Platt-scaled probability in [0, 1].
        n_neighbors_found: Neighbor count (affects signal reliability).
        mean_distance: Mean distance to neighbors (quality indicator).
        calibration_method: Calibration method used (default: "platt").
    """
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    query_ticker: str = Field(description="Ticker symbol")
    query_date: Date = Field(description="Date of the query fingerprint")
    raw_prob: FiniteFloat = Field(ge=0.0, le=1.0, description="Raw KNN vote probability")
    calibrated_prob: FiniteFloat = Field(ge=0.0, le=1.0, description="Platt-scaled probability")
    n_neighbors_found: int = Field(ge=0, description="Number of neighbors used in vote")
    mean_distance: FiniteFloat = Field(ge=0.0, description="Mean Euclidean distance to neighbors")
    calibration_method: str = Field(default="platt", description="Calibration method applied")

    @field_validator("query_ticker")
    @classmethod
    def ticker_uppercase(cls, v: str) -> str:
        """Tickers must be uppercase."""
        if v != v.upper():
            raise ValueError(f"Ticker must be uppercase: got '{v}'")
        return v
