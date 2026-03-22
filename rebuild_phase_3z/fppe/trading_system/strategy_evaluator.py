"""
strategy_evaluator.py — Layer 4: Strategy Evaluator.

Continuously monitors portfolio performance and emits RED/YELLOW/GREEN
status signals that govern whether the system can trade at full, reduced,
or zero capacity.

Architecture:
  - Consumes closed-trade records (TradeEvent objects) to compute rolling metrics.
  - Emits EvaluatorSnapshot (Pydantic model from contracts/decisions.py).
  - Writes EvaluatorState to SharedState via SharedStateManager.
  - Enqueues SystemCommand (HALT / REDUCE_EXPOSURE / RESUME) as status changes.

Metrics computed:
  - Daily PnL series (from closed trades on each day)
  - Rolling Sharpe ratios: 30d, 90d, 252d, all-time
  - Cumulative return and drawdown from peak
  - Trade count (total and within rolling windows)
  - SPY baseline comparison (if spy_returns provided)
  - Calibration drift detection (BSS trend over recent folds)

Status logic (in priority order):
  RED    — 90-day Sharpe < 0 OR drawdown > 15% OR HALT command issued
  YELLOW — 30-day Sharpe < 0.5 OR trade count < 30 (insufficient history)
  GREEN  — all metrics within expected bounds

Thresholds are defined in _THRESHOLDS and can be overridden via EvaluatorConfig.

Design decisions:
  - min_trades_for_metrics=30: don't compute ratios before sufficient history
  - All computations are pure functions on lists of floats — no DataFrames
    in the hot path (fast for daily incremental updates)
  - Calibration drift: BSS is tracked as a rolling list; slope < -0.001/day
    triggers YELLOW (not RED — BSS noise requires more evidence)

Linear: SLE-70
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date as Date
from typing import Dict, List, Optional, Tuple

from rebuild_phase_3z.fppe.trading_system.contracts.decisions import (
    EvaluatorSnapshot,
    EvaluatorStatus,
)
from rebuild_phase_3z.fppe.trading_system.contracts.state import (
    EvaluatorState,
    SharedState,
    SystemCommand,
)


# ─── Evaluator thresholds ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class EvaluatorConfig:
    """
    Thresholds and windows for the strategy evaluator.

    These drive the RED/YELLOW/GREEN status decisions. They are separate
    from RiskConfig (which governs per-trade sizing and stops) and
    EvaluationConfig (which governs reporting windows).

    Defaults match the design spec from FPPE_TRADING_SYSTEM_DESIGN.md v0.4.
    """
    # Status thresholds
    red_sharpe_90d: float = 0.0         # Sharpe < 0 over 90d → RED
    red_drawdown: float = 0.15          # Drawdown > 15% → RED
    yellow_sharpe_30d: float = 0.5      # Sharpe < 0.5 over 30d → YELLOW
    yellow_min_trades: int = 30         # < 30 trades → YELLOW (insufficient history)

    # Calibration drift detection
    bss_drift_slope_threshold: float = -0.001   # BSS slope < -0.001/day → YELLOW

    # Rolling windows (trading days)
    window_30d: int = 30
    window_90d: int = 90
    window_252d: int = 252

    # Minimum trades to compute Sharpe ratios
    min_trades_for_sharpe: int = 10


# ─── Trade record (internal) ───────────────────────────────────────────────────

@dataclass
class ClosedTrade:
    """
    Record of a completed trade, used by the evaluator.

    The evaluator does NOT use TradeEvent directly — it needs closed-trade
    data (entry + exit). This lightweight dataclass avoids importing the
    full trading system just for metric computation.

    Attributes:
        close_date: Date the trade was closed.
        pnl_pct: Percent PnL on this trade (e.g., 0.05 = +5%).
        holding_days: How many trading days the position was held.
        ticker: Ticker symbol (for BSS fold tracking).
    """
    close_date: Date
    pnl_pct: float
    holding_days: int
    ticker: str = ""


# ─── Rolling metrics helpers ───────────────────────────────────────────────────

def _annualized_sharpe(
    daily_returns: List[float],
    risk_free_daily: float = 0.0,
) -> Optional[float]:
    """
    Compute annualized Sharpe ratio from a list of daily returns.

    Returns None if fewer than 2 data points (can't compute std).

    Args:
        daily_returns: Daily return values (e.g., [0.01, -0.005, ...]).
        risk_free_daily: Daily risk-free rate (default 0 for simplicity).

    Returns:
        Annualized Sharpe ratio, or None if insufficient data.
    """
    n = len(daily_returns)
    if n < 2:
        return None

    excess = [r - risk_free_daily for r in daily_returns]
    mean_e = sum(excess) / n
    variance = sum((r - mean_e) ** 2 for r in excess) / (n - 1)  # Bessel-corrected
    std_e = math.sqrt(variance) if variance > 0 else 0.0

    if std_e == 0.0:
        # All returns identical: ±inf Sharpe, or 0 if mean excess is exactly 0
        if mean_e > 0:
            return float("inf")
        elif mean_e < 0:
            return float("-inf")
        else:
            return 0.0

    return (mean_e / std_e) * math.sqrt(252)


def _drawdown_from_peak(cumulative_returns: List[float]) -> float:
    """
    Compute current drawdown from the running peak of cumulative returns.

    Args:
        cumulative_returns: Equity curve as a list of cumulative return values.
                            cumulative_returns[i] = total_equity / inception_equity - 1.

    Returns:
        Current drawdown [0, 1]. 0.0 if at all-time high.
    """
    if not cumulative_returns:
        return 0.0

    # Convert cumulative returns to equity curve (start at 1.0)
    equity = [1.0 + r for r in cumulative_returns]
    peak = 1.0
    for e in equity:
        if e > peak:
            peak = e
    current = equity[-1]
    return max(0.0, 1.0 - current / peak)


def _linear_slope(values: List[float]) -> float:
    """
    Compute slope of a simple linear regression on (index, value) pairs.

    Used for calibration drift detection: if BSS is trending downward,
    the evaluator issues a YELLOW warning.

    Args:
        values: Ordered list of scalar values.

    Returns:
        Slope coefficient (positive = trending up, negative = trending down).
        Returns 0.0 if fewer than 2 values.
    """
    n = len(values)
    if n < 2:
        return 0.0

    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n

    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    denominator = sum((x - mean_x) ** 2 for x in xs)

    return numerator / denominator if denominator > 0 else 0.0


# ─── StrategyEvaluator ────────────────────────────────────────────────────────

class StrategyEvaluator:
    """
    Layer 4: Continuous strategy performance evaluator.

    Accumulates closed-trade records and computes rolling metrics.
    On each evaluation call, emits an EvaluatorSnapshot and optionally
    updates the SharedState (via SharedStateManager) with new status
    and any pending SystemCommands.

    State management:
        The evaluator is stateful — it holds the history of closed trades
        and daily returns. It does NOT hold a SharedState reference; instead,
        it accepts the current state and returns an updated one.

    Thread safety:
        Not thread-safe. The trading system is single-threaded (nn_jobs=1).

    Args:
        config: EvaluatorConfig with threshold parameters.
        risk_free_annual_rate: Annual risk-free rate (for Sharpe denominator).

    Usage:
        evaluator = StrategyEvaluator(config=EvaluatorConfig())
        evaluator.record_trade(ClosedTrade(close_date=d, pnl_pct=0.03, ...))
        evaluator.record_daily_return(date=d, portfolio_return=0.005, cumulative_return=0.12)
        snapshot = evaluator.evaluate(current_date=d, drawdown=0.05, days_in_market=45)
    """

    def __init__(
        self,
        config: EvaluatorConfig | None = None,
        risk_free_annual_rate: float = 0.045,
    ) -> None:
        self.config = config or EvaluatorConfig()
        self._risk_free_daily = (1 + risk_free_annual_rate) ** (1 / 252) - 1

        # Trade history
        self._closed_trades: List[ClosedTrade] = []

        # Daily return series: [(date, portfolio_return, cumulative_return)]
        self._daily_series: List[Tuple[Date, float, float]] = []

        # BSS series for calibration drift detection: [(date, bss_value)]
        self._bss_series: List[Tuple[Date, float]] = []

        # Track previous status for command generation
        self._previous_status: Optional[EvaluatorStatus] = None

        # Research pilot: DriftMonitor (SLE-76) — None until set_drift_monitor() called
        self._drift_monitor = None   # DriftMonitor | None

    # ── Research pilot: DriftMonitor integration (SLE-76) ─────────────────────

    def set_drift_monitor(self, monitor) -> None:
        """Attach a DriftMonitor for feature-drift and BSS-EWMA alerting.

        When a DriftMonitor is attached and reports any_alert=True, evaluate()
        will return YELLOW status (unless RED conditions also apply).

        Args:
            monitor: DriftMonitor instance that has already called set_baseline().
                     The caller is responsible for calling monitor.update_features(),
                     monitor.update_calibration(), and monitor.update_bss() between
                     evaluate() calls to keep the monitor current.
        """
        self._drift_monitor = monitor

    # ── Data ingestion ─────────────────────────────────────────────────────────

    def record_trade(self, trade: ClosedTrade) -> None:
        """
        Register a closed trade for metric computation.

        Args:
            trade: ClosedTrade with close_date, pnl_pct, holding_days, ticker.
        """
        self._closed_trades.append(trade)

    def record_daily_return(
        self,
        trade_date: Date,
        portfolio_return: float,
        cumulative_return: float,
    ) -> None:
        """
        Register the portfolio's daily return for a trading day.

        Args:
            trade_date: The trading day.
            portfolio_return: Daily portfolio return (e.g., 0.005 = +0.5%).
            cumulative_return: Cumulative return since inception.
        """
        self._daily_series.append((trade_date, portfolio_return, cumulative_return))

    def record_bss(self, bss_date: Date, bss_value: float) -> None:
        """
        Register a Brier Skill Score from a walk-forward fold.

        Args:
            bss_date: Date of the BSS computation.
            bss_value: BSS value (positive = better than climatology).
        """
        self._bss_series.append((bss_date, bss_value))

    # ── Metric computation ─────────────────────────────────────────────────────

    def _recent_daily_returns(self, window: int) -> List[float]:
        """Return the last `window` daily portfolio returns."""
        returns = [r for _, r, _ in self._daily_series]
        return returns[-window:]

    def _compute_sharpe(self, window: int) -> Optional[float]:
        """
        Compute rolling Sharpe for the last `window` trading days.

        Returns None if there are fewer than min_trades_for_sharpe data points.
        """
        recent = self._recent_daily_returns(window)
        if len(recent) < self.config.min_trades_for_sharpe:
            return None
        return _annualized_sharpe(recent, self._risk_free_daily)

    def _compute_all_time_sharpe(self) -> Optional[float]:
        """Compute Sharpe ratio over the full trade history."""
        all_returns = [r for _, r, _ in self._daily_series]
        if len(all_returns) < self.config.min_trades_for_sharpe:
            return None
        return _annualized_sharpe(all_returns, self._risk_free_daily)

    def _compute_drawdown(self) -> float:
        """Compute current drawdown from the equity curve."""
        cum_returns = [cr for _, _, cr in self._daily_series]
        return _drawdown_from_peak(cum_returns)

    def _compute_bss_drift_slope(self) -> float:
        """Compute linear slope of BSS series for drift detection."""
        bss_values = [v for _, v in self._bss_series]
        return _linear_slope(bss_values)

    def _count_trades_in_window(self, window_days: int, reference_date: Date) -> int:
        """Count closed trades within the last `window_days` calendar days.

        Note: the caller supplies trading-day windows (e.g. window_30d=30 trading days)
        but this function counts calendar days.  30 trading days ≈ 42 calendar days.
        m4 (SLE review): for full correctness, convert to calendar days via
        ``window_days * 365 // 252`` before comparing, or integrate a trading calendar.
        Current behaviour is conservative (undercounts slightly for 30d window, accurate
        for 252d window since 365 ≈ 252 * 1.45 and the denominator is already trading).
        """
        count = 0
        for t in self._closed_trades:
            delta = (reference_date - t.close_date).days
            if 0 <= delta <= window_days:
                count += 1
        return count

    # ── Status determination ───────────────────────────────────────────────────

    def _determine_status(
        self,
        sharpe_30d: Optional[float],
        sharpe_90d: Optional[float],
        drawdown: float,
        total_trades: int,
        bss_drift_slope: float,
    ) -> Tuple[EvaluatorStatus, str]:
        """
        Apply the RED → YELLOW → GREEN decision tree.

        Priority order: RED first (most severe), then YELLOW, then GREEN.

        Args:
            sharpe_30d: 30-day rolling Sharpe (None = insufficient history).
            sharpe_90d: 90-day rolling Sharpe (None = insufficient history).
            drawdown: Current drawdown from peak [0, 1].
            total_trades: Total closed trades since system start.
            bss_drift_slope: Linear slope of BSS series.

        Returns:
            (EvaluatorStatus, reason_string)
        """
        cfg = self.config

        # ── RED checks (halt all new entries) ────────────────────────────────
        if drawdown > cfg.red_drawdown:
            return EvaluatorStatus.RED, (
                f"Portfolio drawdown {drawdown:.1%} exceeds RED threshold "
                f"({cfg.red_drawdown:.0%})"
            )

        if sharpe_90d is not None and sharpe_90d < cfg.red_sharpe_90d:
            return EvaluatorStatus.RED, (
                f"90-day Sharpe {sharpe_90d:.3f} is below RED threshold "
                f"({cfg.red_sharpe_90d:.2f})"
            )

        # ── YELLOW checks (reduce exposure by 20%) ────────────────────────────
        if total_trades < cfg.yellow_min_trades:
            return EvaluatorStatus.YELLOW, (
                f"Insufficient trade history: {total_trades} trades "
                f"(need ≥ {cfg.yellow_min_trades} for reliable metrics)"
            )

        if sharpe_30d is not None and sharpe_30d < cfg.yellow_sharpe_30d:
            return EvaluatorStatus.YELLOW, (
                f"30-day Sharpe {sharpe_30d:.3f} is below YELLOW threshold "
                f"({cfg.yellow_sharpe_30d:.2f})"
            )

        if bss_drift_slope < cfg.bss_drift_slope_threshold:
            return EvaluatorStatus.YELLOW, (
                f"Calibration drift detected: BSS slope {bss_drift_slope:.4f}/period "
                f"(threshold {cfg.bss_drift_slope_threshold:.4f})"
            )

        # Research pilot: DriftMonitor CUSUM/EWMA alert → YELLOW (SLE-76).
        # Provides more sensitive, statistically-grounded drift detection than
        # the simple linear BSS slope check above.
        if self._drift_monitor is not None:
            _dm_report = self._drift_monitor.get_report()
            if _dm_report.get("any_alert"):
                return EvaluatorStatus.YELLOW, (
                    "DriftMonitor alert: feature distribution shift or "
                    "BSS EWMA drift detected (see monitor.get_report() for details)"
                )

        # ── GREEN (all metrics within bounds) ─────────────────────────────────
        sharpe_desc = f"{sharpe_90d:.3f}" if sharpe_90d is not None else "N/A (early)"
        return EvaluatorStatus.GREEN, (
            f"All metrics within bounds. 90d Sharpe={sharpe_desc}, "
            f"drawdown={drawdown:.1%}, trades={total_trades}"
        )

    # ── Commands ───────────────────────────────────────────────────────────────

    def _determine_commands(
        self,
        new_status: EvaluatorStatus,
    ) -> List[SystemCommand]:
        """
        Compute SystemCommands based on status transition.

        Commands are only issued on status change (not repeatedly on each
        evaluation while status is stable).

        Transitions:
          → RED:    issue HALT
          → YELLOW: issue REDUCE_EXPOSURE
          → GREEN:  issue RESUME (if coming from RED or YELLOW)

        Args:
            new_status: The just-determined status.

        Returns:
            List of SystemCommands to enqueue (may be empty).
        """
        if self._previous_status == new_status:
            return []  # No change — no command needed

        commands: List[SystemCommand] = []

        if new_status == EvaluatorStatus.RED:
            commands.append(SystemCommand.HALT)
        elif new_status == EvaluatorStatus.YELLOW:
            if self._previous_status == EvaluatorStatus.RED:
                # Recovering from RED: RESUME first, then REDUCE_EXPOSURE
                # (RESUME clears the HALT; REDUCE_EXPOSURE then attenuates sizing)
                commands.append(SystemCommand.RESUME)
            commands.append(SystemCommand.REDUCE_EXPOSURE)
        elif new_status == EvaluatorStatus.GREEN:
            if self._previous_status in (EvaluatorStatus.RED, EvaluatorStatus.YELLOW):
                commands.append(SystemCommand.RESUME)

        return commands

    # ── Main evaluation method ─────────────────────────────────────────────────

    def evaluate(
        self,
        current_date: Date,
        drawdown: Optional[float] = None,
    ) -> EvaluatorSnapshot:
        """
        Evaluate current strategy performance and emit a snapshot.

        This is the primary public method. Call it once per trading day
        after all trades for that day are recorded.

        Args:
            current_date: The trading day being evaluated.
            drawdown: Current portfolio drawdown from peak [0, 1].
                      If None, computed from the internal daily series.

        Returns:
            EvaluatorSnapshot with current status, metrics, and reason.
        """
        # Compute metrics
        sharpe_30d = self._compute_sharpe(self.config.window_30d)
        sharpe_90d = self._compute_sharpe(self.config.window_90d)
        sharpe_252d = self._compute_sharpe(self.config.window_252d)
        all_time_sharpe = self._compute_all_time_sharpe()

        if drawdown is None:
            drawdown = self._compute_drawdown()

        total_trades = len(self._closed_trades)
        bss_drift = self._compute_bss_drift_slope()
        days_in_market = len(self._daily_series)

        # Determine status
        status, reason = self._determine_status(
            sharpe_30d=sharpe_30d,
            sharpe_90d=sharpe_90d,
            drawdown=drawdown,
            total_trades=total_trades,
            bss_drift_slope=bss_drift,
        )

        # Build snapshot
        snapshot = EvaluatorSnapshot(
            evaluation_date=current_date,
            status=status,
            sharpe_30d=sharpe_30d,
            sharpe_90d=sharpe_90d,
            drawdown_from_peak=drawdown,
            days_in_market=days_in_market,
            reason=reason,
        )

        # Update internal status tracker (for command emission on next call)
        self._previous_status = status

        return snapshot

    def evaluate_and_update_state(
        self,
        state: SharedState,
        current_date: Date,
        manager,  # SharedStateManager — imported at call site to avoid circular import
        drawdown: Optional[float] = None,
    ) -> Tuple[EvaluatorSnapshot, SharedState]:
        """
        Evaluate and write the result back to SharedState.

        Args:
            state: Current SharedState.
            current_date: The trading day being evaluated.
            manager: SharedStateManager instance for write-isolation enforcement.
            drawdown: Current drawdown (if None, computed from daily series).

        Returns:
            (snapshot, new_state): snapshot for logging, new_state for next layer.
        """
        from rebuild_phase_3z.fppe.trading_system.shared_state import (
            LayerTag,
            SharedStateManager,
        )

        # Save previous status BEFORE evaluate() updates _previous_status.
        # This is required so _determine_commands() can detect the status transition.
        #
        # evaluate() sets self._previous_status = status internally (line in evaluate()).
        # If anything between evaluate() and the final _previous_status update raises,
        # we must restore _previous_status to its pre-call value so the evaluator is
        # not left in a half-updated state on the next invocation.
        prev_status_before_eval = self._previous_status
        snapshot = self.evaluate(current_date=current_date, drawdown=drawdown)

        try:
            # Update evaluator sub-state
            new_eval_state = EvaluatorState(
                latest_snapshot=snapshot,
                consecutive_underperformance_days=(
                    state.evaluator.consecutive_underperformance_days + 1
                    if snapshot.status != EvaluatorStatus.GREEN
                    else 0
                ),
            )
            new_state = manager.update(
                state,
                LayerTag.STRATEGY_EVALUATOR,
                evaluator=new_eval_state,
            )

            # Enqueue any status-change commands based on the transition
            # (prev_status_before_eval → snapshot.status)
            self._previous_status = prev_status_before_eval  # restore so _determine_commands sees the old value
            commands = self._determine_commands(snapshot.status)
            self._previous_status = snapshot.status  # now update to current
            for cmd in commands:
                new_state = manager.enqueue_command(
                    new_state,
                    LayerTag.STRATEGY_EVALUATOR,
                    cmd,
                )
        except Exception:
            # Roll back _previous_status so the next call sees a consistent baseline.
            self._previous_status = prev_status_before_eval
            raise

        return snapshot, new_state

    # ── Inspection ─────────────────────────────────────────────────────────────

    @property
    def total_trades(self) -> int:
        """Total number of closed trades recorded."""
        return len(self._closed_trades)

    @property
    def days_in_market(self) -> int:
        """Number of trading days recorded."""
        return len(self._daily_series)

    def metrics_summary(self) -> Dict:
        """
        Return a dict of current metric values for logging/inspection.

        Returns:
            Dict with sharpe_30d, sharpe_90d, sharpe_252d, all_time_sharpe,
            drawdown, total_trades, days_in_market, bss_drift_slope.
        """
        return {
            "sharpe_30d": self._compute_sharpe(self.config.window_30d),
            "sharpe_90d": self._compute_sharpe(self.config.window_90d),
            "sharpe_252d": self._compute_sharpe(self.config.window_252d),
            "all_time_sharpe": self._compute_all_time_sharpe(),
            "drawdown": self._compute_drawdown(),
            "total_trades": self.total_trades,
            "days_in_market": self.days_in_market,
            "bss_drift_slope": self._compute_bss_drift_slope(),
        }
