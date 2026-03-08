# risk.py
# Position limits, P&L tracking, and unwind logic.
#
# This is the module that prevented the RTG disaster from happening.
# Core rules:
#   1. Unrealized loss per ticker > STOP_LOSS_PER_TICKER → stop + unwind
#   2. Total portfolio loss > STOP_LOSS_PORTFOLIO → full stop everything
#   3. Unwind via aggressive limit orders (NER market orders are unreliable)
#   4. After UNWIND_CROSS_AFTER failed limit attempts → cross the spread

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TickerRisk:
    ticker: str
    inventory: int = 0
    cost_basis: float = 0.0        # average cost per share
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    is_stopped: bool = False        # stop-loss triggered
    unwind_attempts: int = 0        # number of unwind limit orders placed
    last_unwind_time: float = 0.0   # monotonic


@dataclass
class RiskState:
    tickers: dict[str, TickerRisk] = field(default_factory=dict)
    portfolio_stopped: bool = False  # full portfolio stop
    total_unrealized_pnl: float = 0.0
    total_realized_pnl: float = 0.0


class RiskManager:
    """
    Monitors P&L and triggers stops + unwinds.

    Unwind logic (per Cartea & Jaimungal 2015 §7 on liquidation):
      - Post aggressive limit orders inside the spread
      - Retry every UNWIND_RETRY_INTERVAL seconds
      - After UNWIND_CROSS_AFTER attempts, cross the spread entirely
      - Never let a losing position compound indefinitely
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = RiskState()
        self._lock = asyncio.Lock()

    def get_ticker_risk(self, ticker: str) -> TickerRisk:
        if ticker not in self.state.tickers:
            self.state.tickers[ticker] = TickerRisk(ticker=ticker)
        return self.state.tickers[ticker]

    async def update_position(
        self,
        ticker: str,
        inventory: int,
        cost_basis: float,
        mid_price: float,
    ):
        """Update position and recompute unrealized P&L."""
        async with self._lock:
            tr = self.get_ticker_risk(ticker)
            tr.inventory = inventory
            tr.cost_basis = cost_basis

            if inventory != 0 and cost_basis > 0 and mid_price > 0:
                tr.unrealized_pnl = (mid_price - cost_basis) * inventory
            else:
                tr.unrealized_pnl = 0.0

            self._recompute_portfolio_pnl()
            self._check_stops(ticker)

    def _recompute_portfolio_pnl(self):
        self.state.total_unrealized_pnl = sum(
            tr.unrealized_pnl for tr in self.state.tickers.values()
        )
        self.state.total_realized_pnl = sum(
            tr.realized_pnl for tr in self.state.tickers.values()
        )

    def _check_stops(self, ticker: str):
        tr = self.get_ticker_risk(ticker)

        # Per-ticker stop
        if (
            not tr.is_stopped
            and tr.unrealized_pnl < -self.cfg.STOP_LOSS_PER_TICKER
        ):
            logger.warning(
                f"STOP LOSS TRIGGERED: {ticker} unrealized P&L = "
                f"${tr.unrealized_pnl:.2f} < -${self.cfg.STOP_LOSS_PER_TICKER:.0f}"
            )
            tr.is_stopped = True

        # Portfolio stop
        if (
            not self.state.portfolio_stopped
            and self.state.total_unrealized_pnl < -self.cfg.STOP_LOSS_PORTFOLIO
        ):
            logger.warning(
                f"PORTFOLIO STOP TRIGGERED: total unrealized P&L = "
                f"${self.state.total_unrealized_pnl:.2f} < "
                f"-${self.cfg.STOP_LOSS_PORTFOLIO:.0f}"
            )
            self.state.portfolio_stopped = True

    def is_quoting_allowed(self, ticker: str) -> bool:
        """Returns True if quoting is allowed for this ticker."""
        if self.state.portfolio_stopped:
            return False
        tr = self.get_ticker_risk(ticker)
        return not tr.is_stopped

    def needs_unwind(self, ticker: str) -> bool:
        """Returns True if this ticker needs active inventory reduction."""
        tr = self.get_ticker_risk(ticker)
        return tr.is_stopped and tr.inventory != 0

    def should_attempt_unwind(self, ticker: str) -> bool:
        """
        Rate-gate for unwind attempts.
        Returns True if enough time has passed since last unwind attempt.
        """
        tr = self.get_ticker_risk(ticker)
        elapsed = time.monotonic() - tr.last_unwind_time
        return elapsed >= self.cfg.UNWIND_RETRY_INTERVAL

    def should_cross_spread(self, ticker: str) -> bool:
        """
        Returns True if we've tried UNWIND_CROSS_AFTER limit orders
        and should now cross the spread aggressively.
        """
        tr = self.get_ticker_risk(ticker)
        return tr.unwind_attempts >= self.cfg.UNWIND_CROSS_AFTER

    def record_unwind_attempt(self, ticker: str):
        tr = self.get_ticker_risk(ticker)
        tr.unwind_attempts += 1
        tr.last_unwind_time = time.monotonic()
        logger.info(
            f"{ticker}: unwind attempt #{tr.unwind_attempts} "
            f"(cross after {self.cfg.UNWIND_CROSS_AFTER})"
        )

    def record_fill(self, ticker: str, quantity: int, price: float, side: str):
        """
        Update realized P&L after a fill.
        Called when an order disappears from the open orders list.
        """
        tr = self.get_ticker_risk(ticker)
        if side == "sell" and tr.cost_basis > 0:
            realized = (price - tr.cost_basis) * quantity
            tr.realized_pnl += realized
            logger.info(
                f"{ticker}: FILL sell {quantity} @ {price:.4f} "
                f"cost={tr.cost_basis:.4f} realized P&L={realized:+.2f} "
                f"(cumulative={tr.realized_pnl:+.2f})"
            )
        elif side == "buy":
            logger.info(
                f"{ticker}: FILL buy {quantity} @ {price:.4f}"
            )
        # Reset unwind counter on successful fill
        if tr.is_stopped and tr.inventory == 0:
            logger.info(f"{ticker}: position cleared — stop lifted")
            tr.is_stopped = False
            tr.unwind_attempts = 0

    def reset_stop(self, ticker: str):
        """Manually reset a stop (e.g. after successful unwind)."""
        tr = self.get_ticker_risk(ticker)
        tr.is_stopped = False
        tr.unwind_attempts = 0
        logger.info(f"{ticker}: stop manually reset")

    def reset_portfolio_stop(self):
        """Reset portfolio stop — use with caution."""
        self.state.portfolio_stopped = False
        logger.info("Portfolio stop manually reset")

    def log_risk_summary(self):
        logger.info(
            f"RISK SUMMARY | "
            f"total_unrealized={self.state.total_unrealized_pnl:+.2f} "
            f"total_realized={self.state.total_realized_pnl:+.2f} "
            f"portfolio_stopped={self.state.portfolio_stopped}"
        )
        for ticker, tr in self.state.tickers.items():
            if tr.inventory != 0 or tr.unrealized_pnl != 0 or tr.is_stopped:
                logger.info(
                    f"  {ticker}: inv={tr.inventory} "
                    f"cost={tr.cost_basis:.4f} "
                    f"unrealized={tr.unrealized_pnl:+.2f} "
                    f"realized={tr.realized_pnl:+.2f} "
                    f"stopped={tr.is_stopped}"
                )
