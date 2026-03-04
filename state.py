# state.py
# Tracks live bot state: inventory, open orders, session timing, fills.

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OpenOrder:
    order_id: str
    ticker: str
    side: str           # "buy" or "sell"
    price: float
    quantity: int
    placed_at: float    # monotonic time
    expiry_hours: int


@dataclass
class TickerState:
    ticker: str
    inventory: int = 0
    cost_basis: float = 0.0
    market_price: float = 0.0
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    mid: Optional[float] = None
    bid_order: Optional[OpenOrder] = None
    ask_order: Optional[OpenOrder] = None
    last_quote_time: float = 0.0
    allocated_capital: float = 0.0
    total_shares: int = 1000


class BotState:
    """
    Central state container for the MM bot.
    All mutation goes through async methods protected by a lock.
    """

    def __init__(self, config):
        self.config = config
        self._lock = asyncio.Lock()

        self.tickers: dict[str, TickerState] = {}
        self.cash_available: float = 0.0
        self.cash_reserved: float = 0.0
        self.total_equity: float = 0.0

        # Session timing
        self.session_start: float = time.monotonic()
        self.T_hours: float = config.T_HORIZON_HOURS

        # Calibration (set after calibration completes)
        self.calibration = None
        self.ode_solutions: dict = {}
        self.riccati_A: Optional[np.ndarray] = None

        # Webhook liveness
        self.webhook_alive: bool = False
        self.last_webhook_time: float = 0.0

    def t_remaining(self) -> float:
        """Time remaining in current T-horizon session (hours)."""
        elapsed = (time.monotonic() - self.session_start) / 3600.0
        remaining = self.T_hours - (elapsed % self.T_hours)
        return max(remaining, 1e-3)

    def reset_session(self):
        self.session_start = time.monotonic()
        logger.info(f"New MM session started. T={self.T_hours}h")

    async def update_from_portfolio(self, portfolio: dict):
        async with self._lock:
            self.cash_available = portfolio.get("balance", 0.0)
            self.cash_reserved = portfolio.get("reserved_balance", 0.0)
            self.total_equity = portfolio.get("total_equity", 0.0)
            holdings = {h["ticker"]: h for h in portfolio.get("holdings", [])}
            for ticker, ts in self.tickers.items():
                if ticker in holdings:
                    ts.inventory = holdings[ticker]["quantity"]
                    ts.cost_basis = holdings[ticker].get("cost_basis", 0.0)
                else:
                    ts.inventory = 0

    async def update_orderbook(self, ticker: str, ob: dict):
        async with self._lock:
            if ticker not in self.tickers:
                self.tickers[ticker] = TickerState(ticker=ticker)
            ts = self.tickers[ticker]
            ts.best_bid = ob.get("best_bid")
            ts.best_ask = ob.get("best_ask")
            ts.mid = ob.get("mid")
            if ob.get("market_price"):
                ts.market_price = ob["market_price"]

    async def update_market_price(self, ticker: str, price: float):
        async with self._lock:
            if ticker not in self.tickers:
                self.tickers[ticker] = TickerState(ticker=ticker)
            self.tickers[ticker].market_price = price

    async def register_order(self, order: OpenOrder):
        async with self._lock:
            ts = self.tickers.get(order.ticker)
            if not ts:
                return
            if order.side == "buy":
                ts.bid_order = order
            else:
                ts.ask_order = order

    async def clear_order(self, ticker: str, side: str):
        async with self._lock:
            ts = self.tickers.get(ticker)
            if not ts:
                return
            if side == "buy":
                ts.bid_order = None
            else:
                ts.ask_order = None

    async def reconcile_open_orders(self, open_orders: list[dict]):
        """
        Compare known resting orders against the API's live open order list.
        Orders that have disappeared were either filled or expired — clear them.
        """
        async with self._lock:
            open_ids = {o["order_id"] for o in open_orders}
            for ticker, ts in self.tickers.items():
                for side_attr in ("bid_order", "ask_order"):
                    order = getattr(ts, side_attr)
                    if order and order.order_id not in open_ids:
                        side = "buy" if side_attr == "bid_order" else "sell"
                        logger.info(
                            f"Order {order.order_id} ({ticker} {side} @ {order.price}) "
                            f"disappeared — assumed filled or expired"
                        )
                        setattr(ts, side_attr, None)

    def is_quote_stale(self, ticker: str, new_bid: float, new_ask: float) -> bool:
        """
        True if the current resting quotes differ from the new theoretical
        quotes by more than QUOTE_STALE_THRESHOLD, or if no resting orders exist.
        """
        ts = self.tickers.get(ticker)
        if not ts:
            return True

        threshold = self.config.QUOTE_STALE_THRESHOLD

        if ts.bid_order:
            rel_diff = abs(ts.bid_order.price - new_bid) / max(new_bid, 0.01)
            if rel_diff > threshold:
                return True
        else:
            return True

        if ts.ask_order:
            rel_diff = abs(ts.ask_order.price - new_ask) / max(new_ask, 0.01)
            if rel_diff > threshold:
                return True
        else:
            return True

        return False

    def get_inventory_vector(self, ticker_list: list[str]) -> np.ndarray:
        return np.array(
            [self.tickers.get(t, TickerState(t)).inventory for t in ticker_list],
            dtype=float,
        )
