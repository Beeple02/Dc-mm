# state.py
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
    side: str
    price: float
    quantity: int
    placed_at: float
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
    last_quote_time: float = 0.0       # monotonic time of last requote
    last_quoted_mid: Optional[float] = None  # mid price when quotes were last posted
    allocated_capital: float = 0.0
    total_shares: int = 1000


class BotState:
    def __init__(self, config):
        self.config = config
        self._lock = asyncio.Lock()

        self.tickers: dict[str, TickerState] = {}
        self.cash_available: float = 0.0
        self.cash_reserved: float = 0.0
        self.total_equity: float = 0.0

        self.session_start: float = time.monotonic()
        self.T_hours: float = config.T_HORIZON_HOURS

        self.calibration = None
        self.ode_solutions: dict = {}
        self.riccati_A: Optional[np.ndarray] = None

        self.webhook_alive: bool = False
        self.last_webhook_time: float = 0.0

        # FIX 3: Adverse selection pause registry
        # Maps ticker → monotonic time when pause expires
        self._paused_until: dict[str, float] = {}

    def t_remaining(self) -> float:
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

    def should_requote(self, ticker: str, new_bid: float, new_ask: float) -> tuple[bool, str]:
        """
        Returns (should_requote, reason).

        Requote only if ALL of the following are true:
          1. Minimum time interval has elapsed since last requote
          2. Mid price has moved by more than QUOTE_STALE_THRESHOLD since last quote
             OR there are no resting orders at all

        t_remaining drift alone is NOT a reason to requote — on a 24h horizon
        the ODE output barely changes over 10 minutes, and requoting every 15s
        was generating the majority of all NER weekly order volume.
        """
        ts = self.tickers.get(ticker)
        if not ts:
            return True, "no state"

        # Check 1: minimum time interval
        min_interval_secs = self.config.MIN_REQUOTE_INTERVAL_MINUTES * 60
        time_since_last = time.monotonic() - ts.last_quote_time
        if time_since_last < min_interval_secs:
            return False, f"too soon ({time_since_last:.0f}s < {min_interval_secs:.0f}s)"

        # Check 2: no resting orders → always post
        has_bid = ts.bid_order is not None
        has_ask = ts.ask_order is not None
        if not has_bid and not has_ask:
            return True, "no resting orders"

        # Check 3: mid price moved enough since last quote
        current_mid = ts.mid or ts.market_price
        if current_mid and ts.last_quoted_mid:
            rel_move = abs(current_mid - ts.last_quoted_mid) / max(ts.last_quoted_mid, 1e-6)
            if rel_move > self.config.QUOTE_STALE_THRESHOLD:
                return True, f"mid moved {rel_move:.1%}"
            else:
                return False, f"mid only moved {rel_move:.1%} < {self.config.QUOTE_STALE_THRESHOLD:.1%}"
        else:
            # No last mid recorded — post
            return True, "no prior mid recorded"

    def record_quote(self, ticker: str):
        """Call after successfully posting quotes for a ticker."""
        ts = self.tickers.get(ticker)
        if ts:
            ts.last_quote_time = time.monotonic()
            ts.last_quoted_mid = ts.mid or ts.market_price

    def pause_ticker(self, ticker: str, minutes: float):
        """Pause quoting for a ticker for `minutes` minutes (adverse selection cooldown)."""
        self._paused_until[ticker] = time.monotonic() + minutes * 60.0
        logger.warning(f"{ticker}: quoting paused for {minutes:.0f} min (adverse selection)")

    def is_paused(self, ticker: str) -> bool:
        """Returns True if the ticker is currently in an adverse selection cooldown."""
        until = self._paused_until.get(ticker)
        if until is None:
            return False
        if time.monotonic() >= until:
            del self._paused_until[ticker]
            logger.info(f"{ticker}: adverse selection pause lifted — resuming quoting")
            return False
        remaining = (until - time.monotonic()) / 60.0
        logger.debug(f"{ticker}: still paused ({remaining:.1f} min remaining)")
        return True

    def get_inventory_vector(self, ticker_list: list[str]) -> np.ndarray:
        return np.array(
            [self.tickers.get(t, TickerState(t)).inventory for t in ticker_list],
            dtype=float,
        )
