# state.py
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class OpenOrder:
    order_id: str
    ticker: str
    side: str          # "buy" or "sell"
    price: float
    quantity: int
    placed_at: float   # monotonic
    expiry_hours: float
    is_unwind: bool = False   # True if this is a stop-loss unwind order


@dataclass
class TickerState:
    ticker: str
    inventory: int = 0
    cost_basis: float = 0.0
    mid: Optional[float] = None
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    market_price: float = 0.0
    bid_order: Optional[OpenOrder] = None
    ask_order: Optional[OpenOrder] = None
    unwind_order: Optional[OpenOrder] = None
    last_quote_time: float = 0.0
    last_quoted_mid: Optional[float] = None
    allocated_capital: float = 0.0
    q_max: int = 10
    total_shares: int = 1000
    source: str = "ner"              # "ner" or "tse" — determines which client to use
    # Track previous inventory for adverse selection detection
    _prev_inventory: int = 0


class BotState:
    def __init__(self, cfg):
        self.cfg = cfg
        self._lock = asyncio.Lock()

        self.tickers: dict[str, TickerState] = {}
        self.cash_available: float = 0.0
        self.cash_reserved: float = 0.0
        self.total_equity: float = 0.0

        self.session_start: float = time.monotonic()

        # Set by calibration/scorer
        self.calibration = None
        self.scorer_result = None

        # Webhook liveness
        self.webhook_alive: bool = False
        self.last_webhook_time: float = 0.0

    def t_remaining(self) -> float:
        """Hours remaining in current T-horizon window (rolling 24h)."""
        elapsed = (time.monotonic() - self.session_start) / 3600.0
        remaining = self.cfg.T_HORIZON_HOURS - (elapsed % self.cfg.T_HORIZON_HOURS)
        return max(remaining, 1e-3)

    def reset_session(self):
        self.session_start = time.monotonic()
        logger.info(f"Session reset. T={self.cfg.T_HORIZON_HOURS}h")

    # ── Portfolio sync ────────────────────────────────────────────────────────

    async def update_from_portfolio(self, portfolio: dict):
        async with self._lock:
            self.cash_available = portfolio.get("balance", 0.0)
            self.cash_reserved  = portfolio.get("reserved_balance", 0.0)
            self.total_equity   = portfolio.get("total_equity", 0.0)

            holdings = {h["ticker"]: h for h in portfolio.get("holdings", [])}
            for ticker, ts in self.tickers.items():
                if ticker in holdings:
                    ts._prev_inventory = ts.inventory
                    qty = holdings[ticker]["quantity"]
                    ts.inventory  = qty
                    # NER returns cost_basis as TOTAL cost (not per-share).
                    # Convert to per-share for P&L calculations.
                    total_cost = holdings[ticker].get("cost_basis", 0.0)
                    ts.cost_basis = (total_cost / qty) if qty > 0 else 0.0
                else:
                    ts._prev_inventory = ts.inventory
                    ts.inventory  = 0
                    ts.cost_basis = 0.0

    # ── Orderbook updates ─────────────────────────────────────────────────────

    async def update_orderbook(self, ticker: str, ob: dict):
        async with self._lock:
            if ticker not in self.tickers:
                self.tickers[ticker] = TickerState(ticker=ticker)
            ts = self.tickers[ticker]
            ts.best_bid    = ob.get("best_bid")
            ts.best_ask    = ob.get("best_ask")
            ts.mid         = ob.get("mid")
            if ob.get("market_price"):
                ts.market_price = ob["market_price"]

    async def update_market_price(self, ticker: str, price: float):
        async with self._lock:
            if ticker not in self.tickers:
                self.tickers[ticker] = TickerState(ticker=ticker)
            self.tickers[ticker].market_price = price

    # ── Order tracking ────────────────────────────────────────────────────────

    async def register_order(self, order: OpenOrder):
        async with self._lock:
            ts = self.tickers.get(order.ticker)
            if not ts:
                return
            if order.is_unwind:
                ts.unwind_order = order
            elif order.side == "buy":
                ts.bid_order = order
            else:
                ts.ask_order = order

    async def clear_order(self, ticker: str, side: str, is_unwind: bool = False):
        async with self._lock:
            ts = self.tickers.get(ticker)
            if not ts:
                return
            if is_unwind:
                ts.unwind_order = None
            elif side == "buy":
                ts.bid_order = None
            else:
                ts.ask_order = None

    async def reconcile_open_orders(self, open_orders: list[dict]) -> list[dict]:
        """
        Reconcile our tracked orders against the exchange's open order list.
        Returns list of (ticker, side, order) for orders that disappeared
        (assumed filled or expired).
        """
        async with self._lock:
            open_ids = {o["order_id"] for o in open_orders}
            filled_orders = []

            for ticker, ts in self.tickers.items():
                for attr, is_unwind in [
                    ("bid_order", False),
                    ("ask_order", False),
                    ("unwind_order", True),
                ]:
                    order = getattr(ts, attr)
                    if order and order.order_id not in open_ids:
                        side = "buy" if attr == "bid_order" else "sell"
                        logger.info(
                            f"Order {order.order_id} ({ticker} {side} @ {order.price}) "
                            f"disappeared — assumed filled/expired"
                        )
                        filled_orders.append({
                            "ticker": ticker,
                            "side": side,
                            "order": order,
                            "is_unwind": is_unwind,
                        })
                        setattr(ts, attr, None)

            return filled_orders

    # ── Requote gating ────────────────────────────────────────────────────────

    def should_requote(
        self, ticker: str, new_bid: float, new_ask: float
    ) -> tuple[bool, str]:
        """
        Returns (should_requote, reason).
        Only requote if:
          1. Minimum time interval has elapsed
          2. Mid has moved > QUOTE_STALE_THRESHOLD, OR no resting orders
        """
        ts = self.tickers.get(ticker)
        if not ts:
            return True, "no state"

        min_secs = self.cfg.MIN_REQUOTE_INTERVAL_MIN * 60
        since_last = time.monotonic() - ts.last_quote_time

        if since_last < min_secs:
            return False, f"too soon ({since_last:.0f}s < {min_secs:.0f}s)"

        has_orders = ts.bid_order is not None or ts.ask_order is not None
        if not has_orders:
            return True, "no resting orders"

        current_mid = ts.mid or ts.market_price
        if current_mid and ts.last_quoted_mid:
            move = abs(current_mid - ts.last_quoted_mid) / max(ts.last_quoted_mid, 1e-6)
            if move > self.cfg.QUOTE_STALE_THRESHOLD:
                return True, f"mid moved {move:.1%}"
            return False, f"mid only moved {move:.1%}"

        return True, "no prior mid recorded"

    def record_quote(self, ticker: str):
        ts = self.tickers.get(ticker)
        if ts:
            ts.last_quote_time  = time.monotonic()
            ts.last_quoted_mid  = ts.mid or ts.market_price

    # ── Adverse selection detection ───────────────────────────────────────────

    def detect_adverse_selection(self, ticker: str, jump_threshold: int) -> bool:
        """
        Returns True if inventory jumped suspiciously in the last poll cycle.
        This signals someone dumped a block into us (informed trading).
        """
        ts = self.tickers.get(ticker)
        if not ts:
            return False
        jump = abs(ts.inventory - ts._prev_inventory)
        if jump >= jump_threshold:
            logger.warning(
                f"{ticker}: ADVERSE SELECTION — inventory jumped "
                f"{ts._prev_inventory} → {ts.inventory} ({jump:+d} shares)"
            )
            return True
        return False
