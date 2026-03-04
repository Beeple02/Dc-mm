# order_manager.py
# Handles the cancel-replace cycle, active inventory management (Ch.2),
# and rate-limit-aware order placement.

import asyncio
import logging
import time

from state import BotState, OpenOrder
from quoting import OptimalQuotes

logger = logging.getLogger(__name__)


class OrderManager:
    """
    Manages the full lifecycle of MM orders:
      1. Cancel stale quotes
      2. Place fresh quotes within rate limit budget
      3. Active unwind when inventory breaches threshold (Bergault Ch.2)
    """

    def __init__(self, api_client, state: BotState, config):
        self.api = api_client
        self.state = state
        self.config = config
        self._order_times: list[float] = []

    def _can_place_order(self) -> bool:
        """Return True if placing an order won't exceed MAX_ORDERS_PER_MINUTE."""
        now = time.monotonic()
        self._order_times = [t for t in self._order_times if now - t < 60.0]
        return len(self._order_times) < self.config.MAX_ORDERS_PER_MINUTE

    def _record_order(self):
        self._order_times.append(time.monotonic())

    async def refresh_quotes(self, ticker: str, quotes: OptimalQuotes, quote_qty: int):
        """
        Cancel stale resting orders and post updated quotes.
        Only acts if the theoretical quotes have drifted beyond QUOTE_STALE_THRESHOLD.
        """
        ts = self.state.tickers.get(ticker)
        if not ts:
            return

        new_bid = round(quotes.bid, 4)
        new_ask = round(quotes.ask, 4)

        if not self.state.is_quote_stale(ticker, new_bid, new_ask):
            logger.debug(f"{ticker}: quotes still fresh, no refresh needed")
            return

        # Cancel existing bid
        if ts.bid_order:
            await self._cancel_order(ts.bid_order.order_id, ticker, "buy")

        # Cancel existing ask
        if ts.ask_order:
            await self._cancel_order(ts.ask_order.order_id, ticker, "sell")

        # Brief pause after cancels before placing new orders
        await asyncio.sleep(0.15)

        # Place new bid
        await self._place_bid(ticker, new_bid, quote_qty)

        # Place new ask
        await self._place_ask(ticker, new_ask, quote_qty, ts.inventory)

        ts.last_quote_time = time.monotonic()

    async def _cancel_order(self, order_id: str, ticker: str, side: str):
        if not self._can_place_order():
            logger.warning(f"Rate limit: skipping cancel of {order_id}")
            return
        try:
            await self.api.cancel_order(order_id)
            self._record_order()
            await self.state.clear_order(ticker, side)
        except Exception as e:
            logger.warning(f"Cancel {order_id} failed: {e} — clearing from state anyway")
            await self.state.clear_order(ticker, side)

    async def _place_bid(self, ticker: str, price: float, qty: int):
        """Place a limit buy order subject to cash and inventory ceiling checks."""
        # Inventory ceiling check
        cal = self.state.calibration
        if cal and ticker in cal.tickers:
            q_max = cal.tickers[ticker].q_max
            ts = self.state.tickers.get(ticker)
            if ts and ts.inventory >= q_max:
                logger.info(
                    f"{ticker}: at inventory ceiling ({ts.inventory}/{q_max}), "
                    f"skipping bid"
                )
                return

        # Cash check (include commission cost)
        cost = price * qty * (1.0 + self.config.COMMISSION_RATE)
        if self.state.cash_available < cost:
            max_qty = int(
                self.state.cash_available
                / (price * (1.0 + self.config.COMMISSION_RATE))
            )
            if max_qty < 1:
                logger.info(
                    f"{ticker}: insufficient cash for bid "
                    f"(need ${cost:.2f}, have ${self.state.cash_available:.2f})"
                )
                return
            qty = max_qty
            logger.debug(f"{ticker}: bid qty scaled to {qty} due to cash constraint")

        if not self._can_place_order():
            logger.warning(f"Rate limit: skipping bid for {ticker}")
            return

        try:
            result = await self.api.place_buy_limit(
                ticker, qty, price, self.config.ORDER_EXPIRY_HOURS
            )
            self._record_order()
            await self.state.register_order(OpenOrder(
                order_id=result["order_id"],
                ticker=ticker,
                side="buy",
                price=price,
                quantity=qty,
                placed_at=time.monotonic(),
                expiry_hours=self.config.ORDER_EXPIRY_HOURS,
            ))
        except Exception as e:
            logger.error(f"Failed to place bid {ticker} @ {price}: {e}")

    async def _place_ask(self, ticker: str, price: float, qty: int, inventory: int):
        """Place a limit sell order subject to inventory floor check."""
        if inventory < 1:
            logger.info(f"{ticker}: no inventory to post ask")
            return

        qty = min(qty, inventory)

        # Inventory floor check
        cal = self.state.calibration
        if cal and ticker in cal.tickers:
            q_max = cal.tickers[ticker].q_max
            if inventory <= -q_max:
                logger.info(f"{ticker}: at inventory floor, skipping ask")
                return

        if not self._can_place_order():
            logger.warning(f"Rate limit: skipping ask for {ticker}")
            return

        try:
            result = await self.api.place_sell_limit(
                ticker, qty, price, self.config.ORDER_EXPIRY_HOURS
            )
            self._record_order()
            await self.state.register_order(OpenOrder(
                order_id=result["order_id"],
                ticker=ticker,
                side="sell",
                price=price,
                quantity=qty,
                placed_at=time.monotonic(),
                expiry_hours=self.config.ORDER_EXPIRY_HOURS,
            ))
        except Exception as e:
            logger.error(f"Failed to place ask {ticker} @ {price}: {e}")

    async def active_inventory_management(self, ticker: str):
        """
        Chapter 2 active inventory management.

        When inventory exceeds ACTIVE_THRESHOLD_PCT × q_max, the MM switches
        from purely passive (limit orders) to also actively crossing the spread
        via market orders. This approximates the optimal active trading rate ν*(t)
        from Bergault Ch.2 with a threshold rule that fires when inventory risk
        is severe enough to justify the cost of crossing the spread.

        Paper reference: Bergault Ch.2 §2.2 — passive-to-active market maker.
        """
        cal = self.state.calibration
        if not cal or ticker not in cal.tickers:
            return

        ts = self.state.tickers.get(ticker)
        if not ts:
            return

        q_max = cal.tickers[ticker].q_max
        threshold = q_max * self.config.ACTIVE_THRESHOLD_PCT
        target = int(q_max * 0.5)  # Target: bring back to 50% of q_max

        if ts.inventory > threshold:
            excess = ts.inventory - target
            qty = max(1, int(excess * self.config.ACTIVE_UNWIND_FRACTION))
            logger.info(
                f"{ticker}: ACTIVE UNWIND (long) inventory={ts.inventory} "
                f"threshold={threshold:.0f} → selling {qty} at market"
            )
            if not self._can_place_order():
                return
            try:
                await self.api.place_sell_market(ticker, qty)
                self._record_order()
            except Exception as e:
                logger.error(f"Active sell market {ticker}: {e}")

        elif ts.inventory < -threshold:
            excess = abs(ts.inventory) - target
            qty = max(1, int(excess * self.config.ACTIVE_UNWIND_FRACTION))
            logger.info(
                f"{ticker}: ACTIVE UNWIND (short) inventory={ts.inventory} "
                f"threshold={-threshold:.0f} → buying {qty} at market"
            )
            if not self._can_place_order():
                return
            try:
                await self.api.place_buy_market(ticker, qty)
                self._record_order()
            except Exception as e:
                logger.error(f"Active buy market {ticker}: {e}")

    def compute_quote_quantity(
        self, ticker: str, allocated_capital: float, price: float
    ) -> int:
        """
        Quote quantity based on allocated capital.
        qty = floor(allocated_capital / (2 × price))
        Factor of 2: capital must cover both the bid reservation and the
        inventory we might acquire before selling.
        """
        if price <= 0:
            return 1
        qty = int(allocated_capital / (2.0 * price))
        return max(1, qty)
