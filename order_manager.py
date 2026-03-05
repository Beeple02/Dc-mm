# order_manager.py

import asyncio
import logging
import time

from state import BotState, OpenOrder
from quoting import OptimalQuotes, validate_quotes

logger = logging.getLogger(__name__)


class OrderManager:
    def __init__(self, api_client, state: BotState, config):
        self.api = api_client
        self.state = state
        self.config = config
        self._order_times: list[float] = []

    def _can_place_order(self) -> bool:
        now = time.monotonic()
        self._order_times = [t for t in self._order_times if now - t < 60.0]
        return len(self._order_times) < self.config.MAX_ORDERS_PER_MINUTE

    def _record_order(self):
        self._order_times.append(time.monotonic())

    async def refresh_quotes(self, ticker: str, quotes: OptimalQuotes, quote_qty: int):
        """
        Cancel stale resting orders and post updated quotes.

        Gate 1: should_requote() — checks minimum time interval and price move.
                If this returns False, we do nothing (no cancels, no posts).
        Gate 2: validate_quotes() — sanity checks final bid/ask against mid.
                If this fails, we cancel existing orders but do NOT post new ones
                (better to have no quote than a crazy quote).
        """
        ts = self.state.tickers.get(ticker)
        if not ts:
            return

        new_bid = round(quotes.bid, 4)
        new_ask = round(quotes.ask, 4)
        mid = ts.mid or ts.market_price

        # Gate 1: time + price move check
        should, reason = self.state.should_requote(ticker, new_bid, new_ask)
        if not should:
            logger.debug(f"{ticker}: skip requote — {reason}")
            return

        # Gate 2: sanity check final prices
        valid, why = validate_quotes(ticker, new_bid, new_ask, mid or 1.0, self.config)
        if not valid:
            logger.warning(
                f"{ticker}: QUOTE SUPPRESSED — {why} "
                f"(bid={new_bid:.4f} ask={new_ask:.4f} mid={mid:.4f})"
            )
            # Cancel any existing crazy quotes but don't post new ones
            if ts.bid_order:
                await self._cancel_order(ts.bid_order.order_id, ticker, "buy")
            if ts.ask_order:
                await self._cancel_order(ts.ask_order.order_id, ticker, "sell")
            return

        logger.info(f"{ticker}: requoting — {reason}")

        # Cancel existing orders
        if ts.bid_order:
            await self._cancel_order(ts.bid_order.order_id, ticker, "buy")
        if ts.ask_order:
            await self._cancel_order(ts.ask_order.order_id, ticker, "sell")

        await asyncio.sleep(0.2)

        # FIX 2: bid=-1 or ask=inf signals that this side was suppressed
        # by the price-vs-mu gate — skip posting that side entirely
        bid_suppressed = new_bid < 0
        ask_suppressed = new_ask == float('inf')

        if not bid_suppressed:
            await self._place_bid(ticker, new_bid, quote_qty)
        else:
            logger.info(f"{ticker}: skipping bid post (price/mu gate suppressed)")

        if not ask_suppressed:
            await self._place_ask(ticker, new_ask, quote_qty, ts.inventory)
        else:
            logger.info(f"{ticker}: skipping ask post (price/mu gate suppressed)")

        # Record the quote so should_requote() knows when we last posted
        self.state.record_quote(ticker)

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
        # Inventory ceiling check
        cal = self.state.calibration
        if cal and ticker in cal.tickers:
            q_max = cal.tickers[ticker].q_max
            ts = self.state.tickers.get(ticker)
            if ts and ts.inventory >= q_max:
                logger.info(f"{ticker}: at inventory ceiling ({ts.inventory}/{q_max}), skipping bid")
                return

        # Cash check
        cost = price * qty * (1.0 + self.config.COMMISSION_RATE)
        if self.state.cash_available < cost:
            max_qty = int(self.state.cash_available / (price * (1.0 + self.config.COMMISSION_RATE)))
            if max_qty < 1:
                logger.info(f"{ticker}: insufficient cash for bid (need ${cost:.2f}, have ${self.state.cash_available:.2f})")
                return
            qty = max_qty

        if not self._can_place_order():
            logger.warning(f"Rate limit: skipping bid for {ticker}")
            return

        try:
            result = await self.api.place_buy_limit(ticker, qty, price, self.config.ORDER_EXPIRY_HOURS)
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
        if inventory < 1:
            logger.info(f"{ticker}: no inventory to post ask")
            return

        qty = min(qty, inventory)

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
            result = await self.api.place_sell_limit(ticker, qty, price, self.config.ORDER_EXPIRY_HOURS)
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
        Bergault Ch.2 — go active when inventory breaches threshold.

        FIX 4: Uses aggressive limit orders instead of market orders.
        NER's /orders/sell_market returns 400 unless a counterparty exists.
        We instead post a limit order at a discount to the best bid (for sells)
        or a premium to the best ask (for buys) to get fast fills without
        depending on the market order endpoint.
        """
        cal = self.state.calibration
        if not cal or ticker not in cal.tickers:
            return

        ts = self.state.tickers.get(ticker)
        if not ts:
            return

        # FIX 3: Don't attempt unwind if ticker is paused due to adverse selection
        if self.state.is_paused(ticker):
            return

        q_max = cal.tickers[ticker].q_max
        threshold = q_max * self.config.ACTIVE_THRESHOLD_PCT
        target = int(q_max * 0.5)

        if ts.inventory > threshold:
            excess = ts.inventory - target
            qty = max(1, int(excess * self.config.ACTIVE_UNWIND_FRACTION))
            qty = min(qty, getattr(self.config, 'MAX_QUOTE_QTY', 5))

            # FIX 4: Use aggressive limit at best bid minus discount
            mid = ts.mid or ts.market_price or 1.0
            best_bid = ts.best_bid or (mid * 0.99)
            discount = getattr(self.config, 'UNWIND_LIMIT_DISCOUNT', 0.01)
            unwind_price = round(max(0.01, best_bid * (1.0 - discount)), 4)
            expiry = getattr(self.config, 'UNWIND_LIMIT_EXPIRY_HOURS', 0.25)

            logger.info(
                f"{ticker}: ACTIVE UNWIND (long) inv={ts.inventory} → "
                f"sell limit {qty} @ {unwind_price:.4f} (best_bid={best_bid:.4f})"
            )
            if not self._can_place_order():
                return
            try:
                result = await self.api.place_sell_limit(ticker, qty, unwind_price, expiry)
                self._record_order()
                logger.info(f"{ticker}: unwind sell limit placed → {result.get('order_id', '?')}")
            except Exception as e:
                logger.error(f"Active unwind sell limit {ticker}: {e}")

        elif ts.inventory < -threshold:
            excess = abs(ts.inventory) - target
            qty = max(1, int(excess * self.config.ACTIVE_UNWIND_FRACTION))
            qty = min(qty, getattr(self.config, 'MAX_QUOTE_QTY', 5))

            # FIX 4: Use aggressive limit at best ask plus discount
            mid = ts.mid or ts.market_price or 1.0
            best_ask = ts.best_ask or (mid * 1.01)
            discount = getattr(self.config, 'UNWIND_LIMIT_DISCOUNT', 0.01)
            unwind_price = round(best_ask * (1.0 + discount), 4)
            expiry = getattr(self.config, 'UNWIND_LIMIT_EXPIRY_HOURS', 0.25)

            logger.info(
                f"{ticker}: ACTIVE UNWIND (short) inv={ts.inventory} → "
                f"buy limit {qty} @ {unwind_price:.4f} (best_ask={best_ask:.4f})"
            )
            if not self._can_place_order():
                return
            try:
                result = await self.api.place_buy_limit(ticker, qty, unwind_price, expiry)
                self._record_order()
                logger.info(f"{ticker}: unwind buy limit placed → {result.get('order_id', '?')}")
            except Exception as e:
                logger.error(f"Active unwind buy limit {ticker}: {e}")

    def compute_quote_quantity(self, ticker: str, allocated_capital: float, price: float) -> int:
        """
        FIX 1: Cap qty to MAX_QUOTE_QTY to prevent large block adverse selection.
        Previously qty scaled with capital, allowing 98-share bids on RTG.
        """
        if price <= 0:
            return 1
        qty = int(allocated_capital / (2.0 * price))
        qty = max(1, qty)
        max_qty = getattr(self.config, 'MAX_QUOTE_QTY', 5)
        return min(qty, max_qty)
