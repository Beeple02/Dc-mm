# order_manager.py
import asyncio
import logging
import time

from state import BotState, OpenOrder
from quoting import OptimalQuotes
from risk import RiskManager

logger = logging.getLogger(__name__)


class OrderManager:

    def __init__(self, ner_client, state: BotState, risk: RiskManager, cfg):
        self.ner = ner_client
        self.state = state
        self.risk = risk
        self.cfg = cfg
        self._order_times: list[float] = []

    # ── Rate limiting ─────────────────────────────────────────────────────────

    def _can_place_order(self) -> bool:
        now = time.monotonic()
        self._order_times = [t for t in self._order_times if now - t < 60.0]
        return len(self._order_times) < self.cfg.MAX_ORDERS_PER_MINUTE

    def _record_order(self):
        self._order_times.append(time.monotonic())

    # ── Main quoting entry point ──────────────────────────────────────────────

    async def refresh_quotes(
        self,
        ticker: str,
        quotes: OptimalQuotes,
        qty: int,
    ):
        """
        Cancel stale orders and post fresh quotes if conditions are met.

        Gate 1: should_requote() — time + mid move check
        Gate 2: risk.is_quoting_allowed() — stop loss check
        Gate 3: Quote sanity (bid=-1 or ask=inf → suppress that side)
        """
        ts = self.state.tickers.get(ticker)
        if not ts:
            return

        # Gate 2: risk check first — never post into a stopped ticker
        if not self.risk.is_quoting_allowed(ticker):
            logger.debug(f"{ticker}: quoting blocked by risk manager")
            return

        new_bid = quotes.bid
        new_ask = quotes.ask
        mid = ts.mid or ts.market_price or 1.0

        # Gate 1: time + price move check
        should, reason = self.state.should_requote(ticker, new_bid, new_ask)
        if not should:
            logger.debug(f"{ticker}: skip requote — {reason}")
            return

        logger.info(
            f"{ticker}: REQUOTE — {reason} | "
            f"bid={new_bid if new_bid > 0 else 'SUPPRESSED'} "
            f"ask={new_ask if new_ask != float('inf') else 'SUPPRESSED'} | "
            f"r={quotes.reservation_price:.4f} γ={quotes.gamma_effective:.4f} "
            f"spread={quotes.spread_stressed:.4f} "
            f"f_inv={quotes.f_inventory:.2f} f_drift={quotes.f_drift:.2f}"
        )

        # Cancel existing quotes
        if ts.bid_order:
            await self._cancel(ts.bid_order.order_id, ticker, "buy")
        if ts.ask_order:
            await self._cancel(ts.ask_order.order_id, ticker, "sell")

        await asyncio.sleep(0.3)  # small pause between cancel and repost

        # Post bid
        if new_bid > 0 and not (new_bid < 0):
            await self._place_bid(ticker, new_bid, qty)
        else:
            logger.info(f"{ticker}: bid suppressed (sanity/inventory)")

        # Post ask (only if we have inventory)
        if new_ask != float('inf'):
            await self._place_ask(ticker, new_ask, qty)
        else:
            logger.info(f"{ticker}: ask suppressed (sanity/inventory)")

        self.state.record_quote(ticker)

    # ── Unwind logic ──────────────────────────────────────────────────────────

    async def attempt_unwind(self, ticker: str):
        """
        Aggressive inventory unwind after stop-loss trigger.

        Strategy (Cartea & Jaimungal 2015 §7):
          1. First UNWIND_CROSS_AFTER attempts: limit order inside the spread
             (bid - UNWIND_DISCOUNT for sells, ask + UNWIND_DISCOUNT for buys)
          2. After that: cross the spread (market order or limit at best price)

        This guarantees we eventually get out regardless of liquidity.
        """
        ts = self.state.tickers.get(ticker)
        if not ts:
            return

        if ts.inventory == 0:
            return

        if not self.risk.needs_unwind(ticker):
            return

        if not self.risk.should_attempt_unwind(ticker):
            return

        # Cancel existing MM quotes first — don't want conflicting orders
        if ts.bid_order:
            await self._cancel(ts.bid_order.order_id, ticker, "buy")
        if ts.ask_order:
            await self._cancel(ts.ask_order.order_id, ticker, "sell")

        # Cancel previous unwind order if still open
        if ts.unwind_order:
            await self._cancel(ts.unwind_order.order_id, ticker,
                               ts.unwind_order.side, is_unwind=True)

        mid = ts.mid or ts.market_price or 1.0
        inv = ts.inventory

        cross = self.risk.should_cross_spread(ticker)

        if inv > 0:
            # Long inventory: sell to unwind
            if cross:
                # Cross: sell at best bid (take the price)
                price = ts.best_bid or (mid * 0.99)
                logger.warning(
                    f"{ticker}: CROSSING SPREAD to unwind long {inv} shares "
                    f"@ {price:.4f}"
                )
            else:
                # Aggressive limit: just inside best bid
                best_bid = ts.best_bid or (mid * 0.99)
                price = round(max(0.01, best_bid * (1 - self.cfg.UNWIND_DISCOUNT)), 4)
                logger.info(
                    f"{ticker}: UNWIND LIMIT sell {inv} @ {price:.4f} "
                    f"(best_bid={best_bid:.4f} attempt={self.risk.get_ticker_risk(ticker).unwind_attempts + 1})"
                )

            await self._place_unwind_sell(ticker, abs(inv), price)

        elif inv < 0:
            # Short inventory: buy to unwind
            if cross:
                price = ts.best_ask or (mid * 1.01)
                logger.warning(
                    f"{ticker}: CROSSING SPREAD to unwind short {inv} shares "
                    f"@ {price:.4f}"
                )
            else:
                best_ask = ts.best_ask or (mid * 1.01)
                price = round(best_ask * (1 + self.cfg.UNWIND_DISCOUNT), 4)
                logger.info(
                    f"{ticker}: UNWIND LIMIT buy {abs(inv)} @ {price:.4f} "
                    f"(best_ask={best_ask:.4f} attempt={self.risk.get_ticker_risk(ticker).unwind_attempts + 1})"
                )

            await self._place_unwind_buy(ticker, abs(inv), price)

        self.risk.record_unwind_attempt(ticker)

    # ── Private order placement ───────────────────────────────────────────────

    async def _cancel(
        self,
        order_id: str,
        ticker: str,
        side: str,
        is_unwind: bool = False,
    ):
        if not self._can_place_order():
            logger.warning(f"Rate limit: skipping cancel {order_id}")
            return
        try:
            await self.ner.cancel_order(order_id)
            self._record_order()
        except Exception as e:
            logger.warning(f"Cancel {order_id} failed: {e}")
        finally:
            await self.state.clear_order(ticker, side, is_unwind=is_unwind)

    async def _place_bid(self, ticker: str, price: float, qty: int):
        ts = self.state.tickers.get(ticker)
        if not ts:
            return

        # Cash check
        cost = price * qty * (1 + self.cfg.COMMISSION_RATE)
        if self.state.cash_available < cost:
            max_qty = int(self.state.cash_available / (price * (1 + self.cfg.COMMISSION_RATE)))
            if max_qty < 1:
                logger.info(f"{ticker}: insufficient cash for bid (need ${cost:.2f})")
                return
            qty = max_qty

        if not self._can_place_order():
            logger.warning(f"Rate limit: skipping bid {ticker}")
            return

        try:
            result = await self.ner.place_buy_limit(
                ticker, qty, price, self.cfg.ORDER_EXPIRY_HOURS
            )
            self._record_order()
            await self.state.register_order(OpenOrder(
                order_id=result["order_id"],
                ticker=ticker,
                side="buy",
                price=price,
                quantity=qty,
                placed_at=time.monotonic(),
                expiry_hours=self.cfg.ORDER_EXPIRY_HOURS,
            ))
        except Exception as e:
            logger.error(f"Failed to place bid {ticker} @ {price}: {e}")

    async def _place_ask(self, ticker: str, price: float, qty: int):
        ts = self.state.tickers.get(ticker)
        if not ts:
            return

        if ts.inventory < 1:
            logger.info(f"{ticker}: no inventory for ask")
            return

        qty = min(qty, ts.inventory)

        if not self._can_place_order():
            logger.warning(f"Rate limit: skipping ask {ticker}")
            return

        try:
            result = await self.ner.place_sell_limit(
                ticker, qty, price, self.cfg.ORDER_EXPIRY_HOURS
            )
            self._record_order()
            await self.state.register_order(OpenOrder(
                order_id=result["order_id"],
                ticker=ticker,
                side="sell",
                price=price,
                quantity=qty,
                placed_at=time.monotonic(),
                expiry_hours=self.cfg.ORDER_EXPIRY_HOURS,
            ))
        except Exception as e:
            logger.error(f"Failed to place ask {ticker} @ {price}: {e}")

    async def _place_unwind_sell(self, ticker: str, qty: int, price: float):
        if not self._can_place_order():
            logger.warning(f"Rate limit: skipping unwind sell {ticker}")
            return
        try:
            result = await self.ner.place_sell_limit(
                ticker, qty, price, self.cfg.UNWIND_ORDER_EXPIRY
            )
            self._record_order()
            await self.state.register_order(OpenOrder(
                order_id=result["order_id"],
                ticker=ticker,
                side="sell",
                price=price,
                quantity=qty,
                placed_at=time.monotonic(),
                expiry_hours=self.cfg.UNWIND_ORDER_EXPIRY,
                is_unwind=True,
            ))
        except Exception as e:
            logger.error(f"Unwind sell {ticker} failed: {e}")
            # Last resort: try market order
            if self.risk.should_cross_spread(ticker):
                await self._try_market_sell(ticker, qty)

    async def _place_unwind_buy(self, ticker: str, qty: int, price: float):
        if not self._can_place_order():
            logger.warning(f"Rate limit: skipping unwind buy {ticker}")
            return
        try:
            result = await self.ner.place_buy_limit(
                ticker, qty, price, self.cfg.UNWIND_ORDER_EXPIRY
            )
            self._record_order()
            await self.state.register_order(OpenOrder(
                order_id=result["order_id"],
                ticker=ticker,
                side="buy",
                price=price,
                quantity=qty,
                placed_at=time.monotonic(),
                expiry_hours=self.cfg.UNWIND_ORDER_EXPIRY,
                is_unwind=True,
            ))
        except Exception as e:
            logger.error(f"Unwind buy {ticker} failed: {e}")
            if self.risk.should_cross_spread(ticker):
                await self._try_market_buy(ticker, qty)

    async def _try_market_sell(self, ticker: str, qty: int):
        """Last resort market sell — may fail on NER if no counterparty."""
        try:
            await self.ner.place_sell_market(ticker, qty)
            self._record_order()
            logger.warning(f"{ticker}: emergency market sell {qty}")
        except Exception as e:
            logger.error(f"{ticker}: market sell failed (no counterparty?): {e}")

    async def _try_market_buy(self, ticker: str, qty: int):
        """Last resort market buy."""
        try:
            await self.ner.place_buy_market(ticker, qty)
            self._record_order()
            logger.warning(f"{ticker}: emergency market buy {qty}")
        except Exception as e:
            logger.error(f"{ticker}: market buy failed (no counterparty?): {e}")
