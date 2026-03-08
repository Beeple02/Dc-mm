# tse_client.py
# Async wrapper for The Stock Exchange (TSE) trading API.
# Uses TSE's OpenAPI schema: POST /api/v1/orders, DELETE /api/v1/orders/{id}
# Auth: X-API-Key header (same pattern as NER)
# Key differences from NER:
#   - All endpoints prefixed with /api/v1/
#   - Order create uses OrderCreate schema with idempotency_key
#   - limit_price is the field name (not price)
#   - avg_cost in portfolio is per-share (no division needed)
#   - Requires instrument_type="stock" (default) in order body

import asyncio
import logging
import time
import uuid
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self._calls: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            self._calls = [t for t in self._calls if now - t < self.period]
            if len(self._calls) >= self.max_calls:
                sleep_for = self.period - (now - self._calls[0])
                await asyncio.sleep(max(sleep_for, 0))
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < self.period]
            self._calls.append(time.monotonic())


class TSEClient:
    """
    Async HTTP client for The Stock Exchange (TSE) — orders and portfolio only.
    All market data (prices, OB, history) comes from AtlasClient.

    TSE tickers in Atlas are prefixed "TSE:ECO" but the TSE API itself
    uses the bare ticker "ECO". This client always strips the "TSE:" prefix
    before sending requests.
    """

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        }
        # TSE rate limits per OpenAPI spec
        # Default tier: inferred conservative values; respect orders_per_second
        self._order_limiter   = RateLimiter(max_calls=15, period=60.0)
        self._general_limiter = RateLimiter(max_calls=50, period=60.0)
        self._client: Optional[httpx.AsyncClient] = None

    @staticmethod
    def _bare(ticker: str) -> str:
        """Strip 'TSE:' prefix → 'ECO'."""
        return ticker.removeprefix("TSE:")

    @staticmethod
    def _idempotency_key() -> str:
        return str(uuid.uuid4())

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._headers,
            timeout=10.0,
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def _get(self, path: str, params: dict = None) -> dict | list:
        await self._general_limiter.acquire()
        r = await self._client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    # ── Portfolio ─────────────────────────────────────────────────────────────

    async def get_portfolio(self) -> dict:
        """GET /api/v1/account/portfolio — returns PortfolioResponse."""
        return await self._get("/api/v1/account/portfolio")

    async def get_open_orders(self, symbol: str = None) -> list[dict]:
        """
        GET /api/v1/orders?status=open[&symbol=ECO]
        Returns list[OrderResponse]. limit_price is the price field.
        """
        params = {"status": "open", "limit": 100}
        if symbol:
            params["symbol"] = self._bare(symbol)
        return await self._get("/api/v1/orders", params=params)

    async def get_order(self, order_id: str) -> dict:
        """GET /api/v1/orders/{order_id}"""
        return await self._get(f"/api/v1/orders/{order_id}")

    # ── Orders ────────────────────────────────────────────────────────────────

    async def place_buy_limit(
        self,
        ticker: str,
        quantity: int,
        limit_price: float,
        expiry_hours: float = 12,
    ) -> dict:
        await self._order_limiter.acquire()
        body = {
            "instrument_type": "stock",
            "symbol": self._bare(ticker),
            "side": "buy",
            "order_type": "limit",
            "limit_price": round(limit_price, 4),
            "quantity": quantity,
            "idempotency_key": self._idempotency_key(),
            "time_in_force": "GTC",
        }
        r = await self._client.post("/api/v1/orders", json=body)
        r.raise_for_status()
        result = r.json()
        order_id = result.get("order_id", result.get("id", "?"))
        logger.info(
            f"TSE BUY LIMIT {ticker} qty={quantity} @ {limit_price:.4f} "
            f"→ {order_id}"
        )
        # Normalise response to match NER shape our order_manager expects
        result.setdefault("order_id", order_id)
        return result

    async def place_sell_limit(
        self,
        ticker: str,
        quantity: int,
        limit_price: float,
        expiry_hours: float = 12,
    ) -> dict:
        await self._order_limiter.acquire()
        body = {
            "instrument_type": "stock",
            "symbol": self._bare(ticker),
            "side": "sell",
            "order_type": "limit",
            "limit_price": round(limit_price, 4),
            "quantity": quantity,
            "idempotency_key": self._idempotency_key(),
            "time_in_force": "GTC",
        }
        r = await self._client.post("/api/v1/orders", json=body)
        r.raise_for_status()
        result = r.json()
        order_id = result.get("order_id", result.get("id", "?"))
        logger.info(
            f"TSE SELL LIMIT {ticker} qty={quantity} @ {limit_price:.4f} "
            f"→ {order_id}"
        )
        result.setdefault("order_id", order_id)
        return result

    async def place_buy_market(self, ticker: str, quantity: int) -> dict:
        """
        TSE market order (no limit_price).
        Use as last resort — may reject if no counterparty.
        """
        await self._order_limiter.acquire()
        body = {
            "instrument_type": "stock",
            "symbol": self._bare(ticker),
            "side": "buy",
            "order_type": "market",
            "quantity": quantity,
            "idempotency_key": self._idempotency_key(),
        }
        r = await self._client.post("/api/v1/orders", json=body)
        r.raise_for_status()
        result = r.json()
        result.setdefault("order_id", result.get("order_id", "?"))
        logger.warning(f"TSE BUY MARKET {ticker} qty={quantity} → {result['order_id']}")
        return result

    async def place_sell_market(self, ticker: str, quantity: int) -> dict:
        """See note on place_buy_market."""
        await self._order_limiter.acquire()
        body = {
            "instrument_type": "stock",
            "symbol": self._bare(ticker),
            "side": "sell",
            "order_type": "market",
            "quantity": quantity,
            "idempotency_key": self._idempotency_key(),
        }
        r = await self._client.post("/api/v1/orders", json=body)
        r.raise_for_status()
        result = r.json()
        result.setdefault("order_id", result.get("order_id", "?"))
        logger.warning(f"TSE SELL MARKET {ticker} qty={quantity} → {result['order_id']}")
        return result

    async def cancel_order(self, order_id: str) -> dict:
        """DELETE /api/v1/orders/{order_id} → 202 Accepted."""
        await self._order_limiter.acquire()
        r = await self._client.delete(f"/api/v1/orders/{order_id}")
        r.raise_for_status()
        logger.info(f"TSE CANCELLED order {order_id}")
        # TSE returns 202 with empty body or minimal dict
        try:
            return r.json()
        except Exception:
            return {"status": "cancelled", "order_id": order_id}

    # ── Market data (lightweight — prefer Atlas) ──────────────────────────────

    async def get_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """GET /api/v1/market/{symbol}/orderbook — fallback if Atlas is stale."""
        return await self._get(
            f"/api/v1/market/{self._bare(ticker)}/orderbook",
            params={"depth": depth},
        )
