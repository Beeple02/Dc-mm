# ner_client.py
# Async wrapper for NER exchange trading endpoints.
# Market data comes from Atlas — this client handles orders only.

import asyncio
import logging
import time
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


class NERClient:
    """
    Async HTTP client for NER exchange — orders and portfolio only.
    All market data (prices, OB, history) comes from AtlasClient.
    """

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        }
        # Conservative rate limits — well under NER's 20 orders/min hard cap
        self._order_limiter   = RateLimiter(max_calls=10, period=60.0)
        self._general_limiter = RateLimiter(max_calls=50, period=60.0)
        self._client: Optional[httpx.AsyncClient] = None

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
        return await self._get("/portfolio")

    async def get_funds(self) -> dict:
        return await self._get("/funds")

    async def get_transactions(self, limit: int = 200) -> list[dict]:
        """Own transaction history (authenticated)."""
        return await self._get("/transactions", params={"limit": limit})

    # ── Orders ────────────────────────────────────────────────────────────────

    async def get_open_orders(self) -> list[dict]:
        return await self._get("/orders")

    async def get_order(self, order_id: str) -> dict:
        return await self._get(f"/orders/{order_id}")

    async def place_buy_limit(
        self,
        ticker: str,
        quantity: int,
        limit_price: float,
        expiry_hours: float = 12,
    ) -> dict:
        await self._order_limiter.acquire()
        body = {
            "ticker": ticker,
            "quantity": quantity,
            "limit_price": round(limit_price, 4),
            "expiry_hours": expiry_hours,
        }
        r = await self._client.post("/orders/buy_limit", json=body)
        r.raise_for_status()
        result = r.json()
        logger.info(
            f"BUY LIMIT {ticker} qty={quantity} @ {limit_price:.4f} "
            f"→ {result.get('order_id')}"
        )
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
            "ticker": ticker,
            "quantity": quantity,
            "limit_price": round(limit_price, 4),
            "expiry_hours": expiry_hours,
        }
        r = await self._client.post("/orders/sell_limit", json=body)
        r.raise_for_status()
        result = r.json()
        logger.info(
            f"SELL LIMIT {ticker} qty={quantity} @ {limit_price:.4f} "
            f"→ {result.get('order_id')}"
        )
        return result

    async def place_buy_market(self, ticker: str, quantity: int) -> dict:
        """
        NER market orders return 400 if no counterparty exists.
        Use only as last resort after limit unwind attempts fail.
        Caller must handle the 400 gracefully.
        """
        await self._order_limiter.acquire()
        r = await self._client.post(
            "/orders/buy_market",
            json={"ticker": ticker, "quantity": quantity},
        )
        r.raise_for_status()
        result = r.json()
        logger.info(f"BUY MARKET {ticker} qty={quantity} → {result.get('order_id')}")
        return result

    async def place_sell_market(self, ticker: str, quantity: int) -> dict:
        """See note on place_buy_market."""
        await self._order_limiter.acquire()
        r = await self._client.post(
            "/orders/sell_market",
            json={"ticker": ticker, "quantity": quantity},
        )
        r.raise_for_status()
        result = r.json()
        logger.info(f"SELL MARKET {ticker} qty={quantity} → {result.get('order_id')}")
        return result

    async def cancel_order(self, order_id: str) -> dict:
        await self._order_limiter.acquire()
        r = await self._client.delete(f"/orders/{order_id}")
        r.raise_for_status()
        logger.info(f"CANCELLED order {order_id}")
        return r.json()

    # ── Webhook management ────────────────────────────────────────────────────

    async def configure_webhook(self, url: str, secret: str = "") -> dict:
        body = {"webhook_url": url, "enabled": True}
        if secret:
            body["secret"] = secret
        r = await self._client.put(
            "/api-management/config",
            json=body,
        )
        r.raise_for_status()
        return r.json()

    async def subscribe_webhook_all(self) -> dict:
        r = await self._client.post(
            "/api-management/subscriptions",
            json={"ticker": "*"},
        )
        r.raise_for_status()
        return r.json()
