# api_client.py
# Clean async wrapper around every NER API endpoint we use.

import asyncio
import logging
import time
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter — respects the 20 orders/min hard limit."""

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
                logger.debug(f"Rate limit: sleeping {sleep_for:.2f}s")
                await asyncio.sleep(sleep_for)
                now = time.monotonic()
                self._calls = [t for t in self._calls if now - t < self.period]
            self._calls.append(time.monotonic())


class NERClient:
    """
    Async HTTP client for the NER Exchange API.
    All methods return parsed JSON or raise on error.
    """

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key,
        }
        # Separate rate limiters for trading vs general endpoints
        self._order_limiter = RateLimiter(max_calls=18, period=60.0)
        self._general_limiter = RateLimiter(max_calls=55, period=60.0)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.headers,
            timeout=10.0,
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def _get(self, path: str, params: dict = None, auth: bool = True) -> dict:
        await self._general_limiter.acquire()
        h = self.headers if auth else {"Content-Type": "application/json"}
        r = await self._client.get(path, params=params, headers=h)
        r.raise_for_status()
        return r.json()

    async def _post(self, path: str, body: dict) -> dict:
        await self._order_limiter.acquire()
        r = await self._client.post(path, json=body)
        r.raise_for_status()
        return r.json()

    async def _delete(self, path: str) -> dict:
        await self._order_limiter.acquire()
        r = await self._client.delete(path)
        r.raise_for_status()
        return r.json()

    # ── Market data (public) ─────────────────────────────────────────────────

    async def get_securities(self) -> list[dict]:
        return await self._get("/securities", auth=False)

    async def get_market_price(self, ticker: str) -> dict:
        return await self._get(f"/market_price/{ticker}", auth=False)

    async def get_orderbook(self, ticker: str = None) -> dict:
        params = {"ticker": ticker} if ticker else None
        return await self._get("/orderbook", params=params, auth=False)

    async def get_price_history(self, ticker: str, days: int = 90) -> list[dict]:
        return await self._get(
            f"/analytics/price_history/{ticker}",
            params={"days": days},
            auth=False,
        )

    async def get_ohlcv(self, ticker: str, days: int = 90) -> dict:
        return await self._get(
            f"/analytics/ohlcv/{ticker}",
            params={"days": days},
            auth=False,
        )

    async def get_security_stats(self, ticker: str) -> dict:
        return await self._get(f"/securities/{ticker}/stats", auth=False)

    async def get_security_info(self, ticker: str) -> dict:
        return await self._get(f"/securities/{ticker}", auth=False)

    # ── Portfolio ────────────────────────────────────────────────────────────

    async def get_portfolio(self) -> dict:
        return await self._get("/portfolio")

    async def get_funds(self) -> dict:
        return await self._get("/funds")

    async def get_transactions(self, limit: int = 500) -> list[dict]:
        return await self._get("/transactions", params={"limit": limit})

    # ── Orders ───────────────────────────────────────────────────────────────

    async def get_open_orders(self) -> list[dict]:
        return await self._get("/orders")

    async def get_order(self, order_id: str) -> dict:
        return await self._get(f"/orders/{order_id}")

    async def place_buy_limit(
        self, ticker: str, quantity: int, limit_price: float, expiry_hours: int = 2
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
        logger.info(f"BUY LIMIT {ticker} qty={quantity} @ {limit_price:.4f} → {result.get('order_id')}")
        return result

    async def place_sell_limit(
        self, ticker: str, quantity: int, limit_price: float, expiry_hours: int = 2
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
        logger.info(f"SELL LIMIT {ticker} qty={quantity} @ {limit_price:.4f} → {result.get('order_id')}")
        return result

    async def place_buy_market(self, ticker: str, quantity: int) -> dict:
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

    # ── Webhook management ───────────────────────────────────────────────────

    async def configure_webhook(self, url: str, secret: str = "") -> dict:
        r = await self._client.put(
            "/api-management/config",
            json={"webhook_url": url, "enabled": True, "secret": secret},
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
