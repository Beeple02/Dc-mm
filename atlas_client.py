# atlas_client.py
# Async wrapper for the Atlas market data API.
# All trading goes through ner_client.py — this is read-only.

import asyncio
import logging
import time
from typing import Optional
import httpx

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple token bucket — Atlas is read-only so we can be generous."""

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


class AtlasClient:
    """
    Async HTTP client for the Atlas market data API.

    Provides:
      - Securities list and detail
      - Price and orderbook snapshots
      - Full OB + trade history
      - OHLCV candles
      - Ticker stats (vol, spread, trade frequency)
      - Public transactions tape
      - Market summary and breadth
    """

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._limiter = RateLimiter(max_calls=60, period=60.0)
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Content-Type": "application/json",
                "x-atlas-key": self._api_key,
            },
            timeout=15.0,
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()

    async def _get(self, path: str, params: dict = None) -> dict | list:
        await self._limiter.acquire()
        r = await self._client.get(path, params=params)
        r.raise_for_status()
        return r.json()

    # ── Health ────────────────────────────────────────────────────────────────

    async def health(self) -> dict:
        return await self._get("/health")

    async def status(self) -> dict:
        return await self._get("/status")

    # ── Securities ────────────────────────────────────────────────────────────

    async def get_securities(
        self,
        include_derived: bool = True,
        frozen: Optional[bool] = None,
    ) -> list[dict]:
        params = {"include_derived": include_derived}
        if frozen is not None:
            params["frozen"] = frozen
        return await self._get("/securities", params=params)

    async def get_security(self, ticker: str) -> dict:
        return await self._get(f"/securities/{ticker}")

    async def get_securities_by_source(self, source: str) -> list[dict]:
        """source: 'ner' or 'tse'"""
        return await self._get(f"/securities/source/{source}")

    # ── Price & orderbook ─────────────────────────────────────────────────────

    async def get_price(self, ticker: str) -> dict:
        return await self._get(f"/price/{ticker}")

    async def get_orderbook(self, ticker: str) -> dict:
        return await self._get(f"/orderbook/{ticker}")

    async def get_all_orderbooks(self) -> list[dict]:
        return await self._get("/orderbook")

    # ── History ───────────────────────────────────────────────────────────────

    async def get_history(
        self,
        ticker: str,
        days: int = 30,
        limit: int = 500,
        from_ts: Optional[str] = None,
        to_ts: Optional[str] = None,
    ) -> list[dict]:
        params = {"days": days, "limit": limit}
        if from_ts:
            params["from"] = from_ts
        if to_ts:
            params["to"] = to_ts
        return await self._get(f"/history/{ticker}", params=params)

    async def get_ohlcv(self, ticker: str, days: int = 30) -> list[dict]:
        return await self._get(f"/ohlcv/{ticker}", params={"days": days})

    # ── Analytics ─────────────────────────────────────────────────────────────

    async def get_analytics_ohlcv(self, ticker: str, days: int = 365) -> list[dict]:
        return await self._get(
            f"/analytics/ohlcv/{ticker}", params={"days": days}
        )

    async def get_ticker_stats(self, ticker: str, days: int = 365) -> dict:
        """
        Returns precomputed stats: volatility, avg spread, trade frequency,
        VWAP, etc. Use this as the primary calibration input.
        """
        return await self._get(
            f"/analytics/ticker_stats/{ticker}", params={"days": days}
        )

    # ── Transactions tape ─────────────────────────────────────────────────────

    async def get_transactions(
        self,
        ticker: Optional[str] = None,
        limit: int = 500,
        since: Optional[str] = None,
    ) -> list[dict]:
        """Public trade tape — no auth required."""
        params = {"limit": limit}
        if ticker:
            params["ticker"] = ticker
        if since:
            params["since"] = since
        return await self._get("/transactions", params=params)

    # ── Market-wide ───────────────────────────────────────────────────────────

    async def get_market_summary(self) -> dict:
        return await self._get("/market/summary")

    async def get_market_breadth(self, days: int = 7) -> dict:
        return await self._get("/market/breadth", params={"days": days})

    # ── Derived / shareholders ────────────────────────────────────────────────

    async def get_derived(self, ticker: str) -> dict:
        return await self._get(f"/derived/{ticker}")

    async def get_all_derived(self) -> list[dict]:
        return await self._get("/derived")

    async def get_shareholders(self, ticker: str) -> list[dict]:
        return await self._get(f"/shareholders/{ticker}")

    async def get_holder_intel(self, ticker: str) -> dict:
        return await self._get(f"/holder_intel/{ticker}")
