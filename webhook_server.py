# webhook_server.py
# FastAPI webhook receiver for NER market_update events.

import asyncio
import logging
from typing import Callable, Optional

from fastapi import FastAPI, Request, HTTPException, Header
import uvicorn

logger = logging.getLogger(__name__)


class WebhookServer:
    """
    Listens for NER market_update webhook payloads and dispatches them
    to the registered callback.
    """

    def __init__(self, config, on_market_update: Callable):
        self.config = config
        self.on_market_update = on_market_update
        self.app = FastAPI(title="NER MM Webhook")
        self._register_routes()

    def _register_routes(self):

        @self.app.post("/webhook")
        async def receive_webhook(
            request: Request,
            x_webhook_secret: Optional[str] = Header(None),
        ):
            # Validate secret if configured
            if self.config.WEBHOOK_SECRET:
                if x_webhook_secret != self.config.WEBHOOK_SECRET:
                    raise HTTPException(status_code=401, detail="Invalid webhook secret")

            payload = await request.json()
            event = payload.get("event")

            if event == "market_update":
                ticker = payload.get("ticker")
                orderbook = payload.get("orderbook", {})
                market_price = payload.get("market_price")

                if market_price is not None:
                    orderbook["market_price"] = market_price

                logger.debug(
                    f"Webhook: {ticker} mid={orderbook.get('mid')} "
                    f"bid={orderbook.get('best_bid')} ask={orderbook.get('best_ask')}"
                )

                asyncio.create_task(self.on_market_update(ticker, orderbook))

            return {"status": "ok"}

        @self.app.get("/health")
        async def health():
            return {"status": "ok"}

    async def start(self):
        cfg = uvicorn.Config(
            self.app,
            host=self.config.WEBHOOK_HOST,
            port=self.config.WEBHOOK_PORT,
            log_level="warning",
            access_log=False,
        )
        server = uvicorn.Server(cfg)
        await server.serve()
