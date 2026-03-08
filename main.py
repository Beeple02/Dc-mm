# main.py
# NER Market Making Bot — clean rebuild.
#
# Architecture:
#   - Atlas API for all market data (prices, OB, history, stats)
#   - NER API for trading only (orders, portfolio)
#   - GLF (2013) closed-form quoting with dynamic γ
#   - Automatic ticker selection via scored ranking
#   - Hard stop-loss + aggressive unwind per Cartea & Jaimungal (2015) §7

import asyncio
import logging
import os
import signal
import time

import config as cfg
from atlas_client import AtlasClient
from ner_client import NERClient
from calibration import calibrate_all
from scorer import run_scorer
from quoting import compute_quotes, compute_quote_quantity
from risk import RiskManager
from state import BotState, TickerState
from order_manager import OrderManager

import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(cfg.LOG_FILE),
    ],
)
logger = logging.getLogger("main")


def _safe(v):
    """Sanitize float for JSON — replace nan/inf with None."""
    if isinstance(v, float):
        import math
        if math.isnan(v) or math.isinf(v):
            return None
    return v


class MarketMakingBot:

    def __init__(self):
        self.state = BotState(cfg)
        self.risk  = RiskManager(cfg)
        self._running = True
        self._calibration_lock = asyncio.Lock()

    # ── Startup ───────────────────────────────────────────────────────────────

    async def startup(self, atlas: AtlasClient, ner: NERClient):
        logger.info("=" * 60)
        logger.info("NER Market Making Bot — starting up (v2 rebuild)")
        logger.info(f"Capital: ${cfg.TOTAL_CAPITAL:,.0f} | "
                    f"Max deploy: {cfg.CAPITAL_DEPLOY_MAX:.0%} | "
                    f"Stop/ticker: ${cfg.STOP_LOSS_PER_TICKER:.0f} | "
                    f"Portfolio stop: ${cfg.STOP_LOSS_PORTFOLIO:.0f}")
        logger.info(f"Excluded tickers: {cfg.EXCLUDED_TICKERS}")
        logger.info("=" * 60)

        # Validate env
        for var, val in [("ATLAS_BASE_URL", cfg.ATLAS_BASE_URL),
                         ("NER_BASE_URL", cfg.NER_BASE_URL),
                         ("NER_API_KEY", cfg.NER_API_KEY)]:
            if not val:
                raise ValueError(f"{var} environment variable not set")

        # Check Atlas connectivity
        try:
            status = await atlas.status()
            logger.info(f"Atlas status: {status}")
        except Exception as e:
            logger.warning(f"Atlas status check failed: {e}")

        # Check funds
        funds = await ner.get_funds()
        available = funds.get("available", 0.0)
        logger.info(f"NER account cash: ${available:,.2f}")
        if available < cfg.TOTAL_CAPITAL * 0.5:
            logger.warning(
                f"Available cash (${available:,.2f}) is less than half "
                f"the configured MM capital (${cfg.TOTAL_CAPITAL:,.0f}). "
                f"Operating with reduced capital."
            )

        # Sync portfolio
        portfolio = await ner.get_portfolio()
        await self.state.update_from_portfolio(portfolio)
        logger.info(
            f"Portfolio synced: cash=${self.state.cash_available:,.2f} "
            f"equity=${self.state.total_equity:,.2f}"
        )

        # Check for inherited inventory from previous runs
        holdings = portfolio.get("holdings", [])
        if holdings:
            logger.warning(
                f"Inherited inventory from previous run: "
                f"{ {h['ticker']: h['quantity'] for h in holdings} }"
            )

        # Full calibration
        await self._run_calibration(atlas)

        # Snapshot initial orderbooks for selected tickers
        if self.state.scorer_result:
            for ticker in self.state.scorer_result.selected_tickers:
                try:
                    ob = await atlas.get_orderbook(ticker)
                    await self.state.update_orderbook(ticker, ob)
                except Exception as e:
                    logger.warning(f"{ticker}: initial OB fetch failed: {e}")

        # Reconcile open orders
        try:
            open_orders = await ner.get_open_orders()
            await self.state.reconcile_open_orders(open_orders)
            logger.info(f"Reconciled {len(open_orders)} pre-existing open orders")
            if open_orders:
                for o in open_orders:
                    logger.info(
                        f"  Pre-existing: {o.get('ticker')} "
                        f"{o.get('side')} {o.get('quantity')} "
                        f"@ {o.get('limit_price')} [{o.get('order_id')}]"
                    )
        except Exception as e:
            logger.warning(f"Open orders reconciliation failed: {e}")

        # Setup webhook
        await self._setup_webhook(ner)

        logger.info("Startup complete — entering main loop.")

    # ── Calibration & scoring ─────────────────────────────────────────────────

    async def _run_calibration(self, atlas: AtlasClient):
        async with self._calibration_lock:
            logger.info("Running calibration + scoring pipeline...")
            try:
                cal = await calibrate_all(atlas, cfg, cfg.EXCLUDED_TICKERS)
                self.state.calibration = cal

                scorer_result = run_scorer(cal, cfg)
                self.state.scorer_result = scorer_result

                # Update ticker states with scorer output
                for ticker, score in scorer_result.scores.items():
                    if ticker not in self.state.tickers:
                        self.state.tickers[ticker] = TickerState(ticker=ticker)
                    ts = self.state.tickers[ticker]
                    ts.allocated_capital = score.allocated_capital
                    ts.q_max = score.q_max
                    cal.tickers[ticker].q_max = score.q_max

                logger.info(
                    f"Selected tickers: {scorer_result.selected_tickers} | "
                    f"Allocations: { {t: f'${s.allocated_capital:.0f}' for t, s in scorer_result.scores.items()} }"
                )
            except Exception as e:
                logger.error(f"Calibration pipeline failed: {e}", exc_info=True)

    # ── Webhook setup ─────────────────────────────────────────────────────────

    async def _setup_webhook(self, ner: NERClient):
        if not cfg.WEBHOOK_PUBLIC_URL:
            logger.warning(
                "WEBHOOK_PUBLIC_URL not set — running on polling only. "
                "Set this env var on Railway for better responsiveness."
            )
            return
        url = f"{cfg.WEBHOOK_PUBLIC_URL.rstrip('/')}/webhook"
        try:
            await ner.configure_webhook(url, cfg.WEBHOOK_SECRET)
            await ner.subscribe_webhook_all()
            logger.info(f"Webhook registered: {url}")
        except Exception as e:
            # Log response body if available to diagnose NER webhook 500
            body = ""
            try:
                body = e.response.text[:300]
            except Exception:
                pass
            logger.warning(f"Webhook setup failed: {e}{(' | body: ' + body) if body else ''} — polling fallback active")

    # ── Core quoting pipeline ─────────────────────────────────────────────────

    async def _process_ticker(
        self,
        atlas: AtlasClient,
        om: OrderManager,
        ticker: str,
    ):
        """
        Full quoting pipeline for one ticker:
          1. Validate eligibility and risk gates
          2. Fetch live OB from Atlas
          3. Compute dynamic γ and GLF quotes
          4. Refresh orders via OrderManager
          5. Check for unwind needs
        """
        cal = self.state.calibration
        if not cal:
            return

        tc = cal.tickers.get(ticker)
        if not tc or not tc.eligible:
            return

        sr = self.state.scorer_result
        if not sr or ticker not in sr.scores:
            return

        score = sr.scores[ticker]
        ts = self.state.tickers.get(ticker)
        if not ts:
            return

        # Refresh OB from Atlas
        try:
            ob = await atlas.get_orderbook(ticker)
            await self.state.update_orderbook(ticker, ob)
        except Exception as e:
            logger.debug(f"{ticker}: OB fetch failed: {e}")

        ts = self.state.tickers.get(ticker)
        mid = ts.mid or ts.market_price
        if not mid or mid <= 0:
            logger.debug(f"{ticker}: no valid mid price")
            return

        # Update risk manager with current position
        await self.risk.update_position(
            ticker, ts.inventory, ts.cost_basis, mid
        )

        # Check if we need to unwind (stop-loss triggered)
        if self.risk.needs_unwind(ticker):
            await om.attempt_unwind(ticker)
            return  # Don't post MM quotes while unwinding

        # Risk gate
        if not self.risk.is_quoting_allowed(ticker):
            logger.debug(f"{ticker}: quoting blocked")
            return

        # Compute quotes
        t_remaining = self.state.t_remaining()

        quotes = compute_quotes(
            mid=mid,
            inventory=ts.inventory,
            t_remaining=t_remaining,
            sigma=tc.sigma,
            sigma_long_run=tc.sigma_long_run,
            lambda_A=tc.lambda_A,
            k=tc.k,
            q_max=score.q_max,
            drift_score=tc.drift_score,
            mu=tc.mu,
            kappa=tc.kappa,
            is_ou=tc.is_ou,
            cfg=cfg,
        )

        qty = compute_quote_quantity(
            allocated_capital=score.allocated_capital,
            mid=mid,
            q_max=score.q_max,
            cfg=cfg,
        )

        logger.info(
            f"{ticker} | inv={ts.inventory}/{score.q_max} | "
            f"mid={mid:.4f} | "
            f"bid={quotes.bid if quotes.bid > 0 else 'SUP'} "
            f"ask={quotes.ask if quotes.ask != float('inf') else 'SUP'} | "
            f"r={quotes.reservation_price:.4f} γ={quotes.gamma_effective:.4f} | "
            f"spread={quotes.spread_stressed:.4f} "
            f"f_inv={quotes.f_inventory:.2f} f_drift={quotes.f_drift:.2f} | "
            f"τ={t_remaining:.2f}h"
        )

        await om.refresh_quotes(ticker, quotes, qty)

    # ── Polling loop ──────────────────────────────────────────────────────────

    async def _polling_loop(self, atlas: AtlasClient, ner: NERClient, om: OrderManager):
        """
        Main polling loop: syncs state and triggers quoting for all selected tickers.
        Runs every POLLING_INTERVAL_SECONDS.
        """
        while self._running:
            await asyncio.sleep(cfg.POLLING_INTERVAL_SECONDS)

            try:
                # Sync portfolio
                portfolio = await ner.get_portfolio()
                await self.state.update_from_portfolio(portfolio)

                # Reconcile open orders and detect fills
                open_orders = await ner.get_open_orders()
                filled = await self.state.reconcile_open_orders(open_orders)

                # Record fills in risk manager
                for f in filled:
                    self.risk.record_fill(
                        f["ticker"],
                        f["order"].quantity,
                        f["order"].price,
                        f["side"],
                    )

                # Process all selected tickers
                if self.state.scorer_result:
                    for ticker in self.state.scorer_result.selected_tickers:
                        try:
                            await self._process_ticker(atlas, om, ticker)
                        except Exception as e:
                            logger.error(f"{ticker}: process error: {e}", exc_info=True)

                # Also check unwinds for stopped tickers not in selected list
                for ticker, tr in self.risk.state.tickers.items():
                    if tr.is_stopped and tr.inventory != 0:
                        ts = self.state.tickers.get(ticker)
                        if ts:
                            await om.attempt_unwind(ticker)

                # Periodic risk summary
                self.risk.log_risk_summary()

            except Exception as e:
                logger.error(f"Polling loop error: {e}", exc_info=True)

    # ── Recalibration loop ────────────────────────────────────────────────────

    async def _recalibration_loop(self, atlas: AtlasClient):
        """Periodic recalibration and rescoring."""
        while self._running:
            await asyncio.sleep(cfg.RECALIBRATE_INTERVAL_MIN * 60)
            try:
                logger.info("Periodic recalibration starting...")
                await self._run_calibration(atlas)
            except Exception as e:
                logger.error(f"Recalibration error: {e}", exc_info=True)

    # ── Session reset loop ────────────────────────────────────────────────────

    async def _session_reset_loop(self):
        while self._running:
            await asyncio.sleep(cfg.T_RESET_HOURS * 3600)
            self.state.reset_session()

    # ── Webhook server ────────────────────────────────────────────────────────

    async def _start_webhook_server(self, ner: NERClient):
        """
        FastAPI server serving:
          POST /webhook  — NER market update receiver
          GET  /status   — Rich JSON state for dashboard
          GET  /health   — Simple liveness probe
          GET  /         — Dashboard HTML (self-contained, polls /status)
        """
        try:
            from fastapi import FastAPI, Request, HTTPException, Header
            from fastapi.responses import HTMLResponse
            from fastapi.middleware.cors import CORSMiddleware
            import uvicorn
            import os
            from typing import Optional as Opt

            app = FastAPI(title="NER MM Bot")

            # CORS — allows the dashboard to be opened from any origin
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_methods=["GET", "POST"],
                allow_headers=["*"],
            )

            # Load dashboard HTML once at startup
            _dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")
            try:
                with open(_dashboard_path, "r") as f:
                    _dashboard_html = f.read()
                # Patch BOT_URL — regex replace any value so dashboard always
                # uses relative /status (same origin, works on Railway)
                import re as _re
                _dashboard_html = _re.sub(
                    r'const BOT_URL\s*=\s*.+',
                    "const BOT_URL = ''",
                    _dashboard_html
                )
            except FileNotFoundError:
                _dashboard_html = "<h1 style=\'font-family:monospace;color:#c8ff00;background:#000;padding:40px\'>dashboard.html not found alongside main.py</h1>"
                logger.warning("dashboard.html not found — / will show placeholder")

            @app.get("/", response_class=HTMLResponse)
            async def serve_dashboard():
                return HTMLResponse(content=_dashboard_html)

            @app.post("/webhook")
            async def receive_webhook(
                request: Request,
                x_webhook_secret: Opt[str] = Header(None),
            ):
                if cfg.WEBHOOK_SECRET and x_webhook_secret != cfg.WEBHOOK_SECRET:
                    raise HTTPException(status_code=401, detail="Invalid secret")

                payload = await request.json()
                event = payload.get("event")

                if event == "market_update":
                    ticker = payload.get("ticker")
                    ob = payload.get("orderbook", {})
                    mp = payload.get("market_price")
                    if mp:
                        ob["market_price"] = mp

                    if ticker and ticker not in cfg.EXCLUDED_TICKERS:
                        await self.state.update_orderbook(ticker, ob)
                        self.state.webhook_alive = True
                        self.state.last_webhook_time = time.monotonic()

                return {"status": "ok"}

            @app.get("/health")
            async def health():
                return {
                    "status": "ok",
                    "selected": self.state.scorer_result.selected_tickers
                    if self.state.scorer_result else [],
                    "portfolio_stopped": self.risk.state.portfolio_stopped,
                }

            @app.get("/status")
            def _sanitize(obj):
                """Recursively replace nan/inf floats with None for JSON compliance."""
                import math
                if isinstance(obj, float):
                    return None if (math.isnan(obj) or math.isinf(obj)) else obj
                if isinstance(obj, dict):
                    return {k: _sanitize(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [_sanitize(v) for v in obj]
                return obj

            async def status():
                """
                Rich status endpoint for the monitoring dashboard.
                Returns all bot state in one payload.
                """
                from fastapi.responses import JSONResponse

                cal = self.state.calibration
                sr  = self.state.scorer_result

                # Per-ticker status
                tickers_out = []
                for ticker, ts in self.state.tickers.items():
                    tc = cal.tickers.get(ticker) if cal else None
                    sc = sr.scores.get(ticker) if sr else None
                    tr = self.risk.state.tickers.get(ticker)

                    mid = ts.mid or ts.market_price or 0.0

                    tickers_out.append({
                        "ticker": ticker,
                        "selected": ticker in (sr.selected_tickers if sr else []),
                        "mid": _safe(mid),
                        "best_bid": ts.best_bid,
                        "best_ask": ts.best_ask,
                        "inventory": ts.inventory,
                        "cost_basis": ts.cost_basis,
                        "allocated_capital": ts.allocated_capital,
                        "q_max": ts.q_max,
                        # Live quotes
                        "bid_order": {
                            "price": ts.bid_order.price,
                            "qty": ts.bid_order.quantity,
                            "order_id": ts.bid_order.order_id,
                        } if ts.bid_order else None,
                        "ask_order": {
                            "price": ts.ask_order.price,
                            "qty": ts.ask_order.quantity,
                            "order_id": ts.ask_order.order_id,
                        } if ts.ask_order else None,
                        "unwind_order": {
                            "price": ts.unwind_order.price,
                            "qty": ts.unwind_order.quantity,
                            "side": ts.unwind_order.side,
                        } if ts.unwind_order else None,
                        # Risk
                        "unrealized_pnl": tr.unrealized_pnl if tr else 0.0,
                        "realized_pnl": tr.realized_pnl if tr else 0.0,
                        "is_stopped": tr.is_stopped if tr else False,
                        "unwind_attempts": tr.unwind_attempts if tr else 0,
                        # Calibration
                        "sigma": tc.sigma if tc else None,
                        "drift_score": tc.drift_score if tc else None,
                        "mean_reversion_score": tc.mean_reversion_score if tc else None,
                        "trades_per_day": tc.trades_per_day if tc else None,
                        "eligible": tc.eligible if tc else False,
                        # Scorer
                        "score": sc.raw_score if sc else None,
                        "score_components": sc.components if sc else None,
                    })

                # Calibration debug — all tickers, not just selected ones
                cal_debug = {}
                if cal:
                    for t, tc in cal.tickers.items():
                        sc = sr.scores.get(t) if sr else None
                        cal_debug[t] = {
                            "source": tc.source,
                            "mid": _safe(tc.mid),
                            "market_price": _safe(tc.market_price),
                            "sigma": _safe(round(tc.sigma, 6)),
                            "sigma_long_run": _safe(round(tc.sigma_long_run, 6)),
                            "atlas_vol_7d": _safe(tc.atlas_vol_7d),
                            "trades_per_day": _safe(round(tc.trades_per_day, 3)),
                            "drift_score": _safe(round(tc.drift_score, 4)),
                            "mean_reversion_score": _safe(round(tc.mean_reversion_score, 4)),
                            "liquidity_score": _safe(tc.liquidity_score),
                            "eligible": tc.eligible,
                            "score": _safe(round(sc.raw_score, 4)) if sc else None,
                        }

                return JSONResponse(_sanitize({
                    "ts": time.time(),
                    "uptime_hours": (time.monotonic() - self.state.session_start) / 3600.0,
                    "t_remaining_hours": self.state.t_remaining(),
                    # Portfolio
                    "cash_available": self.state.cash_available,
                    "cash_reserved": self.state.cash_reserved,
                    "total_equity": self.state.total_equity,
                    # Risk
                    "portfolio_stopped": self.risk.state.portfolio_stopped,
                    "total_unrealized_pnl": self.risk.state.total_unrealized_pnl,
                    "total_realized_pnl": self.risk.state.total_realized_pnl,
                    # Bot config
                    "total_capital": cfg.TOTAL_CAPITAL,
                    "stop_loss_per_ticker": cfg.STOP_LOSS_PER_TICKER,
                    "stop_loss_portfolio": cfg.STOP_LOSS_PORTFOLIO,
                    "webhook_alive": self.state.webhook_alive,
                    "selected_tickers": sr.selected_tickers if sr else [],
                    "tickers": tickers_out,
                    # Full calibration debug (all tickers, not just selected)
                    "calibration_debug": cal_debug,
                    "calibration_eligible": cal.eligible_tickers if cal else [],
                }))

            server_cfg = uvicorn.Config(
                app,
                host=cfg.WEBHOOK_HOST,
                port=cfg.WEBHOOK_PORT,
                log_level="warning",
                access_log=False,
            )
            server = uvicorn.Server(server_cfg)
            await server.serve()

        except Exception as e:
            logger.error(f"Webhook server error: {e}", exc_info=True)

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self):
        if not cfg.NER_API_KEY:
            raise ValueError("NER_API_KEY not set")

        async with AtlasClient(cfg.ATLAS_BASE_URL, cfg.ATLAS_API_KEY) as atlas, \
                   NERClient(cfg.NER_BASE_URL, cfg.NER_API_KEY) as ner:

            om = OrderManager(ner, self.state, self.risk, cfg)
            await self.startup(atlas, ner)

            tasks = [
                asyncio.create_task(self._polling_loop(atlas, ner, om), name="polling"),
                asyncio.create_task(self._recalibration_loop(atlas), name="recalibration"),
                asyncio.create_task(self._session_reset_loop(), name="session_reset"),
                asyncio.create_task(self._start_webhook_server(ner), name="webhook"),
            ]

            def _shutdown(sig):
                logger.info(f"Signal {sig.name} — shutting down...")
                self._running = False
                for t in tasks:
                    t.cancel()

            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, _shutdown, sig)

            logger.info("All tasks running. Bot is live.")
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                logger.info("Bot shut down cleanly.")
                self.risk.log_risk_summary()


if __name__ == "__main__":
    bot = MarketMakingBot()
    asyncio.run(bot.run())
