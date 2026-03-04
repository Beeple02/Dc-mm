# main.py
# Entry point for the NER Market Making Bot.
# Orchestrates: startup calibration, webhook server, polling fallback,
# quoting loop, active inventory management, periodic recalibration.

import asyncio
import logging
import os
import signal
import time

import config as cfg
from api_client import NERClient
from calibration import calibrate_all, compute_capital_allocation
from order_manager import OrderManager
from quoting import (
    solve_theta_ode,
    compute_riccati_matrix,
    compute_multiasset_skew,
    compute_single_asset_quotes,
    ou_reservation_price_adjustment,
)
from state import BotState, TickerState
from webhook_server import WebhookServer

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


class MarketMakingBot:

    def __init__(self):
        self.state = BotState(cfg)
        self._running = True
        self._calibration_lock = asyncio.Lock()
        self._quote_queue: asyncio.Queue = asyncio.Queue()

    # ── Startup ───────────────────────────────────────────────────────────────

    async def startup(self, api: NERClient):
        logger.info("=" * 60)
        logger.info("NER Market Making Bot — starting up")
        logger.info(f"Configured MM capital: ${cfg.TOTAL_MM_CAPITAL:,.2f}")
        logger.info("=" * 60)

        # 1. Check available funds
        funds = await api.get_funds()
        available = funds.get("available", 0.0)
        logger.info(f"Account cash available: ${available:,.2f}")

        # Cap to configured budget (leave personal funds untouched)
        if available < cfg.TOTAL_MM_CAPITAL:
            logger.warning(
                f"Available (${available:,.2f}) < MM budget (${cfg.TOTAL_MM_CAPITAL:,.2f}). "
                f"Capping to 90% of available."
            )
            effective_capital = available * 0.90
        else:
            effective_capital = cfg.TOTAL_MM_CAPITAL

        logger.info(f"Effective MM capital: ${effective_capital:,.2f}")

        # 2. Sync portfolio
        portfolio = await api.get_portfolio()
        await self.state.update_from_portfolio(portfolio)

        # 3. Full calibration (fetches history, applies fixes, fits parameters)
        await self._run_calibration(api)

        # 4. Initialise ticker states from securities list
        securities = await api.get_securities()
        for sec in securities:
            ticker = sec["ticker"]
            if not sec.get("frozen", False):
                if ticker not in self.state.tickers:
                    self.state.tickers[ticker] = TickerState(ticker=ticker)
                self.state.tickers[ticker].total_shares = sec.get("total_shares", 1000)
                self.state.tickers[ticker].market_price = sec.get("market_price", 10.0)

        # 5. Snapshot all orderbooks
        try:
            all_ob = await api.get_orderbook()
            if isinstance(all_ob, list):
                for ob in all_ob:
                    t = ob.get("ticker")
                    if t:
                        await self.state.update_orderbook(t, ob)
            elif isinstance(all_ob, dict) and all_ob.get("ticker"):
                await self.state.update_orderbook(all_ob["ticker"], all_ob)
        except Exception as e:
            logger.warning(f"Initial orderbook fetch failed: {e}")

        # 6. Reconcile any pre-existing open orders
        try:
            open_orders = await api.get_open_orders()
            await self.state.reconcile_open_orders(open_orders)
            logger.info(f"Reconciled {len(open_orders)} pre-existing open orders")
        except Exception as e:
            logger.warning(f"Open orders reconciliation failed: {e}")

        # 7. Dynamic capital allocation (eligible tickers only)
        await self._reallocate_capital(effective_capital)

        # 8. Register webhook
        await self._setup_webhook(api)

        logger.info("Startup complete — entering main loop.")

    # ── Calibration ───────────────────────────────────────────────────────────

    async def _run_calibration(self, api: NERClient):
        async with self._calibration_lock:
            logger.info("Running calibration pipeline...")
            cal = await calibrate_all(api, cfg, cfg.GAMMA)
            self.state.calibration = cal

            eligible = cal.eligible_tickers
            if not eligible:
                logger.warning("No eligible tickers after calibration.")
                return

            T = cfg.T_HORIZON_HOURS

            # Solve per-ticker ODE for θ(t,q)
            self.state.ode_solutions = {}
            for ticker in eligible:
                tc = cal.tickers[ticker]
                sol = solve_theta_ode(
                    ticker=ticker,
                    sigma=tc.sigma,
                    gamma=cfg.GAMMA,
                    xi=tc.xi,
                    lambda_A=tc.lambda_A,
                    lambda_k=tc.lambda_k,
                    q_max=tc.q_max,
                    T=T,
                    n_steps=cfg.ODE_TIMESTEPS,
                )
                self.state.ode_solutions[ticker] = sol
                logger.info(
                    f"ODE solved for {ticker}: "
                    f"q ∈ [{-tc.q_max}, {tc.q_max}], T={T}h"
                )

            # Solve multi-asset Riccati ODE for cross-inventory skew (Ch.4)
            if len(eligible) > 1:
                lambda_A_vec = np.array([cal.tickers[t].lambda_A for t in eligible])
                lambda_k_vec = np.array([cal.tickers[t].lambda_k for t in eligible])
                xi_vec       = np.array([cal.tickers[t].xi       for t in eligible])
                self.state.riccati_A = compute_riccati_matrix(
                    sigma_matrix=cal.sigma_matrix,
                    gamma=cfg.GAMMA,
                    lambda_A_vec=lambda_A_vec,
                    lambda_k_vec=lambda_k_vec,
                    xi_vec=xi_vec,
                    T=T,
                )
                logger.info(
                    f"Riccati A matrix solved for {len(eligible)} assets"
                )
            else:
                self.state.riccati_A = None

            logger.info("Calibration pipeline complete.")

    async def _reallocate_capital(self, total_capital: float = None):
        """Recompute capital allocation across eligible tickers."""
        if total_capital is None:
            total_capital = cfg.TOTAL_MM_CAPITAL

        if not self.state.calibration:
            return

        orderbooks = {
            t: {
                "best_bid": ts.best_bid,
                "best_ask": ts.best_ask,
                "mid": ts.mid,
            }
            for t, ts in self.state.tickers.items()
        }

        allocation = compute_capital_allocation(
            self.state.calibration.tickers,
            orderbooks,
            total_capital,
            cfg.CAPITAL_RESERVE_BUFFER,
        )

        for ticker, amount in allocation.items():
            if ticker in self.state.tickers:
                self.state.tickers[ticker].allocated_capital = amount

        # Log only eligible allocations
        eligible_alloc = {
            t: f"${v:.2f}"
            for t, v in allocation.items()
            if v > 0
        }
        logger.info(f"Capital allocation: {eligible_alloc}")

    async def _setup_webhook(self, api: NERClient):
        webhook_url = os.environ.get("WEBHOOK_PUBLIC_URL", "")
        if not webhook_url:
            logger.warning(
                "WEBHOOK_PUBLIC_URL not set — webhook disabled. "
                "Running on polling fallback only."
            )
            return
        full_url = f"{webhook_url.rstrip('/')}/webhook"
        try:
            await api.configure_webhook(full_url, cfg.WEBHOOK_SECRET)
            await api.subscribe_webhook_all()
            logger.info(f"Webhook registered: {full_url}")
        except Exception as e:
            logger.warning(f"Webhook setup failed: {e} — using polling fallback")

    # ── Market update (webhook + polling share this) ──────────────────────────

    async def on_market_update(self, ticker: str, orderbook: dict):
        await self.state.update_orderbook(ticker, orderbook)
        self.state.webhook_alive = True
        self.state.last_webhook_time = time.monotonic()
        # Enqueue for quoting — non-blocking
        try:
            self._quote_queue.put_nowait(ticker)
        except asyncio.QueueFull:
            pass  # Already queued

    # ── Core quoting logic ────────────────────────────────────────────────────

    async def _compute_and_post_quotes(
        self, api: NERClient, om: OrderManager, ticker: str
    ):
        """
        Full quoting pipeline for one ticker:
          1. Check eligibility
          2. Compute OU-adjusted reservation price
          3. Compute optimal δ^b, δ^a from θ ODE solution
          4. Apply multi-asset cross-skew (Ch.4 Riccati)
          5. Refresh orders if stale
          6. Check active inventory management trigger (Ch.2)
        """
        cal = self.state.calibration
        if not cal:
            return

        tc = cal.tickers.get(ticker)
        if not tc or not tc.eligible:
            return

        ts = self.state.tickers.get(ticker)
        if not ts:
            return

        ode_sol = self.state.ode_solutions.get(ticker)
        if not ode_sol:
            return

        # Current mid price — prefer live orderbook mid, fall back to market_price
        mid = ts.mid or ts.market_price
        if not mid or mid <= 0:
            logger.debug(f"{ticker}: no valid mid price, skipping quoting cycle")
            return

        t_remaining = self.state.t_remaining()

        # OU reservation price adjustment (Bergault Ch.6)
        ou_adj = ou_reservation_price_adjustment(
            current_price=mid,
            mu=tc.mu,
            kappa=tc.kappa,
            sigma=tc.sigma,
            gamma=cfg.GAMMA,
            inventory=ts.inventory,
            t_remaining=t_remaining,
            is_ou=tc.is_ou,
        )

        # Spread floor: must clear commission on both sides
        spread_floor = mid * cfg.COMMISSION_RATE * cfg.SPREAD_FLOOR_MULTIPLIER

        # Single-asset optimal quotes from ODE solution
        quotes = compute_single_asset_quotes(
            ticker=ticker,
            current_mid=mid,
            current_inventory=ts.inventory,
            t_remaining=t_remaining,
            ode_solution=ode_sol,
            lambda_A=tc.lambda_A,
            lambda_k=tc.lambda_k,
            gamma=cfg.GAMMA,
            xi=tc.xi,
            sigma=tc.sigma,
            spread_floor=spread_floor,
            ou_adjustment=ou_adj,
        )

        # Multi-asset cross-inventory skew (Bergault Ch.4 quadratic approximation)
        # skew_i = 2 · Σ_j A_{ij} · q_j
        if (
            self.state.riccati_A is not None
            and cal.eligible_tickers
            and len(cal.eligible_tickers) > 1
        ):
            inv_vec = self.state.get_inventory_vector(cal.eligible_tickers)
            cross_skews = compute_multiasset_skew(
                inv_vec, self.state.riccati_A, cal.eligible_tickers
            )
            cross_skew_i = cross_skews.get(ticker, 0.0)
            # Shift both bid and ask by the cross-inventory penalty
            quotes.bid -= cross_skew_i
            quotes.ask -= cross_skew_i
            quotes.bid = max(quotes.bid, 0.01)
            quotes.ask = max(quotes.ask, quotes.bid + 2.0 * spread_floor)

        logger.info(
            f"{ticker} | q={ts.inventory} | mid={mid:.4f} | "
            f"bid={quotes.bid:.4f} ask={quotes.ask:.4f} | "
            f"δb={quotes.delta_bid:.4f} δa={quotes.delta_ask:.4f} | "
            f"skew={quotes.skew:.4f} ou_adj={ou_adj:.6f} | "
            f"τ={t_remaining:.2f}h"
        )

        # Determine quote quantity from allocated capital
        allocated = ts.allocated_capital or (
            cfg.TOTAL_MM_CAPITAL / max(len(cal.eligible_tickers), 1)
        )
        qty = om.compute_quote_quantity(ticker, allocated, mid)

        # Cancel stale quotes and post fresh ones
        await om.refresh_quotes(ticker, quotes, qty)

        # Active inventory management check (Ch.2)
        await om.active_inventory_management(ticker)

    # ── Webhook-driven quote loop ─────────────────────────────────────────────

    async def _webhook_quote_loop(self, api: NERClient, om: OrderManager):
        """
        Dequeues tickers from the webhook update queue and triggers requoting.
        Decouples webhook receipt from order placement.
        """
        while self._running:
            try:
                ticker = await asyncio.wait_for(
                    self._quote_queue.get(), timeout=5.0
                )
                await self._compute_and_post_quotes(api, om, ticker)
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                logger.error(f"Webhook quote loop error: {e}", exc_info=True)

    # ── Polling fallback ──────────────────────────────────────────────────────

    async def _polling_loop(self, api: NERClient, om: OrderManager):
        """
        Background poller: runs slower when webhook is live, faster as fallback.
        Also handles portfolio sync and order reconciliation.
        """
        while self._running:
            webhook_recent = (
                self.state.webhook_alive
                and (time.monotonic() - self.state.last_webhook_time) < 30.0
            )
            interval = (
                cfg.POLLING_INTERVAL_SECONDS
                if webhook_recent
                else cfg.POLLING_FALLBACK_SECONDS
            )

            await asyncio.sleep(interval)

            try:
                # Refresh all orderbooks
                for ticker in list(self.state.tickers.keys()):
                    try:
                        ob = await api.get_orderbook(ticker)
                        await self.state.update_orderbook(ticker, ob)
                    except Exception as e:
                        logger.debug(f"Orderbook poll {ticker}: {e}")

                # Reconcile open orders
                try:
                    open_orders = await api.get_open_orders()
                    await self.state.reconcile_open_orders(open_orders)
                except Exception as e:
                    logger.debug(f"Order reconciliation: {e}")

                # Sync portfolio
                try:
                    portfolio = await api.get_portfolio()
                    await self.state.update_from_portfolio(portfolio)
                except Exception as e:
                    logger.debug(f"Portfolio sync: {e}")

                # Compute and post quotes for all eligible tickers
                if self.state.calibration:
                    for ticker in self.state.calibration.eligible_tickers:
                        await self._compute_and_post_quotes(api, om, ticker)

            except Exception as e:
                logger.error(f"Polling loop error: {e}", exc_info=True)

    # ── Periodic recalibration ────────────────────────────────────────────────

    async def _recalibration_loop(self, api: NERClient):
        """Re-run full calibration every REALLOC_INTERVAL_MINUTES."""
        while self._running:
            await asyncio.sleep(cfg.REALLOC_INTERVAL_MINUTES * 60)
            try:
                logger.info("Periodic recalibration starting...")
                await self._run_calibration(api)
                await self._reallocate_capital()
            except Exception as e:
                logger.error(f"Recalibration error: {e}", exc_info=True)

    # ── Session reset loop ────────────────────────────────────────────────────

    async def _session_reset_loop(self):
        """Reset the T-horizon timer every T_HORIZON_HOURS."""
        while self._running:
            await asyncio.sleep(cfg.T_HORIZON_HOURS * 3600)
            self.state.reset_session()

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self):
        if not cfg.API_KEY:
            raise ValueError(
                "NER_API_KEY environment variable is not set. "
                "Set it before running the bot."
            )

        async with NERClient(cfg.API_BASE_URL, cfg.API_KEY) as api:
            om = OrderManager(api, self.state, cfg)
            await self.startup(api)

            webhook_server = WebhookServer(cfg, self.on_market_update)

            tasks = [
                asyncio.create_task(webhook_server.start(),        name="webhook_server"),
                asyncio.create_task(self._webhook_quote_loop(api, om), name="webhook_quotes"),
                asyncio.create_task(self._polling_loop(api, om),   name="polling"),
                asyncio.create_task(self._recalibration_loop(api), name="recalibration"),
                asyncio.create_task(self._session_reset_loop(),    name="session_reset"),
            ]

            def _shutdown(sig):
                logger.info(f"Signal {sig.name} received — shutting down...")
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


if __name__ == "__main__":
    bot = MarketMakingBot()
    asyncio.run(bot.run())
