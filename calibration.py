# calibration.py
# Estimates all model parameters from Atlas market data.
#
# Atlas response shapes (confirmed from live API):
#
#   GET /securities/source/{ner|tse}
#     → list of security dicts:
#       {"ticker": "BB", "source": "ner", "frozen": 0, "market_price": 42,
#        "derived": {"volatility_7d": null, "vwap_7d": 20.6, "spread_pct": null,
#                    "liquidity_score": 41.2, ...}, ...}
#
#   GET /history/{ticker}?days=30&limit=2000
#     → {"count": N, "data": [{"id":..,"price":..,"volume":..,"timestamp":..,"ticker":..}], "ticker": "BB"}
#
#   GET /orderbook/{ticker}
#     → {"ticker":"BB", "mid":42, "best_bid":null, "best_ask":42,
#        "bids":[], "asks":[...], "spread":null, "spread_pct":null, ...}
#
#   GET /transactions?ticker=BB&limit=500
#     → {"count": N, "data": [...]} or plain list
#
# TSE tickers have the format "TSE:ECO". Supported identically to NER.
#
# Paper references:
#   - Realized volatility: Andersen & Bollerslev (1998)
#   - OU mean-reversion: discrete-time MLE
#   - Order arrival rate k: Avellaneda & Stoikov (2008) §3
#   - Drift score: rolling linear regression on log price series

import asyncio
import datetime
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TickerCalibration:
    ticker: str
    source: str = "ner"               # "ner" or "tse"

    mid: float = 0.0
    market_price: float = 0.0

    sigma: float = 0.01               # per-hour realized vol
    sigma_long_run: float = 0.01
    vol_of_vol: float = 0.0
    atlas_vol_7d: Optional[float] = None  # Atlas precomputed, converted to hourly

    kappa: float = 0.0                # OU reversion speed (1/h)
    mu: float = 0.0                   # OU long-run mean
    is_ou: bool = False
    mean_reversion_score: float = 0.0

    drift_score: float = 0.0

    lambda_A: float = 0.02            # order arrival rate (per hour)
    k: float = 1.5                    # AS depth decay parameter

    avg_spread_pct: float = 0.05
    ob_update_freq: float = 0.0       # price history density (rows/hour)
    trades_per_day: float = 0.0

    liquidity_score: float = 0.0      # Atlas precomputed [0-100]
    ob_imbalance: float = 0.0         # Atlas [-1, 1]

    q_max: int = 10                   # set by scorer after capital allocation
    eligible: bool = False


@dataclass
class CalibrationResult:
    tickers: dict[str, TickerCalibration] = field(default_factory=dict)
    eligible_tickers: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.monotonic)


# ─────────────────────────────────────────────────────────────────────────────
# Response unwrapping
# ─────────────────────────────────────────────────────────────────────────────

def _unwrap_list(response) -> list:
    """
    Atlas returns paginated responses as {"count": N, "data": [...], "ticker": ...}.
    Unwrap to plain list regardless of shape.
    """
    if isinstance(response, list):
        return response
    if isinstance(response, dict):
        if "data" in response and isinstance(response["data"], list):
            return response["data"]
        for v in response.values():
            if isinstance(v, list):
                return v
    return []


def _parse_securities(response) -> list[dict]:
    """
    Parse /securities/source/{ner|tse} into list of security dicts.
    Handles: list of dicts, list of strings, paginated dict.
    """
    if isinstance(response, dict):
        items = _unwrap_list(response)
    elif isinstance(response, list):
        items = response
    else:
        return []

    result = []
    for s in items:
        if isinstance(s, str) and s:
            result.append({"ticker": s, "frozen": 0, "source": "unknown"})
        elif isinstance(s, dict) and s.get("ticker"):
            result.append(s)
    return result


def _iso_to_epoch_hours(ts) -> Optional[float]:
    if ts is None:
        return None
    try:
        dt = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt.timestamp() / 3600.0
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Volatility
# ─────────────────────────────────────────────────────────────────────────────

def estimate_realized_vol(prices: np.ndarray, times_hours: np.ndarray) -> float:
    """Realized vol from price series, per-hour units (Andersen & Bollerslev 1998)."""
    if len(prices) < 3:
        return 0.01
    log_returns = np.diff(np.log(np.maximum(prices, 1e-10)))
    dt = np.diff(times_hours)
    dt = np.where(dt > 0, dt, 1e-6)
    rv = np.sum(log_returns ** 2 / dt) / len(log_returns)
    return float(np.sqrt(max(rv, 1e-8)))


def estimate_vol_of_vol(prices: np.ndarray, times_hours: np.ndarray, n_windows: int = 5) -> float:
    """Std dev of rolling realized vol windows."""
    if len(prices) < n_windows * 3:
        return 0.0
    w = len(prices) // n_windows
    vols = [
        estimate_realized_vol(prices[i*w:(i+1)*w], times_hours[i*w:(i+1)*w])
        for i in range(n_windows)
    ]
    return float(np.std(vols)) if len(vols) > 1 else 0.0


def atlas_vol_to_hourly(vol_7d_annualised: Optional[float]) -> Optional[float]:
    """
    Convert Atlas volatility_7d (annualised) → per-hour vol.
    σ_annual → σ_daily = σ_annual/sqrt(365) → σ_hourly = σ_daily/sqrt(24)
    """
    if not vol_7d_annualised or vol_7d_annualised <= 0:
        return None
    return float(vol_7d_annualised / np.sqrt(365) / np.sqrt(24))


# ─────────────────────────────────────────────────────────────────────────────
# OU mean-reversion
# ─────────────────────────────────────────────────────────────────────────────

def fit_ou_process(
    prices: np.ndarray,
    times_hours: np.ndarray,
    halflife_max_hours: float = 48.0,
) -> tuple[float, float, bool]:
    """Fit OU process via discrete-time OLS. Returns (kappa, mu, is_mean_reverting)."""
    if len(prices) < 10:
        return 0.0, float(np.mean(prices)), False

    y = prices[1:]
    x = prices[:-1]

    # Can't fit regression if all x values are identical (constant price series)
    if np.std(x) < 1e-10:
        return 0.0, float(np.mean(prices)), False
    # Guard: linregress crashes if all x values are identical (constant price series)
    if np.std(x) < 1e-10:
        return 0.0, float(np.mean(prices)), False
    slope, intercept, r, p, _ = stats.linregress(x, y)

    if slope >= 1.0 or slope <= 0.0:
        return 0.0, float(np.mean(prices)), False

    dt = max(float(np.mean(np.diff(times_hours))), 1e-6)
    kappa = -np.log(max(slope, 1e-10)) / dt
    mu = intercept / (1.0 - slope)
    halflife = np.log(2) / max(kappa, 1e-10)

    n = len(x)
    residuals = y - (intercept + slope * x)
    se = np.std(residuals) / (np.std(x) * np.sqrt(n)) if np.std(x) > 0 else 1e10
    t_stat = (slope - 1.0) / max(se, 1e-10)
    p_mean_rev = stats.t.cdf(t_stat, df=n - 2)

    is_mean_reverting = (
        p_mean_rev < 0.10
        and halflife < halflife_max_hours
        and kappa > 0.001
        and 0 < mu < prices.max() * 3
    )

    logger.debug(
        f"OU fit: κ={kappa:.4f} μ={mu:.4f} t½={halflife:.1f}h "
        f"p={p_mean_rev:.3f} mr={is_mean_reverting}"
    )
    return float(kappa), float(mu), is_mean_reverting


def mean_reversion_score(kappa: float, halflife_max: float) -> float:
    if kappa <= 0:
        return 0.0
    halflife = np.log(2) / kappa
    return float(np.clip(1.0 / (1.0 + (halflife / 4.0) ** 1.5), 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Drift score
# ─────────────────────────────────────────────────────────────────────────────

def compute_drift_score(prices: np.ndarray, times_hours: np.ndarray, window: int = 20) -> float:
    """R² × slope magnitude on log prices — high = trending = dangerous for MM."""
    window = min(window, max(len(prices), 3))
    recent = prices[-window:]
    times  = times_hours[-window:]
    if len(recent) < 3:
        return 0.0
    log_prices = np.log(np.maximum(recent, 1e-10))
    if np.std(times) < 1e-10 or np.std(log_prices) < 1e-10:
        return 0.0
    slope, _, r, _, _ = stats.linregress(times, log_prices)
    val = r**2 * min(abs(slope) * 10.0, 2.0)
    if np.isnan(val) or np.isinf(val):
        return 0.0
    return float(np.clip(val, 0.0, 2.0))


# ─────────────────────────────────────────────────────────────────────────────
# Order arrival intensity
# ─────────────────────────────────────────────────────────────────────────────

def estimate_arrival_intensity(
    trade_times_hours: list[float],
    default_lambda: float = 0.02,
    default_k: float = 1.5,
) -> tuple[float, float]:
    """Estimate Poisson λ_A (per hour). k falls back to default on sparse data."""
    if len(trade_times_hours) < 3:
        return default_lambda, default_k
    inter = np.diff(sorted(trade_times_hours))
    inter = inter[inter > 0]
    if len(inter) == 0:
        return default_lambda, default_k
    lambda_A = 1.0 / max(float(np.median(inter)), 0.1)
    return float(np.clip(lambda_A, 0.001, 10.0)), default_k


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_ticker(
    ticker: str,
    source: str,
    security_info: dict,
    ob_history: list[dict],
    trade_history: list[dict],
    cfg,
) -> TickerCalibration:
    """
    Full calibration for one ticker.

    ob_history / trade_history are already unwrapped lists of row dicts:
      {"id": int, "price": float, "volume": int, "timestamp": str, "ticker": str}
    """
    cal = TickerCalibration(ticker=ticker, source=source)

    # ── Atlas precomputed derived fields ──────────────────────────────────────
    derived = security_info.get("derived") or {}
    cal.market_price    = float(security_info.get("market_price") or 0.0)
    cal.liquidity_score = float(derived.get("liquidity_score") or 0.0)
    cal.ob_imbalance    = float(derived.get("orderbook_imbalance") or 0.0)
    cal.atlas_vol_7d    = atlas_vol_to_hourly(derived.get("volatility_7d"))

    atlas_spread_pct = derived.get("spread_pct")
    if atlas_spread_pct and atlas_spread_pct > 0:
        cal.avg_spread_pct = float(atlas_spread_pct)

    # ── Extract price time series ─────────────────────────────────────────────
    prices_raw, times_raw = [], []
    for row in ob_history:
        price = row.get("price")
        ts    = row.get("timestamp")
        if price and price > 0 and ts:
            t = _iso_to_epoch_hours(ts)
            if t is not None:
                prices_raw.append(float(price))
                times_raw.append(t)

    # ── Trade history (same row format) ──────────────────────────────────────
    clean_trades = _clean_trades(trade_history)
    trade_times  = _extract_trade_times_hours(clean_trades)
    cal.trades_per_day = _trades_per_day(clean_trades)

    # ── Fallback path: insufficient price history ─────────────────────────────
    if len(prices_raw) < 3:
        logger.warning(f"{ticker} [{source}]: only {len(prices_raw)} price rows — using Atlas precomputed stats")
        cal.mid            = cal.market_price
        cal.sigma          = cal.atlas_vol_7d or 0.01
        cal.sigma_long_run = cal.sigma
        cal.lambda_A, cal.k = estimate_arrival_intensity(
            trade_times, cfg.DEFAULT_LAMBDA_A, cfg.DEFAULT_K
        )
        cal.eligible = (
            cal.market_price > 0
            and cal.sigma > 0
            and cal.trades_per_day >= cfg.SCORER_MIN_TRADES_DAY
        )
        _log_cal(cal)
        return cal

    # ── Full path: compute from price series ──────────────────────────────────
    pairs = sorted(zip(times_raw, prices_raw))
    times_hours = np.array([p[0] for p in pairs])
    prices      = np.array([p[1] for p in pairs])

    prices, times_hours = _remove_outliers(prices, times_hours)

    if len(prices) < 3:
        cal.eligible = False
        return cal

    times_hours = times_hours - times_hours[0]
    total_hours = max(float(times_hours[-1]), 1.0)

    cal.mid = float(prices[-1])

    # Volatility
    recent_n = min(cfg.VOL_ESTIMATION_WINDOW, len(prices))
    cal.sigma          = estimate_realized_vol(prices[-recent_n:], times_hours[-recent_n:])
    cal.sigma_long_run = estimate_realized_vol(prices, times_hours)
    cal.vol_of_vol     = estimate_vol_of_vol(prices, times_hours)

    # Blend with Atlas vol
    if cal.atlas_vol_7d and cal.atlas_vol_7d > 0:
        w_atlas = 0.6 if source == "tse" else 0.3   # TSE Atlas data is more reliable
        cal.sigma          = (1 - w_atlas) * cal.sigma + w_atlas * cal.atlas_vol_7d
        cal.sigma_long_run = max(cal.sigma_long_run, cal.atlas_vol_7d)

    # OU mean-reversion
    cal.kappa, cal.mu, cal.is_ou = fit_ou_process(prices, times_hours, cfg.MEAN_REV_HALFLIFE_MAX)
    cal.mean_reversion_score = mean_reversion_score(cal.kappa, cfg.MEAN_REV_HALFLIFE_MAX)

    # Drift
    cal.drift_score = compute_drift_score(prices, times_hours, cfg.DRIFT_ESTIMATION_WINDOW)

    # OB update freq
    cal.ob_update_freq = len(prices) / total_hours

    # Order arrival
    cal.lambda_A, cal.k = estimate_arrival_intensity(
        trade_times, cfg.DEFAULT_LAMBDA_A, cfg.DEFAULT_K
    )

    # Eligibility
    cal.eligible = (
        cal.trades_per_day >= cfg.SCORER_MIN_TRADES_DAY
        and cal.mid > 0
        and cal.sigma > 0
        and len(prices) >= 3
    )

    _log_cal(cal)
    return cal


def _log_cal(cal: TickerCalibration):
    logger.info(
        f"{cal.ticker} [{cal.source}]: mid={cal.mid:.4f} σ={cal.sigma:.4f} "
        f"σ_lr={cal.sigma_long_run:.4f} κ={cal.kappa:.3f} "
        f"drift={cal.drift_score:.3f} mr={cal.mean_reversion_score:.3f} "
        f"trades/day={cal.trades_per_day:.2f} liq={cal.liquidity_score:.1f} "
        f"eligible={cal.eligible}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main calibration pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def calibrate_all(atlas, cfg, excluded: set[str]) -> CalibrationResult:
    """
    Calibrate all NER + TSE tickers from Atlas.
    Both exchanges use /securities/source/{ner|tse}, /history/{ticker}, /transactions.
    """
    result = CalibrationResult()
    all_securities: list[dict] = []

    for source in ("ner", "tse"):
        try:
            resp   = await atlas.get_securities_by_source(source)
            parsed = _parse_securities(resp)
            for s in parsed:
                s["_source"] = source
            all_securities.extend(parsed)
            logger.info(f"Fetched {len(parsed)} {source.upper()} securities from Atlas")
        except Exception as e:
            logger.error(f"Failed to fetch {source.upper()} securities: {e}")

    if not all_securities:
        logger.error("No securities fetched — aborting calibration")
        return result

    to_calibrate = [
        s for s in all_securities
        if s.get("ticker")
        and not bool(s.get("frozen", 0))
        and s["ticker"] not in excluded
    ]

    logger.info(
        f"Calibrating {len(to_calibrate)} tickers: "
        f"{[s['ticker'] for s in to_calibrate]}"
    )

    async def fetch_one(s: dict):
        ticker = s["ticker"]
        ob, tx = await asyncio.gather(
            atlas.get_history(ticker, days=cfg.CALIBRATION_HISTORY_DAYS, limit=2000),
            atlas.get_transactions(ticker=ticker, limit=500),
            return_exceptions=True,
        )
        if isinstance(ob, Exception):
            logger.warning(f"{ticker}: history failed: {ob}")
            ob = []
        if isinstance(tx, Exception):
            logger.warning(f"{ticker}: transactions failed: {tx}")
            tx = []
        return s, _unwrap_list(ob), _unwrap_list(tx)

    fetch_results = await asyncio.gather(
        *[fetch_one(s) for s in to_calibrate],
        return_exceptions=True,
    )

    for res in fetch_results:
        if isinstance(res, Exception):
            logger.error(f"Fetch task error: {res}")
            continue
        s, ob_hist, trade_hist = res
        ticker = s["ticker"]
        source = s.get("_source", s.get("source", "ner"))
        try:
            cal = calibrate_ticker(ticker, source, s, ob_hist, trade_hist, cfg)
            result.tickers[ticker] = cal
        except Exception as e:
            logger.error(f"{ticker}: calibration error: {e}", exc_info=True)
            result.tickers[ticker] = TickerCalibration(ticker=ticker, source=source, eligible=False)

    result.eligible_tickers = [t for t, c in result.tickers.items() if c.eligible]
    logger.info(
        f"Calibration complete — {len(result.eligible_tickers)}/{len(to_calibrate)} eligible: "
        f"{result.eligible_tickers}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _remove_outliers(
    prices: np.ndarray,
    times: np.ndarray,
    sigma_threshold: float = 5.0,
    window: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    if len(prices) < 3:
        return prices, times
    keep = np.ones(len(prices), dtype=bool)
    for i in range(len(prices)):
        lo = max(0, i - window)
        hi = min(len(prices), i + window + 1)
        w = prices[lo:hi]
        med = np.median(w)
        mad = np.median(np.abs(w - med)) * 1.4826
        if mad > 1e-10:
            if abs(prices[i] - med) > sigma_threshold * mad:
                keep[i] = False
        elif med > 1e-10:
            if abs(prices[i] - med) / med > 0.50:
                keep[i] = False
    return prices[keep], times[keep]


def _clean_trades(trades: list[dict]) -> list[dict]:
    seen, cleaned = set(), []
    for t in trades:
        key = (t.get("timestamp"), t.get("price"), t.get("volume") or t.get("quantity"))
        if key not in seen:
            seen.add(key)
            cleaned.append(t)
    return cleaned


def _extract_trade_times_hours(trades: list[dict]) -> list[float]:
    times = []
    for t in trades:
        h = _iso_to_epoch_hours(t.get("timestamp"))
        if h is not None:
            times.append(h)
    return sorted(times)


def _trades_per_day(trades: list[dict]) -> float:
    times = _extract_trade_times_hours(trades)
    if len(times) < 2:
        return float(len(times))
    span_hours = max(times[-1] - times[0], 1.0)
    return (len(times) / span_hours) * 24.0
