# calibration.py
# Estimates all model parameters from Atlas market data.
#
# Paper references:
#   - Realized volatility from OB mid series: Andersen & Bollerslev (1998)
#   - OU mean-reversion estimation: Ornstein-Uhlenbeck MLE
#   - Order arrival rate k: Avellaneda & Stoikov (2008) §3
#   - Drift score: rolling linear regression on mid time series
#   - Vol-of-vol: variance of rolling realized vol windows

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats, optimize

logger = logging.getLogger(__name__)


@dataclass
class TickerCalibration:
    ticker: str

    # Realized volatility (per hour, from OB mid series)
    sigma: float = 0.01

    # Long-run volatility (for dynamic γ scaling)
    sigma_long_run: float = 0.01

    # Vol-of-vol: variance of rolling sigma windows (scorer penalty)
    vol_of_vol: float = 0.0

    # OU mean-reversion parameters
    kappa: float = 0.0      # reversion speed (1/hours)
    mu: float = 0.0         # long-run mean
    is_ou: bool = False     # did the OU fit pass the stationarity test?

    # Mean-reversion score [0,1]: 1 = strongly mean-reverting, 0 = random walk
    mean_reversion_score: float = 0.0

    # Drift score [0,∞]: rolling directional trend strength (higher = more dangerous)
    drift_score: float = 0.0

    # Order arrival intensity (Poisson, per hour)
    lambda_A: float = 0.02

    # Depth parameter k from AS model (intensity decay with distance from mid)
    k: float = 1.5

    # Average observed bid-ask spread (% of mid)
    avg_spread_pct: float = 0.05

    # OB update frequency (updates/hour — proxy for market activity)
    ob_update_freq: float = 0.0

    # Trades per day (from transactions tape)
    trades_per_day: float = 0.0

    # Current mid price
    mid: float = 0.0

    # Max inventory in shares (derived from capital allocation later)
    q_max: int = 10

    # Is this ticker eligible for quoting?
    eligible: bool = False


@dataclass
class CalibrationResult:
    tickers: dict[str, TickerCalibration] = field(default_factory=dict)
    eligible_tickers: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.monotonic)


# ─────────────────────────────────────────────────────────────────────────────
# Volatility estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_realized_vol(mids: np.ndarray, times_hours: np.ndarray) -> float:
    """
    Realized volatility from OB mid price series.
    Uses log returns, annualised to per-hour units.

    Andersen & Bollerslev (1998): sum of squared log returns.
    We use per-hour rather than per-day to match T_HORIZON_HOURS.
    """
    if len(mids) < 3:
        return 0.01

    log_returns = np.diff(np.log(mids + 1e-10))

    # Time-weight: scale by sqrt of time between observations
    dt = np.diff(times_hours)
    dt = np.where(dt > 0, dt, 1e-6)

    # Realised variance per hour
    rv = np.sum(log_returns ** 2 / dt) / len(log_returns)
    return float(np.sqrt(max(rv, 1e-8)))


def estimate_vol_of_vol(mids: np.ndarray, times_hours: np.ndarray, n_windows: int = 5) -> float:
    """
    Vol-of-vol: standard deviation of rolling realized vol estimates.
    High vol-of-vol means the market's character changes unpredictably — bad for MM.
    """
    if len(mids) < n_windows * 3:
        return 0.0

    window_size = len(mids) // n_windows
    vols = []
    for i in range(n_windows):
        lo = i * window_size
        hi = lo + window_size
        v = estimate_realized_vol(mids[lo:hi], times_hours[lo:hi])
        vols.append(v)

    return float(np.std(vols)) if len(vols) > 1 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# OU mean-reversion
# ─────────────────────────────────────────────────────────────────────────────

def fit_ou_process(
    prices: np.ndarray,
    times_hours: np.ndarray,
    halflife_max_hours: float = 48.0,
) -> tuple[float, float, bool]:
    """
    Fit OU process: dS = κ(μ - S)dt + σdW
    Using discrete-time MLE: S_{t+1} = α + β·S_t + ε

    Returns (kappa, mu, is_mean_reverting).
    is_mean_reverting=True only if:
      - β is statistically significantly < 1 (ADF-style test)
      - implied half-life < halflife_max_hours
    """
    if len(prices) < 10:
        return 0.0, float(np.mean(prices)), False

    y = prices[1:]
    x = prices[:-1]

    # OLS: y = α + β·x
    slope, intercept, r, p, _ = stats.linregress(x, y)

    # β < 1 required for stationarity
    if slope >= 1.0 or slope <= 0.0:
        return 0.0, float(np.mean(prices)), False

    # Average dt
    dt = float(np.mean(np.diff(times_hours)))
    dt = max(dt, 1e-6)

    # κ from β = exp(-κ·dt) → κ = -ln(β)/dt
    kappa = -np.log(max(slope, 1e-10)) / dt
    mu = intercept / (1.0 - slope)

    # Half-life
    halflife = np.log(2) / max(kappa, 1e-10)

    # Significance: use p-value of slope from linregress
    # We want slope significantly < 1, so test (slope - 1)
    n = len(x)
    se = np.std(y - (intercept + slope * x)) / (np.std(x) * np.sqrt(n))
    t_stat = (slope - 1.0) / max(se, 1e-10)
    p_mean_rev = stats.t.cdf(t_stat, df=n - 2)  # one-tailed: slope < 1

    is_mean_reverting = (
        p_mean_rev < 0.10           # 10% significance (loose — sparse data)
        and halflife < halflife_max_hours
        and kappa > 0.001
        and 0 < mu < prices.max() * 3  # mu sanity
    )

    logger.debug(
        f"OU fit: κ={kappa:.4f} μ={mu:.4f} half-life={halflife:.1f}h "
        f"p={p_mean_rev:.3f} mean_rev={is_mean_reverting}"
    )

    return float(kappa), float(mu), is_mean_reverting


def mean_reversion_score(kappa: float, halflife_max: float) -> float:
    """
    Map kappa → [0, 1] score.
    Fast reversion (small half-life) → score close to 1.
    Random walk (κ≈0) → score close to 0.
    """
    if kappa <= 0:
        return 0.0
    halflife = np.log(2) / kappa
    # Sigmoid-ish: score=0.9 at halflife=1h, score=0.5 at halflife=halflife_max/2
    score = 1.0 / (1.0 + (halflife / 4.0) ** 1.5)
    return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Drift score
# ─────────────────────────────────────────────────────────────────────────────

def compute_drift_score(mids: np.ndarray, times_hours: np.ndarray, window: int = 20) -> float:
    """
    Drift score: how strongly is the price trending in one direction?

    Uses R² of a linear regression on the recent window of log prices.
    R²=1 means perfectly trending (dangerous for MM).
    R²=0 means no trend (safe for MM).

    Scaled by the slope magnitude to penalise fast trending more.
    """
    if len(mids) < window:
        window = max(len(mids), 3)

    recent_mids = mids[-window:]
    recent_times = times_hours[-window:]

    if len(recent_mids) < 3:
        return 0.0

    log_prices = np.log(recent_mids + 1e-10)
    slope, intercept, r, p, _ = stats.linregress(recent_times, log_prices)

    r_sq = r ** 2
    # Annualise slope to hourly drift rate
    hourly_drift = abs(slope)

    # Combined: R² × drift magnitude — both must be present to score high
    score = r_sq * min(hourly_drift * 10.0, 2.0)
    return float(np.clip(score, 0.0, 2.0))


# ─────────────────────────────────────────────────────────────────────────────
# Order arrival intensity
# ─────────────────────────────────────────────────────────────────────────────

def estimate_arrival_intensity(
    trade_times_hours: list[float],
    default_lambda: float = 0.02,
    default_k: float = 1.5,
) -> tuple[float, float]:
    """
    Estimate Poisson arrival rate λ_A and depth parameter k.

    From Avellaneda & Stoikov (2008): order arrival intensity is
    Λ(δ) = A · exp(-k·δ), where δ is distance from mid.

    With sparse data we can only reliably estimate λ_A (total arrival rate).
    k is estimated from the distribution of trade prices vs contemporaneous mid
    if we have enough trades; otherwise falls back to default.

    Returns (lambda_A per hour, k).
    """
    if len(trade_times_hours) < 3:
        return default_lambda, default_k

    # λ_A: mean inter-arrival time → rate
    if len(trade_times_hours) >= 2:
        inter_arrivals = np.diff(sorted(trade_times_hours))
        inter_arrivals = inter_arrivals[inter_arrivals > 0]
        if len(inter_arrivals) > 0:
            mean_ia = float(np.median(inter_arrivals))  # median more robust
            lambda_A = 1.0 / max(mean_ia, 0.1)
        else:
            lambda_A = default_lambda
    else:
        lambda_A = default_lambda

    # k: without price-vs-mid data we use default
    # This will be improved when we have OB snapshots paired with trades
    k = default_k

    return float(np.clip(lambda_A, 0.001, 10.0)), k


# ─────────────────────────────────────────────────────────────────────────────
# Main calibration pipeline
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_ticker(
    ticker: str,
    ob_history: list[dict],
    trade_history: list[dict],
    cfg,
) -> TickerCalibration:
    """
    Calibrate all parameters for a single ticker from Atlas data.

    ob_history: list of OB snapshots [{"mid": float, "timestamp": str, ...}]
    trade_history: list of trades [{"price": float, "timestamp": str, ...}]
    """
    cal = TickerCalibration(ticker=ticker)

    # ── Extract mid time series from OB history ───────────────────────────────
    mids_raw = []
    times_raw = []

    for row in ob_history:
        mid = row.get("mid") or row.get("market_price")
        ts  = row.get("timestamp")
        if mid and mid > 0 and ts:
            try:
                # Convert ISO timestamp to hours since first observation
                import datetime
                if isinstance(ts, str):
                    dt = datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    epoch_hours = dt.timestamp() / 3600.0
                else:
                    epoch_hours = float(ts) / 3600.0
                mids_raw.append(float(mid))
                times_raw.append(epoch_hours)
            except Exception:
                continue

    if len(mids_raw) < 3:
        logger.warning(f"{ticker}: insufficient OB history ({len(mids_raw)} rows)")
        cal.eligible = False
        return cal

    # Sort by time
    pairs = sorted(zip(times_raw, mids_raw))
    times_hours = np.array([p[0] for p in pairs])
    mids = np.array([p[1] for p in pairs])

    # Remove obvious outliers (>5 MAD from rolling median)
    mids, times_hours = _remove_outliers(mids, times_hours)

    if len(mids) < 3:
        cal.eligible = False
        return cal

    # Normalise times to start from 0
    times_hours = times_hours - times_hours[0]

    cal.mid = float(mids[-1])

    # ── Volatility ────────────────────────────────────────────────────────────
    # Use recent window for short-run σ, full history for long-run σ
    recent_n = min(cfg.VOL_ESTIMATION_WINDOW, len(mids))
    cal.sigma = estimate_realized_vol(mids[-recent_n:], times_hours[-recent_n:])
    cal.sigma_long_run = estimate_realized_vol(mids, times_hours)
    cal.vol_of_vol = estimate_vol_of_vol(mids, times_hours)

    # ── OU mean-reversion ─────────────────────────────────────────────────────
    cal.kappa, cal.mu, cal.is_ou = fit_ou_process(
        mids, times_hours, cfg.MEAN_REV_HALFLIFE_MAX
    )
    cal.mean_reversion_score = mean_reversion_score(cal.kappa, cfg.MEAN_REV_HALFLIFE_MAX)

    # ── Drift ─────────────────────────────────────────────────────────────────
    cal.drift_score = compute_drift_score(mids, times_hours, cfg.DRIFT_ESTIMATION_WINDOW)

    # ── OB update frequency ───────────────────────────────────────────────────
    total_hours = max(times_hours[-1] - times_hours[0], 1.0)
    cal.ob_update_freq = len(mids) / total_hours

    # ── Average spread ────────────────────────────────────────────────────────
    spreads = []
    for row in ob_history:
        bid = row.get("best_bid") or row.get("bid")
        ask = row.get("best_ask") or row.get("ask")
        mid = row.get("mid")
        if bid and ask and mid and mid > 0 and ask > bid:
            spreads.append((ask - bid) / mid)
    cal.avg_spread_pct = float(np.median(spreads)) if spreads else 0.05

    # ── Trade history ─────────────────────────────────────────────────────────
    clean_trades = _clean_trades(trade_history)
    trade_times = _extract_trade_times_hours(clean_trades)

    if total_hours > 0 and len(clean_trades) > 0:
        cal.trades_per_day = (len(clean_trades) / total_hours) * 24.0
    else:
        cal.trades_per_day = 0.0

    cal.lambda_A, cal.k = estimate_arrival_intensity(
        trade_times,
        default_lambda=cfg.DEFAULT_LAMBDA_A,
        default_k=cfg.DEFAULT_K,
    )

    # ── Eligibility gate ──────────────────────────────────────────────────────
    cal.eligible = (
        cal.trades_per_day >= cfg.SCORER_MIN_TRADES_DAY
        and cal.mid > 0
        and cal.sigma > 0
        and len(mids) >= 5
    )

    logger.info(
        f"{ticker}: σ={cal.sigma:.4f} σ_lr={cal.sigma_long_run:.4f} "
        f"κ={cal.kappa:.3f} drift={cal.drift_score:.3f} "
        f"mr={cal.mean_reversion_score:.3f} trades/day={cal.trades_per_day:.2f} "
        f"eligible={cal.eligible}"
    )

    return cal


async def calibrate_all(atlas, cfg, excluded: set[str]) -> CalibrationResult:
    """
    Run full calibration pipeline for all NER tickers via Atlas.
    Returns CalibrationResult with per-ticker parameters.
    """
    result = CalibrationResult()

    # Fetch NER securities only
    try:
        securities = await atlas.get_securities_by_source("ner")
    except Exception as e:
        logger.error(f"Failed to fetch NER securities: {e}")
        return result

    # Atlas /securities/source/ner may return either:
    #   - list of dicts: [{"ticker": "RTG", "frozen": false, ...}]
    #   - list of strings: ["RTG", "BB", ...]
    # Handle both defensively.
    tickers = []
    for s in securities:
        if isinstance(s, str):
            ticker = s
            frozen = False
        elif isinstance(s, dict):
            ticker = s.get("ticker", "")
            frozen = s.get("frozen", False)
        else:
            continue
        if ticker and not frozen and ticker not in excluded:
            tickers.append(ticker)

    logger.info(f"Calibrating {len(tickers)} NER tickers: {tickers}")

    import asyncio
    for ticker in tickers:
        try:
            ob_hist, trade_hist = await asyncio.gather(
                atlas.get_history(ticker, days=cfg.CALIBRATION_HISTORY_DAYS, limit=2000),
                atlas.get_transactions(ticker=ticker, limit=500),
                return_exceptions=True,
            )

            if isinstance(ob_hist, Exception):
                logger.warning(f"{ticker}: OB history fetch failed: {ob_hist}")
                ob_hist = []
            if isinstance(trade_hist, Exception):
                logger.warning(f"{ticker}: trade history fetch failed: {trade_hist}")
                trade_hist = []

            cal = calibrate_ticker(ticker, ob_hist, trade_hist, cfg)
            result.tickers[ticker] = cal

        except Exception as e:
            logger.error(f"{ticker}: calibration error: {e}", exc_info=True)
            result.tickers[ticker] = TickerCalibration(ticker=ticker, eligible=False)

    result.eligible_tickers = [
        t for t, c in result.tickers.items() if c.eligible
    ]

    logger.info(
        f"Calibration complete. Eligible: {result.eligible_tickers}"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _remove_outliers(
    mids: np.ndarray,
    times: np.ndarray,
    sigma_threshold: float = 5.0,
    window: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove price outliers using rolling MAD filter."""
    if len(mids) < 3:
        return mids, times

    keep = np.ones(len(mids), dtype=bool)
    for i in range(len(mids)):
        lo = max(0, i - window)
        hi = min(len(mids), i + window + 1)
        w = mids[lo:hi]
        med = np.median(w)
        mad = np.median(np.abs(w - med)) * 1.4826
        if mad > 1e-10:
            if abs(mids[i] - med) > sigma_threshold * mad:
                keep[i] = False
        elif med > 1e-10:
            if abs(mids[i] - med) / med > 0.50:
                keep[i] = False

    return mids[keep], times[keep]


def _clean_trades(trades: list[dict]) -> list[dict]:
    """Remove duplicate trades (NER API returns each trade twice)."""
    seen = set()
    cleaned = []
    for t in trades:
        key = (t.get("timestamp"), t.get("price"), t.get("volume") or t.get("quantity"))
        if key not in seen:
            seen.add(key)
            cleaned.append(t)
    return cleaned


def _extract_trade_times_hours(trades: list[dict]) -> list[float]:
    """Extract trade timestamps as hours since epoch."""
    import datetime
    times = []
    for t in trades:
        ts = t.get("timestamp")
        if ts:
            try:
                dt = datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                times.append(dt.timestamp() / 3600.0)
            except Exception:
                continue
    return sorted(times)
