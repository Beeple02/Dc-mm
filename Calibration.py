# calibration.py
# Estimates all model parameters from historical NER data.
#
# Paper references:
#   - OU parameters (κ, μ, σ): Bergault Ch.6 §6.2.1
#   - Intensity function (A, k): AS (2008), Bergault Ch.1 §1.2.2
#   - Covariance matrix Σ: Bergault Ch.4 §4.2.1
#   - Objective function selection: Bergault intro §1.2
#
# Data quality fixes applied (informed by empirical NER data analysis):
#   FIX 1 — Outlier filtering: prices >OUTLIER_SIGMA_THRESHOLD std devs from
#            rolling median are removed before any fitting. Handles the $1 RTG
#            trades and $300/$1000 GCC trades observed in historical data.
#   FIX 2 — Duplicate detection: exact duplicate (timestamp, price, volume) rows
#            are dropped. Every trade in the NER CSV export appears exactly twice.
#   FIX 3 — Minimum trade count gate: tickers with fewer than MIN_TRADES_FOR_QUOTING
#            clean trades, or below MIN_TRADES_PER_DAY frequency, are excluded from
#            quoting and receive zero capital allocation.
#   FIX 4 — Hard exclusion list: tickers that pass the count gate but have
#            pathological price histories (e.g. GCC: $1000 → $10.89) are handled
#            by the outlier filter; the gate ensures they don't sneak through on
#            sparse clean data.

import logging
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from scipy import stats
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data cleaning
# ─────────────────────────────────────────────────────────────────────────────

def remove_duplicate_trades(history: list[dict]) -> list[dict]:
    """
    FIX 2: Drop exact duplicate rows.
    The NER export and potentially the live API return each trade twice
    (observed: every row in NER_pricehistory CSV appears exactly 2x).
    We deduplicate on (timestamp, price, volume).
    """
    seen = set()
    cleaned = []
    for row in history:
        key = (row.get("timestamp"), row.get("price"), row.get("volume"))
        if key not in seen:
            seen.add(key)
            cleaned.append(row)
    n_removed = len(history) - len(cleaned)
    if n_removed > 0:
        logger.debug(f"Removed {n_removed} duplicate trade rows")
    return cleaned


def remove_price_outliers(
    prices: list[float],
    times: list[float],
    volumes: list[float],
    sigma_threshold: float,
    rolling_window: int,
) -> tuple[list[float], list[float], list[float]]:
    """
    FIX 1: Remove price outliers using a rolling median filter.

    A trade is an outlier if its price deviates from the rolling median
    of the surrounding window by more than sigma_threshold standard deviations
    of prices within that window.

    This handles:
    - RTG: eight $1.00 trades when real price was ~$22 (likely test/error trades)
    - GCC: $300 and $1000 trades when real price was ~$17

    Returns cleaned (prices, times, volumes).
    """
    if len(prices) < 3:
        return prices, times, volumes

    arr = np.array(prices, dtype=float)
    keep = np.ones(len(arr), dtype=bool)

    for i in range(len(arr)):
        lo = max(0, i - rolling_window)
        hi = min(len(arr), i + rolling_window + 1)
        window = arr[lo:hi]
        med = np.median(window)
        dev = abs(arr[i] - med)

        # Use MAD (median absolute deviation) — robust to outliers.
        # Scale factor 1.4826 makes it consistent with std for Gaussian data.
        mad = np.median(np.abs(window - med)) * 1.4826

        if mad > 1e-10:
            # Normal case: MAD-based threshold
            is_outlier = dev > sigma_threshold * mad
        elif med > 1e-10:
            # MAD = 0 means most values are identical (e.g. all $22).
            # Fall back to a percentage deviation from the median.
            # Any price deviating by more than 50% from the consensus is an outlier.
            # This correctly catches RTG $1 trades when all other prices are $22
            # (1/22 = 95% deviation) and GCC $300 when others are ~$17 (1664%).
            is_outlier = (dev / med) > 0.50
        else:
            is_outlier = False

        if is_outlier:
            keep[i] = False
            logger.debug(
                f"Outlier removed: price={arr[i]:.4f}, "
                f"window_median={med:.4f}, MAD={mad:.4f}, dev={dev:.4f}"
            )

    n_removed = int((~keep).sum())
    if n_removed > 0:
        logger.info(f"Outlier filter removed {n_removed}/{len(prices)} trades")

    prices_c = [p for p, k in zip(prices, keep) if k]
    times_c = [t for t, k in zip(times, keep) if k]
    vols_c = [v for v, k in zip(volumes, keep) if k]
    return prices_c, times_c, vols_c


def parse_timestamp(ts: str) -> Optional[datetime]:
    """Parse NER timestamp strings — handles both ISO format and legacy dd-mm-yy."""
    ts = str(ts).strip()
    for fmt in [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%d-%m-%y",
        "%d-%m-%Y",
        "%Y-%m-%d",
    ]:
        try:
            return datetime.strptime(ts, fmt)
        except ValueError:
            continue
    logger.warning(f"Could not parse timestamp: {ts!r}")
    return None


def check_quoting_eligibility(
    ticker: str,
    clean_trades: list[dict],
    total_hours_elapsed: float,
    min_trades: int,
    min_trades_per_day: float,
) -> tuple[bool, str]:
    """
    FIX 3: Hard gate on minimum data quality before a ticker is eligible for quoting.

    Returns (is_eligible, reason_string).
    """
    n = len(clean_trades)

    # FIX 3a: Absolute minimum trade count
    if n < min_trades:
        return False, f"only {n} clean trades (minimum {min_trades})"

    # FIX 3b: Minimum frequency
    days_elapsed = max(total_hours_elapsed / 24.0, 1.0)
    trades_per_day = n / days_elapsed
    if trades_per_day < min_trades_per_day:
        return False, (
            f"trade frequency {trades_per_day:.4f}/day below minimum "
            f"{min_trades_per_day}/day"
        )

    return True, "eligible"


# ─────────────────────────────────────────────────────────────────────────────
# Parameter estimation
# ─────────────────────────────────────────────────────────────────────────────

def fit_ou_parameters(
    prices: list[float],
    times_hours: list[float],
) -> tuple[float, float, float, bool]:
    """
    Fit Ornstein-Uhlenbeck parameters via OLS on the discrete-time representation.

    The OU process dS = κ(μ - S)dt + σ dW discretises to:
        S_{t+1} - S_t = κμ·Δt - κ·S_t·Δt + σ√Δt·ε
    which is a linear regression:
        ΔS = a + b·S_t + noise
    where:
        b = -κ·Δt  →  κ = -b/Δt
        a = κμ·Δt  →  μ = a / (κ·Δt) = -a/b

    Input prices/times must already be outlier-filtered.
    Paper reference: Bergault Ch.6 §6.2.1
    """
    if len(prices) < 5:
        return 0.0, float(np.mean(prices)) if prices else 10.0, 1.0, False

    prices_arr = np.array(prices, dtype=float)
    times_arr = np.array(times_hours, dtype=float)

    dt_vec = np.diff(times_arr)
    dt_vec = np.where(dt_vec <= 0, 1e-6, dt_vec)
    dS = np.diff(prices_arr)
    S_t = prices_arr[:-1]

    X = np.column_stack([np.ones_like(S_t), S_t])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, dS, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, float(np.mean(prices_arr)), float(np.std(prices_arr)), False

    a, b = coeffs
    avg_dt = float(np.mean(dt_vec))

    kappa = -b / avg_dt if avg_dt > 0 else 0.0

    res = dS - (a + b * S_t)
    std_res = np.std(res)
    std_S = np.std(S_t)
    se = std_res / (std_S * np.sqrt(len(S_t))) if std_S > 1e-10 else 1e9
    t_stat = b / se if se > 1e-10 else 0.0
    p_value = stats.t.cdf(t_stat, df=len(S_t) - 2)  # one-sided: b < 0

    is_ou = (kappa > 0.01) and (p_value < 0.10)

    mu = (-a / b) if abs(b) > 1e-10 else float(np.mean(prices_arr))

    # Clamp μ to a reasonable range around observed prices
    price_min, price_max = float(np.min(prices_arr)), float(np.max(prices_arr))
    mu = float(np.clip(mu, price_min * 0.5, price_max * 2.0))

    sigma = float(np.std(res) / np.sqrt(avg_dt)) if avg_dt > 0 else float(np.std(dS))
    sigma = max(sigma, 1e-6)

    logger.debug(
        f"OU fit: κ={kappa:.4f}, μ={mu:.4f}, σ={sigma:.4f}, "
        f"is_ou={is_ou}, p={p_value:.3f}"
    )
    return max(kappa, 0.0), mu, sigma, is_ou


def fit_intensity_function(
    history: list[dict],
) -> tuple[float, float]:
    """
    Estimate Λ(δ) = A * exp(-k * δ) parameters from clean, deduped history.

    A = arrival rate at zero spread (trades per hour), estimated from
        observed trade frequency.
    k = spread sensitivity, estimated from the heuristic k ≈ 2/typical_spread
        (from AS 2008: optimal spread ≈ 2/k + inventory term).

    Input history must already be outlier-filtered and deduplicated.
    Paper reference: Bergault Ch.1 §1.2.2, AS (2008)
    """
    if not history or len(history) < 2:
        return 1.0, 1.5

    sorted_h = sorted(history, key=lambda x: x.get("timestamp", ""))

    try:
        t0 = parse_timestamp(sorted_h[0]["timestamp"])
        t1 = parse_timestamp(sorted_h[-1]["timestamp"])
        if t0 and t1:
            hours_elapsed = max((t1 - t0).total_seconds() / 3600.0, 1.0)
        else:
            hours_elapsed = 24.0
    except Exception:
        hours_elapsed = 24.0

    n_trades = len(sorted_h)
    lambda_A = n_trades / hours_elapsed

    prices = [float(h["price"]) for h in sorted_h if h.get("price")]
    if len(prices) > 1:
        mean_price = np.mean(prices)
        # Typical observed spread proxy: std of successive price changes
        typical_move = np.std(np.diff(prices))
        # k in 1/dollar units: k ≈ 2 / natural_spread
        # natural_spread ≈ max(typical_move, 1% of mean price)
        natural_spread = max(typical_move, mean_price * 0.01)
        lambda_k = 2.0 / natural_spread
    else:
        lambda_k = 1.5

    lambda_A = float(np.clip(lambda_A, 1e-4, 100.0))
    lambda_k = float(np.clip(lambda_k, 0.01, 100.0))

    logger.debug(f"Intensity fit: A={lambda_A:.6f}/hr, k={lambda_k:.4f}")
    return lambda_A, lambda_k


def select_objective_function(
    history: list[dict],
    gamma: float,
    pvalue_threshold: float = 0.05,
) -> float:
    """
    Select ξ: 0.0 (Cartea) or γ (AS exponential utility).

    Tests whether inter-trade times are consistent with a Poisson process
    (exponential inter-arrivals, CV ≈ 1). If they are, Cartea's risk-adjusted
    expectation is appropriate. If fills are lumpy/bursty (CV >> 1, KS rejects
    exponential), AS exponential utility is more conservative and correct.

    Empirical finding from NER data: all liquid tickers (RTG CV=3.1, RED CV=2.4,
    BB CV=1.6) strongly reject Poisson → expect ξ=γ to be chosen in almost all cases.

    Paper reference: Bergault intro §1.2, eq (1) vs (2)
    """
    if len(history) < 10:
        logger.info(
            f"Insufficient trades ({len(history)}) for objective function test "
            f"→ defaulting to AS (ξ=γ)"
        )
        return gamma

    sorted_h = sorted(history, key=lambda x: x.get("timestamp", ""))
    inter_times = []

    for i in range(1, len(sorted_h)):
        t0 = parse_timestamp(sorted_h[i - 1]["timestamp"])
        t1 = parse_timestamp(sorted_h[i]["timestamp"])
        if t0 and t1:
            dt = (t1 - t0).total_seconds() / 3600.0
            if dt > 0:
                inter_times.append(dt)

    if len(inter_times) < 5:
        return gamma

    inter_arr = np.array(inter_times)
    cv = np.std(inter_arr) / np.mean(inter_arr) if np.mean(inter_arr) > 0 else 99.0

    rate = 1.0 / np.mean(inter_arr)
    ks_stat, p_value = stats.kstest(inter_arr, "expon", args=(0, 1.0 / rate))

    logger.info(
        f"Inter-trade CV={cv:.3f}, KS p={p_value:.4f} "
        f"(threshold={pvalue_threshold})"
    )

    if p_value > pvalue_threshold and cv < 1.5:
        logger.info("Objective function selected: Cartea (ξ=0) — Poisson arrivals confirmed")
        return 0.0
    else:
        logger.info(
            f"Objective function selected: AS exponential utility (ξ=γ={gamma:.4f})"
            f" — lumpy arrivals (CV={cv:.2f}, p={p_value:.4f})"
        )
        return gamma


def compute_covariance_matrix(
    clean_histories: dict[str, list[dict]],
    ticker_list: list[str],
) -> np.ndarray:
    """
    Compute the variance-covariance matrix Σ from log-returns of clean histories.
    Used in the Chapter 4 quadratic approximation for multi-asset cross-inventory skew.

    Only tickers that pass the quoting eligibility gate are included.
    Per-hour variance (OHLCV daily data divided by 24).

    Paper reference: Bergault Ch.4 §4.2.1, eq. (4.1)
    """
    n = len(ticker_list)
    if n == 0:
        return np.array([[]])

    returns: dict[str, np.ndarray] = {}

    for ticker in ticker_list:
        hist = clean_histories.get(ticker, [])
        if len(hist) < 2:
            returns[ticker] = np.array([0.0])
            continue
        sorted_h = sorted(hist, key=lambda x: x.get("timestamp", ""))
        prices = np.array([float(h["price"]) for h in sorted_h], dtype=float)
        prices = np.maximum(prices, 1e-6)
        log_ret = np.diff(np.log(prices))
        returns[ticker] = log_ret

    min_len = min(
        (len(r) for r in returns.values() if len(r) > 1),
        default=0,
    )

    if min_len < 2:
        logger.warning("Insufficient aligned return history for covariance — using identity")
        # Use per-ticker variance on diagonal, zero off-diagonal
        diag_vars = []
        for ticker in ticker_list:
            h = clean_histories.get(ticker, [])
            if len(h) > 1:
                prices = np.array([float(x["price"]) for x in h])
                diag_vars.append(np.var(np.diff(np.log(np.maximum(prices, 1e-6)))))
            else:
                diag_vars.append(0.01)
        return np.diag(diag_vars)

    aligned = np.column_stack([returns[t][-min_len:] for t in ticker_list])
    if aligned.ndim == 1:
        aligned = aligned.reshape(-1, 1)

    cov = np.cov(aligned, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    # Convert daily returns to per-hour variance
    cov = cov / 24.0

    # Regularise: ensure positive semi-definite
    cov = cov + np.eye(n) * 1e-8

    logger.debug(f"Covariance matrix ({n}×{n}):\n{np.round(cov, 6)}")
    return cov


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TickerCalibration:
    ticker: str
    # OU parameters
    sigma: float = 0.0
    kappa: float = 0.0
    mu: float = 0.0
    is_ou: bool = False

    # Intensity function Λ(δ) = A * exp(-k * δ)
    lambda_A: float = 1.0
    lambda_k: float = 1.5

    # Objective function for this ticker
    xi: float = 0.0        # ξ=γ for AS exponential utility, ξ=0 for Cartea

    # Inventory limits (shares)
    q_max: int = 10
    total_shares: int = 0

    # Capital weight (recomputed dynamically)
    capital_weight: float = 0.0

    # Trade frequency
    trades_per_hour: float = 0.0

    # Eligibility
    eligible: bool = True
    ineligible_reason: str = ""

    # Number of clean trades used for calibration
    n_clean_trades: int = 0


@dataclass
class CalibrationResult:
    tickers: dict[str, TickerCalibration] = field(default_factory=dict)
    sigma_matrix: np.ndarray = field(default_factory=lambda: np.array([[]]))
    # Only eligible tickers (used for ODE solving, Riccati, capital allocation)
    eligible_tickers: list[str] = field(default_factory=list)
    # All tickers seen (including ineligible)
    all_tickers: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Main calibration entry point
# ─────────────────────────────────────────────────────────────────────────────

async def calibrate_all(
    api_client,
    config,
    gamma: float,
) -> CalibrationResult:
    """
    Full calibration pipeline. Called at startup and periodically.
    Fetches all data from the API, applies all data quality fixes,
    and computes all model parameters for eligible tickers.
    """
    logger.info("=" * 50)
    logger.info("Starting full calibration...")

    # 1. Get all non-frozen securities
    securities = await api_client.get_securities()
    all_tickers = [s["ticker"] for s in securities if not s.get("frozen", False)]
    logger.info(f"Non-frozen tickers: {all_tickers}")

    sec_info = {s["ticker"]: s for s in securities}

    # 2. Fetch raw price histories
    raw_histories: dict[str, list[dict]] = {}
    for ticker in all_tickers:
        try:
            hist = await api_client.get_price_history(ticker, days=90)
            raw_histories[ticker] = hist
            logger.info(f"{ticker}: {len(hist)} raw trade records from API")
        except Exception as e:
            logger.warning(f"Failed to fetch history for {ticker}: {e}")
            raw_histories[ticker] = []

    # 3. Apply FIX 2 (deduplication) + FIX 1 (outlier removal) per ticker
    clean_histories: dict[str, list[dict]] = {}

    for ticker in all_tickers:
        raw = raw_histories[ticker]

        # FIX 2: Deduplicate
        if config.DUPLICATE_DETECTION:
            deduped = remove_duplicate_trades(raw)
        else:
            deduped = raw

        # Sort by timestamp before outlier filtering
        parsed = []
        for row in deduped:
            ts = parse_timestamp(str(row.get("timestamp", "")))
            if ts is not None:
                parsed.append((ts, row))
        parsed.sort(key=lambda x: x[0])
        sorted_rows = [r for _, r in parsed]

        if len(sorted_rows) < 2:
            clean_histories[ticker] = sorted_rows
            continue

        prices_raw = [float(r["price"]) for r in sorted_rows]
        times_raw = []
        base = parsed[0][0]
        for ts_dt, _ in parsed:
            times_raw.append((ts_dt - base).total_seconds() / 3600.0)
        vols_raw = [float(r.get("volume", 1)) for r in sorted_rows]

        # FIX 1: Remove outliers
        prices_c, times_c, vols_c = remove_price_outliers(
            prices_raw,
            times_raw,
            vols_raw,
            config.OUTLIER_SIGMA_THRESHOLD,
            config.OUTLIER_ROLLING_WINDOW,
        )

        # Rebuild clean history list
        clean_rows = []
        for i, orig_row in enumerate(sorted_rows):
            p = float(orig_row["price"])
            if p in prices_c:
                idx = prices_c.index(p)
                prices_c.pop(idx)
                times_c.pop(idx)
                vols_c.pop(idx)
                clean_rows.append(orig_row)

        clean_histories[ticker] = clean_rows
        logger.info(
            f"{ticker}: {len(raw)} raw → {len(deduped)} deduped → "
            f"{len(clean_rows)} clean trades"
        )

    # 4. FIX 3: Eligibility gate per ticker
    result = CalibrationResult(all_tickers=all_tickers)

    for ticker in all_tickers:
        clean = clean_histories.get(ticker, [])

        # Compute elapsed time for frequency check
        if len(clean) >= 2:
            t0 = parse_timestamp(str(clean[0].get("timestamp", "")))
            t1 = parse_timestamp(str(clean[-1].get("timestamp", "")))
            if t0 and t1:
                total_hours = max((t1 - t0).total_seconds() / 3600.0, 1.0)
            else:
                total_hours = 1.0
        else:
            total_hours = 1.0

        eligible, reason = check_quoting_eligibility(
            ticker,
            clean,
            total_hours,
            config.MIN_TRADES_FOR_QUOTING,
            config.MIN_TRADES_PER_DAY,
        )

        if not eligible:
            logger.info(f"{ticker}: INELIGIBLE — {reason}")
            sec = sec_info.get(ticker, {})
            cal = TickerCalibration(
                ticker=ticker,
                mu=sec.get("market_price", 10.0),
                sigma=sec.get("market_price", 10.0) * 0.05,
                eligible=False,
                ineligible_reason=reason,
                n_clean_trades=len(clean),
            )
            result.tickers[ticker] = cal
            continue

        # 5. Estimate parameters for eligible tickers
        sec = sec_info.get(ticker, {})
        total_shares = sec.get("total_shares", 1000)

        # Build price/time arrays from clean history
        sorted_clean = sorted(clean, key=lambda x: x.get("timestamp", ""))
        base_dt = parse_timestamp(str(sorted_clean[0]["timestamp"]))

        prices_fit = []
        times_fit = []
        for row in sorted_clean:
            ts = parse_timestamp(str(row["timestamp"]))
            if ts and base_dt:
                prices_fit.append(float(row["price"]))
                times_fit.append((ts - base_dt).total_seconds() / 3600.0)

        # OU fit
        kappa, mu, sigma, is_ou = fit_ou_parameters(prices_fit, times_fit)

        # Fallback μ to current market price if OU fit is unreliable
        current_price = sec.get("market_price", float(np.mean(prices_fit)) if prices_fit else 10.0)
        if not is_ou:
            mu = current_price

        # Intensity function
        lambda_A, lambda_k = fit_intensity_function(sorted_clean)

        # Objective function
        if config.OBJECTIVE_FUNCTION == "auto":
            xi = select_objective_function(
                sorted_clean, gamma, config.OBJECTIVE_AUTO_PVALUE_THRESHOLD
            )
        elif config.OBJECTIVE_FUNCTION == "cartea":
            xi = 0.0
        else:
            xi = gamma

        # Inventory limit
        q_max = max(1, int(total_shares * config.INVENTORY_LIMIT_PCT))

        # Trades per hour
        trades_per_hour = len(sorted_clean) / max(total_hours, 1.0)

        cal = TickerCalibration(
            ticker=ticker,
            sigma=sigma,
            kappa=kappa,
            mu=mu,
            is_ou=is_ou,
            lambda_A=lambda_A,
            lambda_k=lambda_k,
            xi=xi,
            q_max=q_max,
            total_shares=total_shares,
            trades_per_hour=trades_per_hour,
            eligible=True,
            n_clean_trades=len(sorted_clean),
        )
        result.tickers[ticker] = cal
        result.eligible_tickers.append(ticker)

        logger.info(
            f"{ticker} ✓ | n={len(sorted_clean)} | "
            f"σ={sigma:.4f} κ={kappa:.4f} μ={mu:.3f} OU={is_ou} | "
            f"A={lambda_A:.6f}/hr k={lambda_k:.4f} | "
            f"ξ={xi:.4f} | q_max={q_max} | {trades_per_hour:.5f} trades/hr"
        )

    # 6. Covariance matrix for eligible tickers only
    eligible_clean = {t: clean_histories[t] for t in result.eligible_tickers}
    result.sigma_matrix = compute_covariance_matrix(
        eligible_clean, result.eligible_tickers
    )

    logger.info(
        f"Calibration complete: {len(result.eligible_tickers)} eligible, "
        f"{len(all_tickers) - len(result.eligible_tickers)} ineligible"
    )
    logger.info(f"Eligible tickers: {result.eligible_tickers}")
    logger.info("=" * 50)

    return result


def compute_capital_allocation(
    calibrations: dict,
    orderbooks: dict,
    total_capital: float,
    reserve_buffer: float = 0.05,
) -> dict[str, float]:
    """
    Dynamically allocate capital across eligible tickers only.

    Score per ticker = spread_opportunity × liquidity_score
      spread_opportunity = (ask - bid) / mid  (normalised book spread)
                           or 0.5 if book is one-sided (we'd be sole MM)
      liquidity_score    = trades_per_hour (normalised)

    Capital weight = score_i / Σ_j score_j
    Applied to (1 - reserve_buffer) × total_capital.

    Ineligible tickers receive zero allocation.
    """
    deployable = total_capital * (1.0 - reserve_buffer)
    scores = {}

    for ticker, cal in calibrations.items():
        if not cal.eligible:
            scores[ticker] = 0.0
            continue

        ob = orderbooks.get(ticker, {})
        best_bid = ob.get("best_bid")
        best_ask = ob.get("best_ask")
        mid = ob.get("mid")

        if best_bid and best_ask and mid and mid > 0:
            spread_opp = (best_ask - best_bid) / mid
        else:
            # One-sided or empty book — high opportunity (sole MM)
            spread_opp = 0.5

        liquidity = max(cal.trades_per_hour, 1e-6)
        scores[ticker] = spread_opp * liquidity

    total_score = sum(scores.values())
    if total_score < 1e-12:
        n_eligible = sum(1 for c in calibrations.values() if c.eligible)
        if n_eligible == 0:
            return {t: 0.0 for t in calibrations}
        per_ticker = deployable / n_eligible
        return {
            t: per_ticker if calibrations[t].eligible else 0.0
            for t in calibrations
        }

    return {
        t: (scores[t] / total_score) * deployable
        for t in calibrations
    }
