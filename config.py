# config.py
# Central configuration for the NER Market Making Bot
# All parameters are documented with their paper reference

import os

# ── API ──────────────────────────────────────────────────────────────────────
API_BASE_URL = "http://150.230.117.88:8082"
API_KEY = os.environ.get("NER_API_KEY", "")  # Set via environment variable

# ── Capital management ───────────────────────────────────────────────────────
TOTAL_MM_CAPITAL = 10_000.0      # USD allocated to MM activity
CAPITAL_RESERVE_BUFFER = 0.05   # Keep 5% unallocated as buffer
REBALANCE_THRESHOLD = 0.20      # Reallocate if ticker weight drifts >20%

# ── Time horizon ─────────────────────────────────────────────────────────────
# T in the Avellaneda-Stoikov / Bergault model.
# Represents the rolling window over which inventory risk is penalized.
# In a sparse game market, 24h is a natural session length.
T_HORIZON_HOURS = 24.0

# ── Risk aversion ─────────────────────────────────────────────────────────────
# γ (gamma) — the risk aversion parameter from the AS/Bergault model.
# Higher γ → wider spreads, more aggressive skew, faster inventory mean-reversion.
# Paper reference: Avellaneda-Stoikov (2008), eq. (1); Bergault Ch.1 eq. (1.2)
# For a sparse market with high inventory risk, start conservatively high.
GAMMA = 0.1

# ── Inventory limits ─────────────────────────────────────────────────────────
# q_max per ticker: maximum inventory in shares.
# The ODE system is solved on [-Q_MAX, Q_MAX].
# Paper reference: Guéant et al. [65], inventory constraints.
# Expressed as fraction of total securities outstanding — auto-computed per ticker.
INVENTORY_LIMIT_PCT = 0.05      # Max 5% of total shares outstanding per ticker

# ── Intensity function calibration ───────────────────────────────────────────
# Λ(δ) = A * exp(-k * δ)  — Avellaneda-Stoikov exponential intensity.
# A = arrival rate at zero spread (trades per hour)
# k = price sensitivity of order arrival
# Both are estimated from historical data; these are fallback defaults.
# Paper reference: AS (2008), Bergault Ch.1 §1.2.2
DEFAULT_LAMBDA_A = 1.0          # Trades per hour at zero spread (fallback)
DEFAULT_LAMBDA_K = 1.5          # Spread sensitivity (fallback)

# ── Spread floor ─────────────────────────────────────────────────────────────
# Minimum half-spread in absolute price terms.
# Must cover the 0.5% commission per side (1% round trip).
# We add a small buffer above the commission floor.
COMMISSION_RATE = 0.005         # 0.5% per side
SPREAD_FLOOR_MULTIPLIER = 1.5   # Min half-spread = 1.5x commission rate × mid

# ── OU mean-reversion calibration ────────────────────────────────────────────
# For Chapter 6 OU dynamics: dS = κ(μ - S)dt + σ dW
# Estimated per ticker from price history.
# Paper reference: Bergault Ch.6 §6.2.1
MIN_OU_KAPPA = 0.01             # Minimum κ (below this, treat as GBM/ABM)
OU_ESTIMATION_WINDOW_DAYS = 90  # Days of history used for OU fitting

# ── Objective function selection ─────────────────────────────────────────────
# "auto"    → select based on empirical fill distribution (recommended)
# "cartea"  → risk-adjusted expectation (Cartea et al.), ξ=0
# "as"      → exponential utility (Avellaneda-Stoikov), ξ=γ
# Paper reference: Bergault intro §1.2, eq. (1) vs (2)
OBJECTIVE_FUNCTION = "auto"
# Threshold: if Poisson fit p-value < this, use AS (lumpy fills)
OBJECTIVE_AUTO_PVALUE_THRESHOLD = 0.05

# ── ODE solver ───────────────────────────────────────────────────────────────
ODE_TIMESTEPS = 1000            # Number of time steps for θ(t,q) ODE solution
ODE_SOLVER = "RK45"             # scipy solve_ivp method

# ── Multi-asset covariance ───────────────────────────────────────────────────
# Paper reference: Bergault Ch.4 §4.3 — quadratic approximation
# Σ estimated from rolling returns; minimum history required:
MIN_COV_HISTORY_DAYS = 30

# ── Order management ─────────────────────────────────────────────────────────
# Quote refresh: only refresh if theoretical quote differs from resting by > threshold
QUOTE_STALE_THRESHOLD = 0.005   # 0.5% price change triggers requote
ORDER_EXPIRY_HOURS = 2          # Limit order TTL — short to keep book clean
MAX_ORDERS_PER_MINUTE = 18      # Stay under the 20/min hard limit with buffer
POLLING_INTERVAL_SECONDS = 15   # Fallback polling interval when webhook is live
POLLING_FALLBACK_SECONDS = 5    # Polling interval when webhook is down

# ── Active inventory management (Chapter 2) ──────────────────────────────────
# When inventory exceeds this fraction of Q_MAX, go active (cross the spread).
# Paper reference: Bergault Ch.2 — passive-to-active threshold
ACTIVE_THRESHOLD_PCT = 0.80     # 80% of inventory limit triggers active unwind
ACTIVE_UNWIND_FRACTION = 0.25   # Unwind 25% of excess inventory via market order

# ── Capital allocation ────────────────────────────────────────────────────────
# Dynamic allocation: weight each ticker by (spread_opportunity × liquidity_score)
# Recomputed every REALLOC_INTERVAL_MINUTES minutes
REALLOC_INTERVAL_MINUTES = 60

# ── Data quality gates ────────────────────────────────────────────────────────
# Tickers that don't meet these thresholds are excluded from quoting entirely.
# This handles GCC/BB.A style situations with near-zero liquidity or bad data.
MIN_TRADES_FOR_QUOTING = 5              # Minimum clean trades in history
MIN_TRADES_PER_DAY = 0.01              # Minimum trades/day (0.01 = ~1 trade/3 months)
OUTLIER_SIGMA_THRESHOLD = 3.0          # Remove prices > N std devs from rolling median
OUTLIER_ROLLING_WINDOW = 5             # Window size for rolling median outlier filter
DUPLICATE_DETECTION = True             # Drop exact duplicate (ticker, ts, price, vol) rows

# ── Webhook server ────────────────────────────────────────────────────────────
WEBHOOK_HOST = "0.0.0.0"
WEBHOOK_PORT = int(os.environ.get("PORT", 8000))
WEBHOOK_SECRET = os.environ.get("NER_WEBHOOK_SECRET", "")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = "mm_bot.log"
