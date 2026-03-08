# config.py
import os

# ── Atlas (market data) ───────────────────────────────────────────────────────
ATLAS_BASE_URL = os.environ.get("ATLAS_BASE_URL", "")
ATLAS_API_KEY  = os.environ.get("ATLAS_API_KEY", "")

# ── NER (trading) ─────────────────────────────────────────────────────────────
NER_BASE_URL = os.environ.get("NER_BASE_URL", "")
NER_API_KEY  = os.environ.get("NER_API_KEY", "")

# ── Capital management ────────────────────────────────────────────────────────
TOTAL_CAPITAL       = float(os.environ.get("TOTAL_CAPITAL", 10_000))
CAPITAL_DEPLOY_MAX  = 0.60   # never deploy more than 60% simultaneously
CAPITAL_RESERVE     = 0.40   # always keep 40% liquid

MAX_TICKERS_QUOTED  = 3      # quote at most 3 tickers at once

# ── Risk limits ───────────────────────────────────────────────────────────────
STOP_LOSS_PER_TICKER  = 400.0   # unrealized loss per ticker → stop + unwind
STOP_LOSS_PORTFOLIO   = 2000.0  # total unrealized loss → full stop

# ── Unwind parameters ─────────────────────────────────────────────────────────
UNWIND_DISCOUNT        = 0.01   # post limit 1% inside best bid/ask
UNWIND_RETRY_INTERVAL  = 300    # seconds between unwind retries
UNWIND_CROSS_AFTER     = 3      # attempts before crossing spread aggressively
UNWIND_ORDER_EXPIRY    = 0.1    # hours (6 min) per unwind limit order

# ── Quoting model (Guéant-Lehalle-Fernandez-Tapia 2013) ──────────────────────
GAMMA_BASE      = 0.1    # base risk aversion — scales dynamically with vol
GAMMA_VOL_ALPHA = 0.5    # exponent for vol scaling: γ = γ_base · (σ/σ_lr)^α
T_HORIZON_HOURS = 24.0   # rolling session horizon

# Spread stress multipliers
# f_inventory = exp(INVENTORY_SPREAD_BETA · |q/q_max|)
# f_drift     = 1 + DRIFT_SPREAD_LAMBDA · |drift_score|
INVENTORY_SPREAD_BETA  = 1.5
DRIFT_SPREAD_LAMBDA    = 2.0

# Minimum spread floor: always at least 2x commission
COMMISSION_RATE        = 0.005
SPREAD_FLOOR_MULT      = 2.0

# Hard quote sanity bounds relative to mid
QUOTE_MIN_PCT_OF_MID   = 0.70   # bid >= 70% of mid
QUOTE_MAX_PCT_OF_MID   = 1.30   # ask <= 130% of mid

# Max single-side order quantity
MAX_QUOTE_QTY          = 5

# ── Order arrival rate (k) estimation ─────────────────────────────────────────
# k is fitted from the transactions tape (Poisson intensity decay param).
# Fallback used when tape is too sparse to fit.
DEFAULT_K              = 1.5
DEFAULT_LAMBDA_A       = 0.02   # arrivals/hour fallback

# ── Ticker scorer weights ──────────────────────────────────────────────────────
# score = w1·trades_per_day + w2·(1/spread_pct) + w3·ob_update_freq
#       + w4·mean_reversion_score - w5·drift_score - w6·vol_of_vol
SCORER_W1 = 1.5   # trades per day
SCORER_W2 = 1.0   # tightness (1/spread%)
SCORER_W3 = 0.5   # OB update frequency
SCORER_W4 = 2.0   # mean reversion (most important — safety)
SCORER_W5 = 2.5   # drift penalty (most dangerous for MM)
SCORER_W6 = 1.0   # vol-of-vol penalty

# Minimum score to be eligible for quoting at all
SCORER_MIN_ELIGIBLE    = 0.20
# Minimum trades per day to even consider a ticker
SCORER_MIN_TRADES_DAY  = 0.3    # roughly 2+ trades per week

# ── Capital allocation across scored tickers ──────────────────────────────────
# Allocated notional = TOTAL_CAPITAL · CAPITAL_DEPLOY_MAX · liquidity_weight
# liquidity_weight is normalised from scorer output
# Hard cap: no single ticker gets more than this fraction of deployable capital
MAX_SINGLE_TICKER_ALLOC = 0.50  # 50% of deployable = $3,000 max

# ── Calibration ───────────────────────────────────────────────────────────────
CALIBRATION_HISTORY_DAYS  = 30   # days of OB history to use
VOL_ESTIMATION_WINDOW     = 100  # OB snapshots for short-run vol
DRIFT_ESTIMATION_WINDOW   = 20   # snapshots for drift score
MEAN_REV_HALFLIFE_MAX     = 48   # hours — if OU half-life > this, not mean-reverting

# ── Recalibration schedule ────────────────────────────────────────────────────
RECALIBRATE_INTERVAL_MIN  = 120  # full recalibration every 2 hours
RESCORE_INTERVAL_MIN      = 60   # re-score tickers every 1 hour

# ── Order management ──────────────────────────────────────────────────────────
ORDER_EXPIRY_HOURS         = 12
MAX_ORDERS_PER_MINUTE      = 6
MIN_REQUOTE_INTERVAL_MIN   = 10
QUOTE_STALE_THRESHOLD      = 0.02   # 2% mid move triggers requote

# ── Polling ───────────────────────────────────────────────────────────────────
POLLING_INTERVAL_SECONDS   = 60
T_RESET_HOURS              = 24.0

# ── Tickers to always ignore ──────────────────────────────────────────────────
EXCLUDED_TICKERS = {"RNHC", "RNC-B", "VSP3"}

# ── Webhook ───────────────────────────────────────────────────────────────────
WEBHOOK_HOST    = "0.0.0.0"
WEBHOOK_PORT    = int(os.environ.get("PORT", 8000))
WEBHOOK_SECRET  = os.environ.get("NER_WEBHOOK_SECRET", "")
WEBHOOK_PUBLIC_URL = os.environ.get("WEBHOOK_PUBLIC_URL", "")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE  = "mm_bot.log"
