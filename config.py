# config.py
import os

# ── API ───────────────────────────────────────────────────────────────────────
API_BASE_URL = "http://150.230.117.88:8082"
API_KEY = os.environ.get("NER_API_KEY", "")

# ── Capital management ────────────────────────────────────────────────────────
TOTAL_MM_CAPITAL = 10_000.0
CAPITAL_RESERVE_BUFFER = 0.05
REBALANCE_THRESHOLD = 0.20

# ── Time horizon ──────────────────────────────────────────────────────────────
T_HORIZON_HOURS = 24.0

# ── Risk aversion ─────────────────────────────────────────────────────────────
GAMMA = 0.1

# ── Inventory limits ──────────────────────────────────────────────────────────
INVENTORY_LIMIT_PCT = 0.02      # 2% of total shares outstanding
INVENTORY_LIMIT_ABS = 50        # Hard absolute cap: never hold >50 shares

# ── Intensity function (fallback) ─────────────────────────────────────────────
DEFAULT_LAMBDA_A = 0.05
DEFAULT_LAMBDA_K = 0.5

# ── Spread floor ──────────────────────────────────────────────────────────────
COMMISSION_RATE = 0.005
SPREAD_FLOOR_MULTIPLIER = 2.0

# ── OU mean-reversion ─────────────────────────────────────────────────────────
MIN_OU_KAPPA = 0.01
OU_ESTIMATION_WINDOW_DAYS = 90

# OU adjustment clamp: never shift mid by more than this fraction.
# Prevents bad mu fits (e.g. BB calibrating mu=$5 when price=$40) from
# sending quotes to zero or negative.
OU_ADJ_MAX_PCT = 0.10

# ── Quote sanity bounds ───────────────────────────────────────────────────────
# Hard bounds on final bid/ask relative to current mid price.
# If computed quote is outside these, it is suppressed entirely.
QUOTE_MIN_PCT_OF_MID = 0.50     # Bid must be >= 50% of mid
QUOTE_MAX_PCT_OF_MID = 2.00     # Ask must be <= 200% of mid

# ── Intensity function ───────────────────────────────────────────────────────
# NER markets are much more active in the evening (server peak hours).
# Daytime calibration underestimates lambda_A, producing spreads too wide to fill.
# This multiplier scales up the fitted arrival rate to reflect peak-hour activity.
# 3x is conservative — adjust based on observed evening fill rates.
EVENING_BOOST = 3.0

# ── Objective function ────────────────────────────────────────────────────────
OBJECTIVE_FUNCTION = "auto"
OBJECTIVE_AUTO_PVALUE_THRESHOLD = 0.05

# ── ODE solver ────────────────────────────────────────────────────────────────
ODE_TIMESTEPS = 500
ODE_SOLVER = "RK45"

# ── Multi-asset covariance ────────────────────────────────────────────────────
MIN_COV_HISTORY_DAYS = 30

# ── Order management — SPAM FIX ───────────────────────────────────────────────
# NER trades ~0.08x/hour on the most liquid ticker.
# Old setup (15s poll + 0.5% stale threshold) was generating ~75% of all
# weekly NER volume by itself. New rules:
#
#   1. Minimum 10 minutes between ANY requote for a ticker (hard floor)
#   2. Only requote if mid has moved >2% since the last posted quote
#   3. t_remaining drift alone does NOT trigger a requote
#   4. Polling only syncs state — does not directly trigger requoting
QUOTE_STALE_THRESHOLD = 0.02        # 2% mid move triggers requote
MIN_REQUOTE_INTERVAL_MINUTES = 10   # Never requote faster than 10 min per ticker

ORDER_EXPIRY_HOURS = 12
MAX_ORDERS_PER_MINUTE = 6           # Well under 20/min hard limit

POLLING_INTERVAL_SECONDS = 120      # 2 min (was 15s)
POLLING_FALLBACK_SECONDS = 60

# ── Active inventory management ───────────────────────────────────────────────
ACTIVE_THRESHOLD_PCT = 0.80
ACTIVE_UNWIND_FRACTION = 0.25

# FIX 1 — Max fill size per quote order.
# Caps qty posted in any single bid/ask to prevent a single counterparty
# dumping a huge block into us (e.g. 98 RTG @ $13.10 in one fill).
# Capital-derived qty is further capped at this hard limit.
MAX_QUOTE_QTY = 5

# FIX 2 — Price vs mu sanity gate.
# If current market price is more than this multiple above calibrated mu,
# the model believes the stock is significantly overvalued. In that case,
# posting a bid risks being adversely selected by informed sellers.
# We suppress the bid side when price > mu * PRICE_VS_MU_MAX_RATIO.
# e.g. ratio=2.0: RTG trading at $13 with mu=$4.54 → 13/4.54=2.86 > 2.0 → no bid.
PRICE_VS_MU_MAX_RATIO = 2.0  # suppress bid if price > 2x calibrated mu
PRICE_VS_MU_MIN_RATIO = 0.5  # suppress ask if price < 0.5x calibrated mu

# FIX 3 — Adverse selection circuit breaker.
# If inventory for a ticker jumps by more than this amount in a single
# polling cycle, it means someone just hit us for a large block — a classic
# sign of informed trading. Pause quoting that ticker for this many minutes.
ADVERSE_SELECTION_JUMP = 10       # shares: inventory jump that triggers pause
ADVERSE_SELECTION_PAUSE_MINUTES = 30  # how long to pause quoting after detection

# FIX 4 — Active unwind via aggressive limit orders instead of market orders.
# NER's /orders/sell_market endpoint returns 400 unless there's an active
# counterparty. Instead, post a limit order at a small discount to best bid
# to guarantee a fill without requiring a market order endpoint.
# UNWIND_LIMIT_DISCOUNT: how far below best bid to place the unwind limit.
# 0.0 = at best bid (join queue), 0.02 = 2% below best bid (more aggressive).
UNWIND_LIMIT_DISCOUNT = 0.01     # 1% below best bid for fast fills
UNWIND_LIMIT_EXPIRY_HOURS = 0.25  # 15 min expiry — if unfilled, retry next cycle

# ── Capital allocation ────────────────────────────────────────────────────────
REALLOC_INTERVAL_MINUTES = 120

# ── Data quality gates ────────────────────────────────────────────────────────
MIN_TRADES_FOR_QUOTING = 5
MIN_TRADES_PER_DAY = 0.01
OUTLIER_SIGMA_THRESHOLD = 3.0
OUTLIER_ROLLING_WINDOW = 5
DUPLICATE_DETECTION = True

# ── Webhook ───────────────────────────────────────────────────────────────────
WEBHOOK_HOST = "0.0.0.0"
WEBHOOK_PORT = int(os.environ.get("PORT", 8000))
WEBHOOK_SECRET = os.environ.get("NER_WEBHOOK_SECRET", "")

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = "mm_bot.log"
