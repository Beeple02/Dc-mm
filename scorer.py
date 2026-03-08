# scorer.py
# Ticker viability scoring and capital allocation.
#
# Scores each NER ticker on multiple dimensions derived from calibration,
# selects the top N quotable tickers, and allocates capital proportionally.
#
# Key insight: on a thin market, ticker selection is more important than
# spread optimisation. A bad ticker (strongly trending, illiquid) will
# blow up any MM regardless of how well the quotes are computed.

import logging
from dataclasses import dataclass, field

import numpy as np

from calibration import TickerCalibration, CalibrationResult

logger = logging.getLogger(__name__)


@dataclass
class TickerScore:
    ticker: str
    raw_score: float          # composite score before normalisation
    normalised_score: float   # [0, 1]
    allocated_capital: float  # $ allocated for quoting this ticker
    q_max: int                # max inventory in shares
    components: dict = field(default_factory=dict)  # for logging/debug


@dataclass
class ScorerResult:
    scores: dict[str, TickerScore] = field(default_factory=dict)
    selected_tickers: list[str] = field(default_factory=list)


def score_ticker(cal: TickerCalibration, cfg) -> float:
    """
    Compute composite viability score for a single ticker.

    score = w1·trades_per_day_norm
          + w2·tightness_norm
          + w3·ob_freq_norm
          + w4·mean_reversion_score
          - w5·drift_score
          - w6·vol_of_vol_norm

    All positive components are normalised to [0,1] before weighting.
    Negative components are penalties (can push score below 0 → ineligible).
    """
    components = {}

    # Positive: trades per day (normalised — cap at 10 trades/day)
    trades_norm = min(cal.trades_per_day / 10.0, 1.0)
    components["trades"] = trades_norm

    # Positive: tightness (1 / spread_pct, normalised — cap at 20% spread)
    tightness = 1.0 / max(cal.avg_spread_pct, 0.001)
    tightness_norm = min(tightness / (1.0 / 0.01), 1.0)  # 1% spread = max score
    components["tightness"] = tightness_norm

    # Positive: OB update frequency (normalised — cap at 100 updates/hour)
    ob_norm = min(cal.ob_update_freq / 100.0, 1.0)
    components["ob_freq"] = ob_norm

    # Positive: mean reversion score [0,1] — already normalised
    components["mean_reversion"] = cal.mean_reversion_score

    # Negative: drift score [0,2] — normalise to [0,1]
    drift_norm = min(cal.drift_score / 2.0, 1.0)
    components["drift"] = drift_norm

    # Negative: vol-of-vol (normalise — cap at 0.1 hourly)
    vov_norm = min(cal.vol_of_vol / 0.10, 1.0)
    components["vol_of_vol"] = vov_norm

    score = (
        cfg.SCORER_W1 * trades_norm
        + cfg.SCORER_W2 * tightness_norm
        + cfg.SCORER_W3 * ob_norm
        + cfg.SCORER_W4 * cal.mean_reversion_score
        - cfg.SCORER_W5 * drift_norm
        - cfg.SCORER_W6 * vov_norm
    )

    return float(score), components


def run_scorer(
    calibration: CalibrationResult,
    cfg,
) -> ScorerResult:
    """
    Score all eligible tickers, select top N, and allocate capital.

    Capital allocation:
      - Only selected tickers receive capital
      - Allocated proportionally to normalised score
      - Hard cap: MAX_SINGLE_TICKER_ALLOC of deployable capital
      - Total deployed ≤ CAPITAL_DEPLOY_MAX · TOTAL_CAPITAL
    """
    result = ScorerResult()

    eligible = [
        t for t in calibration.eligible_tickers
        if calibration.tickers[t].eligible
    ]

    if not eligible:
        logger.warning("Scorer: no eligible tickers")
        return result

    # Score each ticker
    raw_scores = {}
    components_map = {}
    for ticker in eligible:
        cal = calibration.tickers[ticker]
        s, components = score_ticker(cal, cfg)
        raw_scores[ticker] = s
        components_map[ticker] = components

    # Filter: must exceed minimum score threshold
    viable = {t: s for t, s in raw_scores.items() if s >= cfg.SCORER_MIN_ELIGIBLE}

    if not viable:
        logger.warning(
            f"Scorer: no tickers above minimum score {cfg.SCORER_MIN_ELIGIBLE}. "
            f"Scores: {raw_scores}"
        )
        return result

    # Sort by score descending, take top N
    ranked = sorted(viable.items(), key=lambda x: -x[1])
    selected = [t for t, _ in ranked[:cfg.MAX_TICKERS_QUOTED]]

    # Normalise scores for selected tickers
    selected_scores = {t: raw_scores[t] for t in selected}
    score_sum = sum(max(s, 0) for s in selected_scores.values())

    if score_sum <= 0:
        # Fallback: equal weights
        weights = {t: 1.0 / len(selected) for t in selected}
    else:
        weights = {
            t: max(s, 0) / score_sum
            for t, s in selected_scores.items()
        }

    # Capital allocation
    deployable = cfg.TOTAL_CAPITAL * cfg.CAPITAL_DEPLOY_MAX
    max_per_ticker = deployable * cfg.MAX_SINGLE_TICKER_ALLOC

    for ticker in selected:
        cal = calibration.tickers[ticker]
        raw = raw_scores[ticker]

        # Normalised score [0,1]
        max_raw = max(raw_scores.values())
        norm = raw / max_raw if max_raw > 0 else 0.0

        allocated = min(deployable * weights[ticker], max_per_ticker)

        # q_max: max shares = allocated capital / (2 * mid price)
        # Factor of 2: we only use half for bids, half buffer for inventory
        q_max = max(1, int(allocated / (2.0 * max(cal.mid, 1.0))))

        ts = TickerScore(
            ticker=ticker,
            raw_score=raw,
            normalised_score=norm,
            allocated_capital=allocated,
            q_max=q_max,
            components=components_map[ticker],
        )
        result.scores[ticker] = ts

        logger.info(
            f"SCORER {ticker}: score={raw:.3f} (norm={norm:.3f}) "
            f"alloc=${allocated:.0f} q_max={q_max} | "
            f"mr={components_map[ticker].get('mean_reversion', 0):.2f} "
            f"drift={components_map[ticker].get('drift', 0):.2f} "
            f"trades/day={cal.trades_per_day:.1f}"
        )

    result.selected_tickers = selected

    # Log non-selected eligible tickers
    not_selected = [t for t in viable if t not in selected]
    if not_selected:
        logger.info(
            f"Scorer: eligible but not selected (capped at {cfg.MAX_TICKERS_QUOTED}): "
            f"{not_selected}"
        )

    ineligible = [t for t in raw_scores if t not in viable]
    if ineligible:
        logger.info(
            f"Scorer: below minimum score threshold: "
            f"{ {t: f'{raw_scores[t]:.3f}' for t in ineligible} }"
        )

    return result
