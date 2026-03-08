# quoting.py
# Optimal quote computation based on:
#
#   Primary model:
#     Guéant, Lehalle & Fernandez-Tapia (2013) — "Dealing with the Inventory Risk"
#     Closed-form approximation of the Avellaneda-Stoikov (2008) model.
#
#   Extensions:
#     - Dynamic risk aversion γ(t) from realised vol (Cartea & Jaimungal 2015 §6)
#     - Spread stress multipliers: inventory penalty + drift penalty
#     - OU reservation price shift when ticker is mean-reverting
#
# GLF (2013) closed-form:
#   Reservation price:  r = s - q · γ · σ² · τ
#   Optimal half-spread: δ* = γ·σ²·τ/2 + (1/γ) · ln(1 + γ/k)
#   bid = r - δ*,  ask = r + δ*
#
# Spread stress:
#   spread_final = spread_base · f_inventory(q) · f_drift
#   f_inventory  = exp(β · |q / q_max|)
#   f_drift      = 1 + λ · drift_score

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimalQuotes:
    bid: float
    ask: float
    reservation_price: float
    half_spread: float       # base half-spread before stress
    spread_stressed: float   # final spread after stress multipliers
    gamma_effective: float   # dynamic γ used
    skew: float              # r - s (inventory-induced mid shift)
    f_inventory: float       # inventory stress multiplier
    f_drift: float           # drift stress multiplier


def dynamic_gamma(
    gamma_base: float,
    sigma_current: float,
    sigma_long_run: float,
    alpha: float = 0.5,
) -> float:
    """
    Dynamic risk aversion from Cartea & Jaimungal (2015) §6.
    γ(t) = γ_base · (σ_current / σ_long_run)^α

    When vol is above long-run average → γ increases → spreads widen.
    When vol is below long-run average → γ decreases → spreads tighten.
    Bounded to [γ_base/4, γ_base*8] to prevent degenerate quotes.
    """
    if sigma_long_run <= 0 or sigma_current <= 0:
        return gamma_base

    ratio = sigma_current / sigma_long_run
    gamma = gamma_base * (ratio ** alpha)
    return float(np.clip(gamma, gamma_base / 4.0, gamma_base * 8.0))


def ou_reservation_adjustment(
    current_price: float,
    mu: float,
    kappa: float,
    sigma: float,
    gamma: float,
    inventory: int,
    t_remaining: float,
    is_ou: bool,
) -> float:
    """
    OU-based reservation price adjustment.

    When the ticker is mean-reverting (is_ou=True), the optimal reservation
    price incorporates the expected reversion towards μ over the horizon τ.

    From Bergault & Guéant (2021) §6: the OU adjustment shifts the mid by
    an amount proportional to (μ - s) · (1 - exp(-κτ)) / (γ·σ²·τ).

    Clamped to ±10% of current price to prevent bad μ fits from
    sending quotes to extreme values.
    """
    if not is_ou or kappa <= 0 or t_remaining <= 0 or gamma <= 0 or sigma <= 0:
        return 0.0

    reversion_factor = (1.0 - np.exp(-kappa * t_remaining))
    adj = (mu - current_price) * reversion_factor

    # Clamp to ±10% of current price
    max_adj = current_price * 0.10
    return float(np.clip(adj, -max_adj, max_adj))


def compute_quotes(
    mid: float,
    inventory: int,
    t_remaining: float,
    sigma: float,
    sigma_long_run: float,
    lambda_A: float,
    k: float,
    q_max: int,
    drift_score: float,
    mu: float,
    kappa: float,
    is_ou: bool,
    cfg,
) -> OptimalQuotes:
    """
    Full quoting pipeline for one ticker.

    Steps:
      1. Compute dynamic γ from vol ratio
      2. Compute OU reservation price adjustment (if mean-reverting)
      3. GLF (2013) closed-form: reservation price r, half-spread δ*
      4. Apply inventory stress: f_inventory = exp(β · |q/q_max|)
      5. Apply drift stress: f_drift = 1 + λ · drift_score
      6. Apply spread floor (must clear 2x commission)
      7. Quote sanity bounds (bid ≥ 70% mid, ask ≤ 130% mid)

    Returns OptimalQuotes with bid suppressed (bid=-1) or ask suppressed
    (ask=inf) if sanity bounds are violated — order_manager interprets these.
    """
    if mid <= 0 or t_remaining <= 0:
        return OptimalQuotes(
            bid=-1, ask=float('inf'),
            reservation_price=mid, half_spread=0,
            spread_stressed=0, gamma_effective=cfg.GAMMA_BASE,
            skew=0, f_inventory=1, f_drift=1,
        )

    # 1. Dynamic γ
    gamma = dynamic_gamma(cfg.GAMMA_BASE, sigma, sigma_long_run, cfg.GAMMA_VOL_ALPHA)

    # 2. OU adjustment
    ou_adj = ou_reservation_adjustment(
        mid, mu, kappa, sigma, gamma, inventory, t_remaining, is_ou
    )

    # 3. GLF (2013) reservation price and half-spread
    #    r = s + ou_adj - q · γ · σ² · τ
    #    δ* = γ·σ²·τ/2 + (1/γ) · ln(1 + γ/k)
    variance_term = gamma * (sigma ** 2) * t_remaining
    reservation_price = mid + ou_adj - inventory * variance_term
    skew = reservation_price - mid

    # GLF half-spread
    if gamma > 0 and k > 0:
        half_spread_base = (
            variance_term / 2.0
            + (1.0 / gamma) * np.log(1.0 + gamma / k)
        )
    else:
        half_spread_base = mid * cfg.COMMISSION_RATE * cfg.SPREAD_FLOOR_MULT

    # 4. Inventory stress multiplier
    # f = exp(β · |q/q_max|): rapidly widens as inventory fills
    inv_fraction = abs(inventory) / max(q_max, 1)
    f_inventory = float(np.exp(cfg.INVENTORY_SPREAD_BETA * min(inv_fraction, 1.0)))

    # 5. Drift stress multiplier
    # f = 1 + λ · drift_score: widens when price is trending
    f_drift = 1.0 + cfg.DRIFT_SPREAD_LAMBDA * drift_score

    # 6. Spread floor: must clear commission on both legs
    spread_floor = mid * cfg.COMMISSION_RATE * cfg.SPREAD_FLOOR_MULT

    # Final half-spread
    half_spread_stressed = max(
        half_spread_base * f_inventory * f_drift,
        spread_floor / 2.0,
    )

    spread_stressed = half_spread_stressed * 2.0

    bid = reservation_price - half_spread_stressed
    ask = reservation_price + half_spread_stressed

    # 7. Quote sanity bounds
    bid_min = mid * cfg.QUOTE_MIN_PCT_OF_MID
    ask_max = mid * cfg.QUOTE_MAX_PCT_OF_MID

    bid_suppressed = bid < bid_min or bid <= 0
    ask_suppressed = ask > ask_max

    if bid_suppressed:
        logger.warning(
            f"Bid suppressed: bid={bid:.4f} < min={bid_min:.4f} "
            f"(mid={mid:.4f} inv={inventory} q_max={q_max})"
        )
        bid = -1.0

    if ask_suppressed:
        logger.warning(
            f"Ask suppressed: ask={ask:.4f} > max={ask_max:.4f} "
            f"(mid={mid:.4f} inv={inventory} q_max={q_max})"
        )
        ask = float('inf')

    return OptimalQuotes(
        bid=round(bid, 4) if bid > 0 else bid,
        ask=round(ask, 4) if ask != float('inf') else ask,
        reservation_price=round(reservation_price, 4),
        half_spread=round(half_spread_base, 4),
        spread_stressed=round(spread_stressed, 4),
        gamma_effective=round(gamma, 6),
        skew=round(skew, 4),
        f_inventory=round(f_inventory, 3),
        f_drift=round(f_drift, 3),
    )


def compute_quote_quantity(
    allocated_capital: float,
    mid: float,
    q_max: int,
    cfg,
) -> int:
    """
    Compute order quantity for each side.

    From allocated capital: qty = allocated / (2 * mid)
    (Factor 2: capital split between bid and ask sides.)
    Hard cap at MAX_QUOTE_QTY and q_max.
    """
    if mid <= 0 or allocated_capital <= 0:
        return 1

    qty = int(allocated_capital / (2.0 * mid))
    qty = max(1, qty)
    qty = min(qty, cfg.MAX_QUOTE_QTY)
    qty = min(qty, q_max)
    return qty
