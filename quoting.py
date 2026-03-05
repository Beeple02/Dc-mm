# quoting.py
# Core market making engine — as close as possible to Bergault (2021).
#
# Implements:
#   - Single-asset ODE system for θ(t,q): Bergault Ch.1, eq (1.2) / (3)
#   - Optimal quotes δ^{b,*} and δ^{a,*}: Bergault Ch.1 §1.3
#   - Multi-asset quadratic approximation (Riccati ODE): Bergault Ch.4 §4.3
#   - OU-adjusted mid-price: Bergault Ch.6 §6.2
#   - Spread floor from commission structure

import logging
import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Intensity function and its inverse
# ─────────────────────────────────────────────────────────────────────────────

def lambda_func(delta: float, A: float, k: float) -> float:
    """Λ(δ) = A·exp(-k·δ)  — AS exponential intensity."""
    return A * np.exp(-k * delta)


def lambda_inv(intensity: float, A: float, k: float) -> float:
    """Λ^{-1}(x) = -ln(x/A)/k"""
    if intensity <= 0 or A <= 0:
        return np.inf
    return -np.log(max(intensity / A, 1e-12)) / k


# ─────────────────────────────────────────────────────────────────────────────
# H functions (Hamiltonian terms)
# Bergault §1.3, eq. defining H^b_ξ and H^a_ξ
# ─────────────────────────────────────────────────────────────────────────────

def H_xi(p: float, A: float, k: float, xi: float) -> float:
    """
    H^{b/a}_ξ(p) = sup_{δ} Λ(δ)/ξ · [1 - exp(-ξ(δ - p))]  for ξ > 0
    H^{b/a}_0(p) = sup_{δ} Λ(δ) · (δ - p)                   for ξ = 0

    Closed-form solutions:
      ξ=0: FOC gives δ* = p + 1/k
      ξ>0: FOC gives δ* = p + ln(1 + ξ/k)/ξ

    Paper reference: Bergault §1.3
    """
    if xi < 1e-10:
        # Cartea limit (ξ→0)
        delta_star = float(np.clip(p + 1.0 / k, -100.0, 100.0))
        lam = lambda_func(delta_star, A, k)
        return lam * (delta_star - p)
    else:
        # ξ > 0: AS exponential utility
        # δ* = p + ln(1 + ξ/k)/ξ  — clamped to prevent exp overflow
        delta_star = float(np.clip(p + np.log(1.0 + xi / k) / xi, -100.0, 100.0))
        lam = lambda_func(delta_star, A, k)
        if lam < 1e-300:
            return 0.0
        exponent = float(np.clip(-xi * (delta_star - p), -500.0, 500.0))
        return lam / xi * (1.0 - np.exp(exponent))


def optimal_delta(p: float, A: float, k: float, xi: float) -> float:
    """
    The optimal half-spread δ^* given the marginal inventory value p.

    From the FOC of H_ξ:
      ξ=0: δ* = p + 1/k
      ξ>0: δ* = p + ln(1 + ξ/k)/ξ

    This implements δ̃^b_ξ(p) and δ̃^a_ξ(p) from Bergault §1.3.
    Note: p for the bid side is θ(t,q) - θ(t,q+1) (cost of buying one more unit)
          p for the ask side is θ(t,q) - θ(t,q-1) (cost of selling one more unit)

    Paper reference: Bergault §1.3, optimal controls
    """
    if xi < 1e-10:
        return max(p + 1.0 / k, 0.0)
    else:
        return max(p + np.log(1.0 + xi / k) / xi, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Single-asset ODE system for θ(t, q)
# Bergault Ch.1, eq. (3):
#   ∂_t θ(t,q) - ½γσ²q² + H^b_ξ(θ(t,q)-θ(t,q+1)) + H^a_ξ(θ(t,q)-θ(t,q-1)) = 0
#   θ(T, q) = 0
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SingleAssetODESolution:
    """Result of solving the θ ODE for one ticker."""
    ticker: str
    q_min: int
    q_max: int
    theta_grid: np.ndarray   # shape: (n_timesteps, 2*q_max+1)
    t_grid: np.ndarray       # shape: (n_timesteps,) in [0, T]
    T: float


def solve_theta_ode(
    ticker: str,
    sigma: float,
    gamma: float,
    xi: float,
    lambda_A: float,
    lambda_k: float,
    q_max: int,
    T: float,
    n_steps: int = 1000,
) -> SingleAssetODESolution:
    """
    Solve the backward ODE system for θ(t, q) on q ∈ [-q_max, q_max].

    We solve forward in τ = T - t (time-to-go):
      ∂_τ θ(τ,q) = ½γσ²q² - H^b_ξ(θ(τ,q)-θ(τ,q+1)) - H^a_ξ(θ(τ,q)-θ(τ,q-1))
      θ(0,q) = 0   [τ=0 corresponds to t=T, terminal condition]

    At inventory limits (q = ±q_max), the corresponding H term is set to zero
    (no further buying at q_max, no further selling at -q_max).

    Paper reference: Bergault Ch.1 §1.2.2, eq (3)
    """
    q_range = list(range(-q_max, q_max + 1))
    n_q = len(q_range)
    q_to_idx = {q: i for i, q in enumerate(q_range)}

    def ode_rhs(tau: float, theta: np.ndarray) -> np.ndarray:
        dtheta = np.zeros(n_q)
        for i, q in enumerate(q_range):
            # Running inventory cost: ½γσ²q²
            running_cost = 0.5 * gamma * sigma ** 2 * q ** 2

            # Bid side H term: Δ^b θ = θ(q) - θ(q+1)
            if q + 1 <= q_max:
                delta_b_p = theta[i] - theta[q_to_idx[q + 1]]
                h_b = H_xi(delta_b_p, lambda_A, lambda_k, xi)
            else:
                h_b = 0.0  # At ceiling: no more buying

            # Ask side H term: Δ^a θ = θ(q) - θ(q-1)
            if q - 1 >= -q_max:
                delta_a_p = theta[i] - theta[q_to_idx[q - 1]]
                h_a = H_xi(delta_a_p, lambda_A, lambda_k, xi)
            else:
                h_a = 0.0  # At floor: no more selling

            dtheta[i] = running_cost - h_b - h_a
        return dtheta

    theta_0 = np.zeros(n_q)
    tau_span = (0.0, T)
    tau_eval = np.linspace(0.0, T, n_steps)

    sol = solve_ivp(
        ode_rhs,
        tau_span,
        theta_0,
        method="RK45",
        t_eval=tau_eval,
        max_step=T / 100,
        rtol=1e-4,
        atol=1e-6,
    )

    if not sol.success:
        logger.warning(f"ODE solver for {ticker}: {sol.message}")

    # sol.y shape: (n_q, n_steps)
    # Reverse so that index 0 corresponds to t=0 (τ=T), index -1 to t=T (τ=0)
    theta_grid = sol.y.T[::-1]   # shape: (n_steps, n_q)
    t_grid = T - sol.t[::-1]     # t = T - τ, reversed

    return SingleAssetODESolution(
        ticker=ticker,
        q_min=-q_max,
        q_max=q_max,
        theta_grid=theta_grid,
        t_grid=t_grid,
        T=T,
    )


def get_theta(solution: SingleAssetODESolution, t: float, q: int) -> float:
    """Interpolate θ(t, q) from the ODE solution grid."""
    q_clamped = max(solution.q_min, min(solution.q_max, q))
    idx_q = q_clamped - solution.q_min

    t_clamped = max(0.0, min(solution.T, t))
    idx_t = int(np.searchsorted(solution.t_grid, t_clamped))
    idx_t = max(0, min(len(solution.t_grid) - 1, idx_t))

    return float(solution.theta_grid[idx_t, idx_q])


# ─────────────────────────────────────────────────────────────────────────────
# Optimal single-asset quotes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OptimalQuotes:
    ticker: str
    mid: float
    bid: float
    ask: float
    delta_bid: float        # half-spread on bid side
    delta_ask: float        # half-spread on ask side
    skew: float             # net inventory skew applied
    q: int                  # current inventory
    t_remaining: float      # time remaining in session (hours)


def compute_single_asset_quotes(
    ticker: str,
    current_mid: float,
    current_inventory: int,
    t_remaining: float,
    ode_solution: SingleAssetODESolution,
    lambda_A: float,
    lambda_k: float,
    gamma: float,
    xi: float,
    sigma: float,
    spread_floor: float,
    ou_adjustment: float = 0.0,
) -> OptimalQuotes:
    """
    Compute optimal bid and ask from the θ ODE solution.

    δ^{b,*}(t) = δ̃^b_ξ(θ(t, q) - θ(t, q+1))
    δ^{a,*}(t) = δ̃^a_ξ(θ(t, q) - θ(t, q-1))

    The OU adjustment shifts the mid price (reservation price) before
    computing quotes, implementing the drift+risk correction from Ch.6.

    Paper reference: Bergault §1.3 optimal controls; Ch.6 §6.2.2
    """
    # Current time in [0, T]
    t = max(0.0, ode_solution.T - t_remaining)

    q = current_inventory

    # θ values at q, q+1, q-1
    theta_q       = get_theta(ode_solution, t, q)
    theta_q_plus  = get_theta(ode_solution, t, q + 1)
    theta_q_minus = get_theta(ode_solution, t, q - 1)

    # Marginal value differences (the p arguments to δ̃)
    p_bid = theta_q - theta_q_plus    # cost of buying one more unit
    p_ask = theta_q - theta_q_minus   # cost of selling one more unit

    # Optimal half-spreads from the closed-form formula
    delta_bid = optimal_delta(p_bid, lambda_A, lambda_k, xi)
    delta_ask = optimal_delta(p_ask, lambda_A, lambda_k, xi)

    # Enforce spread floor (must cover commission round-trip)
    delta_bid = max(delta_bid, spread_floor)
    delta_ask = max(delta_ask, spread_floor)

    # OU-adjusted reservation price (Ch.6)
    adjusted_mid = current_mid + ou_adjustment

    # Final quotes
    bid = adjusted_mid - delta_bid
    ask = adjusted_mid + delta_ask

    # Ensure ask > bid with at least 2x spread floor between them
    ask = max(ask, bid + 2.0 * spread_floor)
    bid = max(bid, 0.01)

    # Skew: how far bid is from the symmetric mid
    # Positive skew = quotes shifted down (long inventory, want to sell)
    symmetric_bid = adjusted_mid - 0.5 * (delta_bid + delta_ask)
    skew = symmetric_bid - bid  # positive when long

    return OptimalQuotes(
        ticker=ticker,
        mid=current_mid,
        bid=bid,
        ask=ask,
        delta_bid=delta_bid,
        delta_ask=delta_ask,
        skew=skew,
        q=q,
        t_remaining=t_remaining,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Multi-asset quadratic approximation (Bergault Ch.4)
# Riccati ODE for the A(t) matrix gives cross-inventory skew
# ─────────────────────────────────────────────────────────────────────────────

def compute_riccati_matrix(
    sigma_matrix: np.ndarray,
    gamma: float,
    lambda_A_vec: np.ndarray,
    lambda_k_vec: np.ndarray,
    xi_vec: np.ndarray,
    T: float,
    n_steps: int = 500,
) -> np.ndarray:
    """
    Solve the matrix Riccati ODE for the quadratic approximation A(t).

    The value function approximation θ(t,q) ≈ q^T A(t) q + b(t)^T q + c(t)
    leads to A satisfying:
        dA/dτ = ½γΣ - 2·A·H·A      (τ = T - t, solved forward)
        A(0) = 0

    where H = diag(h_i) with h_i = A_i·k_i / (ξ_i + k_i) is the diagonal
    matrix of per-asset H curvatures at p=0.

    The steady-state (τ→∞) solution is the positive semi-definite root of
    the algebraic Riccati equation: 2·A*·H·A* = ½γΣ.

    Paper reference: Bergault Ch.4 §4.3.2, Proposition 4.1
    """
    n = len(lambda_A_vec)

    # Per-asset effective Hamiltonian curvature at p=0
    h_diag = np.zeros(n)
    for i in range(n):
        A_i  = lambda_A_vec[i]
        k_i  = lambda_k_vec[i]
        xi_i = xi_vec[i]
        # From Taylor expansion of H_ξ around p=0:
        # H_ξ''(0) = A·k / (ξ + k)
        h_diag[i] = A_i * k_i / (xi_i + k_i + 1e-12)

    H_mat = np.diag(h_diag)

    def riccati_rhs(tau: float, A_flat: np.ndarray) -> np.ndarray:
        A = A_flat.reshape(n, n)
        dA = 0.5 * gamma * sigma_matrix - 2.0 * A @ H_mat @ A
        return dA.flatten()

    A_0 = np.zeros(n * n)
    tau_span = (0.0, T)
    tau_eval = np.linspace(0.0, T, n_steps)

    sol = solve_ivp(
        riccati_rhs,
        tau_span,
        A_0,
        method="RK45",
        t_eval=tau_eval,
        rtol=1e-4,
        atol=1e-8,
    )

    if not sol.success:
        logger.warning(f"Riccati ODE: {sol.message}")

    # Return A at τ=T (t=0): the most conservative, widest-spread state
    A_result = sol.y[:, -1].reshape(n, n)

    # Symmetrise to correct for numerical drift
    A_result = 0.5 * (A_result + A_result.T)

    logger.debug(f"Riccati A matrix:\n{np.round(A_result, 6)}")
    return A_result


def compute_multiasset_skew(
    inventory_vector: np.ndarray,
    riccati_A: np.ndarray,
    ticker_list: list[str],
) -> dict[str, float]:
    """
    Multi-asset inventory skew from the Chapter 4 quadratic approximation.

    The gradient of q^T A(t) q with respect to q_i gives the skew for asset i:
        skew_i = 2 · Σ_j A_{ij} · q_j   [= (A + A^T) q = 2Aq since A symmetric]

    Interpretation: your quote on asset i is shifted not just by your own
    inventory q_i (captured by the single-asset ODE) but also by correlated
    positions in other assets. If you're long correlated assets, your ask on
    asset i is shaded lower (you're effectively exposed to the same risk).

    Paper reference: Bergault Ch.4 §4.3.3, heuristic quotes from value function
    """
    skew_vec = 2.0 * riccati_A @ inventory_vector
    return {ticker: float(skew_vec[i]) for i, ticker in enumerate(ticker_list)}


# ─────────────────────────────────────────────────────────────────────────────
# OU reservation price adjustment (Bergault Ch.6)
# ─────────────────────────────────────────────────────────────────────────────

def ou_reservation_price_adjustment(
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
    Reservation price adjustment when prices are mean-reverting (OU dynamics).

    Under OU: dS = κ(μ - S)dt + σ dW
    The optimal reservation price (mid around which to quote) adjusts for:
      1. Expected drift toward μ over the remaining horizon τ = t_remaining
      2. Inventory risk under the OU process

    Adjustment = κ(μ - S)·τ  [drift pull]
               - γσ²·q·(1 - e^{-2κτ})/(2κ)  [inventory risk under OU]

    When κ → 0 (pure ABM), this simplifies to: -γσ²·q·τ
    (the standard AS reservation price adjustment).

    Paper reference: Bergault Ch.6 §6.2.2, Proposition 6.1
    """
    tau = t_remaining

    if not is_ou or kappa < 1e-4:
        # Pure ABM fallback: standard AS reservation price
        return -gamma * sigma ** 2 * inventory * tau

    # OU drift component: price expected to revert toward μ
    drift_adj = kappa * (mu - current_price) * tau

    # OU inventory risk component
    two_kappa_tau = 2.0 * kappa * tau
    if two_kappa_tau > 50:
        # Saturated: e^{-2κτ} ≈ 0
        risk_adj = -gamma * sigma ** 2 * inventory / (2.0 * kappa)
    else:
        risk_adj = (
            -gamma * sigma ** 2 * inventory
            * (1.0 - np.exp(-two_kappa_tau))
            / (2.0 * kappa)
        )

    return drift_adj + risk_adj


# ─────────────────────────────────────────────────────────────────────────────
# Quote sanity checker
# ─────────────────────────────────────────────────────────────────────────────

def validate_quotes(
    ticker: str,
    bid: float,
    ask: float,
    mid: float,
    config,
) -> tuple[bool, str]:
    """
    Hard sanity check on final bid/ask before posting.

    Returns (is_valid, reason). If invalid, the quote should be suppressed
    entirely rather than posted at a nonsense price.

    Catches:
    - OU adjustment gone haywire (bad mu fit sending bid to $0.01)
    - ODE numerical overflow producing extreme spreads
    - Cross-skew pushing quotes far outside any reasonable range
    """
    if mid <= 0:
        return False, "mid price is zero or negative"

    min_bid = mid * config.QUOTE_MIN_PCT_OF_MID
    max_ask = mid * config.QUOTE_MAX_PCT_OF_MID

    if bid < min_bid:
        return False, f"bid {bid:.4f} < floor {min_bid:.4f} (50% of mid {mid:.4f})"
    if ask > max_ask:
        return False, f"ask {ask:.4f} > ceiling {max_ask:.4f} (200% of mid {mid:.4f})"
    if bid <= 0:
        return False, f"bid {bid:.4f} is non-positive"
    if ask <= bid:
        return False, f"ask {ask:.4f} <= bid {bid:.4f}"

    return True, "ok"
