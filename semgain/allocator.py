"""Hard-budget allocation: exact multiple-choice knapsack and baselines."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Allocation:
    levels: np.ndarray        # per-position action indices
    used_data_symbols: int    # complex QPSK symbol uses
    predicted_risk: float
    feasible: bool = True


def _verify(cost: np.ndarray, budget: int, n: int):
    if cost.shape[0] != n or cost.ndim != 2 or np.any(cost <= 0):
        raise ValueError('cost must be positive [L,A]')
    if budget < sum(cost[:, 0]):
        raise ValueError(f'Insufficient data budget: need {sum(cost[:,0])}, have {budget}')


def dp_min_risk(risk: np.ndarray, cost: np.ndarray, budget: int) -> Allocation:
    """Exact multiple-choice knapsack: one action per token, used<=budget.

    Complexity O(L*A*budget). Signed risks are supported, no monotonicity
    assumptions. If multiple equal solutions exist, prefer lower cost.
    """
    risk = np.asarray(risk, dtype=float)
    cost = np.asarray(cost, dtype=int)
    if risk.shape != cost.shape:
        raise ValueError('risk and cost shapes differ')
    l, a = risk.shape
    _verify(cost, budget, l)
    dp = np.full(budget+1, np.inf)
    dp[0] = 0.0
    predecessors = np.full((l, budget+1), -1, dtype=int)
    choices = np.full((l, budget+1), -1, dtype=int)
    for i in range(l):
        nxt = np.full_like(dp, np.inf)
        for act in range(a):
            c = int(cost[i, act])
            if c > budget:
                continue
            candidate = dp[:budget+1-c] + risk[i, act]
            cur = nxt[c:]
            improve = candidate < cur - 1e-12
            positions = np.where(improve)[0] + c
            nxt[positions] = candidate[improve]
            predecessors[i, positions] = positions - c
            choices[i, positions] = act
        dp = nxt
    b = int(np.argmin(dp))
    if not np.isfinite(dp[b]):
        raise RuntimeError('No feasible allocation')
    acts = np.empty(l, dtype=int)
    for i in range(l-1, -1, -1):
        acts[i] = choices[i, b]
        b = predecessors[i, b]
    return Allocation(acts, int(sum(cost[np.arange(l), acts])), float(sum(risk[np.arange(l), acts])))


def uniform(risk: np.ndarray, cost: np.ndarray, budget: int) -> Allocation:
    """Maximum common protection level that fits for every token."""
    risk = np.asarray(risk, dtype=float)
    cost = np.asarray(cost, dtype=int)
    l, a = risk.shape
    _verify(cost, budget, l)
    level = max(j for j in range(a) if int(cost[:, j].sum()) <= budget)
    acts = np.full(l, level, dtype=int)
    return Allocation(acts, int(cost[:, level].sum()), float(risk[:, level].sum()))


def importance_only(q: np.ndarray, risk: np.ndarray, cost: np.ndarray, budget: int) -> Allocation:
    """Importance-ranked upgrades without consulting the predicted risk or CSI.

    Upgrade priority depends ONLY on q, then index for ties. Upgrades use
    the next stronger level; this is an intentionally simple UEP baseline.
    """
    q = np.asarray(q, dtype=float)
    risk = np.asarray(risk, dtype=float)
    cost = np.asarray(cost, dtype=int)
    l, a = risk.shape
    _verify(cost, budget, l)
    if q.shape != (l,):
        raise ValueError('q shape mismatch')
    acts = np.zeros(l, dtype=int)
    used = int(cost[:, 0].sum())
    order = np.argsort(-q, kind='stable')
    # Round-robin upgrade by rank until no upgrades fit.
    while True:
        changed = False
        for i in order:
            t = acts[i]
            if t < a-1:
                d = int(cost[i, t+1] - cost[i, t])
                if used + d <= budget:
                    acts[i] += 1
                    used += d
                    changed = True
        if not changed:
            break
    return Allocation(acts, used, float(risk[np.arange(l), acts].sum()))


def greedy_marginal(risk: np.ndarray, cost: np.ndarray, budget: int) -> Allocation:
    """Iterative maximum positive marginal risk reduction per extra symbol."""
    risk = np.asarray(risk, dtype=float)
    cost = np.asarray(cost, dtype=int)
    l, a = risk.shape
    _verify(cost, budget, l)
    acts = np.zeros(l, dtype=int)
    used = int(cost[:, 0].sum())
    while True:
        best = None
        for i in range(l):
            j = acts[i]
            if j+1 >= a:
                continue
            c = int(cost[i,j+1] - cost[i,j])
            gain = float(risk[i,j] - risk[i,j+1])
            if c > 0 and used+c <= budget and gain > 0:
                score = gain/c
                if best is None or score > best[0]:
                    best = (score, i, c)
        if best is None:
            break
        _, i, c = best
        used += c
        acts[i] += 1
    return Allocation(acts, used, float(risk[np.arange(l), acts].sum()))


def per_token_costs(n: int, levels: tuple[int, ...], nbits: int) -> np.ndarray:
    from .phy import symbol_cost
    return np.tile(np.asarray([symbol_cost(r, nbits) for r in levels], dtype=int), (n, 1))
