import itertools
import numpy as np
import pytest
from semgain.allocator import dp_min_risk,uniform,importance_only,greedy_marginal,per_token_costs
from scripts.run_semgain import choose_allocations


def test_dp_agrees_brute_force():
    r=np.array([[2.0,1.0,0.1],[1.0,0.4,0.3],[3.0,2.0,0.2]])
    c=per_token_costs(3,(1,3,5),8)
    budget=40
    sol=dp_min_risk(r,c,budget)
    brute=min(sum(r[i,a] for i,a in enumerate(choice)) for choice in itertools.product(range(3),repeat=3)
              if sum(c[i,a] for i,a in enumerate(choice))<=budget)
    assert abs(sol.predicted_risk-brute)<1e-12
    assert sol.used_data_symbols<=budget


def test_policy_budget_and_headers():
    r=np.array([[2.0,1.0,0.1]]*10)
    q=np.linspace(1,2,10);q=q/q.sum()
    pol=choose_allocations(q,r,140,(1,3,5),8,3)
    assert pol['uniform'][1]==0
    assert pol['marginal_dp'][1]==42
    for name,(alloc,h) in pol.items():
        assert alloc.used_data_symbols+h<=140,name


def test_infeasible_raises():
    with pytest.raises(ValueError):
        dp_min_risk(np.ones((3,3)),per_token_costs(3,(1,3,5),8),11)


def test_signed_risk():
    risk=np.array([[-2,-5,-3],[4,2,1]],dtype=float)
    cost=np.array([[2,4,6],[2,4,6]])
    sol=dp_min_risk(risk,cost,8)
    assert sol.predicted_risk<=-1
