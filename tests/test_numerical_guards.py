import numpy as np
import pytest
import torch
from semgain.vq import VectorCodebook, fit_kmeans
from semgain.importance import PositionContextImportance
from semgain.risk_model import fit_linear_risk_proxy
from semgain.allocator import dp_min_risk


def test_vq_rejects_nonfinite_vectors():
    with pytest.raises(ValueError):
        fit_kmeans(torch.tensor([[float('nan'), 0.], [1., 1.], [0., 0.], [2., 2.]]), k=4)
    with pytest.raises(ValueError):
        VectorCodebook(torch.full((4, 2), float('inf')))


def test_importance_masks_and_normalization():
    model = PositionContextImportance(d_model=4, max_len=6)
    features = torch.randn(2, 5, 4)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    q = model(features, mask)
    assert torch.allclose(q.sum(-1), torch.ones(2))
    assert torch.all(q[~mask] == 0)
    with pytest.raises(ValueError):
        model(features, torch.zeros_like(mask))
    with pytest.raises(ValueError):
        model(features, mask, temperature=0)


def test_risk_proxy_nonnegative_and_bad_shapes():
    x = np.array([[1., 0., 1.], [1., 1., 2.], [1., 2., 1.]])
    q = np.ones(3)
    y = np.array([0.2, 0.3, 0.5])
    beta = fit_linear_risk_proxy(x, q, y)
    assert np.isfinite(beta).all() and (beta >= 0).all()
    with pytest.raises(ValueError):
        fit_linear_risk_proxy(x, q[:2], y)


def test_dp_rejects_nan_risk():
    with pytest.raises(ValueError):
        dp_min_risk(np.array([[float('nan'), 0.]]), np.array([[1, 2]]), 2)
