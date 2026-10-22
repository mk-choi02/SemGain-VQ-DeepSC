"""Deployable semantic-risk surrogate based on predicted q and codebook geometry.

Fit parameters on a DISJOINT calibration split; never compute test-sentence
finite differences at inference. This low-capacity proxy is a starting point,
not a semantic distortion guarantee.
"""
from __future__ import annotations
import numpy as np


def geometry_features(indices: np.ndarray, codewords: np.ndarray) -> np.ndarray:
    """[L,K,3]: const, squared Euclidean distance, bit Hamming distance."""
    indices = np.asarray(indices, dtype=int)
    c = np.asarray(codewords, dtype=float)
    k = len(c)
    dif = c[indices,None,:] - c[None,:,:]
    dist2 = np.square(dif).sum(-1)
    xor = np.bitwise_xor(indices[:,None], np.arange(k)[None,:])
    hamm = np.vectorize(lambda v: int(v).bit_count(), otypes=[float])(xor)
    return np.stack([np.ones_like(dist2),dist2,hamm],axis=-1)


def fit_linear_risk_proxy(features: np.ndarray, q: np.ndarray, target: np.ndarray,
                          ridge: float=1e-3) -> np.ndarray:
    """Fit nonnegative least squares-like ridge via unconstrained solve and clipping.

    features [N,3] and q [N] are for measured corruption events. No intercept
    added separately: feature[0]=1. Target is signed CE change, but deployment
    surrogate approximates positive harm only, so risk never uses negative values.
    """
    x=np.asarray(features,dtype=float)*np.asarray(q,dtype=float).reshape(-1,1)
    y=np.maximum(np.asarray(target,dtype=float),0)
    return np.maximum(np.linalg.solve(x.T@x+ridge*np.eye(x.shape[1]),x.T@y),0)


def predict_delta(indices: np.ndarray, codewords: np.ndarray, q: np.ndarray,
                  weights: np.ndarray) -> np.ndarray:
    feature=geometry_features(indices,codewords)
    q=np.asarray(q,dtype=float)
    pred=np.maximum(feature@np.asarray(weights,dtype=float),0)*q[:,None]
    pred[np.arange(len(indices)),indices]=0.0
    return pred
