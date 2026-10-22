"""Single-index semantic distortion and confusion-weighted risk."""
from __future__ import annotations
import numpy as np
from .phy import confusion_matrix, repeated_ber


def risk_from_delta(indices: np.ndarray, deltas: np.ndarray,
                    snr_db: float | np.ndarray, levels: tuple[int, ...],
                    approximate_one_bit: bool = False) -> np.ndarray:
    """[L,A] expected *signed* loss change under the independent-bit model.

    Complexity O(L*A*K) instead of building a K-by-K matrix in each loop.
    All missing delta entries in one-bit mode are treated as zero; this
    deliberately produces a truncation approximation, not an exact risk.
    """
    indices = np.asarray(indices, dtype=int)
    deltas = np.asarray(deltas, dtype=float)
    if deltas.ndim != 2 or not np.all(np.isfinite(deltas)):
        raise ValueError('Deltas must be a finite [L,K] array')
    k = deltas.shape[1]
    nbits = (k-1).bit_length()
    if (1 << nbits) != k or indices.shape != (deltas.shape[0],) or np.any(indices < 0) or np.any(indices >= k):
        raise ValueError('Invalid index/delta shapes or out-of-range indices')
    snrs = np.broadcast_to(np.asarray(snr_db,dtype=float),indices.shape)
    hamming = np.array([int(x).bit_count() for x in range(k)],dtype=np.int8)
    alternatives = np.arange(k)
    out = np.zeros((len(indices),len(levels)),dtype=float)
    for i, idx in enumerate(indices):
        ds = hamming[np.bitwise_xor(idx, alternatives)]
        for a, rep in enumerate(levels):
            p = float(repeated_ber(float(snrs[i]),rep))
            probs = p**ds*(1.0-p)**(nbits-ds)
            out[i,a] = probs @ deltas[i]
    return out


def evaluate_additive_risk(indices: np.ndarray, recovered: np.ndarray,
                           delta: np.ndarray) -> float:
    return float(sum(delta[i, int(x)] for i, x in enumerate(recovered)))
