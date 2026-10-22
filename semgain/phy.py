"""Coherent Gray QPSK over AWGN + odd repetition: analyzable PHY MVP.

Index bits are independently repeated `rep` times. Hard QPSK detection is
followed by majority decoding. This is NOT LDPC and not a TDL-OFDM simulator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Protection:
    repeat: int
    @property
    def name(self) -> str:
        return f"rep{self.repeat}"


def validate_levels(levels: tuple[int, ...], nbits: int) -> None:
    if nbits <= 0 or nbits % 2:
        raise ValueError('A QPSK index needs an even number of bits.')
    if not levels or any(r < 1 or r % 2 == 0 for r in levels):
        raise ValueError('All repetition factors must be positive odd integers.')
    if len(set(levels)) != len(levels) or list(levels) != sorted(levels):
        raise ValueError('Repetition levels must be distinct ascending values.')


def index_bits(index: np.ndarray | int, nbits: int) -> np.ndarray:
    index = np.asarray(index, dtype=np.int64)
    return ((index[..., None] >> np.arange(nbits - 1, -1, -1)) & 1).astype(np.int8)


def bits_index(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.int64)
    nbits = bits.shape[-1]
    return np.sum(bits * (1 << np.arange(nbits - 1, -1, -1)), axis=-1).astype(np.int64)


def qpsk_ber(snr_db: float | np.ndarray) -> np.ndarray:
    """Uncoded Gray QPSK bit BER for Es/N0=snr_db, unit-energy symbols."""
    gamma = np.power(10.0, np.asarray(snr_db, dtype=float) / 10.0)
    # math.erfc via numpy.vectorize, no SciPy dependency
    return np.vectorize(lambda g: 0.5 * math.erfc(math.sqrt(float(g) / 2.0)), otypes=[float])(gamma)


def repeated_ber(snr_db: float | np.ndarray, repeat: int) -> np.ndarray:
    if repeat < 1 or repeat % 2 != 1:
        raise ValueError('Only odd positive repetition factors supported')
    p = qpsk_ber(snr_db)
    return sum(math.comb(repeat, j) * p**j * (1.0 - p)**(repeat-j)
               for j in range((repeat + 1) // 2, repeat + 1))


def confusion_matrix(nbits: int, p: float) -> np.ndarray:
    """Exact KxK index confusion under independently flipped bits."""
    k = 1 << nbits
    ids = np.arange(k, dtype=np.uint16)
    xor = np.bitwise_xor(ids[:, None], ids[None, :])
    # bit_count via byte lookup supported by old and new numpy
    count = np.array([int(x).bit_count() for x in range(k)], dtype=np.int8)
    d = count[xor]
    return p**d * (1.0-p)**(nbits-d)


def symbol_cost(repeat: int, nbits: int) -> int:
    """Number of complex QPSK symbol uses per index."""
    return nbits * repeat // 2


def header_cost(n_active: int, n_levels: int, header_repeat: int = 5) -> int:
    """UEP profile + CRC-8 header, repeated QPSK symbol cost.

    Does not charge PHY frame acquisition/synchronization or shared length.
    """
    from .header import header_symbol_cost
    return header_symbol_cost(n_active, n_levels, header_repeat)


def transmit_indices(indices: np.ndarray, repeats: np.ndarray, snr_db: float | np.ndarray,
                     nbits: int, rng: np.random.Generator) -> np.ndarray:
    """Actual unit-energy coherent QPSK noise and repetition hard decisions.

    All indices and repeats arrays are 1-D; each token can have distinct Es/N0.
    This is a per-token AWGN link, perfect carrier phase/CSI, no OFDM.
    """
    indices = np.asarray(indices, dtype=np.int64)
    repeats = np.asarray(repeats, dtype=np.int64)
    snr = np.broadcast_to(np.asarray(snr_db, dtype=float), indices.shape)
    if indices.ndim != 1 or repeats.shape != indices.shape:
        raise ValueError('Expected 1-D indices and matching repetition levels')
    if np.any((indices < 0) | (indices >= (1 << nbits))):
        raise ValueError('Index outside codebook')
    if not np.all((repeats > 0) & (repeats % 2 == 1)):
        raise ValueError('Repetition must be odd positive')
    out = np.empty_like(indices)
    for i, (k, rep) in enumerate(zip(indices, repeats)):
        b = index_bits(int(k), nbits)
        # b[0::2] and b[1::2] label I and Q on the unit QPSK grid.
        real = (1.0 - 2.0*b[0::2]) / math.sqrt(2.0)
        imag = (1.0 - 2.0*b[1::2]) / math.sqrt(2.0)
        x = real + 1j*imag
        gamma = 10.0**(snr[i]/10.0)
        z = (rng.standard_normal((int(rep), len(x))) +
             1j*rng.standard_normal((int(rep), len(x)))) / math.sqrt(2.0*gamma)
        y = x[None, :] + z
        hard = np.empty((int(rep), nbits), dtype=np.int8)
        hard[:, 0::2] = (y.real < 0)
        hard[:, 1::2] = (y.imag < 0)
        out[i] = bits_index((hard.sum(axis=0) > rep // 2).astype(np.int8))
    return out
