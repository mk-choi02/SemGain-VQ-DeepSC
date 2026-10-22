"""Small, explicit UEP action-profile header over coherent QPSK AWGN.

The token count, bit/level convention and packet start are assumed to be
shared beforehand. An 8-bit CRC is sent. Header bit errors cause a frame
failure in this MVP; the PHY payload is not interpreted without a valid
profile. No synchronization, channel estimation or ARQ is modeled.
"""
from __future__ import annotations
import math
import numpy as np


def crc8(bits):
    """CRC-8, generator x^8+x^2+x+1, initial register 0."""
    state = 0
    for bit in np.asarray(bits, dtype=np.uint8).reshape(-1):
        mix = ((state >> 7) & 1) ^ int(bit)
        state = (state << 1) & 255
        if mix:
            state ^= 0x07
    return np.array([(state >> s) & 1 for s in range(7, -1, -1)], dtype=np.uint8)


def profile_bits(level_ids, nlevels):
    """Fixed width profile, index 0,...,nlevels-1; does not include CRC."""
    levels = np.asarray(level_ids, dtype=np.int64)
    if levels.ndim != 1 or nlevels < 1 or np.any(levels < 0) or np.any(levels >= nlevels):
        raise ValueError('Invalid action IDs or number of levels')
    if nlevels == 1:
        return np.zeros(0, dtype=np.uint8)
    bits_per_action = math.ceil(math.log2(nlevels))
    shifts = np.arange(bits_per_action-1,-1,-1,dtype=np.int64)
    return ((levels[:,None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def header_symbol_cost(length, nlevels, repetition=5):
    if nlevels == 1:
        return 0
    if repetition <= 0 or repetition % 2 == 0 or length <= 0:
        raise ValueError('Header repetition must be odd positive; length > 0')
    # Profile + 8 CRC bits, each repeated; pad at most 1 coded bit to QPSK.
    return math.ceil((length * math.ceil(math.log2(nlevels)) + 8) * repetition / 2)


def _qpsk_hard_bits(bits, repetition, snr_db, rng):
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size % 2:
        bits = np.concatenate([bits, np.zeros(1, dtype=np.uint8)])
    mapped = ((1-2*bits[0::2].astype(float)) +
              1j*(1-2*bits[1::2].astype(float)))/math.sqrt(2)
    gamma = 10**(float(snr_db)/10.0)
    noise = (rng.standard_normal(mapped.size) + 1j*rng.standard_normal(mapped.size))/math.sqrt(2*gamma)
    received = mapped+noise
    detected = np.empty(bits.size,dtype=np.uint8)
    detected[0::2] = received.real < 0
    detected[1::2] = received.imag < 0
    return detected


def transmit_profile(action_ids, nlevels, repetition, snr_db, rng):
    """Returns (known_correct, crc_passed, decoded_actions_or_None).

    known_correct is an *evaluation flag* comparing against the sent profile;
    receiver only has the CRC. CRC collisions are recorded as undetected.
    The simulator fails closed for any wrong decoded header to avoid
    inadvertently using oracle action IDs during its payload decoding.
    """
    if nlevels == 1:
        return True,True,np.asarray(action_ids,dtype=int).copy()
    original = profile_bits(action_ids, nlevels)
    packet = np.concatenate([original, crc8(original)])
    coded = np.repeat(packet,repetition)
    hard = _qpsk_hard_bits(coded,repetition,snr_db,rng)
    decoded = (hard[:coded.size].reshape(-1,repetition).sum(1) > repetition//2).astype(np.uint8)
    header = decoded[:-8]
    checked = bool(np.array_equal(crc8(header),decoded[-8:]))
    decoded_actions = None
    if checked:
        width=math.ceil(math.log2(nlevels))
        powers = (1 << np.arange(width-1,-1,-1))
        decoded_actions = (header.reshape(-1,width)*powers).sum(-1).astype(int)
        if np.any(decoded_actions>=nlevels):
            checked=False
            decoded_actions=None
    known_correct = bool(np.array_equal(decoded,packet))
    if not known_correct:
        decoded_actions = None  # conservative framing failure in this study
    return known_correct, checked, decoded_actions
