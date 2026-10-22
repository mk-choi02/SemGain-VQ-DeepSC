# Validation report — SemGain VQ DeepSC v0.2

## Completed (container, Python 3.13, NumPy/PyTorch)

- `python -m pytest -q`: **21 tests passed** on baseline+v0.2 additions.
- Unit/invariants: QPSK BER vs actual repeated physical symbol simulation, single-index risk vs full K×K confusion matrix, DP optimum vs exhaustive integer search, header QPSK repetition+CRC-8, group overhead and total symbol budget, mock DeepSC `transmit/receive/decode_step` compatibility, vocab/encoder fingerprint mismatch detection.
- `python scripts/run_semgain.py --mode toy --snrs 0,6,12 --trials 100 --length 16 --group-size 4 --header-repeat 5 --budget-per-token 14`: completed, JSON+CSV generated. This is **synthetic additive distortion**, not DeepSC semantic quality.
- Compile check (`python -m compileall -q ...`): passed.

## One important negative finding

The real QPSK header matters. In a 100-trial toy, using contiguous groups of 4 and a five-repeat protection header, **the UEP methods experienced 91% header packet failure at 0 dB and ~14–15% at 6 dB**. This is not an indication that the UEP algorithm is inherently bad: it demonstrates that a naive header framing strategy at low Es/N0 can overwhelm gains predicted by a payload-only optimizer. `uniform` uses a profile known a priori and transmits no per-packet UEP header. Conditional payload loss does **not** include semantic consequences of dropped frames.

Remedies that need separate validation: more reliable control channel coding, profile compression/grouping, decoder-known schedules, CRC+ARQ, joint header/data resource optimization. Do not merely omit the header in publishable comparisons.

## Not verified / not implemented

- Real EHeum/DeepSC checkpoint inference, Europarl-120k integration, full DeepSC decoder fine-tuning, BLEU/BERTScore, GPU reproducibility.
- LDPC/Turbo/Polar, actual OFDM/TDL fading, channel estimation and control channel equalization, complete framing and retransmission.
- Robustness of fitted risk proxy to multi-bit errors, empirical correlation with downstream semantic scores, packet-level distortion when the header cannot be decoded.

**Interpretation rule:** Analytical DP optimality refers only to the stipulated *additive expected-risk model*, not measured sentence CE under multiple corruptions nor end-to-end throughput after dropped frames.
