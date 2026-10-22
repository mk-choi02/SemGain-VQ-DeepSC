# Mathematical implementation notes

## AWGN link and codebook errors

K=256 index bits `ell(k) in {0,1}^8`. QPSK unit energy (Es=1),
`Es/N0 = 10^(gamma_db/10)`. With coherent hard QPSK detection

`p_b(gamma) = Q(sqrt(Es/N0)) = 0.5*erfc(sqrt((Es/N0)/2))`.

With odd repetition `r`, majority hard-decision BER:

`p_r = sum_{j=(r+1)/2}^r C(r,j) p_b^j (1-p_b)^(r-j)`.

If bit errors after repetition are independent across bits:

`T_{kk'}(gamma,r) = p_r^{dH(k,k')}(1-p_r)^{8-dH(k,k')}`.

All `T[k,:]` rows sum to 1. Errors across token positions are modeled as independent; the actual DeepSC semantic loss is not assumed additive during final CE measurement.

## Decoder-derived per-index semantic sensitivity

Clean VQ codeword indices `k[1:L]` reconstructed into `(L,16)` array of float real/imag components and passed to the *existing* `DeepSC.receive` (channel decoder) and `DeepSC.decode_step` (semantic decoder). Use teacher-forced cross entropy over non-PAD target tokens:

`D(s, Zhat) = mean_{t nonpad} CE(P_theta(.|s_<t, Zhat), s_t)`.

Set `Delta[i,k'] = D(s, Z_q[i <- C[k']]) - D(s,Z_q)`. Some Delta can be negative: the implementation **does not clamp** offline oracle risk. Exact risk is `R[i,a] = sum_{k'} T[k_i,k';gamma_i,a] Delta[i,k']`. For one-bit teacher only, missing transitions have zero Delta and yield a low-order *approximation*, not exact expectation.

## Risk surrogate and separability

`q_i(s)` is predicted by position-context MLP using frozen semantic encoding, not the corruption effect of the test message.

The demonstrator surrogate uses

`Delta_hat[i,k'] = q_i(s) max(0, beta0 + beta1||C[k_i]-C[k']||² + beta2 Hamming(k_i,k'))` for k' != k_i.

The coefficients are fitted to positive counterfactual CE changes on the original corpus's `train` split. It is intentionally low complexity and does not claim to match sample-specific semantic changes. Learn a richer conditional risk network if it underfits.

## Resource cost and fairness

For 8 index bits and QPSK modulation (2 bits/symbol), repetition r requires exactly `8*r/2 = 4*r` QPSK symbols per token. For three protection classes (`rep1, rep3, rep5`), the current fixed-length profile has 2 header bits per token. With header repetition h=3, the reservation is `ceil(2 * L * 3 / 2) = 3L` complex QPSK symbols. The repeated header is NOT physically transmitted in the demonstrator; its error-free availability is assumed. It is a modeled reservation, not an implemented packet receiver. Uniform reference profile is predetermined, and its per-token profile header is 0.

This design represents **variable data symbols, fixed Es**, not fixed total transmit energy. The fairness claim is limited to a **symbol/RE budget**. Total-energy fairness requires different constraints.

## DP and greedy

Given a fixed risk table `R[L,A]`, cost table `C[L,A]`, and available data symbol budget `B_data = B - header - pilot`, choose actions `a_i` minimizing `sum_i R[i,a_i]` with `sum_i C[i,a_i] <= B_data`. Exact DP is multiple-choice knapsack (`O(L*A*B_data)`). Marginal greedy iteratively maximizes `(R[i,a]-R[i,a+1])/(C[i,a+1]-C[i,a])` among feasible positive-gain upgrades. Greedy need not equal DP.

## What still requires research engineering

- Real LDPC/codeblock link and codeword-to-bit labelling optimization.
- Error-corrupted header with CRC, synchronization and explicit length signalling.
- OFDM multi-carrier/TDL time-frequency response, pilots and imperfect CSI.
- Efficient decoder-error surrogate fully trained on train/val, held-out evaluation.
- Confidence intervals and real semantic similarity metrics.
- Correlated multi-index failure model where blocks span several VQ tokens.
