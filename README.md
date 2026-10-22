# SemGain-VQ-DeepSC (v0.2) — Channel-Aware Marginal Semantic Gain

**Research implementation / 실험용 프로토타입.** This is a **new, independent Python repository** that uses the public [EHeum/DeepSC](https://github.com/EHeum/DeepSC) encoder/decoder through an adapter. It neither vendors the DeepSC source nor includes its checkpoints or datasets.

> **Verification scope:** numerical tests and standalone AWGN/QPSK toy simulations run. The true pretrained DeepSC/Europarl end-to-end pipeline has **not been run here** because its source checkout, matching vocabulary, pretrained checkpoint and dataset are not bundled. Run `verify_environment.py` locally before interpreting real-model results. `toy` is not semantic-communication evidence.

## Research question

Why isn't importance-only UEP necessarily optimal? Index `k_i` reconstructed as `k'` causes a decoder-CE loss difference `Delta[i,k']`. At a given QPSK Es/N0 and repetition factor `a`, index confusion is `T[k_i,k'; SNR,a]`. The predicted *additive* token risk is

```
R[i,a] = sum_{k'=0}^{K-1} T[k_i,k'; SNR,a] * Delta[i,k'].
eta[i,a->a'] = (R[i,a] - R[i,a']) / (cost[a']-cost[a]).
min_{a_i} sum_i R[i,a_i] s.t. sum_i cost[a_i] + header + pilot <= B.
```

Baseline algorithms: `uniform`, `importance_only`, `marginal_greedy`, `marginal_dp` (exact multiple-choice knapsack for this additive risk model). **This does not guarantee minimization of full sentence distortion** when errors interact. Delta may be signed. Finite differences based on one-bit flips leave multi-bit error events unmeasured, so they are only an approximation.

## What's been improved in v0.2

1. **Actually transmit the action-profile header** with QPSK, bit repetition and CRC-8. Profile packet failures are measured, not silently assumed away. Token count, framing and decoder synchronization are still shared assumptions. CRC undetected errors are conservatively counted as a failed profile in evaluation (not a full production protocol).
2. **Contiguous group-level UEP (`--group-size`)** jointly reduces header overhead and the number of independent action choices; compare group sizes 1/2/4/8. This is an overhead–allocation-granularity ablation, not free gain.
3. **Strict artifact compatibility**: newly trained VQ files contain SHA256 fingerprints of the ordered vocabulary and analog encoder; a mismatched checkpoint or codebook is rejected. The original analog checkpoint itself may not contain a tokenizer hash, so externally supplied checkpoints still require manual provenance verification.
4. **Separated calibration ranges**: decoder fine-tuning defaults to first 5000 training examples; importance-teacher defaults to [5000,5100); risk-proxy calibration defaults to [5100,5200). These ranges refer to the **training split only**. Increase sample counts for publishable results.
5. **More verification**: analytical risk vs brute-force confusion matrix, analytical QPSK BER vs Monte Carlo, DP vs exhaustive search, real header/CRC round-trip, strict symbol budgets, and mock DeepSC bridge.
6. **Evaluate clean analog vs clean VQ CE** to detect disastrous VQ degradation before UEP claims.

## Installation (independent GitHub project)

Tested locally using Python 3.13 and installed PyTorch/NumPy; recommended Python **3.11** for your conda/CUDA environment. No GPU is required for synthetic demo.

```bash
# Have two sibling folders in your working directory
# workspace/
#   DeepSC/                      existing upstream checkout
#   SemGain-VQ-DeepSC/            THIS repository
cd SemGain-VQ-DeepSC
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/run_semgain.py --mode toy --snrs 0,6,12 --trials 100 \
  --group-size 4 --header-repeat 5 --budget-per-token 14 \
  --out results/toy.json
```

Expected: all tests pass, and printed rows contain `headerFER`, `empirical`, `pred`, and `uses`; result JSON + CSV appears under `results/`. Some policies may perform worse than `uniform`, especially when low-SNR header failures are prevalent. That is a valid finding, not a test failure.

**Optional plot:** `python -m pip install matplotlib` then

```bash
python scripts/plot_results.py --input results/toy.json --out results/toy.png
```

**Important:** plotted loss is conditional on *successful header delivery*. Inspect `header_frame_error_rate` separately. Conditional loss alone must not be used to rank overall network reliability.

## Use an actual local DeepSC checkpoint

Clone DeepSC **separately** (do not copy upstream code into this new repository):

```bash
cd ..
git clone https://github.com/EHeum/DeepSC.git
cd SemGain-VQ-DeepSC
# Allow the scripts to import the sibling upstream deepsc package
export PYTHONPATH="$(pwd):$(pwd)/../DeepSC"
```

If you already have `../DeepSC`, skip cloning. For Windows PowerShell replace `export` with `$env:PYTHONPATH="$PWD;$PWD\..\DeepSC"`.

**Data schema:** DeepSC's `load_corpus` expects `{"train":["..."],"test":["..."],"vocab":["<PAD>","<START>","<END>","<UNK>",...]}`. If your Europarl-120k uses `train/val/test`, use `scripts/convert_europarl.py` to write that schema and preserve `val` in a sidecar. The **vocabulary order must match the checkpoint**; never silently rebuild it.

```bash
python scripts/convert_europarl.py \
  --input ../DeepSC/data/europarl_en_120k.json \
  --out ../DeepSC/data/europarl_semgain.json
# If the input has no ordered vocabulary, add:
# --vocab-source ../DeepSC/data/<checkpoint_matching_corpus>.json
```

**Stage 0 — model and dataset diagnostics** (before fitting VQ):

```bash
python scripts/verify_environment.py \
  --deepsc-root ../DeepSC \
  --data ../DeepSC/data/europarl_semgain.json \
  --ckpt ../DeepSC/checkpoints/deepsc_awgn.pt
```

If the model has not yet been trained, run the upstream project's training command from `../DeepSC` first. This add-on only supports analog `csi_rx=False` checkpoints without an internal constellation mapper; default DeepSC model uses `n_sym=8`, so its normalized analog latents are 16D.

**Stage 1 — VQ codebook** (checkpoint and corpus must match):

```bash
python scripts/fit_codebook.py \
  --ckpt ../DeepSC/checkpoints/deepsc_awgn.pt \
  --data ../DeepSC/data/europarl_semgain.json \
  --k 256 --max-sents 2000 \
  --out checkpoints/codebook.pt
```

**Stage 2 — adapt frozen encoder's decoder for quantized latents:**

```bash
python scripts/finetune_vq.py \
  --ckpt ../DeepSC/checkpoints/deepsc_awgn.pt \
  --codebook checkpoints/codebook.pt \
  --data ../DeepSC/data/europarl_semgain.json \
  --max-sents 5000 --epochs 5 \
  --out checkpoints/deepsc_vq.pt
```

**Stage 3 — importance teacher + position-context student** (defaults to **training** rows 5000–5099):

```bash
python scripts/train_importance.py \
  --ckpt checkpoints/deepsc_vq.pt \
  --codebook checkpoints/codebook.pt \
  --data ../DeepSC/data/europarl_semgain.json \
  --start-sent 5000 --max-sents 100 --epochs 30 \
  --out checkpoints/importance.pt
```

**Stage 4 — fit risk proxy** (defaults to train rows 5100–5199, *disjoint from importance samples*):

```bash
python scripts/fit_risk_proxy.py \
  --ckpt checkpoints/deepsc_vq.pt \
  --codebook checkpoints/codebook.pt \
  --data ../DeepSC/data/europarl_semgain.json \
  --importance-model checkpoints/importance.pt \
  --start-sent 5100 --max-sents 100 \
  --out checkpoints/risk.json
```

**Stage 5 — TEST-only physical-link evaluation** (risk proxy never queries target sentence oracle):

```bash
python scripts/run_semgain.py --mode deepsc \
  --ckpt checkpoints/deepsc_vq.pt \
  --codebook checkpoints/codebook.pt \
  --data ../DeepSC/data/europarl_semgain.json \
  --importance learned --importance-model checkpoints/importance.pt \
  --risk-source proxy --risk-model checkpoints/risk.json \
  --snrs 0,6,12 --trials 20 --max-sents 50 \
  --group-size 4 --header-repeat 5 --budget-per-token 14 \
  --out results/deepsc_test.json
```

To confirm the trained codebook against this checkpoint:

```bash
python scripts/verify_environment.py --deepsc-root ../DeepSC \
  --data ../DeepSC/data/europarl_semgain.json \
  --ckpt checkpoints/deepsc_vq.pt --codebook checkpoints/codebook.pt
```

**Optional upper bound**: pass `--importance oracle --risk-source oracle --delta-mode all` for a **tiny test subset only**. Never call this oracle approach deployable; it accesses the reference sentence/decoder losses at inference. Using `--delta-mode onebit` gives only a truncated approximation.

### Suggested ablation experiments

Run `--group-size 1`, `2`, `4`, `8` while preserving identical `--budget-per-token`, SNR, seed, and trial count. Compare actual *goodput/packet success rate*, header FER, conditional CE, and consumed symbols. For strict reproducibility archive the JSON `config` and `vocab_sha256` with plots. To compare unconditional semantic quality, define and publish an explicit **frame-erasure semantics or retransmission protocol**, which this MVP does not yet implement.

## Repository map

```text
semgain/                     algorithms
  allocator.py               uniform / rank-only / greedy / exact DP
  risk.py                    confusion-weighted additive semantic risk
  risk_model.py              low-capacity geometry × importance predictor
  importance.py              length prior + contextual residual MLP
  vq.py                      K-means nearest-neighbor quantizer
  phy.py                     QPSK AWGN with bit repetition
  header.py                  actual action-profile header + CRC-8
  validation.py              vocabulary / encoder checkpoint fingerprint
  deepsc_bridge.py           adapter to sibling EHeum/DeepSC
  policy.py                  optional toy-only DP imitation
scripts/                     CLI entry points
  verify_environment.py      upstream checkpoint compatibility diagnostics
  convert_europarl.py        split schema adapter (preserves val separately)
  fit_codebook.py, finetune_vq.py, train_importance.py
  fit_risk_proxy.py, run_semgain.py, plot_results.py, train_policy.py
 tests/                      numerical and protocol regression tests
 docs/GITHUB_PUBLISH_KO.md   step-by-step independent GitHub upload
 docs/VALIDATION_REPORT.md   what has and has not been verified
```

## Scientific limitations — PLEASE READ

- Only **coherent AWGN/QPSK + odd bit repetition** is simulated. No LDPC, real FEC interleaving, OFDM, TDL fading, CSI estimation, CRC-protected **data block**, synchronization, or ARQ yet. QPSK `snr_db` is **Es/N0**, not Eb/N0; a repeated bit uses extra transmitted energy and symbols. The comparison matches symbol-use budget, not total energy normalized across arbitrary waveform durations.
- The header encodes per-group protection levels plus CRC-8, QPSK bit repetition. Header errors are recorded; the prototype *fails closed* rather than reconstructing with a corrupted profile. There is no implemented semantic fallback reconstruction or retransmission. Conditional CE must always be reported alongside header FER.
- A decoder-CE proxy is not BLEU/BERTScore. Protection decisions assume **additive single-index distortion**. Multi-token corruptions may not be additive. The risk proxy extrapolates from **one-bit corruptions** to multi-bit substitutions and is **not calibrated out of the box**.
- Grouped protection assumes frame boundaries, number of VQ index slots, codebook, grouping rule and action alphabet are shared. Their protocol overhead beyond the described header is not modeled.
- Existing EHeum/DeepSC code is not redistributed here; check its license/attribution before distributing a combined derived source tree. No model weights or Europarl datasets are included. No benchmark-quality semantic results are claimed.

See [검증 결과](docs/VALIDATION_REPORT.md) and [GitHub 업로드 가이드](docs/GITHUB_PUBLISH_KO.md).
