"""Learn VQ centroids from pretrained analog DeepSC normalized TX slots."""
import argparse
import json
from pathlib import Path
import torch
from semgain.deepsc_bridge import load_data_and_bridge
from semgain.vq import fit_kmeans
from semgain.validation import vocabulary_fingerprint, encoder_fingerprint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data', default='data/europarl_en.json')
    ap.add_argument('--out', default='checkpoints/semgain_codebook.pt')
    ap.add_argument('--k', type=int, default=256)
    ap.add_argument('--max-sents', type=int, default=2000)
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    train, _, vocab, bridge = load_data_and_bridge(args.ckpt, args.data, args.device)
    samples = []
    for i, s in enumerate(train[:args.max_sents]):
        src = torch.tensor(vocab.encode(s), dtype=torch.long)[None]
        z, mask = bridge.encode(src)
        samples.append(z[mask].cpu())
        if (i+1) % 500 == 0:
            print(f'collected {i+1} sentences')
    if not samples:
        raise RuntimeError('No samples')
    x = torch.cat(samples)
    codebook = fit_kmeans(x, k=args.k, iters=args.iters, seed=args.seed)
    indices = codebook.encode(x)
    error = (codebook.decode(indices)-x).square().mean().item()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'codewords': codebook.weight, 'k':args.k,
                'mean_squared_error':error, 'samples':len(x),
                'checkpoint_source':args.ckpt,
                'vocab_sha256':vocabulary_fingerprint(vocab),
                'encoder_sha256':encoder_fingerprint(bridge.model)}, args.out)
    print(json.dumps({'saved':args.out,'MSE':error,'n_vectors':len(x),
                      'usage':int(indices.unique().numel())}, indent=2))

if __name__ == '__main__':
    main()
