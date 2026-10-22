"""K-means vector quantization of normalized DeepSC transmitter symbols."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


class VectorCodebook(nn.Module):
    def __init__(self, codewords: torch.Tensor):
        super().__init__()
        if codewords.ndim != 2 or codewords.shape[0] < 2:
            raise ValueError('Codewords must have shape [K,D]')
        if codewords.shape[0] & (codewords.shape[0] - 1):
            raise ValueError('Codebook size must be a power of two')
        if codewords.shape[0].bit_length() % 2 != 1:  # log2(K) must be even for QPSK
            raise ValueError('Codebook log2(K) must be even for QPSK')
        if not torch.isfinite(codewords).all():
            raise ValueError('Codewords must be finite')
        self.register_buffer('weight', codewords.detach().clone().float())

    @property
    def nbits(self) -> int:
        return (self.weight.shape[0] - 1).bit_length()

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.weight.shape[1]:
            raise ValueError('Latent feature dimension does not match codebook')
        flat = x.reshape(-1, x.shape[-1])
        dis = (flat.square().sum(-1, keepdim=True)
               + self.weight.square().sum(-1).unsqueeze(0)
               - 2*flat @ self.weight.T)
        return dis.argmin(-1).reshape(x.shape[:-1])

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        return F.embedding(indices.long(), self.weight)


@torch.no_grad()
def fit_kmeans(x: torch.Tensor, k: int = 256, iters: int = 30,
               seed: int = 42, chunk: int = 8192) -> VectorCodebook:
    """Deterministic Lloyd updates with random-point replacement of empty groups."""
    if k < 2 or k & (k-1) or k.bit_length() % 2 != 1:
        raise ValueError('For QPSK use K=4,16,64,256,...')
    if x.ndim < 2 or x.shape[-1] == 0 or chunk <= 0 or iters <= 0:
        raise ValueError('Expected nonempty vectors, positive chunk and iterations')
    x = x.detach().float().reshape(-1, x.shape[-1]).cpu()
    if not torch.isfinite(x).all():
        raise ValueError('Training vectors must be finite')
    if x.shape[0] < k:
        raise ValueError(f'Need >= {k} samples, got {x.shape[0]}')
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=gen)
    centers = x[perm[:k]].clone()
    for _ in range(iters):
        accum = torch.zeros_like(centers)
        counts = torch.zeros(k)
        for batch in x.split(chunk):
            d = (batch.square().sum(-1, keepdim=True) +
                 centers.square().sum(-1)[None, :] - 2*batch @ centers.T)
            ids = d.argmin(-1)
            accum.index_add_(0, ids, batch)
            counts += torch.bincount(ids, minlength=k).float()
        new = accum / counts.clamp(min=1)[:, None]
        empty = counts == 0
        if empty.any():
            new[empty] = x[torch.randint(len(x), (int(empty.sum()),), generator=gen)]
        if torch.allclose(centers, new, rtol=1e-5, atol=1e-6):
            break
        centers = new
    return VectorCodebook(centers)
