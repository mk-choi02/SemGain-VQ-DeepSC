"""Empirical length-conditioned position prior + contextual residual MLP."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionContextImportance(nn.Module):
    def __init__(self, d_model=128, max_len=40, hidden=64):
        super().__init__()
        self.max_len = max_len
        self.register_buffer('prior', torch.zeros(max_len+1, max_len))
        self.prior.fill_(0)
        for length in range(1, max_len+1):
            self.prior[length, :length] = 1.0/length
        self.mlp = nn.Sequential(nn.Linear(2*d_model+2, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    @torch.no_grad()
    def set_empirical_prior(self, targets: torch.Tensor, lengths: torch.Tensor,
                            alpha: float=20.0):
        """targets [N,maxL] normalized per valid sentence; Dirichlet shrinkage."""
        device = self.prior.device
        targets = targets.to(device)
        lengths = lengths.to(device)
        for length in range(1, self.max_len+1):
            chosen = targets[lengths == length]
            count = int(len(chosen))
            if count:
                empirical = chosen[:, :length].mean(0)
                prior = (count*empirical + alpha/length)/(count+alpha)
                self.prior[length, :length] = prior / prior.sum()

    def forward(self, m: torch.Tensor, mask: torch.Tensor, temperature=1.0):
        b, l, d = m.shape
        if l > self.max_len:
            raise ValueError('Sequence longer than max_len')
        lengths = mask.long().sum(-1).clamp(min=1)
        global_m = (m*mask[...,None]).sum(dim=1, keepdim=True)/lengths[:,None,None]
        position = (torch.arange(l, device=m.device, dtype=m.dtype)[None,:,None] + 1)/lengths[:,None,None]
        loglen = torch.log(lengths.float())[:,None,None].expand(b,l,1)
        x = torch.cat([m, global_m.expand(-1,l,-1), position.expand(b,-1,-1),loglen], dim=-1)
        residual = self.mlp(x).squeeze(-1)
        residual = residual - (residual*mask).sum(-1,keepdim=True)/lengths[:,None]
        pi = self.prior[lengths][:,:l].clamp_min(1e-12)
        logits = pi.log() + residual/temperature
        q = F.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
        return q


def teacher_importance(delta: torch.Tensor, indices: torch.Tensor, nbits: int) -> torch.Tensor:
    """Unweighted, positive mean over all one-bit substitutions; not test-channel aware."""
    l = len(indices)
    gathered = torch.stack([delta[torch.arange(l), indices.long() ^ (1 << b)]
                            for b in range(nbits)], dim=-1)
    x = gathered.clamp(min=0).mean(-1)
    if float(x.sum()) == 0:
        return torch.full_like(x, 1.0/l)
    return x/x.sum()


def importance_loss(q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return -(target*torch.log(q.clamp_min(1e-9))).sum(-1).mean()
