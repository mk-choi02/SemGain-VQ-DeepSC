"""Optional low-cost policy network, trained by exact-DP imitation.

A policy never emits an unconstrained argmax at deployment: its logits are
projected through a hard-budget dynamic program. Training is distillation,
NOT backpropagation through the discrete channel or FEC.
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .allocator import dp_min_risk


class AllocationPolicy(nn.Module):
    def __init__(self, n_actions=3, hidden=32):
        super().__init__()
        self.n_actions=n_actions
        self.net=nn.Sequential(nn.Linear(5,hidden),nn.ReLU(),
                               nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,n_actions))

    def forward(self, q: torch.Tensor, snr_db: torch.Tensor, indices: torch.Tensor,
                budget_per_token: float):
        """Features: q*L, relative pos, log10 SNR, normalized index, budget/L."""
        l=len(q)
        inp=torch.stack([q*l,
                         torch.arange(l,device=q.device,dtype=q.dtype)/max(l,1),
                         snr_db.to(dtype=q.dtype)/20.0,
                         indices.to(dtype=q.dtype)/255.0,
                         torch.full_like(q, float(budget_per_token)/20.0)],dim=-1)
        return self.net(inp)

    @torch.no_grad()
    def constrained_select(self,q,snr_db,indices,budget,cost):
        scores=self(q,snr_db,indices,float(budget)/len(q))
        # Maximize network log probability under the true budget.
        neglogp=(-F.log_softmax(scores,dim=-1)).cpu().numpy()
        return dp_min_risk(neglogp,np.asarray(cost),budget)
