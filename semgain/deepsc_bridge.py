"""Adapter to EHeum/DeepSC; must be run from cloned DeepSC repository root.

Tested against published method definitions as of 2026-10-21. For actual
integration tests use your own local repository and pretrained checkpoint.
"""
from __future__ import annotations
import math
from pathlib import Path
import torch
import torch.nn.functional as F


class DeepSCBridge:
    def __init__(self, model: torch.nn.Module, device='cpu'):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        if getattr(model, 'csi_rx', False):
            raise ValueError('Initial index-link adapter supports non-CSI DeepSC receiver only; retrain/freeze csi_rx=False.')
        if getattr(model, 'mapper', None) is not None:
            raise ValueError('Load an analog DeepSC checkpoint: VQ is inserted after its normalized channel encoder, not DeepJSCC-Q mapper.')

    @classmethod
    def from_checkpoint(cls, checkpoint: str, vocab, device='cpu'):
        from deepsc.engine import build_model
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        cfg = state['cfg']
        model = build_model(vocab, cfg)
        model.load_state_dict(state['model'], strict=True)
        return cls(model, device=device)

    @torch.no_grad()
    def encode(self, src: torch.Tensor):
        """Normalized analog TX slots [B,L,2*n_sym], with source padding mask."""
        src = src.to(self.device)
        x, _ = self.model.transmit(src)
        z = torch.stack((x.real, x.imag), -1).reshape(src.shape[0], src.shape[1], 2*self.model.n_sym)
        return z, src.ne(self.model.pad_idx)

    @torch.no_grad()
    def semantic_features(self, src: torch.Tensor):
        src = src.to(self.device)
        pad_mask = src.eq(self.model.pad_idx)
        m = self.model.pos(self.model.embedding(src)*math.sqrt(self.model.d_model))
        return self.model.semantic_encoder(m, src_key_padding_mask=pad_mask)

    def decode_memory(self, z: torch.Tensor, src: torch.Tensor):
        """Feed reconstructed codewords to existing DeepSC channel decoder."""
        if z.ndim != 3 or z.shape[:2] != src.shape or z.shape[-1] != 2*self.model.n_sym:
            raise ValueError('Expected z [B,L,2*n_sym] matching src [B,L]')
        x = torch.complex(z[...,0::2], z[...,1::2]).reshape(src.shape[0], -1)
        return self.model.receive(x, src.eq(self.model.pad_idx))

    def sentence_ce(self, z: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """Teacher-forced reconstruction CE PER SENTENCE (not greedy/BLEU)."""
        src = src.to(self.device)
        z = z.to(self.device)
        memory = self.decode_memory(z, src)
        logits = self.model.decode_step(memory, src.eq(self.model.pad_idx), src[:, :-1])
        tgt = src[:, 1:]
        ce = F.cross_entropy(logits.transpose(1,2), tgt,
                             ignore_index=self.model.pad_idx, reduction='none')
        return ce.sum(-1) / tgt.ne(self.model.pad_idx).sum(-1).clamp(min=1)

    @torch.no_grad()
    def greedy(self, z: torch.Tensor, src: torch.Tensor, vocab, max_len=None) -> torch.Tensor:
        src = src.to(self.device)
        mem = self.decode_memory(z.to(self.device), src)
        max_len = max_len or src.shape[1] + 2
        out = torch.full((src.shape[0],1), vocab.start, dtype=torch.long, device=src.device)
        done = torch.zeros(src.shape[0], dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            new = self.model.decode_step(mem, src.eq(self.model.pad_idx), out)[:,-1].argmax(-1)
            new = new.masked_fill(done, self.model.pad_idx)
            out = torch.cat([out,new[:,None]], dim=-1)
            done |= new.eq(vocab.end)
            if done.all():
                break
        return out


@torch.no_grad()
def delta_table(bridge: DeepSCBridge, src: torch.Tensor, indices: torch.Tensor,
                codebook, mode='onebit', batch_size=16) -> tuple[torch.Tensor, float]:
    """Single-index finite-difference decoder CE, [L,K], signed.

    Modes: 'onebit' only Hamming-1 substitutes (others are set to zero; risk
    is thus a truncated approximation); 'all' evaluates every k' exactly.
    Include BOS/EOS; padding should be removed before calling this function.
    """
    src = src.to(bridge.device)
    if src.shape[0] != 1 or src.eq(bridge.model.pad_idx).any():
        raise ValueError('Pass a single UNPADDED source sentence')
    idx = indices.reshape(-1).to(bridge.device).long()
    l = idx.numel()
    k = codebook.weight.shape[0]
    if src.shape[1] != l:
        raise ValueError('Source and index length differ')
    z0 = codebook.decode(idx).to(bridge.device).unsqueeze(0)
    base = float(bridge.sentence_ce(z0, src)[0])
    neighbors = []
    for i, current in enumerate(idx.tolist()):
        candidates = (range(k) if mode == 'all' else
                      [current ^ (1 << bit) for bit in range(codebook.nbits)])
        for cand in candidates:
            if cand != current:
                neighbors.append((i,int(cand)))
    if mode not in ('all', 'onebit'):
        raise ValueError('Unknown mode')
    result = torch.zeros(l, k, dtype=torch.float32, device=bridge.device)
    for start in range(0, len(neighbors), batch_size):
        part = neighbors[start:start+batch_size]
        z = z0.expand(len(part), -1, -1).clone()
        for j, (i, cand) in enumerate(part):
            z[j,i] = codebook.weight[cand].to(bridge.device)
        tiled_src = src.expand(len(part), -1)
        loss = bridge.sentence_ce(z, tiled_src) - base
        for j, (i, cand) in enumerate(part):
            result[i,cand] = loss[j]
    return result.cpu(), base


def load_data_and_bridge(ckpt, corpus, device='cpu'):
    from deepsc.data import load_corpus
    train, test, vocab = load_corpus(corpus)
    return train, test, vocab, DeepSCBridge.from_checkpoint(ckpt, vocab, device)
