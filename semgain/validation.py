"""Check training and evaluation artifacts refer to identical tokenizer/encoders."""
from __future__ import annotations
import hashlib
import json
import torch


def vocabulary_fingerprint(vocab):
    if not hasattr(vocab,'itos'):
        raise ValueError('Expected DeepSC Vocab with ordered itos tokens')
    return hashlib.sha256(json.dumps(list(vocab.itos),ensure_ascii=False,
                                     separators=(',',':')).encode('utf-8')).hexdigest()


def encoder_fingerprint(model):
    """Includes DeepSC transmitter weights/buffers, not the receiver (fine-tunable)."""
    digest=hashlib.sha256()
    state=model.state_dict()
    for key in sorted(state):
        if key.startswith(('embedding.','pos.','semantic_encoder.','channel_encoder.')):
            item=state[key].detach().cpu().contiguous()
            digest.update(key.encode());digest.update(str(tuple(item.shape)).encode())
            digest.update(item.numpy().tobytes())
    return digest.hexdigest()


def validate_artifact(checkpoint_record, vocab, bridge):
    if checkpoint_record['codewords'].shape[1] != 2 * bridge.model.n_sym:
        raise ValueError('VQ dimensionality does not match DeepSC n_sym')
    expected = checkpoint_record.get('vocab_sha256')
    if expected is not None and expected != vocabulary_fingerprint(vocab):
        raise ValueError('Vocabulary/order mismatch: retrain VQ or use the exact training vocabulary')
    enc = checkpoint_record.get('encoder_sha256')
    if enc is not None and enc != encoder_fingerprint(bridge.model):
        raise ValueError('Semantic/channel encoder weights changed since VQ fitting')
    emb = getattr(bridge.model,'embedding',None)
    if emb is not None and emb.num_embeddings != len(vocab):
        raise ValueError('Checkpoint embedding size does not match vocabulary size')
