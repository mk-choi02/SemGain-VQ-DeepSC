"""Read-only diagnostic: local model/data/codebook compatibility."""
import argparse
import json
import sys
from pathlib import Path
import torch
from semgain.validation import validate_artifact, vocabulary_fingerprint

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--deepsc-root',required=True,help='Path to a separate EHeum/DeepSC checkout')
    ap.add_argument('--data',required=True)
    ap.add_argument('--ckpt',required=True)
    ap.add_argument('--codebook',default=None)
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    upstream=Path(args.deepsc_root).resolve()
    if not (upstream/'deepsc'/'models.py').exists():
        raise SystemExit(f'Not a compatible DeepSC clone: {upstream}')
    sys.path.insert(0,str(upstream))
    from semgain.deepsc_bridge import load_data_and_bridge
    train,test,vocab,bridge=load_data_and_bridge(args.ckpt,args.data,args.device)
    if not train or not test:
        raise ValueError('Dataset must contain nonempty train/test split')
    example=torch.tensor(vocab.encode(test[0]),dtype=torch.long)[None]
    z,mask=bridge.encode(example)
    ce=float(bridge.sentence_ce(z,example.to(bridge.device))[0])
    result={'deepsc_path':str(upstream),'train_sentences':len(train),
            'test_sentences':len(test),'vocab_size':len(vocab),
            'vocab_sha256':vocabulary_fingerprint(vocab),
            'latent_shape':list(z.shape),'n_sym':bridge.model.n_sym,
            'clean_analog_teacher_forcing_ce':ce}
    if args.codebook:
        cb=torch.load(args.codebook,map_location='cpu',weights_only=True)
        validate_artifact(cb,vocab,bridge)
        from semgain.vq import VectorCodebook
        vq=VectorCodebook(cb['codewords']).to(args.device)
        with torch.no_grad():
            idx=vq.encode(z);q=vq.decode(idx)
            result['clean_vq_teacher_forcing_ce']=float(bridge.sentence_ce(q,example.to(bridge.device))[0])
            result['codebook_size']=cb['codewords'].shape[0]
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
