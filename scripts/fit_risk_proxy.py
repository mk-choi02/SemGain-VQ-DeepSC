"""Train risk proxy on offline teacher labels (only 1-bit sampled transitions)."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from semgain.deepsc_bridge import load_data_and_bridge,delta_table
from semgain.vq import VectorCodebook
from semgain.validation import validate_artifact
from semgain.importance import PositionContextImportance,teacher_importance
from semgain.risk_model import geometry_features,fit_linear_risk_proxy


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt',required=True)
    ap.add_argument('--codebook',required=True)
    ap.add_argument('--data',default='data/europarl_en.json')
    ap.add_argument('--importance-model', default=None)
    ap.add_argument('--out',default='checkpoints/semgain_risk.json')
    ap.add_argument('--max-sents',type=int,default=100)
    ap.add_argument('--start-sent',type=int,default=5100,help='Start after decoder/importance training segments (default index 5100)')
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    train,_,vocab,bridge=load_data_and_bridge(args.ckpt,args.data,args.device)
    cb_record=torch.load(args.codebook,map_location='cpu',weights_only=True)
    validate_artifact(cb_record,vocab,bridge)
    codebook=VectorCodebook(cb_record['codewords']).to(args.device)
    predictor=None
    if args.importance_model:
        ck=torch.load(args.importance_model,map_location='cpu',weights_only=True)
        predictor=PositionContextImportance(ck['d_model'],ck['max_len']).to(args.device)
        predictor.load_state_dict(ck['model']);predictor.eval()
    feats,qs,ys=[],[],[]
    for j,s in enumerate(train[args.start_sent:args.start_sent+args.max_sents]):
        src=torch.tensor(vocab.encode(s),dtype=torch.long,device=args.device)[None]
        if predictor is not None and src.shape[1]>predictor.max_len:
            continue
        z,_=bridge.encode(src)
        ids=codebook.encode(z)[0]
        delta,_=delta_table(bridge,src,ids,codebook,'onebit')
        if predictor is None:
            q=teacher_importance(delta,ids.cpu(),codebook.nbits).numpy()
        else:
            with torch.no_grad():
                q=predictor(bridge.semantic_features(src),torch.ones_like(src,dtype=torch.bool))[0].cpu().numpy()
        ids=ids.cpu().numpy()
        feat=geometry_features(ids,codebook.weight.cpu().numpy())
        for i,k in enumerate(ids):
            for bit in range(codebook.nbits):
                alt=int(k)^(1<<bit)
                feats.append(feat[i,alt]);qs.append(q[i]);ys.append(float(delta[i,alt]))
        if (j+1)%20==0:print(f'calibrated {j+1} sentences')
    if not feats:
        raise RuntimeError('No valid samples')
    weights=fit_linear_risk_proxy(np.array(feats),np.array(qs),np.array(ys))
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    with open(args.out,'w') as f:
        json.dump({'weights':weights.tolist(),'n_events':len(ys),
                   'importance_source':'model' if predictor is not None else 'teacher_oracle',
                   'note':'calibrate on train/val only; evaluate on held-out test'},f,indent=2)
    print(f'Saved {args.out}: {weights}')

if __name__=='__main__':main()
