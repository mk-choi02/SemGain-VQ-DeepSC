"""Train optional low-cost importance student from expensive offline CE oracle."""
import argparse
import json
from pathlib import Path
import torch
from semgain.deepsc_bridge import load_data_and_bridge, delta_table
from semgain.vq import VectorCodebook
from semgain.validation import validate_artifact
from semgain.importance import PositionContextImportance, teacher_importance, importance_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--codebook', required=True)
    ap.add_argument('--data', default='data/europarl_en.json')
    ap.add_argument('--out', default='checkpoints/semgain_importance.pt')
    ap.add_argument('--max-sents', type=int, default=100)
    ap.add_argument('--start-sent', type=int, default=5000, help='Use examples after decoder fine-tuning segment')
    ap.add_argument('--max-len', type=int, default=40)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--alpha', type=float, default=20.0)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    train, _, vocab, bridge = load_data_and_bridge(args.ckpt,args.data,args.device)
    saved = torch.load(args.codebook, map_location='cpu', weights_only=True)
    validate_artifact(saved,vocab,bridge)
    codebook = VectorCodebook(saved['codewords']).to(args.device)
    feats, targets, lens = [], [], []
    for t, s in enumerate(train[args.start_sent:args.start_sent+args.max_sents]):
        src = torch.tensor(vocab.encode(s), dtype=torch.long, device=args.device)[None]
        if src.shape[1]>args.max_len:
            continue
        z, _ = bridge.encode(src)
        ids = codebook.encode(z)[0]
        delta,_ = delta_table(bridge,src,ids,codebook,mode='onebit')
        target=teacher_importance(delta,ids.cpu(),codebook.nbits)
        feats.append(bridge.semantic_features(src)[0].cpu())
        targets.append(target.cpu())
        lens.append(src.shape[1])
        if (t+1)%20==0:print(f'cached importance {t+1} sentences')
    if not feats:
        raise RuntimeError('No teacher samples')
    dmodel=feats[0].shape[-1]
    predictor=PositionContextImportance(dmodel,args.max_len).to(args.device)
    full_targets=torch.zeros(len(lens),args.max_len)
    for i,target in enumerate(targets):full_targets[i,:len(target)]=target
    predictor.set_empirical_prior(full_targets,torch.tensor(lens),args.alpha)
    optimizer=torch.optim.Adam(predictor.mlp.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        total=0
        for m,tgt in zip(feats,targets):
            m=m[None].to(args.device)
            mask=torch.ones((1,m.shape[1]),dtype=torch.bool,device=args.device)
            pred=predictor(m,mask)
            loss=importance_loss(pred,tgt[None].to(args.device))
            optimizer.zero_grad();loss.backward();optimizer.step()
            total+=float(loss.detach())
        if epoch==0 or (epoch+1)%5==0:
            print(f'epoch={epoch+1} importance_ce={total/len(feats):.5f}')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model':predictor.cpu().state_dict(),'d_model':dmodel,
                'max_len':args.max_len,'n_teacher':len(feats)},args.out)
    print(json.dumps({'saved':args.out,'train_sentences':len(feats)}, indent=2))

if __name__=='__main__':main()
