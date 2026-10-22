"""Optional stage 1: adapt frozen DeepSC receiver to a fixed VQ codebook.

Train only existing receiver modules, not the source encoder or codebook.
Train/validation losses are CE; no channel noise in this stage.
"""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from semgain.deepsc_bridge import load_data_and_bridge
from semgain.vq import VectorCodebook
from semgain.validation import validate_artifact


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt',required=True)
    ap.add_argument('--codebook',required=True)
    ap.add_argument('--data',default='data/europarl_en.json')
    ap.add_argument('--out',default='checkpoints/deepsc_vq_ft.pt')
    ap.add_argument('--epochs',type=int,default=5)
    ap.add_argument('--max-sents',type=int,default=5000)
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--device',default='cpu')
    args=ap.parse_args()
    from deepsc.data import SentenceDataset
    train,test,vocab,bridge=load_data_and_bridge(args.ckpt,args.data,args.device)
    cb_record=torch.load(args.codebook,map_location='cpu',weights_only=True)
    validate_artifact(cb_record,vocab,bridge)
    cb=VectorCodebook(cb_record['codewords']).to(args.device)
    if cb.weight.shape[1]!=2*bridge.model.n_sym:
        raise ValueError('Codebook latent dimension does not match DeepSC symbols_per_word')
    for par in bridge.model.parameters():par.requires_grad_(False)
    for module in [bridge.model.channel_decoder,bridge.model.semantic_decoder,
                   bridge.model.tgt_embedding,bridge.model.prediction]:
        for par in module.parameters():par.requires_grad_(True)
    optimizer=torch.optim.Adam([p for p in bridge.model.parameters() if p.requires_grad],lr=args.lr)
    loader=DataLoader(SentenceDataset(train[:args.max_sents],vocab),
                      batch_size=args.batch_size,shuffle=True,collate_fn=SentenceDataset([],vocab).collate)
    best=float('inf')
    for epoch in range(args.epochs):
        total=0;count=0
        for src in loader:
            src=src.to(args.device)
            with torch.no_grad():
                z,mask=bridge.encode(src)
                ids=cb.encode(z)
                zq=cb.decode(ids)
            loss=bridge.sentence_ce(zq,src).mean()
            optimizer.zero_grad();loss.backward();optimizer.step()
            total+=loss.detach().item()*src.shape[0];count+=src.shape[0]
        train_ce=total/max(count,1)
        print(f'epoch {epoch+1}: clean VQ train CE={train_ce:.4f}',flush=True)
        if train_ce<best:
            best=train_ce
            original=torch.load(args.ckpt,map_location='cpu',weights_only=False)
            original['model']=bridge.model.state_dict()
            original['vq_stage']={'codebook':args.codebook,'clean_vq_train_ce':best,
                                  'note':'Decoder adapted to fixed, normalized TX VQ codewords'}
            Path(args.out).parent.mkdir(parents=True,exist_ok=True)
            torch.save(original,args.out)
    print('saved',args.out,'best train CE=',best)

if __name__=='__main__':main()
