"""Standalone DP-imitation smoke training on synthetic risk tables.

In real experiments, replace synthetic calibration episodes with held-out
train/validation episodes, and NEVER distill from test-sentence oracle risk.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from semgain.policy import AllocationPolicy
from semgain.risk import risk_from_delta
from semgain.allocator import dp_min_risk,per_token_costs
from semgain.phy import header_cost
from scripts.run_semgain import toy_data


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--episodes',type=int,default=100)
    ap.add_argument('--epochs',type=int,default=10)
    ap.add_argument('--length',type=int,default=16)
    ap.add_argument('--budget-per-token',type=float,default=14)
    ap.add_argument('--out',default='checkpoints/semgain_policy.pt')
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    torch.manual_seed(args.seed)
    rng=np.random.default_rng(args.seed)
    net=AllocationPolicy()
    optim=torch.optim.Adam(net.parameters(),lr=1e-3)
    examples=[]
    levels=(1,3,5); nbits=8
    for ep in range(args.episodes):
        idx,delta,q=toy_data(args.length,nbits,args.seed+ep)
        snr=float(rng.uniform(-3,12)) + np.linspace(-4,4,args.length)
        cost=per_token_costs(args.length,levels,nbits)
        budget=round(args.budget_per_token*args.length)-header_cost(args.length,len(levels),3)
        risk=risk_from_delta(idx,delta,snr,levels)
        label=dp_min_risk(risk,cost,budget).levels
        examples.append((q,snr,idx,label,budget))
    for epoch in range(args.epochs):
        rng.shuffle(examples)
        losses=[]
        for q,snr,idx,labels,b in examples:
            q=torch.tensor(q,dtype=torch.float32)
            snr=torch.tensor(snr,dtype=torch.float32)
            idx=torch.tensor(idx,dtype=torch.long)
            pred=net(q,snr,idx,b/len(q))
            loss=F.cross_entropy(pred,torch.tensor(labels))
            optim.zero_grad();loss.backward();optim.step()
            losses.append(loss.item())
        if epoch==0 or (epoch+1)%5==0:print(f'epoch={epoch+1} CE={np.mean(losses):.4f}')
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':net.cpu().state_dict(),'n_actions':len(levels),'levels':levels},args.out)
    print('saved:',args.out)

if __name__=='__main__':main()
