"""Evaluate fair-budget importance-only versus confusion-aware allocation.

Two modes:
  toy   : fully self-contained, validates math and QPSK/repetition simulations.
  deepsc: requires pretrained EHeum/DeepSC and codebook and test sentences.

Default DeepSC configuration with oracle risk is an ANALYSIS UPPER BOUND,
not a deployable transmitter. Use `--importance learned --risk-source proxy`
with separately fitted models for a deployable risk-prediction experiment.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from semgain.phy import symbol_cost, header_cost, transmit_indices
from semgain.header import transmit_profile
from semgain.allocator import dp_min_risk, greedy_marginal, uniform, importance_only, per_token_costs
from semgain.risk import risk_from_delta, evaluate_additive_risk
from semgain.risk_model import predict_delta
from semgain.validation import validate_artifact


def choose_allocations(q, risk, budget, levels, nbits, header_repeat,
                       learned_policy=None, gammas=None, indices=None, group_size=1):
    """Fair-budget strategies; optional contiguous UEP groups share one header class.

    Grouping controls payload protection granularity AND on-air header length.
    The analytical DP is exact for this group-constrained additive risk model.
    """
    l=len(q)
    q=np.asarray(q,dtype=float)
    risk=np.asarray(risk,dtype=float)
    cost=per_token_costs(l,levels,nbits)
    if group_size < 1:
        raise ValueError('--group-size must be >= 1')
    if learned_policy is not None and group_size != 1:
        raise ValueError('Learned toy policy currently requires --group-size 1')
    starts=list(range(0,l,group_size))
    grouped_q=np.array([q[j:j+group_size].sum() for j in starts])
    grouped_risk=np.stack([risk[j:j+group_size].sum(0) for j in starts])
    grouped_cost=np.stack([cost[j:j+group_size].sum(0) for j in starts])
    ueph=header_cost(len(starts),len(levels),header_repeat)
    uep_budget=budget-ueph
    if uep_budget < int(grouped_cost[:,0].sum()):
        raise ValueError(f'Budget {budget} too small for UEP header ({ueph}) + base data cost')

    def expand(allocation):
        full=np.repeat(allocation.levels,group_size)[:l]
        return type(allocation)(full,int(cost[np.arange(l),full].sum()),
                                float(risk[np.arange(l),full].sum()))
    policies = {
        'uniform': (uniform(risk,cost,budget),0),
        'importance_only': (expand(importance_only(grouped_q,grouped_risk,grouped_cost,uep_budget)),ueph),
        'marginal_greedy': (expand(greedy_marginal(grouped_risk,grouped_cost,uep_budget)),ueph),
        'marginal_dp': (expand(dp_min_risk(grouped_risk,grouped_cost,uep_budget)),ueph),
    }
    if learned_policy is not None:
        q_t = torch.tensor(q,dtype=torch.float32)
        snr_t = torch.tensor(np.broadcast_to(gammas,(l,)),dtype=torch.float32)
        ids_t = torch.tensor(indices,dtype=torch.long)
        learned = learned_policy.constrained_select(q_t,snr_t,ids_t,uep_budget,cost)
        learned.predicted_risk = float(risk[np.arange(l),learned.levels].sum())
        policies['learned_policy']=(learned,ueph)
    return policies


def toy_data(length=16, nbits=8, seed=0):
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,2**nbits,size=length)
    # Heterogeneous local sensitivity with correlation to codeword geometry.
    significance=np.exp(rng.normal(0,0.8,length))
    bitweights=np.exp(rng.normal(0,0.7,(length,nbits)))
    ks=np.arange(2**nbits)
    delta=np.zeros((length,2**nbits))
    for i in range(length):
        diff=np.bitwise_xor(indices[i],ks)
        for b in range(nbits):
            delta[i]+=significance[i]*bitweights[i,b]*((diff>>b)&1)
        # Synergy for two or more bit flips, demonstrating 1-bit truncation risk.
        hd=np.array([int(v).bit_count() for v in diff])
        delta[i]+=0.05*significance[i]*np.maximum(hd-1,0)**2
    q=np.array([sum(delta[i,indices[i]^(1<<bit)] for bit in range(nbits))
                for i in range(length)])
    q=q/q.sum()
    return indices,delta,q


def run_toy(args):
    rows=[]
    policy=load_policy(args.policy_checkpoint) if args.policy_checkpoint else None
    nbits=8
    for snr in args.snrs:
        ids,delta,q=toy_data(args.length,nbits,args.seed)
        # Different link qualities across token positions (known to scheduler).
        gammas=np.clip(snr + np.linspace(-4,4,args.length),-10,30)
        risk=risk_from_delta(ids,delta,gammas,args.levels)
        budget=int(round(args.budget_per_token*args.length))
        policies=choose_allocations(q,risk,budget,args.levels,nbits,args.header_repeat,policy,gammas,ids,args.group_size)
        for name,(alloc,hcost) in policies.items():
            samples=[]
            ser=[]
            fails=0
            undetected=0
            for trial in range(args.trials):
                # Header uses an independent noise stream. Do not assume the
                # receiver knows the allocation without receiving its header.
                hdr_rng=np.random.default_rng(args.seed+100003*trial+77117)
                if hcost and args.simulate_header:
                    correct, crc_passed, decoded_actions=transmit_profile(
                        alloc.levels[::args.group_size],len(args.levels),args.header_repeat,
                        float(np.min(gammas)),hdr_rng)
                    if not correct:
                        fails+=1
                        undetected+=int(crc_passed)
                        continue  # No payload interpretation on header failure.
                received=transmit_indices(ids,np.array(args.levels)[alloc.levels],gammas,nbits,
                                           np.random.default_rng(args.seed+100003*trial))
                samples.append(evaluate_additive_risk(ids,received,delta))
                ser.append((received!=ids).mean())
            rows.append({'mode':'toy','snr_db':snr,'policy':name,'n_sentences':args.trials,
                         'predicted_risk':float(alloc.predicted_risk),
                         'measured_delta_ce':None,'measured_additive_risk':float(np.mean(samples)) if samples else None,
                         'index_error_rate':float(np.mean(ser)) if ser else None,
                         'header_frame_error_rate':fails/args.trials,
                         'header_crc_undetected_rate':undetected/args.trials,
                         'successful_trials':len(samples),
                         'budget':budget,'data_symbols':alloc.used_data_symbols,
                         'header_symbols':hcost,'pilot_symbols':0,
                         'total_symbols':alloc.used_data_symbols+hcost,
                         'levels':np.asarray(args.levels)[alloc.levels].tolist()})
    return rows


def load_policy(path):
    from semgain.policy import AllocationPolicy
    ck=torch.load(path,map_location='cpu',weights_only=True)
    p=AllocationPolicy(ck['n_actions'])
    p.load_state_dict(ck['model']);p.eval()
    return p


def load_predictor(path,device):
    from semgain.importance import PositionContextImportance
    ck=torch.load(path,map_location='cpu',weights_only=True)
    p=PositionContextImportance(ck['d_model'],ck['max_len']).to(device)
    p.load_state_dict(ck['model']);p.eval()
    return p


def run_deepsc(args):
    policy=load_policy(args.policy_checkpoint) if args.policy_checkpoint else None
    from semgain.deepsc_bridge import load_data_and_bridge,delta_table
    from semgain.vq import VectorCodebook
    from semgain.importance import teacher_importance
    train,test,vocab,bridge=load_data_and_bridge(args.ckpt,args.data,args.device)
    ck=torch.load(args.codebook,map_location='cpu',weights_only=True)
    validate_artifact(ck,vocab,bridge)
    codebook=VectorCodebook(ck['codewords']).to(args.device)
    nbits=codebook.nbits
    from semgain.phy import validate_levels
    validate_levels(args.levels,nbits)
    if args.importance=='learned' and not args.importance_model:
        raise ValueError('--importance learned requires --importance-model')
    pmodel=load_predictor(args.importance_model,args.device) if args.importance_model else None
    weights=None
    if args.risk_source=='proxy':
        if not args.risk_model:
            raise ValueError('--risk-source proxy requires --risk-model trained on disjoint calibration set')
        with open(args.risk_model) as f:
            weights=np.asarray(json.load(f)['weights'],dtype=float)
    rows=[]
    for s_ix,s in enumerate(test[:args.max_sents]):
        src=torch.tensor(vocab.encode(s),dtype=torch.long,device=args.device)[None]
        if pmodel is not None and src.shape[1]>pmodel.max_len:
            continue
        z,_=bridge.encode(src)
        ids=codebook.encode(z)[0]
        clean_z=codebook.decode(ids)[None]
        with torch.no_grad():
            clean_ce=float(bridge.sentence_ce(clean_z,src)[0])
            analog_ce=float(bridge.sentence_ce(z,src)[0])
        exact_delta=None
        # During evaluation, do NOT form oracle labels unless requested explicitly.
        if args.risk_source=='oracle' or args.importance=='oracle':
            exact_delta,check_base=delta_table(bridge,src,ids,codebook,args.delta_mode,args.eval_batch)
            assert abs(check_base-clean_ce)<1e-3
        if args.importance=='oracle':
            q=teacher_importance(exact_delta,ids.cpu(),nbits).numpy()
        elif args.importance=='learned':
            with torch.no_grad():
                m=bridge.semantic_features(src)
                q=pmodel(m,src.ne(vocab.pad))[0].cpu().numpy()
        else:
            q=np.ones(len(ids),dtype=float)/len(ids)
        ids_np=ids.cpu().numpy()
        for snr in args.snrs:
            gammas=np.full(len(ids_np),snr,dtype=float)
            if args.risk_source=='oracle':
                delta=exact_delta.numpy()
            else:
                delta=predict_delta(ids_np,codebook.weight.cpu().numpy(),q,weights)
            risk=risk_from_delta(ids_np,delta,gammas,args.levels)
            budget=int(round(args.budget_per_token*len(ids_np)))+args.pilot_symbols
            policies=choose_allocations(q,risk,budget-args.pilot_symbols,
                                        args.levels,nbits,args.header_repeat,policy,gammas,ids_np,args.group_size)
            for name,(alloc,hcost) in policies.items():
                ces=[];iers=[];fails=0;undetected=0
                for trial in range(args.trials):
                    if hcost and args.simulate_header:
                        correct,crc_passed,decoded_actions=transmit_profile(
                            alloc.levels[::args.group_size],len(args.levels),args.header_repeat,
                            float(np.min(gammas)),np.random.default_rng(args.seed+100003*trial+s_ix*10007+77117))
                        if not correct:
                            fails+=1;undetected+=int(crc_passed)
                            continue
                    received=transmit_indices(ids_np,np.array(args.levels)[alloc.levels],
                                              gammas,nbits,np.random.default_rng(args.seed+100003*trial+s_ix*10007))
                    with torch.no_grad():
                        rz=codebook.decode(torch.as_tensor(received,device=args.device))[None]
                        ce=float(bridge.sentence_ce(rz,src)[0])
                    ces.append(ce-clean_ce)
                    iers.append(float(np.mean(received!=ids_np)))
                rows.append({'mode':'deepsc','sample_idx':s_ix,'snr_db':snr,'policy':name,
                             'n_sentences':1,'predicted_risk':float(alloc.predicted_risk),
                             'clean_vq_ce':clean_ce,'clean_analog_ce':analog_ce,
                             'vq_ce_gap':clean_ce-analog_ce,
                             'measured_delta_ce':float(np.mean(ces)) if ces else None,
                             'index_error_rate':float(np.mean(iers)) if iers else None,
                             'header_frame_error_rate':fails/args.trials,
                             'header_crc_undetected_rate':undetected/args.trials,
                             'successful_trials':len(ces),
                             'budget':budget,'data_symbols':alloc.used_data_symbols,
                             'header_symbols':hcost,'pilot_symbols':args.pilot_symbols,
                             'total_symbols':alloc.used_data_symbols+hcost+args.pilot_symbols,
                             'levels':np.asarray(args.levels)[alloc.levels].tolist(),
                             'risk_source':args.risk_source,'importance_source':args.importance,
                             'delta_mode':args.delta_mode})
                assert rows[-1]['total_symbols']<=budget
        print(f'evaluated test {s_ix+1}, L={len(ids_np)}',flush=True)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['toy','deepsc'],default='toy')
    ap.add_argument('--ckpt')
    ap.add_argument('--codebook')
    ap.add_argument('--data',default='data/europarl_en.json')
    ap.add_argument('--importance',choices=['oracle','learned','uniform'],default='uniform')
    ap.add_argument('--importance-model')
    ap.add_argument('--risk-source',choices=['oracle','proxy'],default='proxy')
    ap.add_argument('--risk-model')
    ap.add_argument('--policy-checkpoint', default=None)
    ap.add_argument('--delta-mode',choices=['onebit','all'],default='onebit')
    ap.add_argument('--eval-batch',type=int,default=16)
    ap.add_argument('--snrs',default='0,6,12')
    ap.add_argument('--levels',default='1,3,5')
    ap.add_argument('--budget-per-token',type=float,default=14)
    ap.add_argument('--header-repeat',type=int,default=3)
    ap.add_argument('--group-size',type=int,default=4,help='Tokens per shared UEP action; reduces header overhead (default 4)')
    ap.add_argument('--simulate-header',action=argparse.BooleanOptionalAction,default=True,help='Actually send CRC-8 protected profile over QPSK (default true)')
    ap.add_argument('--pilot-symbols',type=int,default=0)
    ap.add_argument('--length',type=int,default=16)
    ap.add_argument('--max-sents',type=int,default=3)
    ap.add_argument('--trials',type=int,default=30)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--out',default='results/semgain.json')
    args=ap.parse_args()
    args.snrs=[float(x) for x in args.snrs.split(',')]
    args.levels=tuple(int(x) for x in args.levels.split(','))
    from semgain.phy import validate_levels
    validate_levels(args.levels,8)  # toy uses 8 bits; DeepSC validated again after loading its VQ
    if args.group_size<1:
        raise ValueError("--group-size must be >= 1")
    if args.header_repeat%2!=1 or args.header_repeat<1:
        raise ValueError('--header-repeat must be a positive odd integer')
    if args.mode=='deepsc' and (not args.ckpt or not args.codebook):
        raise ValueError('Need both --ckpt and --codebook in DeepSC mode')
    if args.mode=='deepsc' and args.importance=='uniform' and args.risk_source=='proxy' and not args.risk_model:
        raise ValueError('DeepSC mode requires --risk-model, or explicit --risk-source oracle (diagnostic only)')
    rows=run_toy(args) if args.mode=='toy' else run_deepsc(args)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    with open(args.out,'w') as f:json.dump({'config':vars(args),'rows':rows},f,indent=2)
    with open(Path(args.out).with_suffix('.csv'),'w',newline='') as f:
        fields=sorted(set().union(*(r.keys() for r in rows)))
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    for r in rows:
        val=r['measured_additive_risk'] if args.mode=='toy' else r['measured_delta_ce']
        empirical = 'NA' if val is None else f'{val:.4f}'
        idxerr = 'NA' if r['index_error_rate'] is None else f"{r['index_error_rate']:.4f}"
        print(f"SNR={r['snr_db']:5.1f}, {r['policy']:18s}, pred={r['predicted_risk']:.4f}, "
              f"empirical={empirical}, idxERR={idxerr}, headerFER={r['header_frame_error_rate']:.4f}, "
              f"uses={r['total_symbols']}/{r['budget']}")
    print('Saved:',args.out)

if __name__=='__main__':main()
