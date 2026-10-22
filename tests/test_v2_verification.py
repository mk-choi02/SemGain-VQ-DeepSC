"""Numerical invariants and protocol-accounting regression tests."""
import itertools
import numpy as np
import pytest
import torch
from semgain.phy import confusion_matrix,repeated_ber,transmit_indices,header_cost
from semgain.risk import risk_from_delta
from semgain.allocator import dp_min_risk
from semgain.header import profile_bits,crc8,transmit_profile
from semgain.validation import vocabulary_fingerprint, encoder_fingerprint, validate_artifact
from scripts.run_semgain import choose_allocations
from tests.test_bridge import MiniDeepSC
from semgain.deepsc_bridge import DeepSCBridge


def test_risk_matches_brute_confusion_matrix():
    ids=np.array([0,9,14]); delta=np.arange(16*3,dtype=float).reshape(3,16)/100
    snr=np.array([0.,5.,12.]); levels=(1,3,5)
    actual=risk_from_delta(ids,delta,snr,levels)
    expected=np.array([[confusion_matrix(4,float(repeated_ber(s,rep)))[ids[i]]@delta[i]
                        for rep in levels] for i,s in enumerate(snr)])
    assert np.allclose(actual,expected,rtol=1e-12,atol=1e-12)


def test_mc_ber_matches_analytic_model():
    rng=np.random.default_rng(11); indices=rng.integers(0,16,5000)
    for repeat in [1,3]:
        received=transmit_indices(indices,np.full(len(indices),repeat),4.,4,rng)
        errors=(indices[:,None]>>np.arange(4) & 1)!=(received[:,None]>>np.arange(4)&1)
        p=float(repeated_ber(4,repeat))
        assert abs(errors.mean()-p)<0.012


def test_dp_optimality_vs_enumeration():
    rng=np.random.default_rng(0)
    risk=rng.normal(size=(4,3)); costs=np.array([[2,4,7],[3,5,8],[3,6,9],[2,5,7]])
    budget=19
    opt=dp_min_risk(risk,costs,budget)
    alternatives=[(sum(risk[i,a] for i,a in enumerate(z)),z)
                  for z in itertools.product(range(3),repeat=4)
                  if sum(costs[i,a] for i,a in enumerate(z))<=budget]
    assert abs(opt.predicted_risk-min(v for v,_ in alternatives))<1e-10
    assert opt.used_data_symbols<=budget


def test_transmitted_header_and_crc():
    actions=np.array([0,2,1,0,2,1]); src=profile_bits(actions,3)
    assert len(src)==2*len(actions)
    assert len(crc8(src))==8
    assert header_cost(6,3,5)==50
    correct,crc_passed,recovered=transmit_profile(actions,3,5,60.,np.random.default_rng(42))
    assert correct and crc_passed and np.array_equal(recovered,actions)
    rng=np.random.default_rng(21)
    trials=[transmit_profile(actions,3,1,-10.,rng)[0] for _ in range(60)]
    assert sum(trials)<5


def test_grouping_has_correct_airtime_and_same_constraints():
    l=11; risk=np.array([[1.5,.4,.1]]*l); q=np.full(l,1/l)
    choices=choose_allocations(q,risk,168,(1,3,5),8,3,group_size=4)
    expected=header_cost(3,3,3)
    for name,(alloc,header) in choices.items():
        assert alloc.used_data_symbols+header<=168
        if name!='uniform':
            assert header==expected
            for i in (0,4,8):
                assert len(set(alloc.levels[i:i+4]))==1


def test_vocabulary_and_encoder_fingerprints_are_checked():
    class Vocab:
        itos=['<PAD>','<START>','<END>','<UNK>','a','b','c','d','e','f','g','h','i']
        def __len__(self): return len(self.itos)
    model=MiniDeepSC()
    bridge=DeepSCBridge(model)
    record={'codewords':torch.randn(16,4),'vocab_sha256':vocabulary_fingerprint(Vocab()),
            'encoder_sha256':encoder_fingerprint(model)}
    validate_artifact(record,Vocab(),bridge)
    bad={**record,'vocab_sha256':'0'*64}
    with pytest.raises(ValueError,match='Vocabulary'):
        validate_artifact(bad,Vocab(),bridge)
