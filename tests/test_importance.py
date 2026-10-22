import torch
from semgain.vq import fit_kmeans
from semgain.importance import PositionContextImportance,importance_loss


def test_vq_roundtrip_and_fit():
    torch.manual_seed(0)
    code=fit_kmeans(torch.randn(120,16),k=16,iters=3)
    ids=code.encode(torch.randn(2,7,16))
    assert ids.shape==(2,7)
    assert code.decode(ids).shape==(2,7,16)


def test_position_context_baseline_equals_prior():
    model=PositionContextImportance(d_model=8,max_len=10,hidden=16)
    for p in model.mlp.parameters():
        torch.nn.init.zeros_(p)
    m=torch.randn(2,6,8)
    mask=torch.tensor([[True]*6,[True]*4+[False]*2])
    q=model(m,mask)
    torch.testing.assert_close(q[0,:6],torch.full((6,),1/6))
    torch.testing.assert_close(q[1,:4],torch.full((4,),1/4))
    assert q[1,4:].sum()==0
    assert torch.isfinite(importance_loss(q,q.detach()))
