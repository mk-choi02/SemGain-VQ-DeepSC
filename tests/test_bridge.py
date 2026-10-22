import torch
from torch import nn
from semgain.deepsc_bridge import DeepSCBridge,delta_table
from semgain.vq import VectorCodebook


class MiniDeepSC(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_sym=2;self.pad_idx=0;self.d_model=8
        self.csi_rx=False;self.mapper=None;self.embedding=nn.Embedding(13,8,padding_idx=0)
        self.channel_encoder=nn.Linear(8,4)
        self.channel_decoder=nn.Sequential(nn.Linear(4,8),nn.ReLU())
        self.prediction=nn.Linear(8,13)
        self.pos=nn.Identity()
        self.semantic_encoder=nn.Identity()

    def transmit(self,src):
        m=self.embedding(src)
        z=self.channel_encoder(m)
        x=torch.complex(z[...,0::2],z[...,1::2]).reshape(src.shape[0],-1)
        return x,src.ne(0).repeat_interleave(2,dim=1)

    def receive(self,x,pad):
        z=torch.stack([x.real,x.imag],-1).reshape(x.shape[0],-1,4)
        return self.channel_decoder(z)

    def decode_step(self,memory,pad,tgt):
        return self.prediction(memory[:,:tgt.shape[1]])


def test_bridge_teacher_onebit():
    torch.manual_seed(7)
    bridge=DeepSCBridge(MiniDeepSC())
    src=torch.tensor([[1,2,3,4]])
    z,mask=bridge.encode(src)
    assert z.shape==(1,4,4)
    cb=VectorCodebook(torch.randn(16,4))
    ids=cb.encode(z)[0]
    delta,base=delta_table(bridge,src,ids,cb,'onebit',batch_size=4)
    assert delta.shape==(4,16)
    for i,k in enumerate(ids):
        assert abs(delta[i,k])<1e-6
    assert torch.isfinite(delta).all()
