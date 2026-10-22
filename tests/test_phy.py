import numpy as np
from semgain.phy import (confusion_matrix, repeated_ber, qpsk_ber,
                         transmit_indices, symbol_cost,header_cost,index_bits,bits_index)


def test_bits_roundtrip():
    ids=np.array([0,1,17,255])
    np.testing.assert_array_equal(bits_index(index_bits(ids,8)),ids)


def test_confusion_normalized():
    t=confusion_matrix(8,0.1)
    np.testing.assert_allclose(t.sum(1),1,atol=1e-12)
    assert t.shape==(256,256)
    assert np.all(t.diagonal() > 0)


def test_qpsk_and_repetition_ber():
    assert abs(float(qpsk_ber(0))-0.1586552539)<1e-6
    assert float(repeated_ber(0,3))<float(qpsk_ber(0))
    assert float(repeated_ber(0,5))<float(repeated_ber(0,3))


def test_qpsk_empirical_bit_ber():
    rng=np.random.default_rng(13)
    ids=rng.integers(0,256,size=4000)
    rx=transmit_indices(ids,np.ones(len(ids),int),0.0,8,np.random.default_rng(3))
    ber=np.not_equal(index_bits(ids,8),index_bits(rx,8)).mean()
    assert abs(ber-float(qpsk_ber(0)))<0.018


def test_overhead_and_cost():
    assert symbol_cost(1,8)==4
    assert symbol_cost(3,8)==12
    assert symbol_cost(5,8)==20
    assert header_cost(10,3,3)==42
