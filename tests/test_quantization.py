import numpy as np
from src.core.quantization import Quantizer


def test_quantization_roundtrip():
    vec = np.random.rand(768).astype(np.float32)
    q_emb = Quantizer.quantize(vec, bits=4)

    assert q_emb.bits == 4
    assert len(q_emb.data) == 768 // 2

    recon = Quantizer.dequantize(q_emb)
    assert recon.shape == vec.shape

    # Check error is within bounds
    # 4-bit quantization has 16 levels.
    # Max error is roughly range / 30
    max_err = np.max(np.abs(vec - recon))
    print(f"Max reconstruction error: {max_err}")
    # This threshold depends on the range of values, for [0, 1] it should be around 1/15 = 0.06
    assert max_err < 0.1


def test_quantization_zeros():
    vec = np.zeros(768).astype(np.float32)
    q_emb = Quantizer.quantize(vec, bits=4)
    recon = Quantizer.dequantize(q_emb)
    assert np.allclose(vec, recon, atol=1e-5)
