import numpy as np
from src.core.schema import QuantizedEmbedding


class Quantizer:
    """
    Handles 4-bit quantization and dequantization of embedding vectors.
    """

    @staticmethod
    def quantize(vector: np.ndarray, bits: int = 4) -> QuantizedEmbedding:
        """
        Quantizes a float32 vector to 4-bit or 8-bit integers.
        """
        if bits not in [4, 8]:
            raise NotImplementedError(
                f"Only 4-bit and 8-bit quantization are supported. Got {bits}."
            )

        # Ensure vector is float32
        vector = vector.astype(np.float32)

        min_val = float(np.min(vector))
        max_val = float(np.max(vector))

        # Avoid division by zero
        if max_val == min_val:
            scale = 1.0
        else:
            scale = (max_val - min_val) / ((1 << bits) - 1)

        # Quantize: round((x - min) / scale)
        q_vals = np.round((vector - min_val) / scale).astype(np.uint8)

        # Clip to ensure values are within range
        q_vals = np.clip(q_vals, 0, (1 << bits) - 1)

        if bits == 4:
            # Pack (2 values per byte)
            # Ensure even length for packing
            original_len = len(q_vals)
            if original_len % 2 != 0:
                q_vals = np.append(q_vals, 0)  # Pad with 0

            # Pack: high nibble is even index, low nibble is odd index
            # byte = (val[0] << 4) | val[1]
            packed = (q_vals[0::2] << 4) | q_vals[1::2]
            data = packed.tobytes()
        else:  # bits == 8
            # No packing needed, just bytes
            data = q_vals.tobytes()

        return QuantizedEmbedding(
            data=data, scale=scale, min_val=min_val, bits=bits, shape=list(vector.shape)
        )

    @staticmethod
    def dequantize(q_emb: QuantizedEmbedding) -> np.ndarray:
        """
        Dequantizes a QuantizedEmbedding back to float32 vector.
        """
        if q_emb.bits not in [4, 8]:
            raise NotImplementedError(
                f"Only 4-bit and 8-bit quantization are supported. Got {q_emb.bits}."
            )

        packed = np.frombuffer(q_emb.data, dtype=np.uint8)

        if q_emb.bits == 4:
            # Unpack
            high = packed >> 4
            low = packed & 0x0F

            q_vals = np.empty(len(high) * 2, dtype=np.uint8)
            q_vals[0::2] = high
            q_vals[1::2] = low

            # Trim padding if necessary
            original_len = int(np.prod(q_emb.shape))
            q_vals = q_vals[:original_len]
        else:  # bits == 8
            q_vals = packed

        # Dequantize
        # x = x_q * S + x_min
        vector = q_vals.astype(np.float32) * q_emb.scale + q_emb.min_val

        return vector
