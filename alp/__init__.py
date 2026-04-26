"""ALP (Adaptive Lossless floating-Point) compression — Python port.

Public API::

    from alp import encode, decode

Both work in-memory on a numpy float64 array. The wire format matches
``com.wualabs.qtsurfer.alp.AlpCompressor`` byte for byte.

Wire format::

    [int32 BE]   total count
    For each VECTOR_SIZE-sized chunk:
        [uint8]      e (exponent)
        [uint8]      f (factor)
        [uint8]      bitWidth
        [int64 BE]   frame (FOR base)
        [uint16 BE]  exceptionCount
        [uint16 BE]  packedLength
        [packedLength bytes] bit-packed deltas (LE within bytes)
        For each exception:
            [uint16 BE] index
            [int64  BE] original IEEE 754 bits
"""

from __future__ import annotations

import io
import struct

import numpy as np

from . import _bit_packing as bp
from . import _encoder
from ._constants import VECTOR_SIZE

__version__ = "0.8.2"

_SAMPLE_SIZE = 64


def encode(values: np.ndarray | list[float]) -> bytes:
    """Compress a sequence of doubles into the ALP wire format."""
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    count = arr.size

    out = io.BytesIO()
    out.write(struct.pack(">i", count))

    offset = 0
    while offset < count:
        vector_size = min(VECTOR_SIZE, count - offset)
        chunk = arr[offset : offset + vector_size]
        e, f = _encoder.find_best_exponents(chunk, vector_size, _SAMPLE_SIZE)
        encoded, exception_indices, exception_bits = _encoder.encode_vector(
            chunk, vector_size, e, f
        )
        frame = bp.find_frame(encoded, vector_size)
        bit_width = bp.compute_bit_width(encoded, vector_size, frame)
        packed = bp.pack(encoded, vector_size, frame, bit_width)

        out.write(struct.pack(">B", e))
        out.write(struct.pack(">B", f))
        out.write(struct.pack(">B", bit_width))
        out.write(struct.pack(">q", frame))
        out.write(struct.pack(">H", len(exception_indices)))
        out.write(struct.pack(">H", len(packed)))
        out.write(packed)
        for idx, bits in zip(exception_indices, exception_bits, strict=False):
            out.write(struct.pack(">H", int(idx)))
            out.write(struct.pack(">q", int(bits)))

        offset += vector_size

    return out.getvalue()


def decode(data: bytes) -> np.ndarray:
    """Decompress ALP-encoded bytes back into a numpy float64 array."""
    if len(data) < 4:
        raise ValueError(f"truncated ALP blob: missing 4-byte count header (got {len(data)} bytes)")

    in_buf = io.BytesIO(data)
    (count,) = struct.unpack(">i", in_buf.read(4))
    out = np.empty(count, dtype=np.float64)

    offset = 0
    while offset < count:
        vector_size = min(VECTOR_SIZE, count - offset)
        header = in_buf.read(3 + 8 + 2 + 2)
        if len(header) != 15:
            raise ValueError("truncated ALP vector header")
        e = header[0]
        f = header[1]
        bit_width = header[2]
        (frame,) = struct.unpack(">q", header[3:11])
        (exception_count,) = struct.unpack(">H", header[11:13])
        (packed_length,) = struct.unpack(">H", header[13:15])

        packed = in_buf.read(packed_length)
        if len(packed) != packed_length:
            raise ValueError(
                f"truncated ALP packed body: expected {packed_length} bytes, got {len(packed)}"
            )

        encoded = bp.unpack(packed, vector_size, frame, bit_width)
        for i in range(vector_size):
            out[offset + i] = _encoder.decode_value(int(encoded[i]), e, f)

        for _ in range(exception_count):
            tail = in_buf.read(2 + 8)
            if len(tail) != 10:
                raise ValueError("truncated ALP exception entry")
            (idx,) = struct.unpack(">H", tail[:2])
            (orig_bits_signed,) = struct.unpack(">q", tail[2:10])
            unsigned = orig_bits_signed & ((1 << 64) - 1)
            out[offset + idx] = struct.unpack("<d", unsigned.to_bytes(8, "little", signed=False))[0]

        offset += vector_size

    return out
