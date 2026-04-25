"""Fixed-width bit-packing with Frame-Of-Reference subtraction.

Mirrors :class:`com.wualabs.qtsurfer.alp.BitPacking`. The Java side packs
deltas as little-endian within each window of 8 bytes; this module
reproduces that layout byte-for-byte.
"""

from __future__ import annotations

import numpy as np

_GROUP_SIZE = 32
_MASK64 = (1 << 64) - 1


def find_frame(values: np.ndarray, count: int) -> int:
    """Return the minimum integer in ``values[:count]`` (Frame Of Reference base)."""
    if count == 0:
        return 0
    return int(values[:count].min())


def compute_bit_width(values: np.ndarray, count: int, frame: int) -> int:
    """Bits needed to represent ``max(values) - frame`` (0 if all values are equal)."""
    if count == 0:
        return 0
    deltas = np.asarray(values[:count], dtype=np.int64) - np.int64(frame)
    max_delta = int(deltas.max())
    if max_delta == 0:
        return 0
    return max_delta.bit_length()


def pack(values: np.ndarray, count: int, frame: int, bit_width: int) -> bytes:
    """Pack ``values - frame`` into a tightly-packed little-endian bit stream."""
    if bit_width == 0:
        return b""

    total_bits = count * bit_width
    total_bytes = (total_bits + 7) // 8
    buf = bytearray(total_bytes)

    mask = _MASK64 if bit_width == 64 else (1 << bit_width) - 1
    bit_pos = 0
    for i in range(count):
        delta = (int(values[i]) - frame) & mask
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        # Write up to 9 bytes so any bitWidth ≤ 56 fits in one shifted word.
        shifted = delta << bit_offset
        bytes_needed = (bit_offset + bit_width + 7) // 8
        for b in range(bytes_needed):
            if byte_idx + b >= len(buf):
                break
            buf[byte_idx + b] |= (shifted >> (b * 8)) & 0xFF
        bit_pos += bit_width

    return bytes(buf)


def unpack(packed: bytes, count: int, frame: int, bit_width: int) -> np.ndarray:
    """Inverse of :func:`pack`. Returns an ``int64`` numpy array."""
    out = np.empty(count, dtype=np.int64)
    if bit_width == 0:
        out.fill(frame)
        return out

    mask = _MASK64 if bit_width == 64 else (1 << bit_width) - 1
    full_groups = count // _GROUP_SIZE
    remainder = count % _GROUP_SIZE

    bit_pos = 0
    out_idx = 0
    for _ in range(full_groups):
        for i in range(_GROUP_SIZE):
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8
            bytes_avail = min(8, len(packed) - byte_idx)
            raw = 0
            for b in range(bytes_avail):
                raw |= packed[byte_idx + b] << (b * 8)
            out[out_idx + i] = frame + ((raw >> bit_offset) & mask)
            bit_pos += bit_width
        out_idx += _GROUP_SIZE

    for i in range(remainder):
        byte_idx = bit_pos // 8
        bit_offset = bit_pos % 8
        result = 0
        bits_read = 0
        while bits_read < bit_width:
            bits_to_read = min(8 - bit_offset, bit_width - bits_read)
            byte_mask = (1 << bits_to_read) - 1
            result |= ((packed[byte_idx] >> bit_offset) & byte_mask) << bits_read
            bits_read += bits_to_read
            bit_offset = 0
            byte_idx += 1
        out[out_idx + i] = frame + result
        bit_pos += bit_width

    return out
