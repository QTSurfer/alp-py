"""ALP per-value encoder/decoder + parameter search.

Mirror of :class:`com.wualabs.qtsurfer.alp.AlpEncoder`.
"""

from __future__ import annotations

import struct

import numpy as np

from ._constants import (
    ENCODE_MAGIC,
    INV_POW10,
    MAX_EXPONENT,
    MAX_FACTOR,
    POW10_DOUBLE,
)


def _bits(value: float) -> int:
    """Return the raw IEEE 754 bits of a float64 as an unsigned 64-bit int."""
    return int.from_bytes(struct.pack("<d", value), "little", signed=False)


def encode_value(value: float, e: int, f: int) -> int:
    """Encode one double via ``round(value * 10^e / 10^f)``.

    Uses the same branchless ``+ MAGIC - MAGIC`` rounding trick as Java's
    Math (effectively round-half-to-even on a typical x86 build with
    default rounding mode). The Python ``int(...)`` truncation here
    matches Java's ``(long) (...)`` cast behaviour after the magic add.
    """
    scaled = value * POW10_DOUBLE[e] * INV_POW10[f]
    rounded = scaled + ENCODE_MAGIC - ENCODE_MAGIC
    # Java casts a double to long via truncation, which agrees with
    # the rounded value here as long as it fits in [-2^63, 2^63 - 1].
    return int(rounded)


def decode_value(encoded: int, e: int, f: int) -> float:
    return float(encoded) * POW10_DOUBLE[f] / POW10_DOUBLE[e]


def find_best_exponents(values: np.ndarray, count: int, sample_size: int) -> tuple[int, int]:
    """Pick the (e, f) pair with the fewest round-trip exceptions over a sample."""
    n = min(count, sample_size)
    best_e = 0
    best_f = 0
    best_exceptions = (1 << 31) - 1  # Integer.MAX_VALUE
    for e in range(MAX_EXPONENT + 1):
        for f in range(min(e, MAX_FACTOR) + 1):
            exceptions = 0
            for i in range(n):
                v = float(values[i])
                enc = encode_value(v, e, f)
                dec = decode_value(enc, e, f)
                if _bits(dec) != _bits(v):
                    exceptions += 1
            if exceptions < best_exceptions:
                best_exceptions = exceptions
                best_e = e
                best_f = f
                if exceptions == 0:
                    return best_e, best_f
    return best_e, best_f


def encode_vector(
    values: np.ndarray,
    count: int,
    e: int,
    f: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode ``count`` values with parameters ``(e, f)``.

    Returns ``(encoded, exception_indices, exception_value_bits)``:

    - ``encoded``: int64 array of length ``count`` with the FOR-encodable
      integer for every position (including exception positions).
    - ``exception_indices``: int array of positions that didn't round-trip.
    - ``exception_value_bits``: int64 array of the original IEEE 754 bits
      for those exception positions, in the same order.
    """
    encoded = np.empty(count, dtype=np.int64)
    indices: list[int] = []
    bits: list[int] = []
    for i in range(count):
        v = float(values[i])
        enc = encode_value(v, e, f)
        encoded[i] = enc
        if _bits(decode_value(enc, e, f)) != _bits(v):
            indices.append(i)
            # Store the raw IEEE bits as signed int64 so numpy can hold them.
            unsigned = _bits(v)
            signed = unsigned - (1 << 64) if unsigned >= (1 << 63) else unsigned
            bits.append(signed)
    return (
        encoded,
        np.asarray(indices, dtype=np.int32),
        np.asarray(bits, dtype=np.int64),
    )
