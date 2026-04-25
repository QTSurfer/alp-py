"""Constants for the ALP (Adaptive Lossless floating-Point) algorithm.

Reference: Afroozeh & Boncz, *"ALP: Adaptive Lossless floating-Point
Compression"*, ACM SIGMOD 2024 (DOI: 10.1145/3626717).

Mirrors :class:`com.wualabs.qtsurfer.alp.AlpConstants` byte-for-byte —
any change here affects the wire format and must be made in lock-step
with ``alp-java``.
"""

from __future__ import annotations

# Vector size for ALP encoding (matches DuckDB / columnar convention).
VECTOR_SIZE = 1024

# Maximum exponent e (multiply by 10^e to convert to integer).
MAX_EXPONENT = 18

# Maximum factor f (divide by 10^f to keep within 64-bit range).
MAX_FACTOR = 10

# Maximum fraction of values allowed as exceptions (currently informational —
# AlpCompressor in Java doesn't actually fall back to ALP-RD/raw when this
# is exceeded; it just stores all the exceptions). Kept here for parity.
MAX_EXCEPTION_FRACTION = 0.05

# Powers of 10 as Python ints (POW10_LONG in Java).
POW10_LONG: tuple[int, ...] = tuple(10**i for i in range(MAX_EXPONENT + 1))

# Powers of 10 as floats (POW10_DOUBLE in Java).
POW10_DOUBLE: tuple[float, ...] = tuple(float(p) for p in POW10_LONG)

# Reciprocals of powers of 10 (1/10^f) for the factor range.
INV_POW10: tuple[float, ...] = tuple(
    1.0 / POW10_DOUBLE[i] if i > 0 else 1.0
    for i in range(MAX_FACTOR + 1)
)

# Magic number for branchless rounding (= 2^51 + 2^52).
ENCODE_MAGIC = 6755399441055744.0
