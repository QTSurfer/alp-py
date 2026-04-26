# alp-py

[![CI](https://github.com/QTSurfer/alp-py/actions/workflows/ci.yml/badge.svg)](https://github.com/QTSurfer/alp-py/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/alp-codec)](https://pypi.org/project/alp-codec/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/QTSurfer/alp-py/blob/main/LICENSE)

Pure Python implementation of the **ALP** (Adaptive Lossless floating-Point) compression algorithm.

Based on: Afroozeh & Boncz, *"ALP: Adaptive Lossless floating-Point Compression"*, ACM SIGMOD 2024.
([paper](https://dl.acm.org/doi/10.1145/3626717) · [pdf](https://ir.cwi.nl/pub/33334/33334.pdf))

Python port of the [QTSurfer/alp-java](https://github.com/QTSurfer/alp-java) reference implementation. Bit-exact byte-level compatibility with `alp-java` — files written by either implementation can be read by the other.

## Status

`0.8.0` — first published release, bit-exact aligned with `alp-java` 0.2.1. Encode + decode round-trip verified against fixtures emitted by the Java reference.

## Install

```bash
pip install alp-codec
```

## Usage

```python
import numpy as np
from alp import encode, decode

values = np.array([65007.28, 65007.31, 65007.30, 65007.32], dtype=np.float64)

# Encode to bytes
blob = encode(values)

# Decode back to a NumPy float64 array
recovered = decode(blob)
assert np.array_equal(values, recovered)
```

## Why ALP

ALP applies **semantic compression** to decimal floating-point arrays. For a typical financial price column at 2 decimal places it reaches **3-4 bits per value** — about an order of magnitude better than Gorilla XOR or block-compressed PLAIN encodings.

The algorithm picks an `(e, f)` pair per block such that `value × 10^e × 2^-f` is exactly representable as a small `int64`, encodes the integers via Frame-Of-Reference + bit-packing, and stores any non-conforming values as outliers with a position list.

## Reference implementations

| Language | Repo | Status |
|----------|------|--------|
| Java | [QTSurfer/alp-java](https://github.com/QTSurfer/alp-java) | Reference |
| Python | [QTSurfer/alp-py](https://github.com/QTSurfer/alp-py) | This repo |

## License

Copyright 2026 Wualabs LTD. Apache License 2.0 — see [LICENSE](https://github.com/QTSurfer/alp-py/blob/main/LICENSE).
