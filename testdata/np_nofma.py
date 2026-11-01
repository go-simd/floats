"""Reference side of the NoFMA fixtures: NumPy evaluates the same expressions.

Run by TestRegenerateNoFMAFixtures (NOFMA_REGEN=1) through
`uv run --with numpy==<version> python -I np_nofma.py IN OUT`
(or the interpreter named by NUMPY_PYTHON).

IN  : repeated records of u64 words:
        op, n, s0, s1, s2 (scalars as float64 bit patterns),
        then k arrays of n float64 (k = 2, 2, 3, 3 for op 0..3),
        then a mask of ceil(n/8) words holding one byte per element
        (1: a NaN in this element's result must match bit for bit,
         0: any NaN will do).
OUT : one u64 per record: the first 8 bytes (little endian) of the SHA-256 of the
      result bit patterns, every non-strict NaN replaced by 0x7ff8000000000000.

op 0: y + alpha*x            arrays x, y        scalar alpha
op 1: x + beta*y             arrays x, y        scalar beta
op 2: alpha*a + beta*b + gamma*c   arrays a, b, c   scalars alpha, beta, gamma
op 3: a + F*(b - c)          arrays a, b, c     scalar F
"""
import hashlib
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
buf = np.fromfile(src, dtype="<u8")
out = []
pos = 0
arrays_of = {0: 2, 1: 2, 2: 3, 3: 3}
with np.errstate(all="ignore"):
    while pos < len(buf):
        op, n = int(buf[pos]), int(buf[pos + 1])
        s = buf[pos + 2 : pos + 5].view("<f8")
        pos += 5
        arr = []
        for _ in range(arrays_of[op]):
            arr.append(buf[pos : pos + n].view("<f8"))
            pos += n
        words = (n + 7) // 8
        strict = buf[pos : pos + words].view("u1")[:n].astype(bool)
        pos += words
        if op == 0:
            x, y = arr
            r = y + s[0] * x
        elif op == 1:
            x, y = arr
            r = x + s[0] * y
        elif op == 2:
            a, b, c = arr
            r = s[0] * a + s[1] * b + s[2] * c
        else:
            a, b, c = arr
            r = a + s[0] * (b - c)
        bits = np.ascontiguousarray(r, dtype="<f8").view("<u8").copy()
        bits[np.isnan(r) & ~strict] = 0x7FF8000000000000
        out.append(int.from_bytes(hashlib.sha256(bits.tobytes()).digest()[:8], "little"))
np.array(out, dtype="<u8").tofile(dst)
print("numpy", np.__version__, "records", len(out))
