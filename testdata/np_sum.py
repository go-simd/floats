"""Reference side of the SumPairwise fixtures: np.sum of float64 arrays.

Run by TestRegenerateSumPairwiseFixtures (SUMPW_REGEN=1) through
`uv run --with numpy==<version> python -I np_sum.py IN OUT`.

IN  : repeated records  u64 n (little endian), then n float64 values.
OUT : one u64 per record, the bit pattern of np.sum(values).
"""
import sys

import numpy as np

src, dst = sys.argv[1], sys.argv[2]
buf = np.fromfile(src, dtype="<u8")
out = []
pos = 0
while pos < len(buf):
    n = int(buf[pos])
    vals = buf[pos + 1 : pos + 1 + n].view("<f8")
    out.append(np.sum(vals).view("<u8"))
    pos += 1 + n
np.array(out, dtype="<u8").tofile(dst)
print("numpy", np.__version__, "records", len(out))
