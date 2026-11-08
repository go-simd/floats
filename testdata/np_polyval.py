"""Reference side of the Polyval/Chebval fixtures: NumPy's own polyval and chebval.

Run by TestRegeneratePolyvalFixtures (POLYVAL_REGEN=1) through
`uv run --no-project --with numpy==<version> python -I np_polyval.py IN OUT`, or
in a linux/amd64 container (POLYVAL_DOCKER=1), or with the interpreter named by
NUMPY_PYTHON.

IN  : repeated records of u64 words: op, nc, nx, then nc coefficients and nx
      points (float64 bit patterns). op 0: numpy.polynomial.polynomial.polyval,
      op 1: numpy.polynomial.chebyshev.chebval.
OUT : for every record, the nx result bit patterns, concatenated.
"""
import sys

import numpy as np
from numpy.polynomial.chebyshev import chebval
from numpy.polynomial.polynomial import polyval

src, dst = sys.argv[1], sys.argv[2]
buf = np.fromfile(src, dtype="<u8")
out = []
pos = 0
with np.errstate(all="ignore"):
    while pos < len(buf):
        op, nc, nx = int(buf[pos]), int(buf[pos + 1]), int(buf[pos + 2])
        pos += 3
        c = buf[pos : pos + nc].view("<f8")
        pos += nc
        x = buf[pos : pos + nx].view("<f8")
        pos += nx
        r = polyval(x, c, tensor=False) if op == 0 else chebval(x, c, tensor=False)
        r = np.ascontiguousarray(np.broadcast_to(r, (nx,)), dtype="<f8")
        out.append(r.view("<u8"))
np.concatenate(out).tofile(dst)
print("numpy", np.__version__, "records", len(out), "platform", sys.platform)
