package floats

import (
	"encoding/binary"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// sameResult: identical bits, except that any NaN equals any NaN (the payload of a
// NaN an operation creates is the hardware's, in NumPy as here).
func sameResult(a, b float64) bool {
	return math.Float64bits(a) == math.Float64bits(b) || (a != a && b != b)
}

type pvCase struct {
	op   int // 0 Polyval, 1 Chebval
	c, x []float64
}

var pvSpecials = []float64{0, math.Copysign(0, -1), 1, -1, 0.5, math.Inf(1), math.Inf(-1),
	math.NaN(), 5e-324, -5e-324, 1e308, -1e308, 2.2250738585072014e-308}

// pvCases is the deterministic population of the sweep and of the NumPy fixtures:
// both series, 1..40 coefficients, point counts around every block size (16, 8, 4,
// 2, 1 doubles), six magnitudes of point and coefficient, and every few points a
// special value (zeros of both signs, infinities, NaN, subnormals, overflow).
func pvCases() []pvCase {
	rng := rand.New(rand.NewPCG(2026, 11))
	scale := []float64{1, 10, 1e3, 1e-3, 1e150, 1e-150}
	var cs []pvCase
	i := 0
	for op := 0; op < 2; op++ {
		for _, nc := range []int{1, 2, 3, 4, 5, 8, 9, 16, 17, 40} {
			for _, nx := range []int{1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 100} {
				sx, sc := scale[i%len(scale)], scale[(i/3)%len(scale)]
				i++
				c := make([]float64, nc)
				for j := range c {
					c[j] = sc * (2*rng.Float64() - 1)
					if rng.IntN(23) == 0 {
						c[j] = pvSpecials[rng.IntN(len(pvSpecials))]
					}
				}
				x := make([]float64, nx)
				for j := range x {
					x[j] = sx * (2*rng.Float64() - 1)
					if rng.IntN(7) == 0 {
						x[j] = pvSpecials[rng.IntN(len(pvSpecials))]
					}
				}
				cs = append(cs, pvCase{op, c, x})
			}
		}
	}
	return cs
}

func pvRun(op int, dst, x, c []float64) {
	if op == 0 {
		Polyval(dst, x, c)
	} else {
		Chebval(dst, x, c)
	}
}

func pvRef(op int, dst, x, c []float64) {
	if op == 0 {
		polyvalRef(dst, x, c)
	} else {
		chebvalRef(dst, x, c)
	}
}

func TestPolyvalChebvalMatchLoop(t *testing.T) {
	for k, cs := range pvCases() {
		n := len(cs.x)
		got, want := make([]float64, n+3), make([]float64, n+3)
		for j := range got { // sentinel past len(x)
			got[j], want[j] = 7, 7
		}
		pvRun(cs.op, got, cs.x, cs.c)
		pvRef(cs.op, want[:n], cs.x, cs.c)
		for j := range got {
			if !sameResult(got[j], want[j]) {
				t.Fatalf("case %d op %d nc %d n %d: [%d] %016x vs %016x (x=%g)", k, cs.op, len(cs.c), n, j,
					math.Float64bits(got[j]), math.Float64bits(want[j]), cs.x[min(j, n-1)])
			}
		}
		// dst aliasing x exactly.
		y := append([]float64(nil), cs.x...)
		pvRun(cs.op, y, y, cs.c)
		for j := range y {
			if !sameResult(y[j], want[j]) {
				t.Fatalf("case %d op %d aliased: [%d] %016x vs %016x", k, cs.op, j, math.Float64bits(y[j]), math.Float64bits(want[j]))
			}
		}
	}
}

func TestPolyvalChebvalKnownValues(t *testing.T) {
	x := []float64{-2, 0, 0.5, 3}
	// p(t) = 1 + 2t + 3t^2: Horner on exact small integers and halves.
	got := make([]float64, len(x))
	Polyval(got, x, []float64{1, 2, 3})
	for i, v := range x {
		if want := 1 + 2*v + 3*v*v; got[i] != want {
			t.Errorf("Polyval(%g) = %g, want %g", v, got[i], want)
		}
	}
	// 1*T0 + 2*T1 + 3*T2 with T2 = 2t^2 - 1.
	Chebval(got, x, []float64{1, 2, 3})
	for i, v := range x {
		if want := 1 + 2*v + 3*(2*v*v-1); got[i] != want {
			t.Errorf("Chebval(%g) = %g, want %g", v, got[i], want)
		}
	}
	// Inf: c[-1] + x*0 is NaN for x = Inf, as in NumPy.
	r := []float64{0}
	Polyval(r, []float64{math.Inf(1)}, []float64{1})
	if !math.IsNaN(r[0]) {
		t.Errorf("Polyval(Inf, [1]) = %g, want NaN", r[0])
	}
	// Chebval of a constant series is c0 + 0*x: NaN for Inf, c0 otherwise.
	Chebval(r, []float64{math.Inf(1)}, []float64{4})
	if !math.IsNaN(r[0]) {
		t.Errorf("Chebval(Inf, [4]) = %g, want NaN", r[0])
	}
	Chebval(r, []float64{9}, []float64{4})
	if r[0] != 4 {
		t.Errorf("Chebval(9, [4]) = %g, want 4", r[0])
	}
}

func TestPolyvalChebvalPanicsAndEmpty(t *testing.T) {
	for name, f := range map[string]func(dst, x, c []float64){"Polyval": Polyval, "Chebval": Chebval} {
		f(nil, nil, []float64{1}) // no points: nothing to do
		for what, args := range map[string][3][]float64{
			"short dst": {make([]float64, 1), make([]float64, 2), {1}},
			"no coeffs": {make([]float64, 2), make([]float64, 2), nil},
		} {
			func() {
				defer func() {
					if recover() == nil {
						t.Errorf("%s %s: no panic", name, what)
					}
				}()
				f(args[0], args[1], args[2])
			}()
		}
	}
}

// ---- NumPy fixtures ----

const pvFixture = "testdata/polyval_np.bin"

// TestPolyvalChebvalNumPy compares Polyval and Chebval, and the loops, with the
// bits NumPy's polyval and chebval returned for the same inputs. The loops write
// every product through float64(...), so they are unfused; if NumPy had fused
// (arm64 wheels may), the loop and the fixture would disagree here, which is how
// the fixtures are shown FMA-free.
func TestPolyvalChebvalNumPy(t *testing.T) {
	raw, err := os.ReadFile(pvFixture)
	if err != nil {
		t.Fatal(err)
	}
	w := make([]uint64, len(raw)/8)
	for i := range w {
		w[i] = binary.LittleEndian.Uint64(raw[8*i:])
	}
	records := 0
	for pos := 0; pos < len(w); records++ {
		op, nc, nx := int(w[pos]), int(w[pos+1]), int(w[pos+2])
		pos += 3
		f := func(n int) []float64 {
			s := make([]float64, n)
			for i := range s {
				s[i] = math.Float64frombits(w[pos+i])
			}
			pos += n
			return s
		}
		c, x, want := f(nc), f(nx), f(nx)
		got, ref := make([]float64, nx), make([]float64, nx)
		pvRun(op, got, x, c)
		pvRef(op, ref, x, c)
		for j := range want {
			if !sameResult(got[j], want[j]) || !sameResult(ref[j], want[j]) {
				t.Fatalf("record %d op %d nc %d: [%d] x=%g kernel %016x loop %016x numpy %016x", records, op, nc, j, x[j],
					math.Float64bits(got[j]), math.Float64bits(ref[j]), math.Float64bits(want[j]))
			}
		}
	}
	if records != len(pvCases()) {
		t.Fatalf("fixture has %d records for %d cases; regenerate with POLYVAL_REGEN=1", records, len(pvCases()))
	}
}

// TestRegeneratePolyvalFixtures rebuilds testdata/polyval_np.bin with NumPy
// (numpyVersion). Skipped unless POLYVAL_REGEN=1. Interpreter: NUMPY_PYTHON, else
// a linux/amd64 container when POLYVAL_DOCKER=1 (the reference platform), else uv
// on this host, which the fixture must then be regenerated from in the container.
func TestRegeneratePolyvalFixtures(t *testing.T) {
	if os.Getenv("POLYVAL_REGEN") != "1" {
		t.Skip("set POLYVAL_REGEN=1 to rebuild the NumPy fixtures (NUMPY_PYTHON, POLYVAL_DOCKER=1 or uv)")
	}
	dir := t.TempDir()
	if os.Getenv("POLYVAL_DOCKER") == "1" {
		dir = filepath.Join("testdata", ".regen")
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		defer os.RemoveAll(dir)
	}
	in, out := filepath.Join(dir, "in.bin"), filepath.Join(dir, "out.bin")
	cases := pvCases()
	var buf []byte
	put := func(w uint64) { buf = binary.LittleEndian.AppendUint64(buf, w) }
	for _, c := range cases {
		put(uint64(c.op))
		put(uint64(len(c.c)))
		put(uint64(len(c.x)))
		for _, v := range c.c {
			put(math.Float64bits(v))
		}
		for _, v := range c.x {
			put(math.Float64bits(v))
		}
	}
	if err := os.WriteFile(in, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	script, err := filepath.Abs("testdata/np_polyval.py")
	if err != nil {
		t.Fatal(err)
	}
	var cmd *exec.Cmd
	switch py := os.Getenv("NUMPY_PYTHON"); {
	case py != "":
		cmd = exec.Command(py, "-I", script, in, out)
	case os.Getenv("POLYVAL_DOCKER") == "1":
		abs, err := filepath.Abs("testdata")
		if err != nil {
			t.Fatal(err)
		}
		cmd = exec.Command("docker", "run", "--rm", "--platform", "linux/amd64", "-v", abs+":/s", "python:3.13-slim", "sh", "-c",
			"pip install -q --disable-pip-version-check numpy=="+numpyVersion+" && python -I /s/np_polyval.py /s/.regen/in.bin /s/.regen/out.bin")
	default:
		cmd = exec.Command("uv", "run", "--no-project", "--with", "numpy=="+numpyVersion, "python", "-I", script, in, out)
	}
	cmd.Stdin = nil
	msg, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("numpy: %v\n%s", err, msg)
	}
	if !strings.Contains(string(msg), "numpy "+numpyVersion+" ") {
		t.Fatalf("not NumPy %s:\n%s", numpyVersion, msg)
	}
	t.Logf("%s", msg)
	res, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	var fix []byte
	off := 0
	for _, c := range cases {
		fix = binary.LittleEndian.AppendUint64(fix, uint64(c.op))
		fix = binary.LittleEndian.AppendUint64(fix, uint64(len(c.c)))
		fix = binary.LittleEndian.AppendUint64(fix, uint64(len(c.x)))
		for _, v := range c.c {
			fix = binary.LittleEndian.AppendUint64(fix, math.Float64bits(v))
		}
		for _, v := range c.x {
			fix = binary.LittleEndian.AppendUint64(fix, math.Float64bits(v))
		}
		fix = append(fix, res[off:off+8*len(c.x)]...)
		off += 8 * len(c.x)
	}
	if off != len(res) {
		t.Fatalf("numpy returned %d bytes, expected %d", len(res), off)
	}
	if err := os.WriteFile(pvFixture, fix, 0o644); err != nil {
		t.Fatal(err)
	}
}
