package floats

import (
	"crypto/sha256"
	_ "embed"
	"encoding/binary"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// The NumPy equality fixtures of the unfused kernels. As for SumPairwise, an input
// is a (op, kind, n, seed) tuple expanded by genNF (deterministic, PCG), so the
// committed fixture holds only NumPy's answers: for each case the first eight
// bytes of the SHA-256 of the result's bit patterns, with every NaN whose payload
// the hardware is free to choose replaced by one canonical NaN (see strictNF).
// NOFMA_REGEN=1 go test -run RegenerateNoFMA rebuilds it with testdata/np_nofma.py.

//go:embed testdata/nofma.bin
var nofmaFixture []byte

type nfOp int

const (
	opAxpy     nfOp = iota // y + alpha*x
	opXpby                 // x + beta*y
	opAxpby3               // alpha*a + beta*b + gamma*c
	opAxpyDiff             // a + F*(b - c)
	numNFOps
)

func (o nfOp) arrays() int {
	if o < opAxpby3 {
		return 2
	}
	return 3
}

// nfCancel is the kind added to those of SumPairwise: c is b with its low bits
// perturbed, so b - c cancels almost completely and the next rounding matters.
const nfCancel = numKinds

type nfCase struct {
	op   nfOp
	kind pwKind
	n    int
	seed uint64
}

// nfData is an expanded case.
type nfData struct {
	s   [3]float64
	arr [4][]float64
}

func genNF(c nfCase) nfData {
	var d nfData
	r := rand.New(rand.NewPCG(c.seed, uint64(c.op)*64+uint64(c.kind)+7))
	special := c.kind == kindMixed || c.kind == kindPosInf || c.kind == kindOneNaN
	for i := range d.s {
		switch k := r.IntN(12); {
		case k == 0:
			d.s[i] = 0
		case k == 1:
			d.s[i] = math.Copysign(0, -1)
		case k == 2:
			d.s[i] = 1
		case k == 3 && special:
			d.s[i] = math.Inf(1 - 2*r.IntN(2))
		case k == 4 && c.kind == kindMixed:
			d.s[i] = math.Copysign(math.NaN(), float64(1-2*r.IntN(2)))
		case c.kind == kindWide:
			d.s[i] = math.Ldexp(r.Float64()*2-1, r.IntN(121)-60)
		default:
			// float64(...) forbids fusing the multiply-subtract (arm64, ppc64le,
			// ... would otherwise generate different inputs than amd64).
			d.s[i] = float64(r.Float64()*6) - 3
		}
	}
	base := pwKind(c.kind)
	if c.kind == nfCancel {
		base = kindWide
	}
	for j := 0; j < c.op.arrays(); j++ {
		d.arr[j] = genCase(pwCase{base, c.n, c.seed*8 + uint64(j)})
	}
	if c.kind == nfCancel && c.op.arrays() == 3 {
		// a is large, b and c nearly equal.
		for i := 0; i < c.n; i++ {
			b := d.arr[1][i]
			d.arr[2][i] = math.Float64frombits(math.Float64bits(b) ^ uint64(r.IntN(1<<(1+r.IntN(20)))))
			d.arr[0][i] = math.Ldexp(d.arr[0][i], 10)
		}
	}
	return d
}

func nfCases() []nfCase {
	var cs []nfCase
	for op := nfOp(0); op < numNFOps; op++ {
		for k := pwKind(0); k <= nfCancel; k++ {
			for n := 0; n <= 300; n++ {
				cs = append(cs, nfCase{op, k, n, 1})
			}
			// Second seed (it flips the NaN sign and the sparse variant) around the
			// vector widths, the unroll of 16 and the block boundaries.
			for _, n := range []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257} {
				cs = append(cs, nfCase{op, k, n, 2})
			}
			for _, n := range []int{1000, 4097, 100_001} {
				if k == kindUniform || k == kindWide || k == kindMixed || k == nfCancel {
					cs = append(cs, nfCase{op, k, n, 3})
				}
			}
		}
	}
	return cs
}

// nfElem evaluates the operation on one element, unfused, in NumPy's order: the
// scalar specification the strictness test below needs.
func nfElem(op nfOp, s [3]float64, v [4]float64) float64 {
	switch op {
	case opAxpy:
		return v[1] + float64(s[0]*v[0])
	case opXpby:
		return v[0] + float64(s[0]*v[1])
	case opAxpby3:
		return (float64(s[0]*v[0]) + float64(s[1]*v[1])) + float64(s[2]*v[2])
	}
	d := v[1] - v[2]
	return v[0] + float64(s[0]*d)
}

// strictNF marks the elements whose result NaN must match NumPy bit for bit: those
// where exactly one operand is a NaN (it propagates, quieted, with its sign and
// payload) and nothing else in the element creates a NaN of its own (Inf - Inf,
// 0*Inf: which NaN that is — sign, payload — is up to the hardware). The second
// condition is checked by evaluating the element with the NaN operand replaced by
// zero. It also needs hardware that keeps a NaN through an addition.
func strictNF(c nfCase, d nfData) []bool {
	m := make([]bool, c.n)
	if !nanSurvivesAdd {
		return m
	}
	nIn := c.op.arrays()
	for i := range m {
		var v [4]float64
		for j := 0; j < nIn; j++ {
			v[j] = d.arr[j][i]
		}
		s := d.s
		count, where := 0, -1
		for j := 0; j < nIn; j++ {
			if math.IsNaN(v[j]) {
				count++
				where = j
			}
		}
		used := 1
		if c.op == opAxpby3 {
			used = 3
		}
		for j := 0; j < used; j++ {
			if math.IsNaN(s[j]) {
				count++
				where = 10 + j
			}
		}
		if count != 1 {
			continue
		}
		if where >= 10 {
			s[where-10] = 0
		} else {
			v[where] = 0
		}
		m[i] = !math.IsNaN(nfElem(c.op, s, v))
	}
	return m
}

// nfImpl computes the result of a case; the real one calls the package.
type nfImpl func(c nfCase, d nfData) []float64

func nfReal(c nfCase, d nfData) []float64 {
	n := c.n
	cp := func(s []float64) []float64 { return append([]float64(nil), s...) }
	switch c.op {
	case opAxpy:
		y := cp(d.arr[1])
		AxpyNoFMA(d.s[0], d.arr[0], y)
		return y
	case opXpby:
		y := cp(d.arr[1])
		XpbyNoFMA(d.arr[0], d.s[0], y)
		return y
	case opAxpby3:
		dst := make([]float64, n)
		Axpby3NoFMA(dst, d.s[0], d.arr[0], d.s[1], d.arr[1], d.s[2], d.arr[2])
		return dst
	}
	dst := make([]float64, n)
	AxpyDiffNoFMA(dst, d.arr[0], d.s[0], d.arr[1], d.arr[2])
	return dst
}

// nfFused is the planted break "fused": the multiply-add contracted, as the fused
// kernels of the package (and a compiler that is allowed to) compute it.
func nfFused(c nfCase, d nfData) []float64 {
	out := make([]float64, c.n)
	for i := range out {
		switch c.op {
		case opAxpy:
			out[i] = math.FMA(d.s[0], d.arr[0][i], d.arr[1][i])
		case opXpby:
			out[i] = math.FMA(d.s[0], d.arr[1][i], d.arr[0][i])
		case opAxpby3:
			t := d.s[0] * d.arr[0][i]
			t = math.FMA(d.s[1], d.arr[1][i], t)
			out[i] = math.FMA(d.s[2], d.arr[2][i], t)
		default:
			out[i] = math.FMA(d.s[0], d.arr[1][i]-d.arr[2][i], d.arr[0][i])
		}
	}
	return out
}

// nfReordered is the planted break "reordered": the same operations in another
// association (distributing F over the difference; summing the terms right to
// left), each correctly rounded but not NumPy's sequence.
func nfReordered(c nfCase, d nfData) []float64 {
	out := nfReal(c, d)
	for i := range out {
		switch c.op {
		case opAxpby3:
			p, q, r := float64(d.s[0]*d.arr[0][i]), float64(d.s[1]*d.arr[1][i]), float64(d.s[2]*d.arr[2][i])
			out[i] = p + (q + r)
		case opAxpyDiff:
			out[i] = d.arr[0][i] + (float64(d.s[0]*d.arr[1][i]) - float64(d.s[0]*d.arr[2][i]))
		}
	}
	return out
}

func nfHash(res []float64, strict []bool) uint64 {
	buf := make([]byte, 8*len(res))
	for i, v := range res {
		b := math.Float64bits(v)
		if math.IsNaN(v) && !strict[i] {
			b = 0x7ff8000000000000
		}
		binary.LittleEndian.PutUint64(buf[8*i:], b)
	}
	h := sha256.Sum256(buf)
	return binary.LittleEndian.Uint64(h[:8])
}

func nfFixtureWords() []uint64 {
	w := make([]uint64, len(nofmaFixture)/8)
	for i := range w {
		w[i] = binary.LittleEndian.Uint64(nofmaFixture[8*i:])
	}
	return w
}

// nfMismatches runs impl over every case and counts those whose hash differs from
// NumPy's, per operation.
func nfMismatches(t *testing.T, impl nfImpl) (bad [numNFOps]int, total int) {
	t.Helper()
	cs, want := nfCases(), nfFixtureWords()
	if len(cs) != len(want) {
		t.Fatalf("fixture has %d answers for %d cases; regenerate with NOFMA_REGEN=1", len(want), len(cs))
	}
	for i, c := range cs {
		d := genNF(c)
		if nfHash(impl(c, d), strictNF(c, d)) != want[i] {
			bad[c.op]++
			if bad[c.op] <= 3 {
				t.Logf("differs: op %d kind %d n %d seed %d", c.op, c.kind, c.n, c.seed)
			}
		}
	}
	return bad, len(cs)
}

func TestNoFMAMatchesNumPy(t *testing.T) {
	bad, total := nfMismatches(t, nfReal)
	t.Logf("%d cases against numpy %s, mismatches per op (axpy, xpby, axpby3, axpydiff): %v", total, numpyVersion, bad)
	for op, b := range bad {
		if b != 0 {
			t.Errorf("op %d: %d cases differ from NumPy", op, b)
		}
	}
}

// The fixture must be able to see a wrong kernel: a fused one and a reordered one
// each have to be caught, on the operations they touch.
func TestNoFMAFixtureCatchesPlantedBreaks(t *testing.T) {
	bad, total := nfMismatches(t, nfFused)
	for op, b := range bad {
		if b == 0 {
			t.Errorf("fused variant of op %d is not caught by the %d-case fixture", op, total)
		}
	}
	t.Logf("fused: mismatches per op %v of %d", bad, total)
	bad, _ = nfMismatches(t, nfReordered)
	if bad[opAxpby3] == 0 || bad[opAxpyDiff] == 0 {
		t.Errorf("reordered variants not caught: %v", bad)
	}
	if bad[opAxpy] != 0 || bad[opXpby] != 0 {
		t.Errorf("the untouched ops differ: %v", bad)
	}
	t.Logf("reordered: mismatches per op %v", bad)
}

// The strictness rule must really occur in the fixture, or the NaN bits would not
// be compared at all.
func TestNoFMAFixtureHasStrictNaNs(t *testing.T) {
	if !nanSurvivesAdd {
		t.Skip("hardware does not keep NaN payloads through an addition")
	}
	strict, nanOut := 0, 0
	for _, c := range nfCases() {
		if c.n == 0 || c.n > 40 {
			continue
		}
		d := genNF(c)
		m := strictNF(c, d)
		res := nfReal(c, d)
		for i := range m {
			if m[i] && math.IsNaN(res[i]) {
				strict++
			}
			if math.IsNaN(res[i]) {
				nanOut++
			}
		}
	}
	if strict < 100 {
		t.Fatalf("only %d strict NaN elements (%d NaN results): the NaN bits are barely compared", strict, nanOut)
	}
	t.Logf("%d strict NaN elements among %d NaN results (n <= 40)", strict, nanOut)
}

// TestRegenerateNoFMAFixtures rebuilds testdata/nofma.bin with NumPy. Skipped
// unless NOFMA_REGEN=1; uses the interpreter in NUMPY_PYTHON (a NumPy
// numpyVersion) if set, else uv.
func TestRegenerateNoFMAFixtures(t *testing.T) {
	if os.Getenv("NOFMA_REGEN") != "1" {
		t.Skip("set NOFMA_REGEN=1 to rebuild the NumPy fixtures (NUMPY_PYTHON or uv)")
	}
	dir := t.TempDir()
	in, out := filepath.Join(dir, "in.bin"), filepath.Join(dir, "out.bin")
	f, err := os.Create(in)
	if err != nil {
		t.Fatal(err)
	}
	var buf []byte
	put := func(w uint64) { buf = binary.LittleEndian.AppendUint64(buf, w) }
	for _, c := range nfCases() {
		d := genNF(c)
		put(uint64(c.op))
		put(uint64(c.n))
		for _, s := range d.s {
			put(math.Float64bits(s))
		}
		for j := 0; j < c.op.arrays(); j++ {
			for _, v := range d.arr[j] {
				put(math.Float64bits(v))
			}
		}
		mask := make([]byte, 8*((c.n+7)/8))
		for i, s := range strictNF(c, d) {
			if s {
				mask[i] = 1
			}
		}
		buf = append(buf, mask...)
		if len(buf) > 1<<24 {
			if _, err := f.Write(buf); err != nil {
				t.Fatal(err)
			}
			buf = buf[:0]
		}
	}
	if _, err := f.Write(buf); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	script, err := filepath.Abs("testdata/np_nofma.py")
	if err != nil {
		t.Fatal(err)
	}
	var cmd *exec.Cmd
	if py := os.Getenv("NUMPY_PYTHON"); py != "" {
		cmd = exec.Command(py, "-I", script, in, out)
	} else {
		cmd = exec.Command("uv", "run", "--no-project", "--with", "numpy=="+numpyVersion, "python", "-I", script, in, out)
	}
	cmd.Stdin = nil
	if msg, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("numpy: %v\n%s", err, msg)
	} else {
		t.Logf("%s", msg)
	}
	res, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 8*len(nfCases()) {
		t.Fatalf("numpy returned %d bytes for %d cases", len(res), len(nfCases()))
	}
	if err := os.WriteFile("testdata/nofma.bin", res, 0o644); err != nil {
		t.Fatal(err)
	}
}
