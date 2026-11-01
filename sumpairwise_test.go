package floats

import (
	_ "embed"
	"encoding/binary"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// The NumPy equality fixtures. The inputs are not stored: every case is a
// (kind, n, seed) triple expanded by genCase, which is deterministic (PCG), so
// the committed fixture holds only NumPy's answers, one 64-bit pattern per
// case. SUMPW_REGEN=1 go test -run Regenerate rebuilds it by writing the inputs
// to a file and running testdata/np_sum.py through uv.

//go:embed testdata/sumpairwise.bin
var sumpwFixture []byte

// numpyVersion is the NumPy that produced the fixture.
const numpyVersion = "2.5.4"

type pwKind int

const (
	kindUniform  pwKind = iota // uniform in [-1,1)
	kindWide                   // random mantissa, exponent in [-300,300]
	kindCancel                 // ±big pairs hiding small values, shuffled
	kindTies                   // values from {1, 2^-53, 2^-54, -1, ...}: order sensitive
	kindZeros                  // +0 and -0 only
	kindNegZeros               // -0 only
	kindPosInf                 // finite values and +Inf
	kindOneNaN                 // finite values and exactly one NaN (sign from seed)
	kindMixed                  // finite, ±Inf, NaN of both signs, ±0
	numKinds
)

type pwCase struct {
	kind pwKind
	n    int
	seed uint64
}

// strictNaN reports whether a NaN result must match NumPy bit for bit (sign and
// payload). That is only well defined when exactly one NaN enters the sum: with
// several NaNs, or +Inf plus -Inf, which NaN comes out is up to the hardware,
// and the fixture may have been produced on another one.
//
// It also needs hardware that preserves a NaN operand through an addition. RISC-V
// does not: its FP operations return the canonical NaN (0x7ff8000000000000),
// dropping sign and payload (the unprivileged ISA spec, NaN generation and
// propagation), so on riscv64 only "is NaN" can be compared. The test probes
// the hardware rather than naming architectures.
func (c pwCase) strictNaN() bool { return c.kind == kindOneNaN && nanSurvivesAdd }

var probeOne = 1.0

// nanSurvivesAdd reports whether NaN + 1 keeps the NaN's sign and payload.
var nanSurvivesAdd = func() bool {
	for _, b := range []uint64{0x7ff8000000000001, 0xfff8000000000001} {
		if math.Float64bits(math.Float64frombits(b)+probeOne) != b {
			return false
		}
	}
	return true
}()

func genCase(c pwCase) []float64 {
	r := rand.New(rand.NewPCG(c.seed, uint64(c.kind)+1))
	a := make([]float64, c.n)
	for i := range a {
		switch c.kind {
		case kindUniform, kindOneNaN, kindPosInf:
			a[i] = r.Float64()*2 - 1
		case kindWide:
			a[i] = math.Ldexp(r.Float64()*2-1, r.IntN(601)-300)
		case kindCancel:
			if i%2 == 0 {
				a[i] = math.Ldexp(1+r.Float64(), 40+r.IntN(20))
			} else {
				a[i] = -a[i-1]
				if r.IntN(3) == 0 {
					a[i] = r.Float64() * 1e-3
				}
			}
		case kindTies:
			a[i] = [...]float64{1, 0x1p-53, 0x1p-54, -1, -0x1p-53, 0x1p-52, 3, 0x1p-60}[r.IntN(8)]
		case kindZeros:
			a[i] = math.Copysign(0, float64(r.IntN(2)*2-1))
		case kindNegZeros:
			a[i] = math.Copysign(0, -1)
		case kindMixed:
			switch r.IntN(8) {
			case 0:
				a[i] = math.Inf(1)
			case 1:
				a[i] = math.Inf(-1)
			case 2:
				a[i] = math.NaN()
			case 3:
				a[i] = -math.NaN()
			case 4:
				a[i] = math.Copysign(0, -1)
			default:
				a[i] = r.Float64()*2 - 1
			}
		}
	}
	if c.n > 0 {
		switch c.kind {
		case kindPosInf:
			a[r.IntN(c.n)] = math.Inf(1)
		case kindOneNaN:
			nan := math.NaN()
			if c.seed&1 == 1 {
				nan = math.Copysign(nan, -1)
			}
			a[r.IntN(c.n)] = nan
		case kindMixed:
			// Sparse variant: most mixed cases have so many specials that the
			// result is NaN at once; thin them for odd seeds.
			if c.seed&1 == 1 {
				for i := range a {
					if math.IsInf(a[i], 0) || math.IsNaN(a[i]) {
						if r.IntN(16) != 0 {
							a[i] = r.Float64()
						}
					}
				}
			}
		}
	}
	return a
}

// fixtureCases is the ordered case list the fixture was built from.
func fixtureCases() []pwCase {
	var cs []pwCase
	add := func(k pwKind, n int, seed uint64) { cs = append(cs, pwCase{k, n, seed}) }
	// Every length 0..300, every kind, two seeds (the parity of the seed
	// selects the NaN sign and the sparse variant).
	for k := pwKind(0); k < numKinds; k++ {
		for n := 0; n <= 300; n++ {
			add(k, n, 1)
			add(k, n, 2)
		}
	}
	// Both sides of every block boundary: 128 * 2^j for j up to 13, +-12.
	for j := 0; j <= 13; j++ {
		for d := -12; d <= 12; d++ {
			n := 128<<j + d
			if n <= 300 {
				continue
			}
			for _, k := range []pwKind{kindUniform, kindWide, kindCancel, kindTies} {
				add(k, n, 3)
			}
			add(kindMixed, n, 3)
		}
	}
	// Large sizes, odd lengths included.
	for _, n := range []int{100_000, 100_001, 99_999, 1_000_000, 999_999, 1_000_001, 1 << 20, 1<<20 + 1, 1<<20 - 1} {
		for _, k := range []pwKind{kindUniform, kindWide, kindCancel, kindTies} {
			add(k, n, 4)
		}
		add(kindOneNaN, n, 5)
		add(kindOneNaN, n, 6)
		add(kindPosInf, n, 4)
		add(kindMixed, n, 5)
	}
	return cs
}

func fixtureWords() []uint64 {
	w := make([]uint64, len(sumpwFixture)/8)
	for i := range w {
		w[i] = binary.LittleEndian.Uint64(sumpwFixture[8*i:])
	}
	return w
}

func sameSum(c pwCase, got, want float64) bool {
	if math.IsNaN(want) {
		if !math.IsNaN(got) {
			return false
		}
		return !c.strictNaN() || math.Float64bits(got) == math.Float64bits(want)
	}
	return math.Float64bits(got) == math.Float64bits(want)
}

func TestSumPairwiseMatchesNumPy(t *testing.T) {
	cs, want := fixtureCases(), fixtureWords()
	if len(cs) != len(want) {
		t.Fatalf("fixture has %d answers for %d cases; regenerate with SUMPW_REGEN=1", len(want), len(cs))
	}
	nan, bad := 0, 0
	for i, c := range cs {
		a := genCase(c)
		got, w := SumPairwise(a), math.Float64frombits(want[i])
		if math.IsNaN(w) {
			nan++
		}
		if !sameSum(c, got, w) {
			bad++
			if bad <= 20 {
				t.Errorf("kind %d n %d seed %d: got %016x (%v), numpy %016x (%v)",
					c.kind, c.n, c.seed, math.Float64bits(got), got, want[i], w)
			}
		}
	}
	t.Logf("%d cases (%d NaN answers) against numpy %s, %d mismatches", len(cs), nan, numpyVersion, bad)
}

// refPairwise is an independent transcription of the C: it works on indices
// into the full slice, takes the first element as the initial value, and uses
// the identity of the addition for the short block. It is the oracle of the
// fuzz target and of the structural tests.
func refPairwise(a []float64, lo, n int) float64 {
	if n < 8 {
		res := math.Copysign(0, -1)
		for i := 0; i < n; i++ {
			res += a[lo+i]
		}
		return res
	}
	if n <= 128 {
		var r [8]float64
		for j := 0; j < 8; j++ {
			r[j] = a[lo+j]
		}
		i := 8
		for ; i < n-(n%8); i += 8 {
			for j := 0; j < 8; j++ {
				r[j] += a[lo+i+j]
			}
		}
		res := ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
		for ; i < n; i++ {
			res += a[lo+i]
		}
		return res
	}
	n2 := n / 2
	n2 -= n2 % 8
	return refPairwise(a, lo, n2) + refPairwise(a, lo+n2, n-n2)
}

func refSum(a []float64) float64 {
	zero := 0.0
	return zero + refPairwise(a, 0, len(a))
}

func TestSumPairwiseSignedZerosAndEmpty(t *testing.T) {
	negz := pwNegZero
	if b := math.Float64bits(SumPairwise(nil)); b != 0 {
		t.Errorf("empty: %016x", b)
	}
	for _, n := range []int{1, 2, 7, 8, 9, 127, 128, 129, 1000} {
		a := make([]float64, n)
		for i := range a {
			a[i] = negz
		}
		// The block sum keeps -0 (it starts from -0) ...
		if n < 8 {
			if got := pairwise(a); math.Float64bits(got) != math.Float64bits(negz) {
				t.Errorf("n=%d pairwise of all -0: %016x", n, math.Float64bits(got))
			}
		}
		// ... but np.sum starts from +0, so the total is +0.
		if got := SumPairwise(a); math.Float64bits(got) != 0 {
			t.Errorf("n=%d all -0: %016x", n, math.Float64bits(got))
		}
	}
}

func TestSumPairwiseDiffersFromSum(t *testing.T) {
	// The point of SumPairwise: Sum's lane order is not NumPy's. Find at least
	// one input where they differ, so a regression that made SumPairwise
	// delegate to Sum would be caught.
	r := rand.New(rand.NewPCG(7, 7))
	for try := 0; try < 100; try++ {
		a := make([]float64, 1000)
		for i := range a {
			a[i] = r.Float64() - 0.5
		}
		if Sum(a) != SumPairwise(a) {
			return
		}
	}
	t.Fatal("Sum and SumPairwise never differed in 100 tries")
}

func TestSumPairwiseAgainstReference(t *testing.T) {
	r := rand.New(rand.NewPCG(11, 13))
	for n := 0; n < 700; n++ {
		a := make([]float64, n)
		for i := range a {
			a[i] = math.Ldexp(r.Float64()-0.5, r.IntN(80)-40)
		}
		if g, w := SumPairwise(a), refSum(a); math.Float64bits(g) != math.Float64bits(w) {
			t.Fatalf("n=%d: %016x vs %016x", n, math.Float64bits(g), math.Float64bits(w))
		}
	}
}

func FuzzSumPairwise(f *testing.F) {
	f.Add([]byte{})
	f.Add(make([]byte, 8*9))
	seed := make([]byte, 8*300)
	r := rand.New(rand.NewPCG(1, 2))
	for i := 0; i < 300; i++ {
		binary.LittleEndian.PutUint64(seed[8*i:], math.Float64bits(r.Float64()-0.5))
	}
	f.Add(seed)
	f.Fuzz(func(t *testing.T, data []byte) {
		// Raw bit patterns: NaN, Inf, subnormals and signed zeros included.
		a := make([]float64, len(data)/8)
		for i := range a {
			a[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[8*i:]))
		}
		g, w := SumPairwise(a), refSum(a)
		if math.IsNaN(w) {
			if !math.IsNaN(g) {
				t.Fatalf("want NaN, got %v", g)
			}
			return
		}
		if math.Float64bits(g) != math.Float64bits(w) {
			t.Fatalf("n=%d: %016x vs %016x", len(a), math.Float64bits(g), math.Float64bits(w))
		}
	})
}

// TestRegenerateSumPairwiseFixtures rebuilds testdata/sumpairwise.bin with
// NumPy. Skipped unless SUMPW_REGEN=1; needs uv on PATH.
func TestRegenerateSumPairwiseFixtures(t *testing.T) {
	if os.Getenv("SUMPW_REGEN") != "1" {
		t.Skip("set SUMPW_REGEN=1 to rebuild the NumPy fixtures (needs uv)")
	}
	dir := t.TempDir()
	in := filepath.Join(dir, "in.bin")
	out := filepath.Join(dir, "out.bin")
	var buf []byte
	for _, c := range fixtureCases() {
		a := genCase(c)
		buf = binary.LittleEndian.AppendUint64(buf, uint64(len(a)))
		for _, v := range a {
			buf = binary.LittleEndian.AppendUint64(buf, math.Float64bits(v))
		}
	}
	if err := os.WriteFile(in, buf, 0o600); err != nil {
		t.Fatal(err)
	}
	script, err := filepath.Abs("testdata/np_sum.py")
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command("uv", "run", "--no-project", "--with", "numpy=="+numpyVersion, "python", "-I", script, in, out)
	cmd.Stdin = nil
	if msg, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("uv: %v\n%s", err, msg)
	} else {
		t.Logf("%s", msg)
	}
	res, err := os.ReadFile(out)
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 8*len(fixtureCases()) {
		t.Fatalf("numpy returned %d bytes for %d cases", len(res), len(fixtureCases()))
	}
	if err := os.WriteFile("testdata/sumpairwise.bin", res, 0o644); err != nil {
		t.Fatal(err)
	}
}
