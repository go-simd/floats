package floats

import (
	"math"
	"math/rand"
	"runtime"
	"testing"
)

// The reference of every kernel is the Go expression itself, spelled out here
// (not absRef and its siblings, which are tested against it too): each kernel
// must agree with it bit for bit on every length, alignment and special value,
// and must not write past dst.

type roundOp struct {
	name string
	f    func(dst, src []float64)
	ref  func(dst, src []float64)
	expr func(x float64) float64
	// exact: a sign-bit operation, so every NaN keeps its exact bits. The others
	// may quiet a signalling NaN (and riscv64 produces the canonical NaN).
	exact bool
}

var roundOps = []roundOp{
	{"Abs", Abs, absRef, math.Abs, true},
	{"Neg", Neg, negRef, func(x float64) float64 { return -x }, true},
	{"Square", Square, squareRef, func(x float64) float64 { return x * x }, false},
	{"Floor", Floor, floorRef, math.Floor, false},
	{"Ceil", Ceil, ceilRef, math.Ceil, false},
	{"Trunc", Trunc, truncRef, math.Trunc, false},
	{"RoundEven", RoundEven, rintRef, math.RoundToEven, false},
}

const quietBit = 1 << 51

// roundOK reports whether got is the contract's result for in, whose Go
// expression gives want.
func roundOK(op roundOp, in, got, want float64) bool {
	gb, wb := math.Float64bits(got), math.Float64bits(want)
	if gb == wb {
		return true
	}
	if op.exact || !math.IsNaN(in) {
		return false
	}
	// A NaN input: the hardware may quiet a signalling NaN (never changes
	// anything else), and RISC-V arithmetic returns the canonical NaN.
	if gb == math.Float64bits(in)|quietBit {
		return true
	}
	return runtime.GOARCH == "riscv64" && math.IsNaN(got)
}

func checkRound(t *testing.T, op roundOp, what string, got, src []float64, n, off int) {
	t.Helper()
	for i := 0; i < n; i++ {
		want := op.expr(src[i])
		if !roundOK(op, src[i], got[i], want) {
			t.Fatalf("%s %s n=%d off=%d: [%d] src %v (%#x) = %v (%#x), want %v (%#x)", op.name, what, n, off, i,
				src[i], math.Float64bits(src[i]), got[i], math.Float64bits(got[i]), want, math.Float64bits(want))
		}
	}
}

// roundEdges: signed zeros, infinities, NaNs of every kind, subnormals, the
// halfway cases and their neighbours, and the values around 2^52 (the first
// binade whose values are all integers) and 2^63 (where a float-to-int round
// trip breaks).
func roundEdges() []float64 {
	var e []float64
	add := func(v float64) { e = append(e, v, -v) }
	for _, v := range []float64{
		0, math.SmallestNonzeroFloat64, 2 * math.SmallestNonzeroFloat64, 1e-310,
		0x1p-1022 - 0x1p-1074, 0x1p-1022, 0x1p-1022 + 0x1p-1074, 1e-300, 1e-17,
		0.1, 0.25, math.Nextafter(0.5, 0), 0.5, math.Nextafter(0.5, 1),
		math.Nextafter(1, 0), 1, math.Nextafter(1, 2),
		1.25, 1.5, math.Nextafter(1.5, 1), math.Nextafter(1.5, 2), 1.75,
		2, 2.5, math.Nextafter(2.5, 2), math.Nextafter(2.5, 3), 3.5, 4.5, 5.5,
		0.5 + 0x1p51, 1.5 + 0x1p50, 0x1p51 - 0.5, 0x1p51, 0x1p51 + 1,
		0x1p52 - 1, 0x1p52 - 0.5, math.Nextafter(0x1p52, 0), 0x1p52, 0x1p52 + 1, 0x1p52 + 2,
		math.Nextafter(0x1p52, math.Inf(1)), 0x1p53 - 1, 0x1p53, 0x1p53 + 2,
		0x1p62, 0x1p63, math.Nextafter(0x1p63, 0), 0x1p63 + 0x1p11, 0x1p64, 0x1p100,
		1e100, 0x1p1000, math.MaxFloat64, math.Nextafter(math.MaxFloat64, 0),
		math.Inf(1), 123456.789, 7, 8, 9.999999999999998,
	} {
		add(v)
	}
	// Every kind of NaN: quiet and signalling, both signs, with payloads.
	e = append(e, math.NaN(), negNaN, sigNaN, nsigNaN,
		math.Float64frombits(0x7ff8000000000001), math.Float64frombits(0xfff8000000000001),
		math.Float64frombits(0x7fffffffffffffff), math.Float64frombits(0xffffffffffffffff),
		math.Float64frombits(0x7ff7ffffffffffff), math.Float64frombits(0xfff7ffffffffffff),
		math.Float64frombits(0x7ff0000000000002), math.Float64frombits(0x7ff4000000000000))
	return e
}

func TestRoundEdgeTable(t *testing.T) {
	edges := roundEdges()
	rng := rand.New(rand.NewSource(1))
	for _, op := range roundOps {
		for n := 0; n <= 33; n++ {
			for off := 0; off < 4; off++ {
				// src[i] = edges[(i+s) % len]: every edge value lands on every
				// position (every vector lane, every tail slot) of every length.
				for s := range edges {
					src, _ := fillE(n, off, rng, false)
					for i := range src {
						src[i] = edges[(i+s)%len(edges)]
					}
					dst, buf := fillE(n, (off+2)%4, rng, false)
					op.f(dst, src)
					checkRound(t, op, "table", dst, src, n, off)
					checkGuard(t, op.name, buf, n, (off+2)%4)
				}
			}
		}
	}
}

// randomWords returns n float64s: random bit patterns biased to the binades where
// rounding matters (|x| from 2^-30 to 2^60), exact halves, a half above and
// below an integer, and the planted specials.
func randomWords(n, off int, rng *rand.Rand, sp bool) (s, buf []float64) {
	s, buf = fillE(n, off, rng, sp)
	for i := range s {
		switch rng.Intn(5) {
		case 0:
			u := rng.Uint64()
			e := uint64(1023-30+rng.Intn(91)) << 52
			s[i] = math.Float64frombits(u&^(0x7ff<<52) | e)
		case 1:
			k := float64(rng.Int63n(1<<52)) - 0x1p51
			s[i] = k + 0.5
		case 2:
			k := float64(rng.Int63n(1<<20)) - 0x1p19
			s[i] = math.Nextafter(k+0.5, float64(rng.Intn(3)-1)*math.Inf(1))
		case 3:
			s[i] = math.Float64frombits(rng.Uint64())
		}
	}
	return s, buf
}

func TestRoundRandomDifferential(t *testing.T) {
	for _, op := range roundOps {
		forEachCase(t, 71, func(n, off int, sp bool, rng *rand.Rand) {
			src, _ := randomWords(n, off, rng, sp)
			dst, buf := fillE(n, (off+2)%4, rng, sp)
			op.f(dst, src)
			checkRound(t, op, "random", dst, src, n, off)
			checkGuard(t, op.name, buf, n, (off+2)%4)
			op.ref(dst, src)
			checkRound(t, op, "ref", dst, src, n, off)
		})
	}
}

// dst may be the very slice of src; the other elements of dst past len(src) are
// untouched.
func TestRoundAliasing(t *testing.T) {
	rng := rand.New(rand.NewSource(72))
	for _, op := range roundOps {
		for n := 0; n <= 70; n++ {
			src, _ := randomWords(n, 1, rng, true)
			x := append([]float64(nil), src...)
			op.f(x, x)
			checkRound(t, op, "alias", x, src, n, 0)
		}
	}
}

// Only the first len(src) elements are written, and a short dst panics.
func TestRoundLengths(t *testing.T) {
	src := []float64{1.5, -2.5, 3.5, -0.5, 0.5, 8.5, -9.5, 10.5, 11.5, -12.5}
	for _, op := range roundOps {
		dst := make([]float64, len(src)+3)
		for i := range dst {
			dst[i] = 99
		}
		op.f(dst, src[:9])
		for i := 0; i < 9; i++ {
			if want := op.expr(src[i]); !roundOK(op, src[i], dst[i], want) {
				t.Fatalf("%s: [%d] = %v, want %v", op.name, i, dst[i], want)
			}
		}
		if dst[9] != 99 || dst[10] != 99 || dst[11] != 99 {
			t.Fatalf("%s touched dst past len(src): %v", op.name, dst)
		}
		func() {
			defer func() {
				if recover() == nil {
					t.Fatalf("%s did not panic on a short dst", op.name)
				}
			}()
			op.f(dst[:len(src)-1], src)
		}()
	}
}

// The documented values, by hand.
func TestRoundDocumentedValues(t *testing.T) {
	src := []float64{-2.5, -1.5, -0.5, negZero, 0.5, 1.5, 2.5, 2.4999, -2.5001, 3.5, 7.5, 0x1p52 + 1, -0x1p53}
	src = append(src, negZero)
	want := map[string][]float64{
		"Floor":     {-3, -2, -1, negZero, 0, 1, 2, 2, -3, 3, 7, 0x1p52 + 1, -0x1p53, negZero},
		"Ceil":      {-2, -1, negZero, negZero, 1, 2, 3, 3, -2, 4, 8, 0x1p52 + 1, -0x1p53, negZero},
		"Trunc":     {-2, -1, negZero, negZero, 0, 1, 2, 2, -2, 3, 7, 0x1p52 + 1, -0x1p53, negZero},
		"RoundEven": {-2, -2, negZero, negZero, 0, 2, 2, 2, -3, 4, 8, 0x1p52 + 1, -0x1p53, negZero},
	}
	// Pad to two blocks so the vector kernels run, not only the scalar tail.
	pad := func(s []float64) []float64 { return append(append([]float64(nil), s...), s...) }
	for _, op := range roundOps {
		w, ok := want[op.name]
		if !ok {
			continue
		}
		got := make([]float64, 2*len(src))
		op.f(got, pad(src))
		for i, v := range pad(w) {
			if math.Float64bits(got[i]) != math.Float64bits(v) {
				t.Fatalf("%s(%v) = %v (%#x), want %v", op.name, pad(src)[i], got[i], math.Float64bits(got[i]), v)
			}
		}
	}
}

// FuzzRound: any bytes read as float64 words, any operation, any alignment:
// the kernel agrees with the Go expression, in place and out of place.
func FuzzRound(f *testing.F) {
	f.Add(uint8(0), uint8(0), []byte{})
	f.Add(uint8(3), uint8(1), []byte("halfway: 0.5, 1.5, 2.5 and a tail!"))
	f.Fuzz(func(t *testing.T, which, off uint8, data []byte) {
		op := roundOps[int(which)%len(roundOps)]
		n := len(data) / 8
		if n > 96 {
			n = 96
		}
		src, _ := fillE(n, int(off%4), rand.New(rand.NewSource(0)), false)
		for i := range src {
			var u uint64
			for j := 0; j < 8; j++ {
				u |= uint64(data[8*i+j]) << (8 * j)
			}
			src[i] = math.Float64frombits(u)
		}
		dst := make([]float64, n)
		op.f(dst, src)
		checkRound(t, op, "fuzz", dst, src, n, 0)
		cp := append([]float64(nil), src...)
		op.f(cp, cp)
		checkRound(t, op, "fuzz in place", cp, src, n, 0)
	})
}

// roundOracle is a second reference, independent of package math and of the
// hardware: for 0 < |x| < 2^52 the int64 conversion is the exact truncation
// (every such float64 is below 2^63), the fraction x - t is exact, and the sign
// of the result is the sign of x (which restores -0 where the integer zero has
// none: Ceil(-0.5) = -0). Zeros, infinities, NaNs and |x| >= 2^52 (already
// integers) are returned unchanged.
func roundOracle(name string, x float64) float64 {
	if math.IsNaN(x) || math.IsInf(x, 0) || x == 0 || math.Abs(x) >= 0x1p52 {
		return x
	}
	t := float64(int64(x))
	r := t
	switch name {
	case "Floor":
		if t > x {
			r = t - 1
		}
	case "Ceil":
		if t < x {
			r = t + 1
		}
	case "RoundEven":
		d, odd := x-t, math.Mod(t, 2) != 0
		switch {
		case d > 0.5 || d == 0.5 && odd:
			r = t + 1
		case d < -0.5 || d == -0.5 && odd:
			r = t - 1
		}
	}
	return math.Copysign(r, x)
}

// The Go expressions the kernels are held to are themselves exact: on this
// machine they equal the oracle, and so do the kernels.
func TestRoundMatchesOracle(t *testing.T) {
	rng := rand.New(rand.NewSource(73))
	src := roundEdges()
	for i := 0; i < 1<<16; i++ {
		src = append(src, randomWordsOne(rng))
	}
	for _, op := range roundOps {
		switch op.name {
		case "Floor", "Ceil", "Trunc", "RoundEven":
		default:
			continue
		}
		dst := make([]float64, len(src))
		op.f(dst, src)
		for i, x := range src {
			want := roundOracle(op.name, x)
			if math.IsNaN(x) {
				continue // the NaN rules are checked by roundOK
			}
			if e := op.expr(x); math.Float64bits(e) != math.Float64bits(want) {
				t.Fatalf("%s(%v): Go expression %v (%#x), oracle %v (%#x)", op.name, x, e, math.Float64bits(e), want, math.Float64bits(want))
			}
			if math.Float64bits(dst[i]) != math.Float64bits(want) {
				t.Fatalf("%s(%v): kernel %v (%#x), oracle %v (%#x)", op.name, x, dst[i], math.Float64bits(dst[i]), want, math.Float64bits(want))
			}
		}
	}
}

func randomWordsOne(rng *rand.Rand) float64 {
	s, _ := randomWords(1, 0, rng, false)
	return s[0]
}
