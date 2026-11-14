package floats

import (
	"math"
	"math/rand"
	"testing"
	"unsafe"
)

func conjRevLengths() []int {
	var l []int
	for n := 0; n <= 70; n++ {
		l = append(l, n)
	}
	return append(l, 127, 128, 129, 255, 256, 257, 1000, 1024, 4097)
}

// conjRevInputs holds n complex numbers (2n floats) with signed zeros and NaN payloads
// among ordinary values, so that a kernel that negates by subtraction or canonicalises
// a NaN is caught.
func conjRevInputs(rng *rand.Rand, n int) []float64 {
	c := make([]float64, 2*n)
	for i := range c {
		switch rng.Intn(16) {
		case 0:
			c[i] = 0
		case 1:
			c[i] = math.Copysign(0, -1)
		case 2:
			c[i] = math.Float64frombits(0x7ff8000000000000 | uint64(rng.Intn(1<<20)))
		case 3:
			c[i] = math.Inf(1 - 2*rng.Intn(2))
		default:
			c[i] = rng.NormFloat64() * 100
		}
	}
	return c
}

// conjRevWant is the definition, written straight from the contract.
func conjRevWant(src []float64, n int) []float64 {
	want := make([]float64, 2*n)
	for k := 0; k < n; k++ {
		want[2*k] = src[2*(n-1-k)]
		want[2*k+1] = math.Float64frombits(math.Float64bits(src[2*(n-1-k)+1]) ^ (1 << 63))
	}
	return want
}

func conjBitDiff(a, b []float64) int {
	for i := range a {
		if math.Float64bits(a[i]) != math.Float64bits(b[i]) {
			return i
		}
	}
	return -1
}

// TestConjReverseExact: the kernel and the portable loop equal the definition bit for
// bit at every length from 0 to 70 and at the sizes of a transform's mirror.
func TestConjReverseExact(t *testing.T) {
	rng := rand.New(rand.NewSource(101))
	for _, n := range conjRevLengths() {
		src := conjRevInputs(rng, n)
		want := conjRevWant(src, n)
		got, ref := make([]float64, 2*n), make([]float64, 2*n)
		ConjReverse(got, src)
		conjRevTail(ref, src, 0)
		if i := conjBitDiff(got, want); i >= 0 {
			t.Fatalf("ConjReverse n=%d element %d: %x, want %x", n, i, math.Float64bits(got[i]), math.Float64bits(want[i]))
		}
		if i := conjBitDiff(ref, want); i >= 0 {
			t.Fatalf("conjRevTail n=%d element %d differs from the definition", n, i)
		}
	}
}

func TestConjReversePanicsAndBounds(t *testing.T) {
	mustPanic := func(name, want string, f func()) {
		t.Helper()
		defer func() {
			r, _ := recover().(string)
			if r != want {
				t.Errorf("%s: panic %q, want %q", name, r, want)
			}
		}()
		f()
	}
	mustPanic("odd dst", "floats: ConjReverse: odd len(dst)", func() { ConjReverse(make([]float64, 3), make([]float64, 4)) })
	mustPanic("short src", "floats: ConjReverse: len(src) < len(dst)", func() { ConjReverse(make([]float64, 4), make([]float64, 3)) })
	ConjReverse(nil, nil) // n = 0: nothing
	buf := make([]float64, 64)
	mustPanic("dst == src", "floats: ConjReverse: dst overlaps src", func() { ConjReverse(buf[:16], buf[:16]) })
	mustPanic("shifted", "floats: ConjReverse: dst overlaps src", func() { ConjReverse(buf[8:24], buf[:16]) })
	mustPanic("src in dst", "floats: ConjReverse: dst overlaps src", func() { ConjReverse(buf[:16], buf[4:36]) })
	mustPanic("across element types", "floats: ConjReverse: dst overlaps src", func() {
		ConjReverse(unsafe.Slice((*float64)(unsafe.Pointer(&buf[8])), 16), buf[:16])
	})
	// adjacent slices of one array do not overlap
	rng := rand.New(rand.NewSource(102))
	c := conjRevInputs(rng, 8)
	copy(buf[:16], c)
	ConjReverse(buf[16:32], buf[:16])
	if i := conjBitDiff(buf[16:32], conjRevWant(c, 8)); i >= 0 {
		t.Fatalf("adjacent: element %d differs", i)
	}
	// nothing written outside dst, nothing read past len(dst), src unmodified
	for _, n := range conjRevLengths() {
		src := conjRevInputs(rng, n)
		s0 := append([]float64(nil), src...)
		long := append(append([]float64(nil), src...), 1e300, 1e300, 1e300, 1e300)
		pad := make([]float64, 2*n+16)
		for i := range pad {
			pad[i] = 12345
		}
		ConjReverse(pad[8:8+2*n], long)
		for i := range pad {
			if (i < 8 || i >= 8+2*n) && pad[i] != 12345 {
				t.Fatalf("n=%d: wrote outside dst at %d", n, i)
			}
		}
		if i := conjBitDiff(src, s0); i >= 0 {
			t.Fatalf("n=%d: src changed", n)
		}
		for i := 8; i < 8+2*n; i++ {
			if math.Abs(pad[i]) > 1e8 && !math.IsInf(pad[i], 0) {
				t.Fatalf("n=%d: an operand past len(dst) was read (%g)", n, pad[i])
			}
		}
	}
}

// FuzzConjReverse: for any length 0 to 130 complex numbers and any bytes, the public
// function equals the definition bit for bit.
func FuzzConjReverse(f *testing.F) {
	f.Add(uint8(0), []byte{1, 2, 3})
	f.Add(uint8(1), []byte("conjugate reverse"))
	f.Add(uint8(3), []byte{255, 0, 128})
	f.Add(uint8(64), []byte{})
	f.Fuzz(func(t *testing.T, k uint8, data []byte) {
		n := int(k) % 131
		src := make([]float64, 2*n)
		for i := range src {
			if len(data) > 0 {
				src[i] = math.Float64frombits(uint64(data[(i*3+1)%len(data)])<<56 | uint64(data[i%len(data)])<<48 | uint64(i))
			}
		}
		got := make([]float64, 2*n)
		ConjReverse(got, src)
		if i := conjBitDiff(got, conjRevWant(src, n)); i >= 0 {
			t.Fatalf("n=%d element %d differs from the definition", n, i)
		}
	})
}

func BenchmarkConjReverse(b *testing.B) {
	rng := rand.New(rand.NewSource(103))
	for _, n := range []int{511, 2047} { // the mirror of a 1,024 and a 4,096 point real transform
		src := conjRevInputs(rng, n)
		d := make([]float64, 2*n)
		name := map[int]string{511: "511", 2047: "2047"}[n]
		b.Run("kernel/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				ConjReverse(d, src)
			}
		})
		b.Run("portable/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				conjRevTail(d, src, 0)
			}
		})
	}
}
