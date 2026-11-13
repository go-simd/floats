package floats

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
	"unsafe"
)

var foldSizes = []int{4, 8, 12, 16, 20, 28, 32, 36, 60, 64, 68, 124, 128, 132, 252, 256, 260, 508, 512, 1000, 1024, 2048, 4096}

func foldInputs(rng *rand.Rand, n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		x[i] = rng.NormFloat64() * 100
	}
	return x
}

func eq(a, b float64) bool { return a == b || (math.IsNaN(a) && math.IsNaN(b)) } // -0 == +0

// TestMDCTFoldHandChecked: the formulas against the article's (a, b, c, d) ->
// (-c_r - d, a - b_r) on a block of eight samples, and the unfold against
// (u, v) -> (v, -v_r, -u_r, -u).
func TestMDCTFoldHandChecked(t *testing.T) {
	x := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	got := make([]float64, 4)
	MDCTFold(got, x)
	want := []float64{-(6 + 7), -(5 + 8), 1 - 4, 2 - 3}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("fold: %v, want %v", got, want)
		}
	}
	u := []float64{10, 20, 30, 40}
	d := make([]float64, 8)
	MDCTUnfold(d, u)
	wantU := []float64{30, 40, -40, -30, -20, -10, -10, -20}
	for i := range wantU {
		if d[i] != wantU[i] {
			t.Fatalf("unfold: %v, want %v", d, wantU)
		}
	}
}

// TestMDCTFoldExact: every output of the fold is one correctly rounded operation, so
// it must equal the 200-bit result rounded to float64, for the kernels, the portable
// loop and the float32 form; the unfold must equal the formula exactly.
func TestMDCTFoldExact(t *testing.T) {
	rng := rand.New(rand.NewSource(81))
	for _, n := range foldSizes {
		m, h := n/2, n/4
		x := foldInputs(rng, n)
		x32 := to32(x)
		for name, got := range map[string][]float64{
			"MDCTFold":       func() []float64 { d := make([]float64, m); MDCTFold(d, x); return d }(),
			"foldTail":       func() []float64 { d := make([]float64, m); foldTail(d, x, 0); return d }(),
			"MDCTFoldFrom32": func() []float64 { d := make([]float64, m); MDCTFoldFrom32(d, x32); return d }(),
		} {
			xs := x
			if name == "MDCTFoldFrom32" {
				xs = to64(x32)
			}
			for i := 0; i < h; i++ {
				a := new(big.Float).SetPrec(200).Add(bigOf(xs[m+h-1-i]), bigOf(xs[m+h+i]))
				wantLo, _ := a.Neg(a).Float64()
				b := new(big.Float).SetPrec(200).Sub(bigOf(xs[i]), bigOf(xs[m-1-i]))
				wantHi, _ := b.Float64()
				if !eq(got[i], wantLo) || !eq(got[h+i], wantHi) {
					t.Fatalf("%s n=%d i=%d: %g %g, want %g %g", name, n, i, got[i], got[h+i], wantLo, wantHi)
				}
			}
		}
		// unfold: u has m values
		u := foldInputs(rng, m)
		d := make([]float64, n)
		MDCTUnfold(d, u)
		d32 := make([]float32, n)
		MDCTUnfoldTo32(d32, u)
		for i := 0; i < h; i++ {
			s, tt := u[i], u[h+i]
			for _, c := range []struct {
				idx  int
				want float64
			}{{i, tt}, {m - 1 - i, -tt}, {m + h - 1 - i, -s}, {m + h + i, -s}} {
				if !eq(d[c.idx], c.want) {
					t.Fatalf("unfold n=%d: dst[%d] = %g, want %g", n, c.idx, d[c.idx], c.want)
				}
				if d32[c.idx] != float32(c.want) {
					t.Fatalf("unfold32 n=%d: dst[%d] = %g, want %g", n, c.idx, d32[c.idx], float32(c.want))
				}
			}
		}
	}
}

func TestMDCTFoldPanicsAndBounds(t *testing.T) {
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
	f8, f6, f7 := make([]float64, 8), make([]float64, 6), make([]float64, 7)
	g8, g6 := make([]float32, 8), make([]float32, 6)
	mustPanic("fold odd quarter", "floats: MDCTFold: len(x) is not a multiple of 4, or len(dst) < len(x)/2", func() { MDCTFold(f8, f6) })
	mustPanic("fold short dst", "floats: MDCTFold: len(x) is not a multiple of 4, or len(dst) < len(x)/2", func() { MDCTFold(f8[:3], f8) })
	mustPanic("fold32 odd quarter", "floats: MDCTFoldFrom32: len(x) is not a multiple of 4, or len(dst) < len(x)/2", func() { MDCTFoldFrom32(f8, g6) })
	mustPanic("fold32 short dst", "floats: MDCTFoldFrom32: len(x) is not a multiple of 4, or len(dst) < len(x)/2", func() { MDCTFoldFrom32(f8[:3], g8) })
	mustPanic("unfold odd", "floats: MDCTUnfold: len(u) is odd, or len(dst) < 2*len(u)", func() { MDCTUnfold(f8, f7) })
	mustPanic("unfold short", "floats: MDCTUnfold: len(u) is odd, or len(dst) < 2*len(u)", func() { MDCTUnfold(f6, f6) })
	mustPanic("unfold32 odd", "floats: MDCTUnfoldTo32: len(u) is odd, or len(dst) < 2*len(u)", func() { MDCTUnfoldTo32(g8, f7) })
	mustPanic("unfold32 short", "floats: MDCTUnfoldTo32: len(u) is odd, or len(dst) < 2*len(u)", func() { MDCTUnfoldTo32(g6, f6) })
	MDCTFold(nil, nil) // n = 0: nothing
	MDCTFoldFrom32(nil, nil)
	MDCTUnfold(nil, nil)
	MDCTUnfoldTo32(nil, nil)

	// overlap: refused, exactly or shifted, also across element types
	buf := make([]float64, 64)
	mustPanic("fold dst == x", "floats: MDCTFold: dst overlaps x", func() { MDCTFold(buf[:8], buf[:16]) })
	mustPanic("fold shifted", "floats: MDCTFold: dst overlaps x", func() { MDCTFold(buf[10:18], buf[:16]) })
	mustPanic("fold32 overlap", "floats: MDCTFoldFrom32: dst overlaps x", func() {
		MDCTFoldFrom32(buf[:8], unsafe.Slice((*float32)(unsafe.Pointer(&buf[0])), 16))
	})
	mustPanic("unfold dst == u", "floats: MDCTUnfold: dst overlaps u", func() { MDCTUnfold(buf[:16], buf[:8]) })
	mustPanic("unfold32 overlap", "floats: MDCTUnfoldTo32: dst overlaps u", func() {
		MDCTUnfoldTo32(unsafe.Slice((*float32)(unsafe.Pointer(&buf[4])), 16), buf[:8])
	})

	// adjacent slices of one array do not overlap; nothing is written outside
	// dst[:n/2] (or dst[:2m]) and nothing is read past the lengths
	rng := rand.New(rand.NewSource(82))
	for _, n := range foldSizes {
		x := foldInputs(rng, n)
		pad := make([]float64, n/2+16)
		for i := range pad {
			pad[i] = 12345
		}
		long := append(append([]float64(nil), x...), 1e300, 1e300, 1e300, 1e300)
		MDCTFold(pad[8:8+n/2], long[:n])
		for i := range pad {
			if (i < 8 || i >= 8+n/2) && pad[i] != 12345 {
				t.Fatalf("MDCTFold n=%d: wrote outside dst at %d", n, i)
			}
		}
		for i := 8; i < 8+n/2; i++ {
			if math.Abs(pad[i]) > 1e6 {
				t.Fatalf("MDCTFold n=%d: an operand past len was read (%g)", n, pad[i])
			}
		}
		u := foldInputs(rng, n/2)
		pad2 := make([]float64, n+16)
		pad32 := make([]float32, n+16)
		for i := range pad2 {
			pad2[i], pad32[i] = 12345, 12345
		}
		ulong := append(append([]float64(nil), u...), 1e300, 1e300)
		MDCTUnfold(pad2[8:8+n], ulong[:n/2])
		MDCTUnfoldTo32(pad32[8:8+n], ulong[:n/2])
		for i := range pad2 {
			if (i < 8 || i >= 8+n) && (pad2[i] != 12345 || pad32[i] != 12345) {
				t.Fatalf("unfold n=%d: wrote outside dst at %d", n, i)
			}
		}
	}
}

// FuzzMDCTFold: for any block length 4 to 260 and any bytes, the kernels and the
// portable loops agree.
func FuzzMDCTFold(f *testing.F) {
	f.Add(uint8(0), []byte{1, 2, 3})
	f.Add(uint8(1), []byte("fold and unfold"))
	f.Add(uint8(2), []byte{255, 0, 128})
	f.Add(uint8(64), []byte{})
	f.Fuzz(func(t *testing.T, k uint8, data []byte) {
		n := 4 * (int(k)%65 + 1)
		at := func(i, mul int) float64 {
			if len(data) == 0 {
				return 0
			}
			return float64(int(data[(i*mul+1)%len(data)])-128) / 8
		}
		x := make([]float64, n)
		u := make([]float64, n/2)
		for i := range x {
			x[i] = at(i, 3)
		}
		for i := range u {
			u[i] = at(i, 5)
		}
		got, ref := make([]float64, n/2), make([]float64, n/2)
		MDCTFold(got, x)
		foldTail(ref, x, 0)
		x32 := to32(x)
		g32, r32 := make([]float64, n/2), make([]float64, n/2)
		MDCTFoldFrom32(g32, x32)
		fold32Tail(r32, x32, 0)
		for i := range got {
			if !eq(got[i], ref[i]) || !eq(g32[i], r32[i]) {
				t.Fatalf("fold n=%d i=%d", n, i)
			}
		}
		d, rd := make([]float64, n), make([]float64, n)
		MDCTUnfold(d, u)
		unfoldTail(rd, u, 0)
		d32, rd32 := make([]float32, n), make([]float32, n)
		MDCTUnfoldTo32(d32, u)
		unfold32Tail(rd32, u, 0)
		for i := range d {
			if !eq(d[i], rd[i]) || d32[i] != rd32[i] {
				t.Fatalf("unfold n=%d i=%d", n, i)
			}
		}
	})
}

func BenchmarkMDCTFold(b *testing.B) {
	rng := rand.New(rand.NewSource(83))
	for _, n := range []int{128, 256, 2048} { // the block length N
		x := foldInputs(rng, n)
		u := foldInputs(rng, n/2)
		x32 := to32(x)
		d, d2 := make([]float64, n/2), make([]float64, n)
		d32 := make([]float32, n)
		name := map[int]string{128: "128", 256: "256", 2048: "2048"}[n]
		b.Run("fold/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				MDCTFold(d, x)
			}
		})
		b.Run("fold-portable/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				foldTail(d, x, 0)
			}
		})
		b.Run("foldFrom32/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				MDCTFoldFrom32(d, x32)
			}
		})
		b.Run("unfold/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				MDCTUnfold(d2, u)
			}
		})
		b.Run("unfold-portable/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				unfoldTail(d2, u, 0)
			}
		})
		b.Run("unfoldTo32/"+name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				MDCTUnfoldTo32(d32, u)
			}
		})
	}
}
