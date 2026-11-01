package floats

import (
	"encoding/binary"
	"math"
	"math/rand"
	"testing"
)

// ---- SIMD kernel versus portable reference ----

func scalarsNF(rng *rand.Rand) []float64 {
	return []float64{0, negZero, 1, -1, 0.5, 1 + 0x1p-27, rng.NormFloat64(), math.NaN(), math.Inf(1), math.Inf(-1), 0x1p600}
}

func TestAxpyNoFMAMatchesReference(t *testing.T) {
	forEachCase(t, 31, func(n, off int, sp bool, rng *rand.Rand) {
		x, _ := fillE(n, off, rng, sp)
		y0, _ := fillE(n, (off+1)%4, rng, sp)
		for _, al := range scalarsNF(rng)[:6+rng.Intn(5)] {
			want := append([]float64(nil), y0...)
			for i := range want {
				want[i] = y0[i] + float64(al*x[i])
			}
			y, buf := fill(n, off, rng, false)
			copy(y, y0)
			AxpyNoFMA(al, x, y)
			checkSame(t, "AxpyNoFMA", y, want, n, off)
			checkGuard(t, "AxpyNoFMA", buf, n, off)
			copy(y, y0)
			axpyNoFMARef(al, x, y)
			checkSame(t, "axpyNoFMARef", y, want, n, off)
		}
	})
}

func TestXpbyNoFMAMatchesReference(t *testing.T) {
	forEachCase(t, 32, func(n, off int, sp bool, rng *rand.Rand) {
		x, _ := fillE(n, off, rng, sp)
		y0, _ := fillE(n, (off+1)%4, rng, sp)
		for _, be := range scalarsNF(rng)[:6+rng.Intn(5)] {
			want := make([]float64, n)
			for i := range want {
				want[i] = x[i] + float64(be*y0[i])
			}
			y, buf := fill(n, off, rng, false)
			copy(y, y0)
			XpbyNoFMA(x, be, y)
			checkSame(t, "XpbyNoFMA", y, want, n, off)
			checkGuard(t, "XpbyNoFMA", buf, n, off)
			copy(y, y0)
			xpbyNoFMARef(be, x, y)
			checkSame(t, "xpbyNoFMARef", y, want, n, off)
		}
	})
}

func TestAxpby3NoFMAMatchesReference(t *testing.T) {
	forEachCase(t, 33, func(n, off int, sp bool, rng *rand.Rand) {
		a, _ := fillE(n, off, rng, sp)
		b, _ := fillE(n, (off+1)%4, rng, sp)
		c, _ := fillE(n, (off+2)%4, rng, sp)
		sc := scalarsNF(rng)
		al, be, ga := sc[rng.Intn(len(sc))], sc[rng.Intn(len(sc))], sc[rng.Intn(len(sc))]
		want := make([]float64, n)
		for i := range want {
			p, q, r := float64(al*a[i]), float64(be*b[i]), float64(ga*c[i])
			want[i] = (p + q) + r
		}
		dst, buf := fill(n, (off+3)%4, rng, false)
		Axpby3NoFMA(dst, al, a, be, b, ga, c)
		checkSame(t, "Axpby3NoFMA", dst, want, n, off)
		checkGuard(t, "Axpby3NoFMA", buf, n, (off+3)%4)
		axpby3NoFMARef(dst, al, a, be, b, ga, c)
		checkSame(t, "axpby3NoFMARef", dst, want, n, off)
	})
}

func TestAxpyDiffNoFMAMatchesReference(t *testing.T) {
	forEachCase(t, 34, func(n, off int, sp bool, rng *rand.Rand) {
		a, _ := fillE(n, off, rng, sp)
		b, _ := fillE(n, (off+1)%4, rng, sp)
		c, _ := fillE(n, (off+2)%4, rng, sp)
		sc := scalarsNF(rng)
		f := sc[rng.Intn(len(sc))]
		want := make([]float64, n)
		for i := range want {
			d := b[i] - c[i]
			want[i] = a[i] + float64(f*d)
		}
		dst, buf := fill(n, (off+3)%4, rng, false)
		AxpyDiffNoFMA(dst, a, f, b, c)
		checkSame(t, "AxpyDiffNoFMA", dst, want, n, off)
		checkGuard(t, "AxpyDiffNoFMA", buf, n, (off+3)%4)
		axpyDiffNoFMARef(dst, a, f, b, c)
		checkSame(t, "axpyDiffNoFMARef", dst, want, n, off)
	})
}

// ---- the unfused kernels are not the fused ones ----

// nfProbe holds a value whose square is not representable: with e = 1+2^-27,
// e*e = 1 + 2^-26 + 2^-54, which rounds to 1 + 2^-26, so an unfused
// -(1+2^-26) + e*e is exactly 0 while a fused one leaves 2^-54.
const nfE = 1 + 0x1p-27

func TestNoFMAKernelsAreNotFused(t *testing.T) {
	prod := nfE * nfE // rounded: 1 + 2^-26
	if math.FMA(nfE, nfE, -prod) != 0x1p-54 {
		t.Fatal("probe is not sensitive to fusion")
	}
	for _, n := range []int{1, 3, 5, 17, 70} {
		fill := func(v float64) []float64 {
			s := make([]float64, n)
			for i := range s {
				s[i] = v
			}
			return s
		}
		allZero := func(what string, s []float64) {
			t.Helper()
			for i, v := range s {
				if v != 0 {
					t.Fatalf("%s n=%d [%d] = %v, want 0 (unfused)", what, n, i, v)
				}
			}
		}
		y := fill(-prod)
		AxpyNoFMA(nfE, fill(nfE), y)
		allZero("AxpyNoFMA", y)

		y = fill(nfE)
		XpbyNoFMA(fill(-prod), nfE, y)
		allZero("XpbyNoFMA", y)

		dst := make([]float64, n)
		Axpby3NoFMA(dst, 1, fill(-prod), 0, fill(0), nfE, fill(nfE))
		allZero("Axpby3NoFMA", dst)

		AxpyDiffNoFMA(dst, fill(-prod), nfE, fill(nfE), fill(0))
		allZero("AxpyDiffNoFMA", dst)

		// And the fused kernels on the same data are NOT zero: the two families
		// are different on purpose.
		y = fill(-prod)
		Axpy(nfE, fill(nfE), y)
		if y[0] == 0 {
			t.Fatal("Axpy unexpectedly unfused")
		}
	}
}

func TestNoFMAAliasing(t *testing.T) {
	rng := rand.New(rand.NewSource(35))
	for _, n := range []int{0, 1, 5, 16, 17, 33, 100} {
		a, b, c := randF64(n, rng), randF64(n, rng), randF64(n, rng)
		f := 0.37

		// dst == a: a += f*(b-c)
		want := make([]float64, n)
		axpyDiffNoFMARef(want, a, f, b, c)
		got := append([]float64(nil), a...)
		AxpyDiffNoFMA(got, got, f, b, c)
		checkSame(t, "AxpyDiffNoFMA dst==a", got, want, n, 0)

		// rand-to-best/1 step: b += f*(x - b): dst = a = c = b.
		axpyDiffNoFMARef(want, b, f, a, b)
		got = append([]float64(nil), b...)
		AxpyDiffNoFMA(got, got, f, a, got)
		checkSame(t, "AxpyDiffNoFMA dst==a==c", got, want, n, 0)

		// dst == b.
		axpyDiffNoFMARef(want, a, f, b, c)
		got = append([]float64(nil), b...)
		AxpyDiffNoFMA(got, a, f, got, c)
		checkSame(t, "AxpyDiffNoFMA dst==b", got, want, n, 0)

		// Axpby3NoFMA with dst == a and dst == c.
		axpby3NoFMARef(want, 0.3, a, -1.1, b, 2.2, c)
		got = append([]float64(nil), a...)
		Axpby3NoFMA(got, 0.3, got, -1.1, b, 2.2, c)
		checkSame(t, "Axpby3NoFMA dst==a", got, want, n, 0)
		got = append([]float64(nil), c...)
		Axpby3NoFMA(got, 0.3, a, -1.1, b, 2.2, got)
		checkSame(t, "Axpby3NoFMA dst==c", got, want, n, 0)

		// Axpy / Xpby with y == x.
		want = append([]float64(nil), a...)
		axpyNoFMARef(0.7, want, want)
		got = append([]float64(nil), a...)
		AxpyNoFMA(0.7, got, got)
		checkSame(t, "AxpyNoFMA y==x", got, want, n, 0)
		want = append([]float64(nil), a...)
		xpbyNoFMARef(0.7, want, want)
		got = append([]float64(nil), a...)
		XpbyNoFMA(got, 0.7, got)
		checkSame(t, "XpbyNoFMA y==x", got, want, n, 0)
	}
}

func TestNoFMALongerSlices(t *testing.T) {
	x := []float64{1, 2, 3}
	y := []float64{10, 20, 30, 40, 50}
	AxpyNoFMA(2, x, y)
	if y[0] != 12 || y[2] != 36 || y[3] != 40 || y[4] != 50 {
		t.Fatalf("AxpyNoFMA with a longer y: %v", y)
	}
	y = []float64{10, 20, 30, 40, 50}
	XpbyNoFMA(x, 2, y)
	if y[0] != 21 || y[2] != 63 || y[3] != 40 {
		t.Fatalf("XpbyNoFMA with a longer y: %v", y)
	}
	dst := []float64{9, 9, 9, 9, 9}
	b, c := []float64{5, 6, 7, 8}, []float64{1, 1, 1, 1}
	AxpyDiffNoFMA(dst, x, 2, b, c)
	if dst[0] != 9 || dst[1] != 12 || dst[2] != 15 || dst[3] != 9 { // a + 2*(b-c)
		t.Fatalf("AxpyDiffNoFMA with longer slices: %v", dst)
	}
	Axpby3NoFMA(dst, 1, x, 1, x, 1, x)
	if dst[0] != 3 || dst[1] != 6 || dst[2] != 9 || dst[3] != 9 {
		t.Fatalf("Axpby3NoFMA with longer dst: %v", dst)
	}
}

func TestNoFMAPanics(t *testing.T) {
	short := []float64{1}
	long := []float64{1, 2}
	cases := map[string]func(){
		"AxpyNoFMA":    func() { AxpyNoFMA(1, long, short) },
		"XpbyNoFMA":    func() { XpbyNoFMA(long, 1, short) },
		"Axpby3 b":     func() { Axpby3NoFMA(long, 1, long, 1, short, 1, long) },
		"Axpby3 c":     func() { Axpby3NoFMA(long, 1, long, 1, long, 1, short) },
		"Axpby3 dst":   func() { Axpby3NoFMA(short, 1, long, 1, long, 1, long) },
		"AxpyDiff b":   func() { AxpyDiffNoFMA(long, long, 1, short, long) },
		"AxpyDiff c":   func() { AxpyDiffNoFMA(long, long, 1, long, short) },
		"AxpyDiff dst": func() { AxpyDiffNoFMA(short, long, 1, long, long) },
	}
	for name, f := range cases {
		func() {
			defer func() {
				if recover() == nil {
					t.Errorf("%s did not panic", name)
				}
			}()
			f()
		}()
	}
}

// ---- fuzz ----

func f64s(data []byte) []float64 {
	a := make([]float64, len(data)/8)
	for i := range a {
		a[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[8*i:]))
	}
	return a
}

func FuzzNoFMAKernels(f *testing.F) {
	f.Add([]byte{})
	seed := make([]byte, 8*40)
	r := rand.New(rand.NewSource(1))
	for i := 0; i < 40; i++ {
		binary.LittleEndian.PutUint64(seed[8*i:], math.Float64bits(r.NormFloat64()))
	}
	f.Add(seed)
	f.Fuzz(func(t *testing.T, data []byte) {
		v := f64s(data)
		// Four equal parts: a, b, c, and scalars+y.
		n := len(v) / 4
		if n == 0 {
			return
		}
		a, b, c, d := v[:n], v[n:2*n], v[2*n:3*n], v[3*n:4*n]
		al, be, ga := d[0], d[n/2], d[n-1]
		got, want := make([]float64, n), make([]float64, n)

		axpyDiffNoFMARef(want, a, al, b, c)
		AxpyDiffNoFMA(got, a, al, b, c)
		checkSame(t, "AxpyDiffNoFMA", got, want, n, 0)

		axpby3NoFMARef(want, al, a, be, b, ga, c)
		Axpby3NoFMA(got, al, a, be, b, ga, c)
		checkSame(t, "Axpby3NoFMA", got, want, n, 0)

		w := append([]float64(nil), b...)
		axpyNoFMARef(al, a, w)
		g := append([]float64(nil), b...)
		AxpyNoFMA(al, a, g)
		checkSame(t, "AxpyNoFMA", g, w, n, 0)

		w = append([]float64(nil), b...)
		xpbyNoFMARef(be, a, w)
		g = append([]float64(nil), b...)
		XpbyNoFMA(a, be, g)
		checkSame(t, "XpbyNoFMA", g, w, n, 0)
	})
}

// ---- guard pages: no access outside the slices ----

func TestNoFMAScalarSpecMatchesKernels(t *testing.T) {
	rng := rand.New(rand.NewSource(36))
	for n := 0; n < 40; n++ {
		a, b, c := randF64(n, rng), randF64(n, rng), randF64(n, rng)
		got := make([]float64, n)
		AxpyDiffNoFMA(got, a, 0.9, b, c)
		for i := range got {
			if w := nfElem(opAxpyDiff, [3]float64{0.9}, [4]float64{a[i], b[i], c[i]}); !sameBits(got[i], w) {
				t.Fatalf("nfElem diff n=%d [%d]: %v vs %v", n, i, got[i], w)
			}
		}
	}
}
