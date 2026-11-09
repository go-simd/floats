package floats

import (
	"math"
	"math/rand"
	"testing"
)

// rowsWant is the contract: len(a) successive AxpyNoFMA calls, one per row.
func rowsWant(a, x []float64, ldx int, y0 []float64) []float64 {
	want := append([]float64(nil), y0...)
	for k, ak := range a {
		axpyNoFMARef(ak, x[k*ldx:k*ldx+len(y0)], want)
	}
	return want
}

// TestAxpyRowsNoFMAMatchesAxpyLoop: every column count 0..70 (all block
// boundaries of the 32/16/8/4/2/1 loops), several row counts, unaligned starts,
// a row stride with and without padding, planted specials (NaN, Inf, signed
// zeros), against r AxpyNoFMA calls and the portable reference.
func TestAxpyRowsNoFMAMatchesAxpyLoop(t *testing.T) {
	rng := rand.New(rand.NewSource(77))
	for n := 0; n <= 70; n++ {
		for _, r := range []int{0, 1, 2, 3, 4, 5, 8, 9, 17} {
			for off := 0; off < 4; off++ {
				for _, pad := range []int{0, 3} {
					for _, sp := range []bool{false, true} {
						ldx := n + pad
						x, _ := fill(r*ldx+n, off, rng, sp)
						a, _ := fill(r, (off+1)%4, rng, sp)
						y0, _ := fillE(n, (off+2)%4, rng, sp)
						want := rowsWant(a, x, ldx, y0)
						y, buf := fill(n, off, rng, false)
						copy(y, y0)
						AxpyRowsNoFMA(a, x, ldx, y)
						checkSame(t, "AxpyRowsNoFMA", y, want, n, off)
						checkGuard(t, "AxpyRowsNoFMA", buf, n, off)
						copy(y, y0)
						axpyRowsNoFMARef(a, x, ldx, y)
						checkSame(t, "axpyRowsNoFMARef", y, want, n, off)
						// the same bits as real AxpyNoFMA calls
						copy(y, y0)
						for k := range a {
							AxpyNoFMA(a[k], x[k*ldx:k*ldx+n], y)
						}
						checkSame(t, "AxpyNoFMA loop", y, want, n, off)
					}
				}
			}
		}
	}
}

// TestAxpyRowsNoFMAIsNotFused: the probe of TestNoFMAKernelsAreNotFused, across
// several rows: -(1+2^-26) + e*e rounds to exactly 0 unfused.
func TestAxpyRowsNoFMAIsNotFused(t *testing.T) {
	prod := nfE * nfE
	for _, n := range []int{1, 3, 5, 17, 70} {
		y := make([]float64, n)
		x := make([]float64, 2*n)
		for i := range y {
			y[i] = -prod
		}
		for i := range x {
			x[i] = nfE
		}
		AxpyRowsNoFMA([]float64{nfE, 0}, x, n, y)
		for i, v := range y {
			if v != 0 {
				t.Fatalf("n=%d [%d] = %v, want 0 (unfused)", n, i, v)
			}
		}
	}
}

func TestAxpyRowsNoFMAPanics(t *testing.T) {
	mustPanic := func(what string, f func()) {
		t.Helper()
		defer func() {
			if recover() == nil {
				t.Errorf("%s: no panic", what)
			}
		}()
		f()
	}
	y := make([]float64, 4)
	mustPanic("short x", func() { AxpyRowsNoFMA(make([]float64, 3), make([]float64, 4+2*4-1), 4, y) })
	mustPanic("negative ldx", func() { AxpyRowsNoFMA(make([]float64, 3), make([]float64, 100), -1, y) })
	mustPanic("huge ldx", func() { AxpyRowsNoFMA(make([]float64, 3), make([]float64, 100), math.MaxInt, y) })
	AxpyRowsNoFMA(make([]float64, 3), make([]float64, 4+2*4), 4, y) // exact fit: fine
	AxpyRowsNoFMA(nil, nil, -5, y)                                  // no rows: nothing checked
	AxpyRowsNoFMA([]float64{1}, nil, 0, nil)                        // no columns: nothing checked
}

// FuzzAxpyRowsNoFMA: arbitrary shapes and finite data against the AxpyNoFMA loop,
// bit for bit.
func FuzzAxpyRowsNoFMA(f *testing.F) {
	f.Add([]byte{3, 5, 2, 1, 2, 3, 4, 5, 6, 7, 8}, uint8(3))
	f.Add(make([]byte, 8*40), uint8(7))
	f.Fuzz(func(t *testing.T, data []byte, shape uint8) {
		v := bytesToF64(data)
		r := int(shape&7) + 1
		pad := int(shape >> 6)
		if len(v) < r+1 {
			return
		}
		a, v := v[:r], v[r:]
		n := len(v) / (r + 1)
		if n == 0 {
			return
		}
		ldx := n + pad
		x := make([]float64, (r-1)*ldx+n)
		copy(x, v)
		y0 := make([]float64, n)
		copy(y0, v[len(v)-n:])
		want := rowsWant(a, x, ldx, y0)
		y := append([]float64(nil), y0...)
		AxpyRowsNoFMA(a, x, ldx, y)
		checkSame(t, "AxpyRowsNoFMA", y, want, n, 0)
	})
}
