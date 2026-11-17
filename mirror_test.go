package floats

import (
	"math"
	"math/big"
	"math/rand"
	"testing"
)

// mirrorOracle is r[k] in 300-bit arithmetic, with the scale |w1||u| + |w2||v| the
// error bound is relative to.
func mirrorOracle(k, n int, z, w1, w2 []float64) (rr, ri, scale float64) {
	f := func(x float64) *big.Float { return new(big.Float).SetPrec(300).SetFloat64(x) }
	add := func(a, b *big.Float) *big.Float { return new(big.Float).SetPrec(300).Add(a, b) }
	sub := func(a, b *big.Float) *big.Float { return new(big.Float).SetPrec(300).Sub(a, b) }
	mul := func(a, b *big.Float) *big.Float { return new(big.Float).SetPrec(300).Mul(a, b) }
	ar, ai := f(z[2*k]), f(z[2*k+1])
	br, bi := f(z[2*(n-k)]), f(z[2*(n-k)+1])
	ur, ui := add(ar, br), sub(ai, bi)
	vr, vi := add(ai, bi), sub(br, ar)
	w1r, w1i, w2r, w2i := f(w1[2*k]), f(w1[2*k+1]), f(w2[2*k]), f(w2[2*k+1])
	r := add(sub(mul(w1r, ur), mul(w1i, ui)), sub(mul(w2r, vr), mul(w2i, vi)))
	i := add(add(mul(w1r, ui), mul(w1i, ur)), add(mul(w2r, vi), mul(w2i, vr)))
	rr, _ = r.Float64()
	ri, _ = i.Float64()
	fu, _ := ur.Float64()
	fv, _ := ui.Float64()
	gu, _ := vr.Float64()
	gv, _ := vi.Float64()
	scale = math.Hypot(w1[2*k], w1[2*k+1])*math.Hypot(fu, fv) + math.Hypot(w2[2*k], w2[2*k+1])*math.Hypot(gu, gv)
	return rr, ri, scale
}

const mirrorSentinel = 1234.5

func mirrorInputs(rng *rand.Rand, n int) (z, w1, w2 []float64) {
	z, w1, w2 = make([]float64, 2*n), make([]float64, 2*n), make([]float64, 2*n)
	for i := range z {
		z[i] = rng.NormFloat64() * 4
		w1[i] = rng.NormFloat64()
		w2[i] = rng.NormFloat64()
	}
	return
}

// checkMirror asserts both functions against the oracle for every n in lens, and
// that the entries they do not own are untouched.
func checkMirror(t *testing.T, lens []int) {
	t.Helper()
	const bound = 4 * 0x1p-52
	rng := rand.New(rand.NewSource(1))
	for _, n := range lens {
		z, w1, w2 := mirrorInputs(rng, n)
		dst := make([]float64, 2*n+3)
		for i := range dst {
			dst[i] = mirrorSentinel
		}
		MirrorCombine(dst[:2*n], z, w1, w2)
		for k := 0; k < n; k++ {
			if k == 0 {
				if dst[0] != mirrorSentinel || dst[1] != mirrorSentinel {
					t.Fatalf("n=%d: MirrorCombine wrote entry 0", n)
				}
				continue
			}
			rr, ri, sc := mirrorOracle(k, n, z, w1, w2)
			if math.Abs(dst[2*k]-rr) > bound*sc || math.Abs(dst[2*k+1]-ri) > bound*sc {
				t.Fatalf("n=%d k=%d: got (%v,%v) want (%v,%v) scale %v", n, k, dst[2*k], dst[2*k+1], rr, ri, sc)
			}
		}
		for i := 2 * n; i < len(dst); i++ {
			if dst[i] != mirrorSentinel {
				t.Fatalf("n=%d: MirrorCombine wrote past 2n at %d", n, i)
			}
		}
		for i := range dst {
			dst[i] = mirrorSentinel
		}
		MirrorCombineSplit(dst[:2*n], z, w1, w2)
		written := make([]bool, len(dst))
		for k := 1; k < n; k++ {
			rr, ri, sc := mirrorOracle(k, n, z, w1, w2)
			if math.Abs(dst[k]-rr) > bound*sc || math.Abs(dst[2*n-k]+ri) > bound*sc {
				t.Fatalf("n=%d k=%d: split got (%v,%v) want (%v,%v) scale %v", n, k, dst[k], dst[2*n-k], rr, -ri, sc)
			}
			written[k], written[2*n-k] = true, true
		}
		for i := range dst {
			if !written[i] && dst[i] != mirrorSentinel {
				t.Fatalf("n=%d: MirrorCombineSplit wrote dst[%d], which is not r's", n, i)
			}
		}
	}
}

func mirrorLens() []int {
	var l []int
	for n := 0; n <= 130; n++ {
		l = append(l, n)
	}
	return append(l, 255, 256, 257, 1000, 4097)
}

func TestMirrorCombineOracle(t *testing.T) { checkMirror(t, mirrorLens()) }

// TestMirrorCombineMutation: the oracle must see a wrong kernel, the way a swapped
// sign of v would be wrong; here a reference with the sign of v flipped differs from
// the oracle by far more than the bound at every k.
func TestMirrorCombineMutation(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	n := 16
	z, w1, w2 := mirrorInputs(rng, n)
	for k := 1; k < n; k++ {
		for i := range w2 {
			w2[i] = -w2[i]
		}
		rr, ri := mirrorAt(k, n, z, w1, w2)
		for i := range w2 {
			w2[i] = -w2[i]
		}
		or, oi, sc := mirrorOracle(k, n, z, w1, w2)
		if math.Abs(rr-or) <= 1e-9*sc && math.Abs(ri-oi) <= 1e-9*sc {
			t.Fatalf("k=%d: a flipped w2 is not told from the oracle", k)
		}
	}
}

func TestMirrorCombineImpulse(t *testing.T) {
	// z = delta at entry j: r[k] is exactly w1[k] u + w2[k] v for the two k that see it.
	n := 9
	for j := 1; j < n; j++ {
		z := make([]float64, 2*n)
		z[2*j] = 1 // real impulse
		w1, w2 := make([]float64, 2*n), make([]float64, 2*n)
		for k := 0; k < n; k++ {
			w1[2*k] = 1
			w2[2*k+1] = 1 // w2 = i
		}
		dst := make([]float64, 2*n)
		MirrorCombine(dst, z, w1, w2)
		for _, k := range []int{j, n - j} {
			// a or b is the impulse: u = 1, v = -i(1)=-i  (k = j) or  u = 1, v = -i(-1) = i
			wantI := -1.0
			if k == n-j {
				wantI = 1
			}
			// r = 1*u + i*v = 1 + i*(0 + wantI*i) = 1 - wantI
			want := 1 - wantI
			if j == n-j {
				continue
			}
			if dst[2*k] != want {
				t.Errorf("n=%d j=%d k=%d: Re r = %v want %v", n, j, k, dst[2*k], want)
			}
		}
	}
}

func TestMirrorCombinePanics(t *testing.T) {
	ok := func() (dst, z, w1, w2 []float64) {
		return make([]float64, 8), make([]float64, 8), make([]float64, 8), make([]float64, 8)
	}
	cases := []struct {
		name string
		cut  func(dst, z, w1, w2 []float64) (a, b, c, d []float64)
	}{
		{"odd z", func(dst, z, w1, w2 []float64) (a, b, c, d []float64) { return dst, z[:7], w1, w2 }},
		{"short w1", func(dst, z, w1, w2 []float64) (a, b, c, d []float64) { return dst, z, w1[:7], w2 }},
		{"short w2", func(dst, z, w1, w2 []float64) (a, b, c, d []float64) { return dst, z, w1, w2[:7] }},
		{"short dst", func(dst, z, w1, w2 []float64) (a, b, c, d []float64) { return dst[:7], z, w1, w2 }},
		{"dst is z", func(dst, z, w1, w2 []float64) (a, b, c, d []float64) { return z, z, w1, w2 }},
		{"dst is w1", func(dst, z, w1, w2 []float64) (a, b, c, d []float64) { return w1, z, w1, w2 }},
	}
	for _, c := range cases {
		for name, f := range map[string]func(dst, z, w1, w2 []float64){"MirrorCombine": MirrorCombine, "MirrorCombineSplit": MirrorCombineSplit} {
			dst, z, w1, w2 := ok()
			for i := range dst {
				dst[i], z[i], w1[i], w2[i] = mirrorSentinel, 1, 1, 1
			}
			a, b, cc, d := c.cut(dst, z, w1, w2)
			func() {
				defer func() {
					if recover() == nil {
						t.Errorf("%s %s: no panic", name, c.name)
					}
				}()
				f(a, b, cc, d)
			}()
			if c.name != "dst is z" && c.name != "dst is w1" {
				for i, v := range dst {
					if v != mirrorSentinel {
						t.Errorf("%s %s: dst[%d] written before the panic", name, c.name, i)
					}
				}
			}
		}
	}
}

func FuzzMirrorCombine(f *testing.F) {
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8}, uint8(3))
	f.Add(make([]byte, 8*40), uint8(7))
	f.Fuzz(func(t *testing.T, data []byte, nn uint8) {
		a := bytesToF64(data)
		n := int(nn) % 70
		if len(a) < 6*n {
			return
		}
		z, w1, w2 := a[:2*n], a[2*n:4*n], a[4*n:6*n]
		dst := make([]float64, 2*n)
		MirrorCombine(dst, z, w1, w2)
		dst2 := make([]float64, 2*n)
		MirrorCombineSplit(dst2, z, w1, w2)
		for k := 1; k < n; k++ {
			rr, ri, sc := mirrorOracle(k, n, z, w1, w2)
			tol := 4 * 0x1p-52 * sc
			if math.Abs(dst[2*k]-rr) > tol || math.Abs(dst[2*k+1]-ri) > tol ||
				math.Abs(dst2[k]-rr) > tol || math.Abs(dst2[2*n-k]+ri) > tol {
				t.Fatalf("n=%d k=%d: kernel (%v,%v | %v,%v) oracle (%v,%v) tol %v", n, k, dst[2*k], dst[2*k+1], dst2[k], dst2[2*n-k], rr, ri, tol)
			}
		}
	})
}
