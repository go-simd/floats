package floats

import (
	"math/rand"
	"testing"
)

// allPass3Go is the loop a decoder writes with complex128 arithmetic: the baseline the
// kernel is measured against (it rounds every product, so it is not bit-equal).
func allPass3Go(n int, out, in []complex128, t *AllPassRows, old, cur *[3][]complex128) {
	q := [3][]complex128{t.Q0, t.Q1, t.Q2}
	cc := [3][]float64{t.C0, t.C1, t.C2}
	for k := 0; k < n; k++ {
		x := t.Phi[k] * in[k]
		for l := 0; l < 3; l++ {
			p := q[l][k] * old[l][k]
			c := x + complex(cc[l][k]*real(p), cc[l][k]*imag(p))
			cur[l][k] = c
			x = p - complex(cc[l][k]*real(c), cc[l][k]*imag(c))
		}
		out[k] = x
	}
}

// TestAllPass3GoBaselineWithinBound: the plain complex128 loop of the benchmark stays within
// twice the documented bound (it is the loop the kernel replaces, so the bound is the
// justification for the last-bit differences a caller sees).
func TestAllPass3GoBaselineWithinBound(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	for _, n := range []int{1, 7, 34} {
		f := newAPFix(rng, n, 0, 0)
		in0 := f.clone()
		allPass3Go(n, f.out, f.in, &f.t, &f.old, &f.cur)
		for k := 0; k < n; k++ {
			if w, msg := apBound(f, in0, k); w > 2 {
				t.Fatalf("n=%d: baseline at %.2f of the bound: %s", n, w, msg)
			}
		}
	}
}

func BenchmarkAllPass3(b *testing.B) {
	const n = 34
	f := newAPFix(rand.New(rand.NewSource(1)), n, 0, 0)
	b.Run("kernel", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			AllPass3(f.out, f.in, &f.t, &f.old, &f.cur)
		}
	})
	b.Run("portable-fma", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			allPass3Ref(n, f.out, f.in, &f.t, &f.old, &f.cur)
		}
	})
	b.Run("complex128-loop", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			allPass3Go(n, f.out, f.in, &f.t, &f.old, &f.cur)
		}
	})
}
