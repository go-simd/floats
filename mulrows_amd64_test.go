//go:build amd64

package floats

import (
	"math/rand"
	"testing"
)

// TestMulAddRowsPortableForced: the path of a CPU without AVX2+FMA, forced on any host,
// gives the same bits as the definition at every shape.
func TestMulAddRowsPortableForced(t *testing.T) {
	saved := hasAVX2FMA
	defer func() { hasAVX2FMA = saved }()
	hasAVX2FMA = false
	TestMulAddRowsExact(t)
}

// TestMulAddRowsAVX2Dispatch: on a CPU with AVX2+FMA the public function runs the
// kernel, and the kernel equals the portable loop bit for bit at every shape.
func TestMulAddRowsAVX2Dispatch(t *testing.T) {
	if !hasAVX2FMA {
		t.Skip("no AVX2+FMA on this CPU: the portable loop runs, covered by TestMulAddRowsPortableForced")
	}
	rng := rand.New(rand.NewSource(205))
	for _, s := range mulRowsShapes() {
		n, rows, lda, ldb := s[0], s[1], s[2], s[3]
		a := mulRowsInputs(rng, (rows-1)*lda+n)
		b := mulRowsInputs(rng, (rows-1)*ldb+n)
		d0 := mulRowsInputs(rng, n)
		ker, ref := append([]float64(nil), d0...), append([]float64(nil), d0...)
		mulAddRowsAVX2(ker, a, lda, b, ldb, rows)
		mulAddRowsRef(ref, a, lda, b, ldb, rows)
		if i := mulRowsDiff(ker, ref); i >= 0 {
			t.Fatalf("AVX2 kernel n=%d rows=%d lda=%d ldb=%d element %d differs from the portable loop", n, rows, lda, ldb, i)
		}
	}
}
