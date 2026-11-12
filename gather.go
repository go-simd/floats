package floats

import "math"

// Gather kernels: the dot products and the AXPY of a vector read through an
// index list, the free-set operations of bound-constrained solvers (L-BFGS-B's
// subspace minimisation, active-set QP). Requested by go-numerics/optimize
// (lbfgsb_fast.go carries a local copy until this lands). The reference below
// is portable Go on four FMA accumulators; a SIMD gather (AVX2 vgatherdpd,
// SVE ld1d gather) can replace it per architecture behind the same signature.
// Sums are lane-blocked like Dot's: not the left-to-right order.

// checkLen panics when d is shorter than idx. A bare d[:len(idx)] would not:
// reslicing up to cap(d) is legal, so a short d with spare capacity would be
// read or written past its length.
func checkLen(d []float64, idx []int) {
	if len(d) < len(idx) {
		panic("floats: len(d) < len(idx)")
	}
}

// DotIdx returns Σ a[idx[k]]·b[idx[k]]. It panics if an index is out of range.
func DotIdx(a, b []float64, idx []int) float64 {
	var s0, s1, s2, s3 float64
	k := 0
	for ; k+4 <= len(idx); k += 4 {
		i0, i1, i2, i3 := idx[k], idx[k+1], idx[k+2], idx[k+3]
		s0 = math.FMA(a[i0], b[i0], s0)
		s1 = math.FMA(a[i1], b[i1], s1)
		s2 = math.FMA(a[i2], b[i2], s2)
		s3 = math.FMA(a[i3], b[i3], s3)
	}
	for ; k < len(idx); k++ {
		i := idx[k]
		s0 = math.FMA(a[i], b[i], s0)
	}
	return (s0 + s1) + (s2 + s3)
}

// DotGather returns Σ a[idx[k]]·d[k] (d is contiguous, a is gathered). It
// panics if an index is out of range or len(d) < len(idx).
func DotGather(a []float64, idx []int, d []float64) float64 {
	checkLen(d, idx)
	d = d[:len(idx)]
	var s0, s1, s2, s3 float64
	k := 0
	for ; k+4 <= len(idx); k += 4 {
		s0 = math.FMA(a[idx[k]], d[k], s0)
		s1 = math.FMA(a[idx[k+1]], d[k+1], s1)
		s2 = math.FMA(a[idx[k+2]], d[k+2], s2)
		s3 = math.FMA(a[idx[k+3]], d[k+3], s3)
	}
	for ; k < len(idx); k++ {
		s0 = math.FMA(a[idx[k]], d[k], s0)
	}
	return (s0 + s1) + (s2 + s3)
}

// AxpyGather does d[k] = fma(alpha, a[idx[k]], d[k]) for every k (each element
// rounded once, bit-identical on every architecture). It panics if an index is
// out of range or len(d) < len(idx).
func AxpyGather(alpha float64, a []float64, idx []int, d []float64) {
	checkLen(d, idx)
	d = d[:len(idx)]
	for k, i := range idx {
		d[k] = math.FMA(alpha, a[i], d[k])
	}
}
