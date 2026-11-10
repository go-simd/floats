//go:build ppc64le

package floats

import "math"

// VSX float32 Max/Min kernels (extrema32_ppc64le_gen.go): whole blocks of 16
// reduced with xvmaxsp/xvminsp, then the four lanes and the tail folded with
// Go's max/min. NaN is reported separately by an integer magnitude test, and a
// zero result goes to the scalar loop for its sign. Gated on hasVSX.
//
//go:noescape
func max32VSX(a *float32, n int, out *[4]float32) int64

//go:noescape
func min32VSX(a *float32, n int, out *[4]float32) int64

func maxF32(a []float32) float32 {
	n := len(a) &^ 15
	if !hasVSX || n == 0 {
		return maxRef32(a)
	}
	var out [4]float32
	if max32VSX(&a[0], n, &out) != 0 {
		return float32(math.NaN())
	}
	m := max(max(out[0], out[1]), max(out[2], out[3]))
	for _, v := range a[n:] {
		m = max(m, v)
	}
	if m == 0 {
		return maxRef32(a)
	}
	return m
}

func minF32(a []float32) float32 {
	n := len(a) &^ 15
	if !hasVSX || n == 0 {
		return minRef32(a)
	}
	var out [4]float32
	if min32VSX(&a[0], n, &out) != 0 {
		return float32(math.NaN())
	}
	m := min(min(out[0], out[1]), min(out[2], out[3]))
	for _, v := range a[n:] {
		m = min(m, v)
	}
	if m == 0 {
		return minRef32(a)
	}
	return m
}
