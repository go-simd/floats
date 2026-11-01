//go:build !amd64 && !arm64

package floats

// On architectures without an unfused SIMD kernel the portable loops are the
// kernels; they give the same bits (nofma.go, sumpairwise.go).

func axpyNoFMA(alpha float64, x, y []float64) { axpyNoFMARef(alpha, x, y) }
func xpbyNoFMA(beta float64, x, y []float64)  { xpbyNoFMARef(beta, x, y) }
func axpby3NoFMA(dst []float64, alpha float64, a []float64, beta float64, b []float64, gamma float64, c []float64) {
	axpby3NoFMARef(dst, alpha, a, beta, b, gamma, c)
}
func axpyDiffNoFMA(dst, a []float64, f float64, b, c []float64) {
	axpyDiffNoFMARef(dst, a, f, b, c)
}
func pairwiseBlock(a []float64) float64 { return pairwiseBlockRef(a) }
