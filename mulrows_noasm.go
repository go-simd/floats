//go:build !arm64 && !amd64

package floats

// No multiply-accumulate-rows kernel on this architecture: the portable math.FMA loop of
// mulrows.go, which the compiler lowers to one hardware FMA per term where there is one.

func mulAddRows(dst, a []float64, lda int, b []float64, ldb int, rows int) {
	mulAddRowsRef(dst, a, lda, b, ldb, rows)
}
