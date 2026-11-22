//go:build !arm64 && !amd64

package floats

// No float32 multiply-accumulate-rows kernel on this architecture: the portable loop of
// mulrows32.go.

func mulAddRows32(dst, a []float32, lda int, b []float32, ldb int, rows int) {
	mulAddRows32Ref(dst, a, lda, b, ldb, rows)
}
