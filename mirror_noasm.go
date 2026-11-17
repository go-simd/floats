//go:build !amd64

package floats

// No mirror-combiner kernel on this architecture: the portable loops of mirror.go.

func mirrorCombine(dst, z, w1, w2 []float64, n int) { mirrorTail(dst, z, w1, w2, n, 1) }

func mirrorCombineSplit(dst, z, w1, w2 []float64, n int) {
	mirrorSplitTail(dst, z, w1, w2, n, 1)
}
