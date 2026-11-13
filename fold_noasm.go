//go:build !arm64 && !amd64

package floats

// No fold or unfold kernel on this architecture: the portable loops of fold.go.

func mdctFold(dst, x []float64)               { foldTail(dst, x, 0) }
func mdctFold32(dst []float64, x []float32)   { fold32Tail(dst, x, 0) }
func mdctUnfold(dst, u []float64)             { unfoldTail(dst, u, 0) }
func mdctUnfold32(dst []float32, u []float64) { unfold32Tail(dst, u, 0) }
