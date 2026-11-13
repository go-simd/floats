//go:build !arm64

package floats

// No twiddle kernel on this architecture: the portable loops of twiddle.go.

func cmulGather(dst, x, tw []float64)                         { cmulGatherTail(dst, x, tw, 0) }
func cmulGather32(dst []float64, x []float32, tw []float64)   { cmulGather32Tail(dst, x, tw, 0) }
func cmulScatter(dst, c, tw []float64, f float64)             { cmulScatterRef(dst, c, tw, f) }
func cmulScatter32(dst []float32, c, tw []float64, f float64) { cmulScatter32Ref(dst, c, tw, f) }
