//go:build !arm64 && !amd64

package floats

// No overlap-add kernel on this architecture (yet): the portable loop.

func overlapWindow64(dst, a, b, win []float64) { overlapRef64(dst, a, b, win) }
func overlapWindow32(dst, a, b, win []float32) { overlapRef32(dst, a, b, win) }
