//go:build amd64 || arm64

package floats

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

// TestSqrt32Exhaustive runs Sqrt32 over ALL 2^32 float32 bit patterns and checks
// each result against an oracle that does not use a square root at all:
//
//   - NaN and negative non-zero inputs give NaN; ±0 gives the same zero; +Inf
//     gives +Inf;
//   - otherwise r is the correctly rounded binary32 root of x iff x lies strictly
//     between the squares of the two midpoints around r. A midpoint has at most
//     25 significant bits, so its square (at most 50 bits) and the midpoint itself
//     are exact in float64, and a tie is impossible (an odd 25-bit significand
//     squared has at least 49 significant bits, x has at most 24).
//
// On amd64 both the AVX2 and the SSE2 kernels are run. About 10 s per kernel on
// one core of an Apple M-series; the work is spread over every CPU. Skipped with
// -short. The other architectures (qemu lanes, no float32 kernel) are covered by
// the sampled tests of binop32_test.go.
func TestSqrt32Exhaustive(t *testing.T) {
	if testing.Short() {
		t.Skip("exhaustive: 2^32 inputs")
	}
	runExhaustive := func(label string) {
		const chunk = 1 << 16
		var next, bad atomic.Uint64
		var wg sync.WaitGroup
		for w := 0; w < runtime.GOMAXPROCS(0); w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				in, out := make([]float32, chunk), make([]float32, chunk)
				for bad.Load() == 0 {
					c := next.Add(1) - 1
					if c >= 1<<32/chunk {
						return
					}
					base := uint32(c * chunk)
					for i := range in {
						in[i] = math.Float32frombits(base + uint32(i))
					}
					Sqrt32(out, in)
					for i, x := range in {
						if !sqrt32Correct(x, out[i]) {
							if bad.CompareAndSwap(0, uint64(base)+uint64(i)+1) {
								return
							}
						}
					}
				}
			}()
		}
		wg.Wait()
		if b := bad.Load(); b != 0 {
			x := math.Float32frombits(uint32(b - 1))
			var out [1]float32
			Sqrt32(out[:], []float32{x})
			t.Fatalf("%s: Sqrt32(%v = %#x) = %v (%#x) is not the correctly rounded root", label, x,
				math.Float32bits(x), out[0], math.Float32bits(out[0]))
		}
	}
	runExhaustiveKernels(runExhaustive)
}

// sqrt32Correct reports whether r is the correctly rounded binary32 square root
// of x, by the squared-midpoint test above.
func sqrt32Correct(x, r float32) bool {
	switch {
	case x != x:
		return r != r
	case x == 0:
		return math.Float32bits(r) == math.Float32bits(x)
	case x < 0:
		return r != r
	case math.IsInf(float64(x), 1):
		return math.IsInf(float64(r), 1)
	}
	rb := math.Float32bits(r)
	if r != r || r <= 0 || rb >= 0x7f800000-1 { // finite positive x never roots to 0, Inf or the last finite
		return false
	}
	lo := (float64(math.Float32frombits(rb-1)) + float64(r)) / 2
	hi := (float64(r) + float64(math.Float32frombits(rb+1))) / 2
	return lo*lo < float64(x) && float64(x) < hi*hi
}
