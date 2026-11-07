package floats

import (
	"fmt"
	"math"
	"math/bits"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

// TestLogCRMatchesLog32 holds logCR to log32 bit for bit (NaN to NaN) on
// every float32 input (NDARRAY_EXHAUSTIVE=1; otherwise every 65537th).
// log32 is proven correctly rounded over all 2^32 inputs by
// TestLog32CorrectlyRounded, so equality everywhere proves logCR is too. It
// also counts the inputs logCRSlow decides.
func TestLogCRMatchesLog32(t *testing.T) {
	stride := exhaustiveStride()
	const batch = 4096
	var next atomic.Uint64
	var bad, slow, checked atomic.Int64
	var mu sync.Mutex
	var first []string
	var wg sync.WaitGroup
	for range runtime.GOMAXPROCS(0) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			xs := make([]float32, 0, batch)
			want := make([]float32, batch)
			for {
				start := next.Add(batch*stride) - batch*stride
				if start >= 1<<32 {
					return
				}
				xs = xs[:0]
				for b := start; b < start+batch*stride && b < 1<<32; b += stride {
					xs = append(xs, math.Float32frombits(uint32(b)))
				}
				log32(want[:len(xs)], xs)
				for i, x := range xs {
					checked.Add(1)
					got := logCR(x)
					if logCRTakesSlow(x) {
						slow.Add(1)
					}
					w := want[i]
					if math.Float32bits(got) != math.Float32bits(w) && !(got != got && w != w) {
						bad.Add(1)
						mu.Lock()
						if len(first) < 8 {
							first = append(first, "logCR("+formatF32(x)+") = "+formatF32(got)+", log32 "+formatF32(w))
						}
						mu.Unlock()
					}
				}
			}
		}()
	}
	wg.Wait()
	t.Logf("%d inputs checked, %d through the slow path, %d differ from log32", checked.Load(), slow.Load(), bad.Load())
	for _, s := range first {
		t.Error(s)
	}
}

// logCRTakesSlow repeats logCR's fast path and reports whether its bounds
// straddle a rounding boundary.
func logCRTakesSlow(x float32) bool {
	ux := math.Float32bits(x)
	if ux == 0 || ux >= 0x7f800000 || ux == 127<<23 {
		return false
	}
	if ux < 1<<23 {
		n := bits.LeadingZeros32(ux) - 8
		ux <<= n
		ux -= uint32(n) << 23
	}
	m := ux & (1<<23 - 1)
	j := (m + 1<<(23-7)) >> (23 - 6)
	e := float64(int32(ux)>>23 - 127)
	z := float64(math.Float64frombits((uint64(m)|1023<<23)<<(52-23))*logCRTr[j]) - 1
	z2 := float64(z * z)
	r := ((float64(e*0x1.62e42fefa39efp-1) + logCRTl[j]) + float64(z*logCRB[0])) +
		float64(z2*(logCRB[1]+float64(z*logCRB[2])))
	return float32(r) != float32(r+0x1.f06p-33)
}

// BenchmarkLog32Kernels compares the float32 log kernels on two inputs in
// [0.25, 24.25] at 16 Ki and 1 Mi: a ramp, and the multiples of 0.25 that
// benchmarks/BenchmarkDTypeExpLog uses (0.25 + (i%97)*0.25). It reports the
// blocks of 8 the AVX2 kernel hands back to logCR per call.
func BenchmarkLog32Kernels(b *testing.B) {
	for _, data := range []struct {
		name string
		at   func(i, n int) float32
	}{
		{"ramp", func(i, n int) float32 { return 0.25 + 24*float32(i)/float32(n) }},
		{"steps", func(i, n int) float32 { return float32(0.25 + float64(i%97)*0.25) }},
	} {
		for _, n := range []int{16 << 10, 1 << 20} {
			src := make([]float32, n)
			for i := range src {
				src[i] = data.at(i, n)
			}
			handed := 0
			for i := 0; i+8 <= n; i += 8 {
				for _, x := range src[i : i+8] {
					if logCRTakesSlow(x) {
						handed++
						break
					}
				}
			}
			dst := make([]float32, n)
			for _, k := range []struct {
				name string
				f    func(dst, src []float32)
			}{{"via64", log32}, {"logCR", logCRSlice}, {"fast", log32Fast}} {
				b.Run(fmt.Sprintf("%s/n=%d/%s", data.name, n, k.name), func(b *testing.B) {
					b.SetBytes(int64(4 * n))
					for b.Loop() {
						k.f(dst, src)
					}
					b.ReportMetric(float64(handed), "handed-blocks/op")
				})
			}
		}
	}
}

func TestLogCRSpecial(t *testing.T) {
	for _, c := range []struct{ x, want float32 }{
		{0, float32(math.Inf(-1))},
		{float32(math.Copysign(0, -1)), float32(math.Inf(-1))},
		{float32(math.Inf(1)), float32(math.Inf(1))},
		{1, 0},
		{math.SmallestNonzeroFloat32, -103.27893},
	} {
		if got := logCR(c.x); got != c.want {
			t.Errorf("logCR(%v) = %v, want %v", c.x, got, c.want)
		}
	}
	for _, x := range []float32{-1, float32(math.Inf(-1)), float32(math.NaN()), -math.SmallestNonzeroFloat32} {
		if got := logCR(x); got == got {
			t.Errorf("logCR(%v) = %v, want NaN", x, got)
		}
	}
}
