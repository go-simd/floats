package floats

import (
	"math"
	"math/big"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
)

// Exp32 and Log32 are correctly rounded: proven over every float32.
//
// A float32 function has 2^32 inputs, so correct rounding can be checked
// exhaustively instead of sampled. For each input the float64 intermediate y
// (Exp or Log of the widened input, the value Exp32 and Log32 round) has an
// error of at most 0.504 float64 ULP (TestExpULP, TestLogULP). If y is more
// than 2 float64 ULP away from both float32 rounding midpoints around
// float32(y), the exact value rounds to the same float32: certified. The few
// inputs closer than that are decided by the 300-bit references refExp and
// refLog, rounded by big.Float.Float32 (round to nearest, ties to even).
// Results far beyond float32's range are certified by bounds: |y| > 2^129
// rounds to ±Inf, |y| < 2^-152 to ±0, and so does the exact value.
//
// The full run (2^32 inputs per function) takes seconds on all cores and is
// set off by NDARRAY_EXHAUSTIVE=1; without it every 65537th bit pattern is
// checked, about 65 000 inputs, so CI stays fast.

// certified32 rounds the float64 intermediate y to float32 and reports
// whether that rounding is provably the correct rounding of the exact value.
func certified32(y float64) (float32, bool) { return certified32Margin(y, 2) }

// certified32Margin is certified32 for an intermediate whose error may reach
// margin/2 float64 ULP (the margin is kept at twice the measured worst).
func certified32Margin(y float64, margin float64) (float32, bool) {
	f := float32(y)
	ay := math.Abs(y)
	if math.IsNaN(y) || math.IsInf(y, 0) || ay > 0x1p129 || ay < 0x1p-152 {
		return f, true
	}
	eps := margin * (math.Nextafter(ay, math.Inf(1)) - ay)
	if math.IsInf(float64(f), 0) {
		// y rounded to ±Inf although |y| <= 2^129: the only boundary is
		// half an ULP past MaxFloat32, where ties-to-even goes to Inf.
		// (TestCertified32 plants exactly this value.)
		return f, ay-(math.MaxFloat32+0x1p103) > eps
	}
	ff := float64(f)
	near := func(neighbour float32) bool {
		var m float64
		if math.IsInf(float64(neighbour), 0) { // f is ±MaxFloat32: the boundary is half an ULP past it
			m = ff + math.Copysign(0x1p103, ff)
		} else {
			m = (ff + float64(neighbour)) / 2 // exact: adjacent float32s have 24-bit mantissas
		}
		return math.Abs(y-m) <= eps
	}
	if near(math.Nextafter32(f, float32(math.Inf(-1)))) || near(math.Nextafter32(f, float32(math.Inf(1)))) {
		return f, false
	}
	return f, true
}

type exhaustiveCase struct {
	name string
	k32  func(dst, src []float32)
	k64  func(dst, src []float64)
	ref  func(float64) *big.Float
	// special returns the expected result for inputs the reference does not
	// take (NaN, ±Inf, and for log 0 and negatives), and ok=false otherwise.
	special func(x float32) (want float32, isNaN, ok bool)
	// margin is the certifier's distance to a midpoint, in float64 ULP
	// (0 means 2, for the 0.504-ULP kernels).
	margin float64
}

func exp32Special(x float32) (float32, bool, bool) {
	switch {
	case x != x:
		return 0, true, true
	case math.IsInf(float64(x), 1):
		return float32(math.Inf(1)), false, true
	case math.IsInf(float64(x), -1):
		return 0, false, true
	}
	return 0, false, false
}

func log32Special(x float32) (float32, bool, bool) {
	switch {
	case x != x || x < 0:
		return 0, true, true
	case x == 0:
		return float32(math.Inf(-1)), false, true
	case math.IsInf(float64(x), 1):
		return float32(math.Inf(1)), false, true
	}
	return 0, false, false
}

func runExhaustive(t *testing.T, c exhaustiveCase, stride uint64) {
	const batch = 4096
	workers := runtime.GOMAXPROCS(0)
	var next atomic.Uint64
	var bad, hard, checked atomic.Int64
	var mu sync.Mutex
	var firstBad []string
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			xs := make([]float32, 0, batch)
			got := make([]float32, batch)
			w64 := make([]float64, batch)
			y64 := make([]float64, batch)
			for {
				start := next.Add(batch*stride) - batch*stride
				if start >= 1<<32 {
					return
				}
				xs = xs[:0]
				for b := start; b < start+batch*stride && b < 1<<32; b += stride {
					xs = append(xs, math.Float32frombits(uint32(b)))
				}
				n := len(xs)
				c.k32(got[:n], xs)
				for i, x := range xs {
					w64[i] = float64(x)
				}
				c.k64(y64[:n], w64[:n])
				for i, x := range xs {
					checked.Add(1)
					var want float32
					if w, isNaN, ok := c.special(x); ok {
						if isNaN {
							if got[i] == got[i] {
								bad.Add(1)
								mu.Lock()
								if len(firstBad) < 8 {
									firstBad = append(firstBad, c.name+": NaN wanted")
								}
								mu.Unlock()
							}
							continue
						}
						want = w
					} else {
						margin := c.margin
						if margin == 0 {
							margin = 2
						}
						f, sure := certified32Margin(y64[i], margin)
						if sure {
							want = f
						} else {
							hard.Add(1)
							want, _ = c.ref(float64(x)).Float32()
							if os.Getenv("NDARRAY_DUMP_HARD") != "" {
								mu.Lock()
								t.Logf("hard %s 0x%s -> 0x%s", c.name, hex32(math.Float32bits(x)), hex32(math.Float32bits(want)))
								mu.Unlock()
							}
						}
					}
					if math.Float32bits(got[i]) != math.Float32bits(want) {
						bad.Add(1)
						mu.Lock()
						if len(firstBad) < 8 {
							firstBad = append(firstBad, c.name+"("+formatF32(x)+") = "+formatF32(got[i])+", correctly rounded "+formatF32(want))
						}
						mu.Unlock()
					}
				}
			}
		}()
	}
	wg.Wait()
	t.Logf("%s: %d inputs checked, %d decided by the 300-bit reference, %d not correctly rounded",
		c.name, checked.Load(), hard.Load(), bad.Load())
	for _, s := range firstBad {
		t.Error(s)
	}
}

func formatF32(x float32) string {
	return big.NewFloat(float64(x)).Text('g', 9) + " (0x" + hex32(math.Float32bits(x)) + ")"
}

func hex32(u uint32) string {
	const digits = "0123456789abcdef"
	var b [8]byte
	for i := 7; i >= 0; i-- {
		b[i] = digits[u&15]
		u >>= 4
	}
	return string(b[:])
}

func exhaustiveStride() uint64 {
	if os.Getenv("NDARRAY_EXHAUSTIVE") != "" {
		return 1
	}
	return 65537
}

func TestExp32CorrectlyRounded(t *testing.T) {
	// Exp32 rounds expCR (1.006 float64 ULP): the certifier's margin is 4.
	runExhaustive(t, exhaustiveCase{"exp", Float32Exp, expCRSlice, refExp, exp32Special, 4}, exhaustiveStride())
}

func TestLog32CorrectlyRounded(t *testing.T) {
	runExhaustive(t, exhaustiveCase{"log", Float32Log, Log, refLog, log32Special, 2}, exhaustiveStride())
}

// TestLog32Exceptions runs Log32 on every input of the exception table (the
// sampled run rarely meets one): each result must be the table's, and the
// table's must be the 300-bit reference rounded to float32. An input outside
// the table keeps the computed value.
func TestLog32Exceptions(t *testing.T) {
	src := make([]float32, len(log32Exceptions))
	for i, e := range log32Exceptions {
		src[i] = math.Float32frombits(e[0])
	}
	got := make([]float32, len(src))
	Float32Log(got, src)
	for i, e := range log32Exceptions {
		want, _ := refLog(float64(src[i])).Float32()
		if math.Float32bits(want) != e[1] {
			t.Errorf("table entry 0x%s: 0x%s, the reference rounds to 0x%s", hex32(e[0]), hex32(e[1]), hex32(math.Float32bits(want)))
		}
		if math.Float32bits(got[i]) != e[1] {
			t.Errorf("Log32(0x%s) = 0x%s, want the table's 0x%s", hex32(e[0]), hex32(math.Float32bits(got[i])), hex32(e[1]))
		}
	}
	if f := log32Exception(2, 0.5); f != 0.5 {
		t.Errorf("an input outside the table changed: %v", f)
	}
}

// TestCertified32 checks the certifier itself on planted values: a float64
// exactly on a float32 midpoint, and one ULP from it, must be sent to the
// reference; a value in the middle of a float32 interval is certified.
func TestCertified32(t *testing.T) {
	one, next := float64(float32(1)), float64(math.Nextafter32(1, 2))
	mid := (one + next) / 2
	if _, sure := certified32(mid); sure {
		t.Error("a float32 midpoint was certified")
	}
	if _, sure := certified32(math.Nextafter(mid, 2)); sure {
		t.Error("one float64 ULP above a midpoint was certified")
	}
	if f, sure := certified32(one + (next-one)/8); !sure || f != 1 {
		t.Errorf("1 + ulp/8: %v, %v", f, sure)
	}
	if f, sure := certified32(0x1p130); !sure || !math.IsInf(float64(f), 1) {
		t.Errorf("2^130: %v, %v", f, sure)
	}
	if f, sure := certified32(0x1p-160); !sure || f != 0 {
		t.Errorf("2^-160: %v, %v", f, sure)
	}
	maxMid := float64(math.MaxFloat32) + 0x1p103
	if _, sure := certified32(maxMid); sure {
		t.Error("the overflow boundary was certified")
	}
}
