//go:build unix

package floats

import (
	"math"
	"testing"
	"unsafe"
)

// guardedPage32 is guardedPage (axpy_guardpage_test.go) viewed as float32s: the
// same middle page between two inaccessible ones.
func guardedPage32(t *testing.T) []float32 {
	t.Helper()
	p := guardedPage(t)
	return unsafe.Slice((*float32)(unsafe.Pointer(&p[0])), len(p)*2)
}

// TestBinop32GuardPages runs Add32, Sub32, Mul32, Div32 and Sqrt32 on slices that
// end exactly at a page boundary followed by an inaccessible page (and,
// separately, start right after one), for every length 0..131: a vector load or
// store past either end of any operand would fault.
func TestBinop32GuardPages(t *testing.T) {
	pa, pb, pd := guardedPage32(t), guardedPage32(t), guardedPage32(t)
	for n := 0; n <= 131; n++ {
		for _, atEnd := range []bool{true, false} {
			cut := func(p []float32) []float32 {
				if atEnd {
					return p[len(p)-n:]
				}
				return p[:n]
			}
			a, b, dst := cut(pa), cut(pb), cut(pd)
			for i := 0; i < n; i++ {
				a[i], b[i] = float32(i)+0.5, float32(2*i)-1
			}
			want := make([]float32, n)
			for _, op := range binop32Ops {
				op.ref(want, a, b)
				op.f(dst, a, b)
				checkSame32(t, op.name+" guard", dst, want, n, 0)
			}
			for i := range a {
				a[i] = float32(math.Abs(float64(a[i])))
			}
			sqrt32Ref(want, a)
			Sqrt32(dst, a)
			checkSame32(t, "Sqrt32 guard", dst, want, n, 0)
		}
	}
}
