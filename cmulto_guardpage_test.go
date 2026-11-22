//go:build unix

package floats

import (
	"testing"
	"unsafe"
)

// TestCMulToGuardPages runs CMulTo with dst, a and b ending exactly at a page boundary
// followed by an inaccessible page (and starting right after one), for every length
// 0..40: a block loaded or stored past the last number would fault.
func TestCMulToGuardPages(t *testing.T) {
	asC := func(p []float64) []complex128 {
		return unsafe.Slice((*complex128)(unsafe.Pointer(&p[0])), len(p)/2)
	}
	pd, pa, pb := asC(guardedPage(t)), asC(guardedPage(t)), asC(guardedPage(t))
	for n := 0; n <= 40; n++ {
		for _, atEnd := range []bool{true, false} {
			cut := func(p []complex128) []complex128 {
				if atEnd {
					return p[len(p)-n:]
				}
				return p[:n]
			}
			d, a, b := cut(pd), cut(pa), cut(pb)
			for i := range a {
				a[i], b[i] = complex(float64(i)*0.25+1, -float64(i)), complex(1/float64(i+1), 0.5)
			}
			CMulTo(d, a, b)
			for i := range d {
				if !cmulOK(d[i], a[i], b[i]) {
					t.Fatalf("n=%d atEnd=%v i=%d: got %v", n, atEnd, i, d[i])
				}
			}
			CMulTo(a, a, b) // in place at the edge
		}
	}
}
