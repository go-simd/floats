//go:build unix

package floats

import (
	"math"
	"testing"
	"unsafe"

	"golang.org/x/sys/unix"
)

// guardedPage maps three pages and makes the outer two inaccessible, returning
// the middle page as float64s. Any read or write one element before its start
// or one element past its end faults, which turns an out-of-bounds vector load
// in a kernel tail into a crash instead of a silent read of a neighbour.
func guardedPage(t *testing.T) []float64 {
	t.Helper()
	ps := unix.Getpagesize()
	mem, err := unix.Mmap(-1, 0, 3*ps, unix.PROT_READ|unix.PROT_WRITE, unix.MAP_ANON|unix.MAP_PRIVATE)
	if err != nil {
		t.Skipf("mmap: %v", err)
	}
	t.Cleanup(func() { _ = unix.Munmap(mem) })
	if err := unix.Mprotect(mem[:ps], unix.PROT_NONE); err != nil {
		t.Skipf("mprotect: %v", err)
	}
	if err := unix.Mprotect(mem[2*ps:], unix.PROT_NONE); err != nil {
		t.Skipf("mprotect: %v", err)
	}
	return unsafe.Slice((*float64)(unsafe.Pointer(&mem[ps])), ps/8)
}

// TestAxpyGuardPages runs every elementwise kernel on slices that end exactly
// at a page boundary followed by an inaccessible page (and, separately, start
// right after one), for every length 0..40 — the tails that a vector kernel
// would get wrong by loading a whole register past the last element.
func TestAxpyGuardPages(t *testing.T) {
	px, py, pw := guardedPage(t), guardedPage(t), guardedPage(t)
	for n := 0; n <= 40; n++ {
		for _, atEnd := range []bool{true, false} {
			cut := func(p []float64) []float64 {
				if atEnd {
					return p[len(p)-n:]
				}
				return p[:n]
			}
			x, y, w := cut(px), cut(py), cut(pw)
			for i := 0; i < n; i++ {
				x[i], y[i], w[i] = float64(i)+0.5, float64(2*i)-1, 1/float64(i+1)
			}
			want := append([]float64(nil), y...)
			axpyRef(1.25, x, want)
			Axpy(1.25, x, y)
			checkSame(t, "Axpy guard", y, want, n, 0)

			xpbyRef(-0.5, x, want)
			Xpby(x, -0.5, y)
			checkSame(t, "Xpby guard", y, want, n, 0)

			wantDot := axpyDotRef(0.25, x, want, w)
			got := AxpyDot(0.25, x, y, w)
			checkSame(t, "AxpyDot guard", y, want, n, 0)
			if !closeDot(got, wantDot, absSumProd64(want, w), relTol64) {
				t.Fatalf("AxpyDot guard n=%d: %v want %v", n, got, wantDot)
			}

			scaleRef(3, want)
			Scale(3, y)
			checkSame(t, "Scale guard", y, want, n, 0)
			if math.IsNaN(got) {
				t.Fatalf("AxpyDot guard n=%d: NaN", n)
			}
		}
	}
}
