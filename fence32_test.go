//go:build (amd64 || arm64) && (linux || darwin)

package floats

import (
	"os"
	"syscall"
	"testing"
	"unsafe"
)

// The assembly kernels take raw pointers and trust their Go wrappers for the
// lengths: an off-by-one in a tail loop, or a load issued one iteration early,
// reads or writes memory that is not the slice's, and Go's bounds checks cannot
// see inside a .s file. A fence makes it fault instead: each operand is placed
// flush against an inaccessible page (PROT_NONE), once ending at the page and once
// starting right after one, so an access a single element past either end is a
// SIGSEGV. (Moved from go-ndarray's internal/kernels/fence_test.go.)

type fenced struct {
	mem  []byte
	data []float64 // the whole region between the guards
}

func newFenced(t *testing.T, n int) *fenced {
	t.Helper()
	pg := os.Getpagesize()
	dataBytes := (n*8 + pg - 1) / pg * pg
	mem, err := syscall.Mmap(-1, 0, dataBytes+2*pg,
		syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_ANON|syscall.MAP_PRIVATE)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	t.Cleanup(func() { syscall.Munmap(mem) })
	for _, g := range [][]byte{mem[:pg], mem[pg+dataBytes:]} {
		if err := syscall.Mprotect(g, syscall.PROT_NONE); err != nil {
			t.Fatalf("mprotect: %v", err)
		}
	}
	d := unsafe.Slice((*float64)(unsafe.Pointer(&mem[pg])), dataBytes/8)
	for i := range d {
		d[i] = float64(i%7) + 1 // positive, so log and sqrt stay finite
	}
	return &fenced{mem: mem, data: d}
}

// fences returns, for n float64 elements, the operand placements to try.
func fences(t *testing.T, n int) [][]float64 {
	f := newFenced(t, n)
	return [][]float64{f.data[:n:n], f.data[len(f.data)-n:]}
}

// fences32 is fences for float32 operands: the same mappings, viewed as
// float32, with n elements flush against each guard.
func fences32(t *testing.T, n int) [][]float32 {
	f := newFenced(t, (n+1)/2)
	d := unsafe.Slice((*float32)(unsafe.Pointer(&f.data[0])), 2*len(f.data))
	return [][]float32{d[:n:n], d[len(d)-n:]}
}
