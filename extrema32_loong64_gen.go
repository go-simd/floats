//go:build ignore

// Command gen produces extrema32_loong64.s with go-asmgen: the LSX float32 Max
// and Min kernels max32LSX/min32LSX (go-ndarray's, moved here so the kernel
// lives once), NAME(a *float32, n int, out *[16]byte) int64 for n a positive
// multiple of 16, entirely in LSX integer arithmetic: Go's loong64 assembler has
// no vector float max, min or compare.
//
//   - Order: t = x XOR ((x >>a 31) >>l 1) flips the magnitude bits of the
//     negatives, so the signed order of t is the float order, with -0 below +0,
//     Go's rule. The map is its own inverse.
//   - Extremum: the max of t and acc is acc XOR ((acc XOR t) AND (acc < t))
//     (vslt.w), and the min swaps the compare.
//   - NaN: a lane is NaN iff its magnitude bits (x AND abs) exceed +Inf's bits
//     as an unsigned integer (vsltu). The masks are OR-accumulated, and a
//     non-zero result is returned so the wrapper can answer NaN.
//
// Four accumulators (V0-V3) are folded the same way into one vector, which is
// stored, still in the t order, at out. The Go wrapper maps it back and folds
// the lanes and the tail with Go's max/min. Registers: V4-V7 the block, V8-V19
// scratch, V20 abs, V21 +Inf, V22 the NaN mask.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> extrema32_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
)

func main() {
	f := emit.NewFile("loong64")
	abs32 := f.Data("ext32abs", lanes(4, 0x7fffffff))
	inf32 := f.Data("ext32inf", lanes(4, 0x7f800000))
	f.Add(extremum32Kernel("max32LSX", true, abs32, inf32))
	f.Add(extremum32Kernel("min32LSX", false, abs32, inf32))
	if err := os.WriteFile("extrema32_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote extrema32_loong64.s")
}

// lanes is a 16-byte vector constant: v in every lane of size bytes, each lane
// little-endian.
func lanes(size int, v uint64) []byte {
	b := make([]byte, 0, 16)
	for len(b) < 16 {
		for i := 0; i < size; i++ {
			b = append(b, byte(v>>(8*i)))
		}
	}
	return b
}

func extremum32Kernel(name string, isMax bool, abs, inf string) *emit.Function {
	sig := loong64.Layout([]string{"a", "n", "out"},
		[]loong64.Type{loong64.Ptr, loong64.Int64, loong64.Ptr}, []string{"ret"}, []loong64.Type{loong64.Int64})
	b := loong64.NewFunc(name, sig, 0)
	b.LoadArg("a", "R4").LoadArg("n", "R5").LoadArg("out", "R6")
	const bits, block = 32, 16
	sra, srl, slt, sltu := "VSRAW", "VSRLW", "VSLTW", "VSLTWU"
	b.Raw("MOVV $%s+0(SB), R12", abs).Raw("VMOVQ (R12), V20").
		Raw("MOVV $%s+0(SB), R12", inf).Raw("VMOVQ (R12), V21").
		Raw("VXORV V22, V22, V22")
	// sel sets acc to the extremum of acc and t, with m and d as scratch.
	sel := func(acc, t, m, d int) {
		if isMax {
			b.Raw("%s V%d, V%d, V%d", slt, t, acc, m) // m = acc < t
		} else {
			b.Raw("%s V%d, V%d, V%d", slt, acc, t, m) // m = t < acc
		}
		b.Raw("VXORV V%d, V%d, V%d", t, acc, d).
			Raw("VANDV V%d, V%d, V%d", m, d, d).
			Raw("VXORV V%d, V%d, V%d", d, acc, acc)
	}
	// load reads one block into V4-V7, ORs its NaN lanes into V22 and leaves
	// the ordered integers in V8-V11.
	load := func() {
		for k := 0; k < 4; k++ {
			b.Raw("VMOVQ %d(R4), V%d", 16*k, 4+k)
		}
		for k := 0; k < 4; k++ {
			b.Raw("VANDV V20, V%d, V%d", 4+k, 12+k).
				Raw("%s V%d, V21, V%d", sltu, 12+k, 12+k). // +Inf < |x|: NaN
				Raw("VORV V%d, V22, V22", 12+k).
				Raw("%s $%d, V%d, V%d", sra, bits-1, 4+k, 8+k).
				Raw("%s $1, V%d, V%d", srl, 8+k, 8+k).
				Raw("VXORV V%d, V%d, V%d", 4+k, 8+k, 8+k)
		}
		b.Raw("ADDV $64, R4").Raw("ADDV $-%d, R5", block)
	}
	load()
	for k := 0; k < 4; k++ {
		b.Raw("VORV V%d, V%d, V%d", 8+k, 8+k, k) // the first block seeds acc
	}
	b.Label(name+"loop").Raw("BEQ R5, %sfold", name)
	load()
	for k := 0; k < 4; k++ {
		sel(k, 8+k, 12+k, 16+k)
	}
	b.Raw("JMP %sloop", name)
	b.Label(name + "fold")
	sel(0, 1, 12, 16)
	sel(2, 3, 13, 17)
	sel(0, 2, 14, 18)
	b.Raw("VMOVQ V0, (R6)").
		Raw("VMOVQ V22.V[0], R13").Raw("VMOVQ V22.V[1], R14").Raw("OR R14, R13").
		StoreRet("R13", "ret").Ret()
	return b.Func()
}
