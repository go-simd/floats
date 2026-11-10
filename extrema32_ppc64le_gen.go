//go:build ignore

// Command gen produces extrema32_ppc64le.s with go-asmgen: the VSX float32 Max
// and Min kernels max32VSX/min32VSX (go-ndarray's, moved here so the kernel
// lives once), NAME(a *float32, n int, out *[16]byte) int64 for n a positive
// multiple of 16. It reduces a with xvmaxsp/xvminsp into four accumulators,
// folds them vertically into one vector stored at out, and returns non-zero if
// any element was NaN. The Go wrapper folds the lanes and the tail with Go's
// max/min. So neither the instruction's NaN rule nor its sign of a zero tie
// matters: NaN is detected separately, and a zero result goes to the scalar loop.
//
// NaN detection is integer arithmetic on the VMX registers, because Go's
// assembler has no VSX float compare: a lane is NaN iff its magnitude bits
// (x AND abs) exceed +Inf's bits as an unsigned integer (vcmpgtuw). The masks
// are OR-accumulated in V22. Registers: VS0-VS3 accumulators; V4-V7 (VS36-VS39)
// the block; V8-V11 scratch; V20, V21 the abs and +Inf constants; V22 the NaN
// mask. LXVW4X/STXVW4X put each word at the same lane in every operand.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> extrema32_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

func main() {
	f := emit.NewFile("ppc64le")
	abs32 := f.Data("ext32abs", lanes(4, 0x7fffffff))
	inf32 := f.Data("ext32inf", lanes(4, 0x7f800000))
	f.Add(extremum32Kernel("max32VSX", true, abs32, inf32))
	f.Add(extremum32Kernel("min32VSX", false, abs32, inf32))
	if err := os.WriteFile("extrema32_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote extrema32_ppc64le.s")
}

// lanes is a 16-byte vector constant: v in every lane of size bytes, each lane
// little-endian (every lane equal, so LXVW4X's lane order does not matter).
func lanes(size int, v uint64) []byte {
	b := make([]byte, 0, 16)
	for len(b) < 16 {
		for i := 0; i < size; i++ {
			b = append(b, byte(v>>(8*i)))
		}
	}
	return b
}

// vidx is the index register for vector k (0..3) of a 64-byte block.
func vidx(k int) string { return [...]string{"R0", "R14", "R15", "R16"}[k] }

func extremum32Kernel(name string, isMax bool, abs, inf string) *emit.Function {
	sig := ppc64.Layout([]string{"a", "n", "out"},
		[]ppc64.Type{ppc64.Ptr, ppc64.Int64, ppc64.Ptr}, []string{"ret"}, []ppc64.Type{ppc64.Int64})
	b := ppc64.NewFunc(name, sig, 0)
	b.LoadArg("a", "R3").LoadArg("n", "R4").LoadArg("out", "R5")
	// LXVW4X's indexed form takes only (base)(index): the byte offsets 16, 32, 48.
	b.Raw("MOVD $16, R14").Raw("MOVD $32, R15").Raw("MOVD $48, R16")
	const ld, st, cmp, block = "LXVW4X", "STXVW4X", "VCMPGTUW", 16
	op := b.XVMAXSP
	if !isMax {
		op = b.XVMINSP
	}
	b.Raw("MOVD $%s+0(SB), R7", abs).Raw("%s (R7)(R0), VS52", ld).
		Raw("MOVD $%s+0(SB), R7", inf).Raw("%s (R7)(R0), VS53", ld).
		Raw("XXLXOR VS54, VS54, VS54")
	loadAndScan := func() {
		for k := 0; k < 4; k++ {
			b.Raw("%s (R3)(%s), VS%d", ld, vidx(k), 36+k)
		}
		for k := 0; k < 4; k++ {
			b.Raw("VAND V%d, V20, V%d", 4+k, 8+k).
				Raw("%s V%d, V21, V%d", cmp, 8+k, 8+k). // magnitude > +Inf: NaN
				Raw("VOR V%d, V22, V22", 8+k)
		}
		b.Raw("ADD $64, R3").Raw("ADD $-%d, R4", block)
	}
	// The first block seeds the accumulators.
	loadAndScan()
	for k := 0; k < 4; k++ {
		b.Raw("XXLOR VS%d, VS%d, VS%d", 36+k, 36+k, k)
	}
	b.Label(name+"loop").Raw("CMP R4, $0").Raw("BEQ %sfold", name)
	loadAndScan()
	for k := 0; k < 4; k++ {
		op(k, k, 36+k)
	}
	b.Raw("BR %sloop", name)
	b.Label(name + "fold")
	op(0, 0, 1)
	op(2, 2, 3)
	op(0, 0, 2)
	b.Raw("%s VS0, (R5)(R0)", st).
		// The NaN mask's two doublewords, ORed into R6.
		Raw("XXPERMDI VS54, VS54, $2, VS55").
		Raw("MFVSRD VS54, R6").Raw("MFVSRD VS55, R7").Raw("OR R7, R6").
		StoreRet("R6", "ret").Ret()
	return b.Func()
}
