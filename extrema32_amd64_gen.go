//go:build ignore

// Command gen produces extrema32_amd64.s with go-asmgen: the SSE2 float32 Max
// and Min kernels max32SSE2/min32SSE2 (go-ndarray's, moved here so the kernel
// lives once), for n >= 1: the extremum over 16 floats per iteration in four
// XMM accumulators (MAXPS/MINPS), X8 the OR of every CMPPS-unordered ($3) mask,
// so a NaN anywhere gives NaN. MAXPS/MINPS return their source operand when both
// inputs are zeros, so the sign of a ZERO result is unspecified: the Go wrapper
// (extrema32_amd64.go) resolves a zero result with the scalar loop, which costs
// nothing on any other result. SSE2 is the amd64 baseline, so no feature test.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> extrema32_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("amd64")
	f.Add(extremum32Kernel("max32SSE2", "MAXPS", "MAXSS"))
	f.Add(extremum32Kernel("min32SSE2", "MINPS", "MINSS"))
	if err := os.WriteFile("extrema32_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote extrema32_amd64.s")
}

// extremum32Kernel builds NAME(a *float32, n int) float32 from the packed (vec)
// and scalar (sca) extreme mnemonics.
func extremum32Kernel(name, vec, sca string) *emit.Function {
	sig := amd64.Layout([]string{"a", "n"}, []amd64.Type{amd64.Ptr, amd64.Int64},
		[]string{"ret"}, []amd64.Type{amd64.Float32})
	b := amd64.NewFunc(name, sig, 0)
	b.LoadArg("a", "AX").LoadArg("n", "CX")
	// Seed the accumulators with a[0] in every lane.
	b.Raw("MOVSS (AX), X0").Raw("SHUFPS $0x00, X0, X0").
		Raw("MOVAPS X0, X1").Raw("MOVAPS X0, X2").Raw("MOVAPS X0, X3").
		Raw("XORPS X8, X8")
	nanScan32(b, "X0")
	b.Raw("xblock:").Raw("CMPQ CX, $16").Raw("JL xfold")
	for r := 0; r < 4; r++ {
		b.Raw("MOVUPS %d(AX), X%d", 16*r, 4+r)
	}
	for r := 0; r < 4; r++ {
		nanScan32(b, fmt.Sprintf("X%d", 4+r))
	}
	for r := 0; r < 4; r++ {
		b.Raw("%s X%d, X%d", vec, 4+r, r)
	}
	b.Raw("ADDQ $64, AX").Raw("SUBQ $16, CX").Raw("JMP xblock")
	// Fold the accumulators, then the four lanes, into X0's low lane.
	b.Raw("xfold:").
		Raw("%s X1, X0", vec).Raw("%s X3, X2", vec).Raw("%s X2, X0", vec).
		Raw("MOVAPS X0, X1").Raw("MOVHLPS X1, X1").Raw("%s X1, X0", vec).
		Raw("MOVAPS X0, X1").Raw("SHUFPS $0x55, X1, X1").Raw("%s X1, X0", sca).
		Raw("xtail:").Raw("TESTQ CX, CX").Raw("JZ xnan").
		Raw("MOVSS (AX), X4")
	nanScan32(b, "X4")
	b.Raw("%s X4, X0", sca).
		Raw("ADDQ $4, AX").Raw("DECQ CX").Raw("JMP xtail").
		// Any lane of X8 set: some element was NaN.
		Raw("xnan:").Raw("MOVMSKPS X8, DX").Raw("TESTQ DX, DX").Raw("JZ xdone").
		Raw("XORPS X1, X1").Raw("DIVSS X1, X1"). // 0/0 = NaN
		Raw("MOVSS X1, X0").
		Raw("xdone:").
		StoreRet("X0", "ret").
		Ret()
	return b.Func()
}

// nanScan32 ORs the unordered (NaN) lanes of reg into X8.
func nanScan32(b *amd64.Builder, reg string) {
	b.Raw("MOVAPS %s, X9", reg).Raw("CMPPS X9, X9, $3").Raw("ORPS X9, X8")
}
