//go:build ignore

// Command gen produces extrema_amd64.s with go-asmgen: the SSE2 float64 Max and
// Min kernels maxSSE2/minSSE2 (go-ndarray's, moved here: ndarray v0.37.0 had
// switched to floats.Max/Min, which were a scalar loop). SSE2 is the amd64
// baseline, so no feature test.
//
// X0..X3 hold the running packed extreme (MAXPD/MINPD), X8 the OR of every
// CMPPD-unordered ($3) mask, which records "saw a NaN"; the fold returns NaN if
// any lane ever held one, otherwise the reduced extreme. MAXPD/MINPD return their
// source operand when both inputs are zeros, so the sign of a ZERO result is
// unspecified: the Go wrapper (extrema_amd64.go) resolves a zero result with the
// scalar loop, which costs nothing on any other result.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> extrema_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("amd64")
	f.Add(extremeKernel("maxSSE2", "MAXPD", "MAXSD"))
	f.Add(extremeKernel("minSSE2", "MINPD", "MINSD"))
	if err := os.WriteFile("extrema_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote extrema_amd64.s")
}

// nanScan ORs the unordered (NaN) lanes of reg into X8.
func nanScan(b *amd64.Builder, reg string) {
	b.Raw("MOVAPS %s, X9", reg).Raw("CMPPD X9, X9, $3").Raw("ORPD X9, X8")
}

// extremeKernel builds NAME(a *float64, n int) float64 from the packed (vec) and
// scalar (sca) extreme mnemonics. a is non-empty (the caller guarantees n >= 1).
func extremeKernel(name, vec, sca string) *emit.Function {
	sig := amd64.Layout([]string{"a", "n"}, []amd64.Type{amd64.Ptr, amd64.Int64},
		[]string{"ret"}, []amd64.Type{amd64.Float64})
	b := amd64.NewFunc(name, sig, 0)
	b.LoadArg("a", "AX")
	b.LoadArg("n", "CX")
	// Seed the four packed accumulators with a[0] broadcast to both lanes.
	b.Raw("MOVSD (AX), X0") // SSE2 only (MOVDDUP is SSE3)
	b.Raw("UNPCKLPD X0, X0")
	b.Raw("MOVAPS X0, X1")
	b.Raw("MOVAPS X0, X2")
	b.Raw("MOVAPS X0, X3")
	b.Raw("XORPS X8, X8")
	nanScan(b, "X0")
	b.Raw("eblock:")
	b.Raw("CMPQ CX, $8")
	b.Raw("JL efold")
	b.Raw("MOVUPD (AX), X4")
	b.Raw("MOVUPD 16(AX), X5")
	b.Raw("MOVUPD 32(AX), X6")
	b.Raw("MOVUPD 48(AX), X7")
	nanScan(b, "X4")
	nanScan(b, "X5")
	nanScan(b, "X6")
	nanScan(b, "X7")
	b.Raw(vec + " X4, X0")
	b.Raw(vec + " X5, X1")
	b.Raw(vec + " X6, X2")
	b.Raw(vec + " X7, X3")
	b.Raw("ADDQ $64, AX")
	b.Raw("SUBQ $8, CX")
	b.Raw("JMP eblock")
	b.Raw("efold:")
	b.Raw(vec + " X1, X0")
	b.Raw(vec + " X3, X2")
	b.Raw(vec + " X2, X0")
	b.Raw("MOVAPS X0, X1")
	b.Raw("UNPCKHPD X1, X1")
	b.Raw(sca + " X1, X0")
	b.Raw("etail:")
	b.Raw("TESTQ CX, CX")
	b.Raw("JZ enan")
	b.Raw("MOVSD (AX), X4")
	nanScan(b, "X4")
	b.Raw(sca + " X4, X0")
	b.Raw("ADDQ $8, AX")
	b.Raw("DECQ CX")
	b.Raw("JMP etail")
	b.Raw("enan:")
	b.Raw("MOVAPS X8, X9")
	b.Raw("UNPCKHPD X9, X9")
	b.Raw("ORPD X9, X8")
	b.Raw("MOVQ X8, DX")
	b.Raw("TESTQ DX, DX")
	b.Raw("JZ edone")
	b.Raw("XORPS X1, X1")
	b.Raw("DIVSD X1, X1") // 0.0/0.0 = NaN
	b.Raw("MOVSD X1, X0")
	b.Raw("edone:")
	b.StoreRet("X0", "ret")
	b.Ret()
	return b.Func()
}
