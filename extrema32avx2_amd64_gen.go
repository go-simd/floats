//go:build ignore

// Command gen produces extrema32avx2_amd64.s with go-asmgen: the AVX2 float32
// Max and Min kernels max32AVX2/min32AVX2, for n >= 32 (floats#46: the SSE2
// kernels of extrema32_amd64.s were 1.17x/1.33x slower than NumPy at n = 1M,
// which dispatches AVX2). Four YMM accumulators (VMAXPS/VMINPS, 32 floats per
// iteration) are seeded with a[0] in every lane; Y8 is the OR of every
// VCMPPS-unordered ($3) mask, so a NaN anywhere gives NaN. VMAXPS/VMINPS return
// the second source when both inputs are zeros, so the sign of a ZERO result is
// unspecified: the Go wrapper resolves a zero result with the scalar loop, as for
// the SSE2 kernels. The accumulators and the mask are folded to 128 bits, VZEROUPPER
// is issued, and the up to 31 tail elements finish in scalar SSE.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> extrema32avx2_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("amd64")
	f.Add(kernel("max32AVX2", "VMAXPS", "MAXSS"))
	f.Add(kernel("min32AVX2", "VMINPS", "MINSS"))
	if err := os.WriteFile("extrema32avx2_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote extrema32avx2_amd64.s")
}

// nanScan ORs the unordered (NaN) lanes of reg into Y8.
func nanScan(b *amd64.Builder, reg string) {
	b.Raw("VCMPPS $3, %s, %s, Y9", reg, reg).Raw("VORPS Y9, Y8, Y8")
}

func kernel(name, vec, sca string) *emit.Function {
	sig := amd64.Layout([]string{"a", "n"}, []amd64.Type{amd64.Ptr, amd64.Int64},
		[]string{"ret"}, []amd64.Type{amd64.Float32})
	b := amd64.NewFunc(name, sig, 0)
	b.LoadArg("a", "AX").LoadArg("n", "CX")
	b.Raw("VBROADCASTSS (AX), Y0").
		Raw("VMOVAPS Y0, Y1").Raw("VMOVAPS Y0, Y2").Raw("VMOVAPS Y0, Y3").
		Raw("VXORPS Y8, Y8, Y8")
	nanScan(b, "Y0")
	b.Raw("yblock:").Raw("CMPQ CX, $32").Raw("JL yfold")
	for r := 0; r < 4; r++ {
		b.Raw("VMOVUPS %d(AX), Y%d", 32*r, 4+r)
	}
	for r := 0; r < 4; r++ {
		nanScan(b, fmt.Sprintf("Y%d", 4+r))
	}
	for r := 0; r < 4; r++ {
		b.Raw("%s Y%d, Y%d, Y%d", vec, 4+r, r, r)
	}
	b.Raw("ADDQ $128, AX").Raw("SUBQ $32, CX").Raw("JMP yblock")
	b.Raw("yfold:").
		Raw("%s Y1, Y0, Y0", vec).Raw("%s Y3, Y2, Y2", vec).Raw("%s Y2, Y0, Y0", vec).
		Raw("VEXTRACTF128 $1, Y0, X1").Raw("%s X1, X0, X0", vec).
		Raw("VEXTRACTF128 $1, Y8, X9").Raw("VORPS X9, X8, X8").
		Raw("VZEROUPPER").
		Raw("MOVAPS X0, X1").Raw("MOVHLPS X1, X1").Raw("%s X1, X0", sca[:3]+"PS").
		Raw("MOVAPS X0, X1").Raw("SHUFPS $0x55, X1, X1").Raw("%s X1, X0", sca).
		Raw("ytail:").Raw("TESTQ CX, CX").Raw("JZ ynan").
		Raw("MOVSS (AX), X4").
		Raw("MOVAPS X4, X9").Raw("CMPPS X9, X9, $3").Raw("ORPS X9, X8").
		Raw("%s X4, X0", sca).
		Raw("ADDQ $4, AX").Raw("DECQ CX").Raw("JMP ytail").
		Raw("ynan:").Raw("MOVMSKPS X8, DX").Raw("TESTQ DX, DX").Raw("JZ ydone").
		Raw("XORPS X1, X1").Raw("DIVSS X1, X1").
		Raw("MOVSS X1, X0").
		Raw("ydone:").
		StoreRet("X0", "ret").
		Ret()
	return b.Func()
}
