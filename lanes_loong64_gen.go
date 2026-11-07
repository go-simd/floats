//go:build ignore

// Command gen produces lanes_loong64.s with go-asmgen: the loong64 LASX float64
// elementwise kernels Add, Sub, Mul, Div, Sqrt of go-ndarray
// (internal/kernels/asmgen/loong64, moved here: floats#30) and the
// eight-accumulator pairwise-sum block of SumPairwise.
//
// Go's loong64 assembler has the LASX vector float64 add/sub/mul/div (XVADDD,
// XVSUBD, XVMULD, XVDIVD), the square root (XVFSQRTD) and the 256-bit loads and
// stores (XVMOVQ). One IEEE operation per element, no fused instruction:
// bit-identical to the scalar loops. LASX is not on every LoongArch CPU, so the
// Go side calls these only when cpu.Loong64.HasLASX.
//
// pwBlockLASX(a, n): 8 <= n <= 128. The eight accumulators r[0..7] of NumPy's
// DOUBLE_pairwise_sum are two four-lane registers X0 = r0..r3 and X1 = r4..r7,
// loaded from a[0:8] and then added to each following block of eight, so every
// lane receives exactly the additions of the scalar r[j]. The lanes are moved to
// the scalar registers F0..F7 and folded ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7));
// the leftover n % 8 elements are added to the result one at a time, as NumPy
// does.
//
// Registers: R4.. hold the arguments; R12..R15 are scratch; X0..X31 are the
// 256-bit LASX registers, F0..F31 their low lanes.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> lanes_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
)

func main() {
	f := emit.NewFile("loong64")
	f.Add(sqrtKernel())
	for _, k := range []struct{ name, vec, sca string }{
		{"addLASX", "XVADDD", "ADDD"}, {"subLASX", "XVSUBD", "SUBD"},
		{"mulLASX", "XVMULD", "MULD"}, {"divLASX", "XVDIVD", "DIVD"},
	} {
		f.Add(binKernel(k.name, k.vec, k.sca))
	}
	f.Add(pwBlockKernel())
	if err := os.WriteFile("lanes_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote lanes_loong64.s")
}

// sqrtKernel builds sqrtLASX(dst, src *float64, n int).
func sqrtKernel() *emit.Function {
	sig := loong64.Layout([]string{"dst", "src", "n"}, []loong64.Type{loong64.Ptr, loong64.Ptr, loong64.Int64}, nil, nil)
	b := loong64.NewFunc("sqrtLASX", sig, 0)
	b.LoadArg("dst", "R4").LoadArg("src", "R6").LoadArg("n", "R5")
	b.Label("qloop").Raw("MOVV $16, R12").Raw("BLT R5, R12, qtail")
	for k := 0; k < 4; k++ {
		b.Raw("XVMOVQ %d(R6), X%d", 32*k, k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("XVFSQRTD X%d, X%d", k, 4+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("XVMOVQ X%d, %d(R4)", 4+k, 32*k)
	}
	b.Raw("ADDV $128, R4").Raw("ADDV $128, R6").Raw("ADDV $-16, R5").Raw("JMP qloop")
	b.Label("qtail").Raw("BEQ R5, qdone").
		Raw("MOVD 0(R6), F1").Raw("SQRTD F1, F1").Raw("MOVD F1, 0(R4)").
		Raw("ADDV $8, R4").Raw("ADDV $8, R6").Raw("ADDV $-1, R5").Raw("JMP qtail")
	b.Label("qdone").Ret()
	return b.Func()
}

// binKernel builds name(dst, a, b *float64, n int): dst[i] = a[i] OP b[i], 16
// doubles per iteration with the LASX op, a scalar tail with sca. Each block is
// loaded and computed before it is stored, so dst may alias an input exactly.
func binKernel(name, vec, sca string) *emit.Function {
	sig := loong64.Layout([]string{"dst", "a", "b", "n"},
		[]loong64.Type{loong64.Ptr, loong64.Ptr, loong64.Ptr, loong64.Int64}, nil, nil)
	b := loong64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "R4").LoadArg("a", "R6").LoadArg("b", "R7").LoadArg("n", "R5")
	l := name[:3]
	b.Label(l+"loop").Raw("MOVV $16, R12").Raw("BLT R5, R12, %stail", l)
	for k := 0; k < 4; k++ {
		b.Raw("XVMOVQ %d(R6), X%d", 32*k, k)
		b.Raw("XVMOVQ %d(R7), X%d", 32*k, 4+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("%s X%d, X%d, X%d", vec, 4+k, k, 8+k) // X(8+k) = X(k) OP X(4+k)
	}
	for k := 0; k < 4; k++ {
		b.Raw("XVMOVQ X%d, %d(R4)", 8+k, 32*k)
	}
	b.Raw("ADDV $128, R4").Raw("ADDV $128, R6").Raw("ADDV $128, R7").Raw("ADDV $-16, R5").Raw("JMP %sloop", l)
	b.Label(l+"tail").Raw("BEQ R5, %sdone", l).
		Raw("MOVD 0(R6), F1").Raw("MOVD 0(R7), F2").Raw("%s F2, F1", sca).Raw("MOVD F1, 0(R4)").
		Raw("ADDV $8, R4").Raw("ADDV $8, R6").Raw("ADDV $8, R7").Raw("ADDV $-1, R5").Raw("JMP %stail", l)
	b.Label(l + "done").Ret()
	return b.Func()
}

// pwBlockKernel builds pwBlockLASX(a *float64, n int) float64 for 8 <= n <= 128.
func pwBlockKernel() *emit.Function {
	sig := loong64.Layout([]string{"a", "n"}, []loong64.Type{loong64.Ptr, loong64.Int64},
		[]string{"ret"}, []loong64.Type{loong64.Float64})
	b := loong64.NewFunc("pwBlockLASX", sig, 0)
	b.LoadArg("a", "R4").LoadArg("n", "R5").
		Raw("XVMOVQ 0(R4), X0").Raw("XVMOVQ 32(R4), X1"). // r[0..7] = a[0..7]
		Raw("ADDV $64, R4").Raw("ADDV $-8, R5")
	b.Label("ploop").Raw("MOVV $8, R12").Raw("BLT R5, R12, pfold").
		Raw("XVMOVQ 0(R4), X2").Raw("XVMOVQ 32(R4), X3").
		Raw("XVADDD X2, X0, X0").Raw("XVADDD X3, X1, X1").
		Raw("ADDV $64, R4").Raw("ADDV $-8, R5").Raw("JMP ploop")
	b.Label("pfold")
	for l := 0; l < 4; l++ {
		b.Raw("XVMOVQ X0.V[%d], R%d", l, 12+l)
	}
	b.Raw("MOVV R12, F0").Raw("MOVV R13, F1").Raw("MOVV R14, F2").Raw("MOVV R15, F3")
	for l := 0; l < 4; l++ {
		b.Raw("XVMOVQ X1.V[%d], R%d", l, 12+l)
	}
	b.Raw("MOVV R12, F4").Raw("MOVV R13, F5").Raw("MOVV R14, F6").Raw("MOVV R15, F7")
	b.Raw("ADDD F1, F0").Raw("ADDD F3, F2").Raw("ADDD F2, F0"). // (r0+r1)+(r2+r3)
									Raw("ADDD F5, F4").Raw("ADDD F7, F6").Raw("ADDD F6, F4"). // (r4+r5)+(r6+r7)
									Raw("ADDD F4, F0")
	b.Label("ptail").Raw("BEQ R5, pdone").
		Raw("MOVD 0(R4), F1").Raw("ADDD F1, F0").
		Raw("ADDV $8, R4").Raw("ADDV $-1, R5").Raw("JMP ptail")
	b.Label("pdone").StoreRet("F0", "ret").Ret()
	return b.Func()
}
