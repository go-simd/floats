//go:build ignore

// Command gen produces lanes_riscv64.s with go-asmgen: the riscv64 RVV 1.0
// float64 elementwise kernels Add, Sub, Mul, Div, Sqrt of go-ndarray
// (internal/kernels/asmgen/riscv64, moved here: floats#30) and the
// eight-accumulator pairwise-sum block of SumPairwise.
//
// Go's riscv64 assembler names the vector instructions (VSETVLI, VLE64V,
// VSE64V, VFADDVV, VFSUBVV, VFMULVV, VFDIVVV, VFSQRTV), so nothing is a WORD. The
// elementwise loops are vector-length agnostic: VSETVLI hands out as many
// elements as the hardware holds (LMUL=8), including a short last strip, so there
// is no scalar tail. One IEEE operation per element, no fused instruction:
// bit-identical to the scalar loops. The V extension is optional, so the Go side
// calls these only when cpu.RISCV64.HasV.
//
// Operand order, decoded from Go's own testdata encodings (ndarray): VFADDVV V1,
// V2, V3 computes V3 = V2 + V1 (likewise SUB and DIV: V3 = V2 - V1, V2 / V1).
//
// pwBlockRVV(a, n): 8 <= n <= 128. NumPy's eight accumulators r[0..7] are one
// register group (LMUL=4, vl=8: a group holds 8 doubles whatever VLEN >= 128 is),
// loaded from a[0:8] and then added to each following block of eight, so every
// lane receives exactly the additions of the scalar r[j]. The group is stored to
// the frame, loaded into F0..F7 and folded ((r0+r1)+(r2+r3)) + ((r4+r5)+(r6+r7))
// with scalar FADDD; the leftover n % 8 elements are added to the result one at
// a time, as NumPy does. The frame slot starts at 8(SP): the first word is left
// alone whatever the assembler keeps there.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> lanes_riscv64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
)

func main() {
	f := emit.NewFile("riscv64")
	f.Add(unaryKernel("sqrtRVV", "VFSQRTV V0, V16"))
	for _, k := range []struct{ name, op string }{
		{"addRVV", "VFADDVV"}, {"subRVV", "VFSUBVV"},
		{"mulRVV", "VFMULVV"}, {"divRVV", "VFDIVVV"},
	} {
		f.Add(binKernel(k.name, k.op))
	}
	f.Add(pwBlockKernel())
	if err := os.WriteFile("lanes_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote lanes_riscv64.s")
}

// advance moves each pointer register past vl (in X) float64s.
func advance(b *riscv64.Builder, vl string, ptrs ...string) {
	b.Raw("SLLI $3, %s, X17", vl)
	for _, p := range ptrs {
		b.Raw("ADD X17, %s, %s", p, p)
	}
}

// unaryKernel builds NAME(dst, src *float64, n int) for n >= 1: op reads V0 and
// writes V16.
func unaryKernel(name, op string) *emit.Function {
	sig := riscv64.Layout([]string{"dst", "src", "n"},
		[]riscv64.Type{riscv64.Ptr, riscv64.Ptr, riscv64.Int64}, nil, nil)
	b := riscv64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "X10").LoadArg("src", "X11").LoadArg("n", "X13")
	b.Raw("loop:").
		Raw("VSETVLI X13, E64, M8, TA, MA, X14").
		Raw("VLE64V (X11), V0").
		Raw(op).
		Raw("VSE64V V16, (X10)")
	advance(b, "X14", "X10", "X11")
	b.Raw("SUB X14, X13, X13").Raw("BNEZ X13, loop").Ret()
	return b.Func()
}

// binKernel builds NAME(dst, a, b *float64, n int) for n >= 1: dst = a OP b.
func binKernel(name, op string) *emit.Function {
	sig := riscv64.Layout([]string{"dst", "a", "b", "n"},
		[]riscv64.Type{riscv64.Ptr, riscv64.Ptr, riscv64.Ptr, riscv64.Int64}, nil, nil)
	b := riscv64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "X10").LoadArg("a", "X11").LoadArg("b", "X12").LoadArg("n", "X13")
	b.Raw("loop:").
		Raw("VSETVLI X13, E64, M8, TA, MA, X14").
		Raw("VLE64V (X11), V0").
		Raw("VLE64V (X12), V8").
		Raw("%s V8, V0, V16", op). // V16 = a OP b
		Raw("VSE64V V16, (X10)")
	advance(b, "X14", "X10", "X11", "X12")
	b.Raw("SUB X14, X13, X13").Raw("BNEZ X13, loop").Ret()
	return b.Func()
}

// pwBlockKernel builds pwBlockRVV(a *float64, n int) float64 for 8 <= n <= 128.
func pwBlockKernel() *emit.Function {
	sig := riscv64.Layout([]string{"a", "n"}, []riscv64.Type{riscv64.Ptr, riscv64.Int64},
		[]string{"ret"}, []riscv64.Type{riscv64.Float64})
	b := riscv64.NewFunc("pwBlockRVV", sig, 80)
	b.LoadArg("a", "X10").LoadArg("n", "X11").
		Raw("MOV $8, X5").
		Raw("VSETVLI X5, E64, M4, TA, MA, X6"). // vl = 8
		Raw("VLE64V (X10), V8").                // r[0..7] = a[0..7]
		Raw("ADD $64, X10").Raw("ADD $-8, X11").
		Raw("loop:").
		Raw("SLTIU $8, X11, X5").Raw("BNEZ X5, fold"). // fewer than 8 left (unsigned compare, n >= 0)
		Raw("VLE64V (X10), V0").
		Raw("VFADDVV V0, V8, V8"). // r += block
		Raw("ADD $64, X10").Raw("ADD $-8, X11").Raw("JMP loop").
		Raw("fold:").
		Raw("ADD $8, X2, X5").
		Raw("VSE64V V8, (X5)")
	for k := 0; k < 8; k++ {
		b.Raw("MOVD %d(X5), F%d", 8*k, k)
	}
	b.Raw("FADDD F1, F0, F8").Raw("FADDD F3, F2, F9").Raw("FADDD F9, F8, F8"). // (r0+r1)+(r2+r3)
											Raw("FADDD F5, F4, F10").Raw("FADDD F7, F6, F11").Raw("FADDD F11, F10, F10"). // (r4+r5)+(r6+r7)
											Raw("FADDD F10, F8, F8").
											Raw("tail:").Raw("BEQZ X11, done").
											Raw("MOVD (X10), F1").Raw("FADDD F1, F8, F8").
											Raw("ADD $8, X10").Raw("ADD $-1, X11").Raw("JMP tail").
											Raw("done:").
											StoreRet("F8", "ret").
											Ret()
	return b.Func()
}
