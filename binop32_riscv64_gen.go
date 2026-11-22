//go:build ignore

// Command gen produces binop32_riscv64.s with go-asmgen: the float32 elementwise
// Add32, Sub32, Mul32, Div32 and Sqrt32 RVV kernels (go-ndarray/ndarray's, moved
// here so the kernel lives once).
//
// VFADDVV / VFSUBVV / VFMULVV / VFDIVVV / VFSQRTV on 32-bit elements are one IEEE
// binary32 operation per lane, so every result is the correctly rounded one,
// bit-identical to the scalar Go loop. The loops are vector-length agnostic:
// VSETVLI (E32, LMUL=8) hands out as many elements as the hardware holds,
// including a short last strip, so there is no scalar tail. Each strip is loaded,
// computed and stored before the next, so dst may alias an input exactly. The V
// extension is optional: the Go side calls these only when hasRVV, and for n >= 1.
//
// Operand order (decoded from Go's own testdata encodings): VFADDVV V8, V0, V16
// computes V16 = V0 + V8 (likewise SUB and DIV: V0 - V8, V0 / V8).
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> binop32_riscv64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
)

func main() {
	f := emit.NewFile("riscv64")
	for _, k := range []struct{ name, op string }{
		{"add32RVV", "VFADDVV"}, {"sub32RVV", "VFSUBVV"},
		{"mul32RVV", "VFMULVV"}, {"div32RVV", "VFDIVVV"},
	} {
		f.Add(bin32Kernel(k.name, k.op))
	}
	f.Add(sqrt32Kernel())
	if err := os.WriteFile("binop32_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote binop32_riscv64.s")
}

// sqrt32Kernel builds sqrt32RVV(dst, src *float32, n int) for n >= 1.
func sqrt32Kernel() *emit.Function {
	sig := riscv64.Layout([]string{"dst", "src", "n"},
		[]riscv64.Type{riscv64.Ptr, riscv64.Ptr, riscv64.Int64}, nil, nil)
	b := riscv64.NewFunc("sqrt32RVV", sig, 0)
	b.LoadArg("dst", "X10").LoadArg("src", "X11").LoadArg("n", "X13")
	b.Raw("loop:").
		Raw("VSETVLI X13, E32, M8, TA, MA, X14").
		Raw("VLE32V (X11), V0").
		Raw("VFSQRTV V0, V16").
		Raw("VSE32V V16, (X10)").
		Raw("SLLI $2, X14, X17") // vl elements of 4 bytes
	for _, p := range []string{"X10", "X11"} {
		b.Raw("ADD X17, %s, %s", p, p)
	}
	b.Raw("SUB X14, X13, X13").Raw("BNEZ X13, loop").Ret()
	return b.Func()
}

// bin32Kernel builds NAME(dst, a, b *float32, n int) for n >= 1.
func bin32Kernel(name, op string) *emit.Function {
	sig := riscv64.Layout([]string{"dst", "a", "b", "n"},
		[]riscv64.Type{riscv64.Ptr, riscv64.Ptr, riscv64.Ptr, riscv64.Int64}, nil, nil)
	b := riscv64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "X10").LoadArg("a", "X11").LoadArg("b", "X12").LoadArg("n", "X13")
	b.Raw("loop:").
		Raw("VSETVLI X13, E32, M8, TA, MA, X14").
		Raw("VLE32V (X11), V0").
		Raw("VLE32V (X12), V8").
		Raw("%s V8, V0, V16", op). // V16 = a OP b
		Raw("VSE32V V16, (X10)").
		Raw("SLLI $2, X14, X17") // vl elements of 4 bytes
	for _, p := range []string{"X10", "X11", "X12"} {
		b.Raw("ADD X17, %s, %s", p, p)
	}
	b.Raw("SUB X14, X13, X13").Raw("BNEZ X13, loop").Ret()
	return b.Func()
}
