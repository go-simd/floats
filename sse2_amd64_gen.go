//go:build ignore

// Command gen produces sse2_amd64.s with go-asmgen: SSE2 versions of the float64
// elementwise Add, Sub, Mul, Div, Sqrt and ClampScalar kernels, for amd64 CPUs
// without AVX2+FMA (pre-Haswell parts, some virtualised and emulated CPUs). SSE2
// is part of the amd64 baseline, so these need no feature test (floats#31;
// go-ndarray had them as addSSE2 ... sqrtSSE2 before v0.37.0).
//
// Every instruction is a plain SSE2 one, one IEEE rounding per element, and no
// FMA appears, so the result is bit-identical to the scalar Go loop and to NumPy:
//
//   - ADDPD / SUBPD / MULPD / DIVPD / SQRTPD (packed double) and ADDSD / SUBSD /
//     MULSD / DIVSD / SQRTSD (scalar double): Intel SDM vol. 2.
//   - ClampScalar: MAXPD / MINPD return the second (source) operand when the
//     operands are equal (±0) or either is a NaN. `MAXPD src, dst` is
//     dst > src ? dst : src, so with dst = a copy of the broadcast bound and
//     src = v the result is lo > v ? lo : v, as the AVX2 kernel's `VMAXPD v, lo,
//     r`; `MINPD r, copy-of-hi` is hi < r ? hi : r.
//
// SSE2 memory operands of packed instructions must be 16-byte aligned, so every
// input is loaded with MOVUPD into a register first. Each 8-element block is
// fully loaded, computed and stored before the next, so dst may alias an input
// exactly.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> sse2_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("amd64")
	f.Add(binop("addSSE2", "ADDPD", "ADDSD").Func())
	f.Add(binop("subSSE2", "SUBPD", "SUBSD").Func())
	f.Add(binop("mulSSE2", "MULPD", "MULSD").Func())
	f.Add(binop("divSSE2", "DIVPD", "DIVSD").Func())
	f.Add(sqrt().Func())
	f.Add(clampScalar().Func())
	if err := os.WriteFile("sse2_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote sse2_amd64.s")
}

// elemLoop emits an x4-vector (8 elements) / x1-vector (2 elements) / scalar loop
// skeleton; vec(pairs, off) emits one block of `pairs` XMM registers starting at
// byte offset 16*k, sca the scalar step. AX is the element index, CX the length.
func elemLoop(b *amd64.Builder, vec func(pairs int), sca func()) {
	b.Raw("XORQ AX, AX").
		Label("uloop").
		Raw("LEAQ 8(AX), R8").Raw("CMPQ R8, CX").Raw("JGT vloop")
	vec(4)
	b.Raw("ADDQ $8, AX").Raw("JMP uloop").
		Label("vloop").
		Raw("LEAQ 2(AX), R8").Raw("CMPQ R8, CX").Raw("JGT sloop")
	vec(1)
	b.Raw("ADDQ $2, AX").Raw("JMP vloop").
		Label("sloop").
		Raw("CMPQ AX, CX").Raw("JGE done")
	sca()
	b.Raw("ADDQ $1, AX").Raw("JMP sloop").
		Label("done")
}

func mem(base string, off int) string {
	if off == 0 {
		return fmt.Sprintf("(%s)(AX*8)", base)
	}
	return fmt.Sprintf("%d(%s)(AX*8)", off, base)
}

// binop: dst[i] = a[i] OP b[i]; func(dst, a, b []float64). X1..X4 hold a, X5..X8 b.
func binop(name, vop, sop string) *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Slice("b")}, nil)
	b := amd64.NewFunc(name, sig, 0)
	b.LoadArg("dst_base", "DI").LoadArg("a_base", "SI").LoadArg("a_len", "CX").LoadArg("b_base", "DX")
	elemLoop(b, func(pairs int) {
		for k := 0; k < pairs; k++ {
			b.Raw("MOVUPD %s, X%d", mem("SI", 16*k), 1+k).
				Raw("MOVUPD %s, X%d", mem("DX", 16*k), 5+k)
		}
		for k := 0; k < pairs; k++ {
			b.Raw("%s X%d, X%d", vop, 5+k, 1+k) // X(1+k) = a OP b
		}
		for k := 0; k < pairs; k++ {
			b.Raw("MOVUPD X%d, %s", 1+k, mem("DI", 16*k))
		}
	}, func() {
		b.Raw("MOVSD (SI)(AX*8), X1").
			Raw("MOVSD (DX)(AX*8), X5").
			Raw("%s X5, X1", sop).
			Raw("MOVSD X1, (DI)(AX*8)")
	})
	b.Ret()
	return b
}

// sqrt: dst[i] = sqrt(src[i]); func(dst, src []float64).
func sqrt() *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("src")}, nil)
	b := amd64.NewFunc("sqrtSSE2", sig, 0)
	b.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("src_len", "CX")
	elemLoop(b, func(pairs int) {
		for k := 0; k < pairs; k++ {
			b.Raw("MOVUPD %s, X%d", mem("SI", 16*k), 1+k)
		}
		for k := 0; k < pairs; k++ {
			b.Raw("SQRTPD X%d, X%d", 1+k, 1+k)
		}
		for k := 0; k < pairs; k++ {
			b.Raw("MOVUPD X%d, %s", 1+k, mem("DI", 16*k))
		}
	}, func() {
		b.Raw("MOVSD (SI)(AX*8), X1").
			Raw("SQRTSD X1, X1").
			Raw("MOVSD X1, (DI)(AX*8)")
	})
	b.Ret()
	return b
}

// clampScalar: dst[i] = (lo > v ? lo : v), then (hi < r ? hi : r), scalar bounds;
// func(dst, v []float64, lo, hi float64). X0 / X5 = lo / hi in both lanes. X1..X4
// load v, X6..X9 are the bound copies the destructive MAXPD / MINPD consume.
func clampScalar() *amd64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("v"),
		abi.Scalar("lo", abi.Float64), abi.Scalar("hi", abi.Float64)}, nil)
	b := amd64.NewFunc("clampScalarSSE2", sig, 0)
	b.LoadArg("dst_base", "DI").LoadArg("v_base", "SI").LoadArg("v_len", "CX").
		LoadArg("lo", "X0").LoadArg("hi", "X5").
		Raw("UNPCKLPD X0, X0").Raw("UNPCKLPD X5, X5")
	elemLoop(b, func(pairs int) {
		for k := 0; k < pairs; k++ {
			b.Raw("MOVUPD %s, X%d", mem("SI", 16*k), 1+k).
				Raw("MOVAPD X0, X%d", 6+k)
		}
		for k := 0; k < pairs; k++ {
			b.Raw("MAXPD X%d, X%d", 1+k, 6+k) // X(6+k) = lo > v ? lo : v
		}
		for k := 0; k < pairs; k++ {
			b.Raw("MOVAPD X5, X%d", 1+k)
		}
		for k := 0; k < pairs; k++ {
			b.Raw("MINPD X%d, X%d", 6+k, 1+k) // X(1+k) = hi < r ? hi : r
		}
		for k := 0; k < pairs; k++ {
			b.Raw("MOVUPD X%d, %s", 1+k, mem("DI", 16*k))
		}
	}, func() {
		b.Raw("MOVSD (SI)(AX*8), X1").
			Raw("MOVAPD X0, X6").
			Raw("MAXSD X1, X6").
			Raw("MOVAPD X5, X1").
			Raw("MINSD X6, X1").
			Raw("MOVSD X1, (DI)(AX*8)")
	})
	b.Ret()
	return b
}
