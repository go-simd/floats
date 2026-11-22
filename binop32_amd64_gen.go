//go:build ignore

// Command gen produces binop32_amd64.s with go-asmgen: the float32 elementwise
// Add32, Sub32, Mul32, Div32 and Sqrt32 kernels (go-ndarray kept SSE2 copies of
// them privately in internal/kernels), in two flavours:
//
//   - AVX2 (add32AVX2 ...): eight floats per YMM register, four registers (32
//     elements) per iteration, then one register (8), then one float. VADDPS /
//     VSUBPS / VMULPS / VDIVPS / VSQRTPS and VADDSS / VSUBSS / VMULSS / VDIVSS /
//     VSQRTSS (Intel SDM vol. 2). They share the gate of the other AVX2 kernels
//     (hasAVX2FMA); no FMA instruction appears.
//   - SSE2 (add32SSE2 ...): four floats per XMM register, 16 / 4 / 1 elements per
//     step, ADDPS ... SQRTPS and ADDSS ... SQRTSS. SSE2 is the amd64 baseline, so
//     it runs on a CPU without AVX2+FMA. Memory operands of packed SSE
//     instructions must be 16-byte aligned, so every input is loaded with MOVUPS.
//
// Every instruction is one IEEE-754 binary32 operation (one rounding) per lane,
// so each result is the correctly rounded one: bit-identical to the scalar Go
// loop (and Sqrt32 to float32(math.Sqrt(float64(x))): sqrt of a float32 widened
// to float64 and rounded back is the correctly rounded float32 root, there is no
// double-rounding issue for sqrt). Each block is loaded, computed and stored
// before the next, so dst may alias an input exactly.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> binop32_amd64_gen.go
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
	for _, op := range []struct{ name, avxP, avxS, sseP, sseS string }{
		{"add32", "VADDPS", "VADDSS", "ADDPS", "ADDSS"},
		{"sub32", "VSUBPS", "VSUBSS", "SUBPS", "SUBSS"},
		{"mul32", "VMULPS", "VMULSS", "MULPS", "MULSS"},
		{"div32", "VDIVPS", "VDIVSS", "DIVPS", "DIVSS"},
	} {
		f.Add(binopAVX(op.name+"AVX2", op.avxP, op.avxS).Func())
		f.Add(binopSSE(op.name+"SSE2", op.sseP, op.sseS).Func())
	}
	f.Add(sqrtAVX().Func())
	f.Add(sqrtSSE().Func())
	if err := os.WriteFile("binop32_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote binop32_amd64.s")
}

// elemLoop emits the three-stage loop skeleton: blk elements per big step (4
// registers of w lanes), w per single-register step, then one element. AX is the
// element index, CX the length. vec(k, regs) emits the block of `regs` registers
// starting at register k's byte offset; sca the scalar step.
func elemLoop(b *amd64.Builder, w int, vec func(regs int), sca func()) {
	big := 4 * w
	b.Raw("XORQ AX, AX").
		Label("uloop").
		Raw("LEAQ %d(AX), R8", big).Raw("CMPQ R8, CX").Raw("JGT vloop")
	vec(4)
	b.Raw("ADDQ $%d, AX", big).Raw("JMP uloop").
		Label("vloop").
		Raw("LEAQ %d(AX), R8", w).Raw("CMPQ R8, CX").Raw("JGT sloop")
	vec(1)
	b.Raw("ADDQ $%d, AX", w).Raw("JMP vloop").
		Label("sloop").
		Raw("CMPQ AX, CX").Raw("JGE done")
	sca()
	b.Raw("ADDQ $1, AX").Raw("JMP sloop").
		Label("done")
}

func mem(base string, off int) string {
	if off == 0 {
		return fmt.Sprintf("(%s)(AX*4)", base)
	}
	return fmt.Sprintf("%d(%s)(AX*4)", off, base)
}

func sig3() []abi.Arg { return []abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Slice("b")} }

func sig2() []abi.Arg { return []abi.Arg{abi.Slice("dst"), abi.Slice("src")} }

// binopAVX: dst[i] = a[i] OP b[i]; func(dst, a, b []float32). Y1..Y4 hold a.
func binopAVX(name, vop, sop string) *amd64.Builder {
	b := amd64.NewFunc(name, abi.LayoutArgs(sig3(), nil), 0)
	b.LoadArg("dst_base", "DI").LoadArg("a_base", "SI").LoadArg("a_len", "CX").LoadArg("b_base", "DX")
	elemLoop(b, 8, func(regs int) {
		for k := 0; k < regs; k++ {
			b.Raw("VMOVUPS %s, Y%d", mem("SI", 32*k), 1+k).
				Raw("%s %s, Y%d, Y%d", vop, mem("DX", 32*k), 1+k, 1+k). // Y = a OP b
				Raw("VMOVUPS Y%d, %s", 1+k, mem("DI", 32*k))
		}
	}, func() {
		b.Raw("VMOVSS (SI)(AX*4), X1").
			Raw("%s (DX)(AX*4), X1, X1", sop).
			Raw("VMOVSS X1, (DI)(AX*4)")
	})
	b.Raw("VZEROUPPER").Ret()
	return b
}

// sqrtAVX: dst[i] = sqrt(src[i]); func(dst, src []float32).
func sqrtAVX() *amd64.Builder {
	b := amd64.NewFunc("sqrt32AVX2", abi.LayoutArgs(sig2(), nil), 0)
	b.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("src_len", "CX")
	elemLoop(b, 8, func(regs int) {
		for k := 0; k < regs; k++ {
			b.Raw("VSQRTPS %s, Y%d", mem("SI", 32*k), 1+k).
				Raw("VMOVUPS Y%d, %s", 1+k, mem("DI", 32*k))
		}
	}, func() {
		b.Raw("VMOVSS (SI)(AX*4), X1").
			Raw("VSQRTSS X1, X1, X1").
			Raw("VMOVSS X1, (DI)(AX*4)")
	})
	b.Raw("VZEROUPPER").Ret()
	return b
}

// binopSSE: X1..X4 hold a, X5..X8 hold b; the result replaces a.
func binopSSE(name, vop, sop string) *amd64.Builder {
	b := amd64.NewFunc(name, abi.LayoutArgs(sig3(), nil), 0)
	b.LoadArg("dst_base", "DI").LoadArg("a_base", "SI").LoadArg("a_len", "CX").LoadArg("b_base", "DX")
	elemLoop(b, 4, func(regs int) {
		for k := 0; k < regs; k++ {
			b.Raw("MOVUPS %s, X%d", mem("SI", 16*k), 1+k).
				Raw("MOVUPS %s, X%d", mem("DX", 16*k), 5+k)
		}
		for k := 0; k < regs; k++ {
			b.Raw("%s X%d, X%d", vop, 5+k, 1+k) // X(1+k) = a OP b
		}
		for k := 0; k < regs; k++ {
			b.Raw("MOVUPS X%d, %s", 1+k, mem("DI", 16*k))
		}
	}, func() {
		b.Raw("MOVSS (SI)(AX*4), X1").
			Raw("MOVSS (DX)(AX*4), X5").
			Raw("%s X5, X1", sop).
			Raw("MOVSS X1, (DI)(AX*4)")
	})
	b.Ret()
	return b
}

func sqrtSSE() *amd64.Builder {
	b := amd64.NewFunc("sqrt32SSE2", abi.LayoutArgs(sig2(), nil), 0)
	b.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("src_len", "CX")
	elemLoop(b, 4, func(regs int) {
		for k := 0; k < regs; k++ {
			b.Raw("MOVUPS %s, X%d", mem("SI", 16*k), 1+k)
		}
		for k := 0; k < regs; k++ {
			b.Raw("SQRTPS X%d, X%d", 1+k, 1+k)
		}
		for k := 0; k < regs; k++ {
			b.Raw("MOVUPS X%d, %s", 1+k, mem("DI", 16*k))
		}
	}, func() {
		b.Raw("MOVSS (SI)(AX*4), X1").
			Raw("SQRTSS X1, X1").
			Raw("MOVSS X1, (DI)(AX*4)")
	})
	b.Ret()
	return b
}
