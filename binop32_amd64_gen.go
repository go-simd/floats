//go:build ignore

// Command gen produces binop32_amd64.s with go-asmgen: the float32 elementwise
// Add32, Sub32, Mul32, Div32 and Sqrt32 kernels (go-ndarray kept SSE2 copies of
// them privately in internal/kernels), in two flavours:
//
//   - AVX2 (add32AVX2 ...): eight floats per YMM register, four registers (32
//     elements) per iteration, then one register (8), then one float. VADDPS /
//     VSUBPS / VMULPS / VDIVPS / VSQRTPS and VADDSS / VSUBSS / VMULSS / VDIVSS /
//     VSQRTSS (Intel SDM vol. 2). They share the gate of the other AVX2 kernels
//     (hasAVX2FMA); no FMA instruction appears. div32AVX2Halves is a second Div32
//     kernel: YMM divides fed and drained by 128-bit halves, two registers per
//     iteration (divAVX2Halves), chosen on Zen 3 only (div32HalvesCPU).
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
		if op.name == "div32" {
			f.Add(divAVX2Halves("div32AVX2Halves").Func())
		}
		f.Add(binopSSE(op.name+"SSE2", op.sseP, op.sseS).Func())
	}
	f.Add(sqrtAVX().Func())
	f.Add(sqrtSSE().Func())
	f.Add(cpuidFunc().Func())
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
	elemLoopN(b, w, 4, vec, sca)
}

// elemLoopN is elemLoop with `unroll` registers per big step.
func elemLoopN(b *amd64.Builder, w, unroll int, vec func(regs int), sca func()) {
	big := unroll * w
	b.Raw("XORQ AX, AX").
		Label("uloop").
		Raw("LEAQ %d(AX), R8", big).Raw("CMPQ R8, CX").Raw("JGT vloop")
	vec(unroll)
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

// divAVX2Halves: dst[i] = a[i] / b[i] with VDIVPS on YMM registers fed and
// drained by 128-bit halves (VMOVUPS xmm + VINSERTF128 from memory in, VMOVUPS
// xmm + VEXTRACTF128 to memory out), two YMM registers (16 elements) per
// iteration, loads of a and b first, then the divides, then the stores: the
// shape of NumPy's float32 divide loop (numpy/_core/src/umath/
// loops_arithm_fp.dispatch.c.src, v2.5.5, lines 103-112: two vectors per
// iteration, all loads, then the divides, then the stores) with the memory side
// of the 128-bit SSE2 loop. On Zen 3 the divider keeps its 256-bit throughput
// while the memory system sees the access pattern that moves data best
// (floats#93). Y1,Y2 hold a, Y9,Y10 hold b. Tail: one YMM (8), then one float.
func divAVX2Halves(name string) *amd64.Builder {
	b := amd64.NewFunc(name, abi.LayoutArgs(sig3(), nil), 0)
	b.LoadArg("dst_base", "DI").LoadArg("a_base", "SI").LoadArg("a_len", "CX").LoadArg("b_base", "DX")
	elemLoopN(b, 8, 2, func(regs int) {
		for k := 0; k < regs; k++ {
			b.Raw("VMOVUPS %s, X%d", mem("SI", 32*k), 1+k).
				Raw("VINSERTF128 $1, %s, Y%d, Y%d", mem("SI", 32*k+16), 1+k, 1+k).
				Raw("VMOVUPS %s, X%d", mem("DX", 32*k), 9+k).
				Raw("VINSERTF128 $1, %s, Y%d, Y%d", mem("DX", 32*k+16), 9+k, 9+k)
		}
		for k := 0; k < regs; k++ {
			b.Raw("VDIVPS Y%d, Y%d, Y%d", 9+k, 1+k, 1+k) // Y = a / b
		}
		for k := 0; k < regs; k++ {
			b.Raw("VMOVUPS X%d, %s", 1+k, mem("DI", 32*k)).
				Raw("VEXTRACTF128 $1, Y%d, %s", 1+k, mem("DI", 32*k+16))
		}
	}, func() {
		b.Raw("VMOVSS (SI)(AX*4), X1").
			Raw("VDIVSS (DX)(AX*4), X1, X1").
			Raw("VMOVSS X1, (DI)(AX*4)")
	})
	b.Raw("VZEROUPPER").Ret()
	return b
}

// cpuidFunc: func cpuid(op, sub uint32) (eax, ebx, ecx, edx uint32), the CPUID
// instruction (Intel SDM vol. 2, CPUID). It selects the Div32 kernel by CPU
// family and model (div32HalvesCPU).
func cpuidFunc() *amd64.Builder {
	u := abi.Uint32
	b := amd64.NewFunc("cpuid", abi.LayoutArgs(
		[]abi.Arg{abi.Scalar("op", u), abi.Scalar("sub", u)},
		[]abi.Arg{abi.Scalar("eax", u), abi.Scalar("ebx", u), abi.Scalar("ecx", u), abi.Scalar("edx", u)}), 0)
	b.LoadArg("op", "AX").LoadArg("sub", "CX").Raw("CPUID").
		StoreRet("AX", "eax").StoreRet("BX", "ebx").StoreRet("CX", "ecx").StoreRet("DX", "edx").Ret()
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
