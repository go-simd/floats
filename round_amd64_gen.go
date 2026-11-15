//go:build ignore

// Command gen produces round_amd64.s with go-asmgen: the float64 elementwise Abs,
// Neg, Square, Floor, Ceil, Trunc and RoundEven kernels (go-ndarray/ndarray#118).
//
// Sources (the SDM page of ROUNDPD, felixcloutier.com/x86/roundpd, was opened):
//
//   - Intel SDM vol. 2: VROUNDPD (ymm, imm8): imm8[1:0] is the rounding mode
//     (00 nearest even, 01 down, 10 up, 11 toward zero) when imm8[2] = 0, imm8[3]
//     suppresses the precision exception. So 8 = RoundToEven, 9 = Floor, 10 =
//     Ceil, 11 = Trunc, all exact: the result of a roundToIntegral is an integer
//     of the same format, a NaN comes back quieted (a signalling NaN gets its
//     quiet bit set, a quiet NaN is unchanged), infinities and signed zeros are
//     unchanged, -0.5 rounds up/toward zero to -0. VROUNDPD needs AVX only.
//   - VANDPD / VXORPD / VMULPD (and ANDPD / XORPD / MULPD for SSE2): bitwise
//     AND, bitwise XOR, packed double multiply.
//   - VPCMPEQQ (AVX2) / PCMPEQL (SSE2) of a register with itself is all ones;
//     VPSRLQ $1 / PSRLQ $1 of it is 0x7FFF...F (the Abs mask), VPSLLQ $63 /
//     PSLLQ $63 is 0x8000...0 (the Neg mask). No constant is loaded from memory.
//
// Abs and Neg are pure bit operations, so every NaN keeps its payload. Each
// kernel does n/8 blocks of eight elements (n a positive multiple of 8, sliced by
// the Go caller, which finishes the last seven elements); a block is loaded,
// computed and stored before the next, so dst may alias src exactly.
//
// SSE2 has no ROUNDPD (it is SSE4.1): the rounding functions have no SSE2 kernel
// and run the scalar loop on a CPU without AVX2+FMA.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> round_amd64_gen.go
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
	for _, k := range []struct{ name, setup, op string }{
		{"absAVX2", "VPCMPEQQ Y0, Y0, Y0\n\tVPSRLQ $1, Y0, Y0", "VANDPD Y0, %s, %s"},
		{"negAVX2", "VPCMPEQQ Y0, Y0, Y0\n\tVPSLLQ $63, Y0, Y0", "VXORPD Y0, %s, %s"},
		{"squareAVX2", "", "VMULPD %s, %s, %s"},
		{"floorAVX2", "", "VROUNDPD $9, %s, %s"},
		{"ceilAVX2", "", "VROUNDPD $10, %s, %s"},
		{"truncAVX2", "", "VROUNDPD $11, %s, %s"},
		{"rintAVX2", "", "VROUNDPD $8, %s, %s"},
	} {
		f.Add(avx2(k.name, k.setup, k.op).Func())
	}
	for _, k := range []struct{ name, setup, op string }{
		{"absSSE2", "PCMPEQL X0, X0\n\tPSRLQ $1, X0", "ANDPD X0, %s"},
		{"negSSE2", "PCMPEQL X0, X0\n\tPSLLQ $63, X0", "XORPD X0, %s"},
		{"squareSSE2", "", "MULPD %s, %s"},
	} {
		f.Add(sse2(k.name, k.setup, k.op).Func())
	}
	if err := os.WriteFile("round_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote round_amd64.s")
}

func sig() []abi.Arg { return []abi.Arg{abi.Slice("dst"), abi.Slice("src")} }

// avx2: two YMM registers (8 doubles) per step. op has one %s per register
// operand: the source/destination register Yk (Y1, Y2).
func avx2(name, setup, op string) *amd64.Builder {
	b := amd64.NewFunc(name, abi.LayoutArgs(sig(), nil), 0)
	b.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("src_len", "CX")
	if setup != "" {
		b.Raw(setup)
	}
	b.Raw("XORQ AX, AX").Label("loop").
		Raw("VMOVUPD (SI)(AX*8), Y1").Raw("VMOVUPD 32(SI)(AX*8), Y2")
	for _, r := range []string{"Y1", "Y2"} {
		switch n := countVerbs(op); n {
		case 2:
			b.Raw(op, r, r)
		default:
			b.Raw(op, r, r, r)
		}
	}
	b.Raw("VMOVUPD Y1, (DI)(AX*8)").Raw("VMOVUPD Y2, 32(DI)(AX*8)").
		Raw("ADDQ $8, AX").Raw("CMPQ AX, CX").Raw("JLT loop").
		Raw("VZEROUPPER").Ret()
	return b
}

// sse2: four XMM registers (8 doubles) per step. op has one or two %s: the
// register (X1..X4), twice for MULPD.
func sse2(name, setup, op string) *amd64.Builder {
	b := amd64.NewFunc(name, abi.LayoutArgs(sig(), nil), 0)
	b.LoadArg("dst_base", "DI").LoadArg("src_base", "SI").LoadArg("src_len", "CX")
	if setup != "" {
		b.Raw(setup)
	}
	b.Raw("XORQ AX, AX").Label("loop")
	for k := 0; k < 4; k++ {
		b.Raw("MOVUPD %d(SI)(AX*8), X%d", 16*k, 1+k)
	}
	for k := 0; k < 4; k++ {
		r := fmt.Sprintf("X%d", 1+k)
		if countVerbs(op) == 2 {
			b.Raw(op, r, r)
		} else {
			b.Raw(op, r)
		}
	}
	for k := 0; k < 4; k++ {
		b.Raw("MOVUPD X%d, %d(DI)(AX*8)", 1+k, 16*k)
	}
	b.Raw("ADDQ $8, AX").Raw("CMPQ AX, CX").Raw("JLT loop").Ret()
	return b
}

func countVerbs(s string) int {
	n := 0
	for i := 0; i+1 < len(s); i++ {
		if s[i] == '%' && s[i+1] == 's' {
			n++
		}
	}
	return n
}
