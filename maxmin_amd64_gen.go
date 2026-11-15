//go:build ignore

// Command gen produces maxmin_amd64.s with go-asmgen: the float64 pairwise
// Maximum and Minimum kernels (go-ndarray/ndarray#98, #118), AVX2 and SSE2.
//
// The contract is the Go builtin max / min as runtime/minmax.go spells it, bit for
// bit (maxmin.go): a NaN b[i] wins, else a NaN a[i] wins, else the larger
// (smaller) operand, and for two zeros the bitwise AND (OR) of the two, so +0 is
// above -0 whatever the operand order.
//
// MAXPD / MINPD cannot give it: they return the SECOND operand when either operand
// is a NaN or both are zero, so a NaN a[i] is lost and (+0, -0) depends on the
// order. So both kernels use a compare-and-blend, with the instruction semantics
// of Intel SDM vol. 2 (CMPPD / VCMPPD: predicate 0 = EQ, 1 = LT, 3 = UNORD,
// every comparison false when an operand is a NaN, except UNORD and NEQ; VBLENDVPD
// picks, per element, the sign bit of the mask register; ANDPD, ANDNPD, ORPD
// are bitwise), checked by the execution tests (maxmin_test.go, bit for bit, on a
// real Zen 3 core and under the SSE2 gate), not by the SDM text, which was not
// opened for this generator:
//
//	takeB = (a < b) | unordered(b, b)      Maximum        (b < a) for Minimum
//	r     = takeB ? b : a
//	r     = (a == b) ? (a & b) : r         (a | b for Minimum)
//
// takeB and (a == b) never hold together (a == b is false for a NaN), and a == b of
// two equal non-zero numbers is the same bits, so the third line only changes the
// two-zeros case: the AND of -0 and +0 is +0, the OR is -0, and the result of a
// pair of equal zeros of the same sign is that zero. Every selected value is a
// copy of a or b, never an arithmetic result, so a NaN keeps its sign, payload and
// quiet bit.
//
// Each kernel does n/8 blocks of eight elements (n a positive multiple of 8,
// sliced by the Go caller, which finishes the last seven elements); a block is
// loaded, computed and stored before the next, so dst may alias a or b exactly.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> maxmin_amd64_gen.go
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
	f.Add(avx2("maximumAVX2", true).Func())
	f.Add(avx2("minimumAVX2", false).Func())
	f.Add(sse2("maximumSSE2", true).Func())
	f.Add(sse2("minimumSSE2", false).Func())
	if err := os.WriteFile("maxmin_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote maxmin_amd64.s")
}

func sig() []abi.Arg { return []abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Slice("b")} }

// avx2: two YMM groups (8 doubles) per step. Group g uses Y(4g)..Y(4g+3) as a, b and
// two temporaries, the result lands in the a register.
func avx2(name string, isMax bool) *amd64.Builder {
	b := amd64.NewFunc(name, abi.LayoutArgs(sig(), nil), 0)
	b.LoadArg("dst_base", "DI").LoadArg("a_base", "SI").LoadArg("a_len", "CX").LoadArg("b_base", "DX")
	b.Raw("XORQ AX, AX").Label("loop")
	for g := 0; g < 2; g++ {
		b.Raw("VMOVUPD %d(SI)(AX*8), Y%d", 32*g, 4*g).
			Raw("VMOVUPD %d(DX)(AX*8), Y%d", 32*g, 4*g+1)
	}
	for g := 0; g < 2; g++ {
		a, bb, t, u := fmt.Sprintf("Y%d", 4*g), fmt.Sprintf("Y%d", 4*g+1), fmt.Sprintf("Y%d", 4*g+2), fmt.Sprintf("Y%d", 4*g+3)
		lt, op := "VCMPPD $1, %s, %s, %s", "VANDPD"
		x, y := bb, a // t = a < b: VCMPPD imm, src3, src2, dst computes src2 < src3
		if !isMax {
			x, y, op = a, bb, "VORPD" // t = b < a
		}
		b.Raw(lt, x, y, t).
			Raw("VCMPPD $3, %s, %s, %s", bb, bb, u).
			Raw("VORPD %s, %s, %s", u, t, t).             // takeB
			Raw("VBLENDVPD %s, %s, %s, %s", t, bb, a, t). // t = takeB ? b : a
			Raw("VCMPPD $0, %s, %s, %s", bb, a, u).       // u = a == b
			Raw("%s %s, %s, %s", op, bb, a, a).           // a = a AND/OR b
			Raw("VBLENDVPD %s, %s, %s, %s", u, a, t, a)   // a = eq ? (a op b) : t
	}
	for g := 0; g < 2; g++ {
		b.Raw("VMOVUPD Y%d, %d(DI)(AX*8)", 4*g, 32*g)
	}
	b.Raw("ADDQ $8, AX").Raw("CMPQ AX, CX").Raw("JLT loop").Raw("VZEROUPPER").Ret()
	return b
}

// sse2: four XMM vectors (8 doubles) per step, one at a time with the registers
// X0 (a), X1 (b) and X2..X5 as temporaries; the result is stored from X5.
func sse2(name string, isMax bool) *amd64.Builder {
	b := amd64.NewFunc(name, abi.LayoutArgs(sig(), nil), 0)
	b.LoadArg("dst_base", "DI").LoadArg("a_base", "SI").LoadArg("a_len", "CX").LoadArg("b_base", "DX")
	b.Raw("XORQ AX, AX").Label("loop")
	for k := 0; k < 4; k++ {
		o := 16 * k
		b.Raw("MOVUPD %d(SI)(AX*8), X0", o).Raw("MOVUPD %d(DX)(AX*8), X1", o)
		op := "ANDPD"
		if isMax {
			b.Raw("MOVAPD X0, X2").Raw("CMPPD X1, X2, $1") // X2 = a < b
		} else {
			op = "ORPD"
			b.Raw("MOVAPD X1, X2").Raw("CMPPD X0, X2, $1") // X2 = b < a
		}
		b.Raw("MOVAPD X1, X3").Raw("CMPPD X3, X3, $3"). // X3 = b is a NaN
								Raw("ORPD X3, X2").                           // X2 = takeB
								Raw("MOVAPD X0, X3").Raw("CMPPD X1, X3, $0"). // X3 = a == b
								Raw("MOVAPD X0, X4").Raw("%s X1, X4", op).    // X4 = a AND/OR b
								Raw("MOVAPD X2, X5").Raw("ANDNPD X0, X5").    // X5 = a where not takeB
								Raw("ANDPD X2, X1").                          // X1 = b where takeB
								Raw("ORPD X1, X5").                           // X5 = takeB ? b : a
								Raw("ANDPD X3, X4").                          // X4 = eq ? a op b : 0
								Raw("ANDNPD X5, X3").                         // X3 = eq ? 0 : X5
								Raw("ORPD X4, X3").                           // X3 = the result
								Raw("MOVUPD X3, %d(DI)(AX*8)", o)
	}
	b.Raw("ADDQ $8, AX").Raw("CMPQ AX, CX").Raw("JLT loop").Ret()
	return b
}
