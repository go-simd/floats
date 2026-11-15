//go:build ignore

// Command gen produces maxmin_arm64.s with go-asmgen: the float64 pairwise
// Maximum and Minimum kernels (go-ndarray/ndarray#98, #118), NEON.
//
// The contract is the Go builtin max / min as runtime/minmax.go spells it, bit for
// bit (maxmin.go): a NaN b[i] wins, else a NaN a[i] wins, else the larger
// (smaller) operand, and for two zeros the bitwise AND (OR) of the two, so +0 is
// above -0 whatever the operand order.
//
// Sources: Arm Architecture Reference Manual (DDI 0602, AdvSIMD floating-point
// instructions, 2D arrangement). The online page did not render when this was
// written, so the semantics below are those of the manual as known and were
// verified by execution on Apple silicon (maxmin_test.go, bit for bit):
//
//   - FMAX (vector) returns a NaN when either operand is a NaN, and orders the
//     zeros: max(-0, +0) is +0 (FMIN: -0), so it is the Go rule for numbers and
//     for zeros, in either operand order. It quiets a signalling NaN and chooses
//     among two NaNs by its own rule, so it is NOT the Go rule for NaNs: that is
//     fixed up below. (FMAXNM / FMINNM return the number when one operand is a
//     quiet NaN, the wrong answer.) Go names them VFMAX and VFMIN.
//   - FCMEQ (vector, register) sets each lane to all ones when the operands are
//     equal and to zero otherwise, including when either is a NaN: FCMEQ of a
//     register with itself is the "not a NaN" mask. Go: VFCMEQ.
//   - BSL (bitwise select, byte arrangement): Vd = (Vd AND Vn) OR (NOT Vd AND Vm),
//     Vd being the mask. Go: VBSL Vm, Vn, Vd.
//
// Per element: r = FMAX(a, b); r = a not-NaN ? r : a; r = b not-NaN ? r : b. The
// last two selects copy a or b bit for bit, so a NaN keeps its sign, payload and
// quiet bit; a NaN b is applied last, so it wins over a NaN a.
//
// Go's arm64 assembler writes a three-operand vector form as `OP Vm, Vn, Vd`. Each
// kernel does n/8 blocks of eight elements (n a positive multiple of 8, sliced by
// the Go caller, which finishes the last seven elements); a block is loaded,
// computed and stored before the next, so dst may alias a or b exactly.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> maxmin_arm64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("arm64")
	f.Add(kernel("maximumNEON", "VFMAX").Func())
	f.Add(kernel("minimumNEON", "VFMIN").Func())
	if err := os.WriteFile("maxmin_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote maxmin_arm64.s")
}

// kernel: a in V0..V3, b in V4..V7, FMAX in V8..V11, the a mask in V12..V15, the
// result (the b mask, selected) in V16..V19.
func kernel(name, op string) *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Slice("b")}, nil)
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("dst_base", "R2").LoadArg("a_base", "R0").LoadArg("a_len", "R1").LoadArg("b_base", "R7").
		Raw("MOVD $0, R3").Label("loop").
		Raw("ADD R3<<3, R0, R5").
		Raw("ADD R3<<3, R7, R8").
		Raw("VLD1 (R5), [V0.D2, V1.D2, V2.D2, V3.D2]").
		Raw("VLD1 (R8), [V4.D2, V5.D2, V6.D2, V7.D2]")
	for k := 0; k < 4; k++ {
		a, bb, r, ma, mb := k, 4+k, 8+k, 12+k, 16+k
		b.Raw("%s V%d.D2, V%d.D2, V%d.D2", op, bb, a, r).
			Raw("VFCMEQ V%d.D2, V%d.D2, V%d.D2", a, a, ma).
			Raw("VFCMEQ V%d.D2, V%d.D2, V%d.D2", bb, bb, mb).
			Raw("VBSL V%d.B16, V%d.B16, V%d.B16", a, r, ma).  // ma = a not-NaN ? FMAX : a
			Raw("VBSL V%d.B16, V%d.B16, V%d.B16", bb, ma, mb) // mb = b not-NaN ? ma : b
	}
	b.Raw("ADD R3<<3, R2, R6").
		Raw("VST1 [V16.D2, V17.D2, V18.D2, V19.D2], (R6)").
		Raw("ADD $8, R3, R3").Raw("CMP R1, R3").Raw("BLT loop").Ret()
	return b
}
