//go:build ignore

// Command gen produces maxabsmul_arm64.s with go-asmgen: the NEON kernel of
// MaxAbsMul (maxabsmul.go), max_i |v[i]|*w[i] with the semantics of C's fmax
// folded from +0 (a NaN product loses).
//
// Per element: VFABS, VFMUL (one rounding, like the scalar product), then the NaN
// product is replaced by +0 with a compare-and-mask (VFCMGT against a broadcast
// -Inf is false for a NaN and for -Inf, both of which cannot win against the
// non-negative running maximum anyway), and VFMAX folds it in. No NaN ever reaches
// VFMAX, so its NaN propagation is never exercised. The kernel returns the value
// of the maximum; its zero has no meaningful sign (the Go wrapper redoes an
// all-zero result with the scalar loop, which carries the signed-zero rule).
//
// Registers: R0 = v, R7 = w, R1 = n, R3 = i; V29 = broadcast -Inf; V0..V3
// accumulators (zero); V4.. v then the product, V8.. w, V16.. masks. The
// 1-element stage loads D1 (the upper half zero, |0|*0 = 0 neutral).
//
// Operand order: `VFCMGT Vm, Vn, Vd` is Vd = Vn > Vm; `VAND Vm, Vn, Vd` is Vd = Vn & Vm.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> maxabsmul_arm64_gen.go
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
	f.Add(kernel().Func())
	if err := os.WriteFile("maxabsmul_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote maxabsmul_arm64.s")
}

func vregs(first, n int, arr string) string {
	s := "["
	for k := 0; k < n; k++ {
		if k > 0 {
			s += ", "
		}
		s += fmt.Sprintf("V%d.%s", first+k, arr)
	}
	return s + "]"
}

func kernel() *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("v"), abi.Slice("w")},
		[]abi.Arg{abi.Scalar("ret", abi.Float64)})
	b := arm64.NewFunc("maxAbsMulNEON", sig, 0)
	b.LoadArg("v_base", "R0").LoadArg("v_len", "R1").LoadArg("w_base", "R7").
		Raw("VEOR V0.B16, V0.B16, V0.B16").Raw("VEOR V1.B16, V1.B16, V1.B16").
		Raw("VEOR V2.B16, V2.B16, V2.B16").Raw("VEOR V3.B16, V3.B16, V3.B16").
		Raw("MOVD $0xfff0000000000000, R9").Raw("VDUP R9, V29.D2").
		Raw("MOVD $0, R3")
	stages := []struct {
		label, next string
		elems, regs int
		arr         string
	}{
		{"uloop", "vcomb", 8, 4, "D2"},
		{"vloop", "sloop", 2, 1, "D2"},
		{"sloop", "done", 1, 1, "D1"},
	}
	for _, s := range stages {
		b.Label(s.label).
			Raw(fmt.Sprintf("ADD $%d, R3, R4", s.elems)).Raw("CMP R1, R4").Raw("BGT " + s.next).
			Raw("ADD R3<<3, R0, R5").Raw(fmt.Sprintf("VLD1 (R5), %s", vregs(4, s.regs, s.arr))).
			Raw("ADD R3<<3, R7, R8").Raw(fmt.Sprintf("VLD1 (R8), %s", vregs(8, s.regs, s.arr)))
		for k := 0; k < s.regs; k++ {
			p, m := 4+k, 16+k
			b.Raw(fmt.Sprintf("VFABS V%d.D2, V%d.D2", p, p)).
				Raw(fmt.Sprintf("VFMUL V%d.D2, V%d.D2, V%d.D2", 8+k, p, p)).
				Raw(fmt.Sprintf("VFCMGT V29.D2, V%d.D2, V%d.D2", p, m)).
				Raw(fmt.Sprintf("VAND V%d.B16, V%d.B16, V%d.B16", m, p, p))
			if s.label == "uloop" {
				b.Raw(fmt.Sprintf("VFMAX V%d.D2, V%d.D2, V%d.D2", p, k, k))
			} else {
				b.Raw(fmt.Sprintf("VFMAX V%d.D2, V0.D2, V0.D2", p))
			}
		}
		b.Raw(fmt.Sprintf("ADD $%d, R3, R3", s.elems)).Raw("B " + s.label)
		if s.label == "uloop" {
			b.Label("vcomb").
				Raw("VFMAX V1.D2, V0.D2, V0.D2").Raw("VFMAX V3.D2, V2.D2, V2.D2").
				Raw("VFMAX V2.D2, V0.D2, V0.D2").Raw("B vloop")
		}
	}
	b.Label("done").
		Raw("VMOV V0.D[0], R9").Raw("VMOV V0.D[1], R10").
		Raw("FMOVD R9, F1").Raw("FMOVD R10, F2").
		Raw("FMAXD F2, F1, F0").
		StoreRet("F0", "ret").Ret()
	return b
}
