//go:build ignore

// Command gen produces allpass_arm64.s with go-asmgen: the NEON kernel of AllPass3
// (allpass.go). One row per 128-bit register (a complex128 is a lane pair), the same
// operations as allPass3Ref:
//
//	complex product d*a (a the coefficient): VEXT swaps d, VFMUL multiplies it by the
//	imaginary part of a duplicated with its even lane negated (exact), VFMLA fuses d times
//	the real part of a duplicated, so the even lane is FMA(dr, ar, -(di*ai)) and the odd
//	FMA(di, ar, dr*ai) as on AVX2 (NEON has no fused add-subtract);
//	c = x + cl*p: VFMLA; x = p - cl*c: VFMLS.
//
// R4 = byte offset of the row (16 per row), R5 = byte offset in the float64 tables (8 per
// row), R3 = rows left, V31 = (-0.0, 0). Every operand pointer is reloaded from its
// argument slot where it is used (R6): there are more pointers than registers. The caller
// guarantees n >= 1 and the lengths.
//
// Operand order: `VFMLA Vm, Vn, Vd` is Vd += Vn*Vm; `VFMLS Vm, Vn, Vd` is Vd -= Vn*Vm.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> allpass_arm64_gen.go
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
	if err := os.WriteFile("allpass_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote allpass_arm64.s")
}

func v(i int) string { return fmt.Sprintf("V%d", i) }

// addr leaves the address of the row of operand arg in R7: base R6 plus the byte offset off.
func addr(b *arm64.Builder, arg, off string) {
	b.LoadArg(arg, "R6").Raw("ADD %s, R6, R7", off)
}

func kernel() *arm64.Builder {
	args := []abi.Arg{abi.Scalar("n", abi.Int64)}
	for _, n := range []string{"out", "in", "phi", "q0", "q1", "q2", "c0", "c1", "c2", "old0", "new0", "old1", "new1", "old2", "new2"} {
		args = append(args, abi.Scalar(n, abi.Int64))
	}
	b := arm64.NewFunc("allPass3NEON", abi.LayoutArgs(args, nil), 0)
	load := func(arg string, dst int) {
		addr(b, arg, "R4")
		b.Raw("VLD1 (R7), [%s.D2]", v(dst))
	}
	// cmul leaves coefficient (register ci) times data (register di) in register t.
	cmul := func(di, ci, t, u, w int) {
		b.Raw("VEXT $8, %s.B16, %s.B16, %s.B16", v(di), v(di), v(t)).
			Raw("VZIP2 %s.D2, %s.D2, %s.D2", v(ci), v(ci), v(u)).
			Raw("VEOR V31.B16, %s.B16, %s.B16", v(u), v(u)).
			Raw("VFMUL %s.D2, %s.D2, %s.D2", v(u), v(t), v(t)).
			Raw("VZIP1 %s.D2, %s.D2, %s.D2", v(ci), v(ci), v(w)).
			Raw("VFMLA %s.D2, %s.D2, %s.D2", v(w), v(di), v(t))
	}
	b.LoadArg("n", "R3").Raw("MOVD $0, R4").Raw("MOVD $0, R5").
		Raw("MOVD $0x8000000000000000, R8").Raw("VEOR V31.B16, V31.B16, V31.B16").Raw("VMOV R8, V31.D[0]").
		Label("arow")
	load("in", 0)
	load("phi", 1)
	cmul(0, 1, 2, 3, 4) // x = phi*in in 2
	for _, names := range [3][4]string{{"old0", "q0", "c0", "new0"}, {"old1", "q1", "c1", "new1"}, {"old2", "q2", "c2", "new2"}} {
		load(names[0], 5)
		load(names[1], 6)
		cmul(5, 6, 7, 8, 9) // p = q*old in 7
		addr(b, names[2], "R5")
		b.Raw("VLD1R (R7), [V10.D2]").
			Raw("VMOV V2.B16, V11.B16").
			Raw("VFMLA V7.D2, V10.D2, V11.D2") // c = x + cl*p
		addr(b, names[3], "R4")
		b.Raw("VST1 [V11.D2], (R7)").
			Raw("VMOV V7.B16, V2.B16").
			Raw("VFMLS V11.D2, V10.D2, V2.D2") // x = p - cl*c
	}
	addr(b, "out", "R4")
	b.Raw("VST1 [V2.D2], (R7)").
		Raw("ADD $16, R4, R4").Raw("ADD $8, R5, R5").Raw("SUBS $1, R3, R3").Raw("BNE arow").Ret()
	return b
}
