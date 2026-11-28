//go:build ignore

// Command gen produces allpass32_arm64.s with go-asmgen: the NEON kernel of
// AllPass3Split32 (allpass32.go). Four rows per step, one 128-bit register per plane (real
// and imaginary parts are separate float32 planes), every link of the chain in registers,
// the same operations as the AVX2 kernel:
//
//	complex product d*a: or = dr*ar (VFMUL), or -= di*ai (VFMLS),
//	                     oi = dr*ai (VFMUL), oi += di*ar (VFMLA);
//	c = x + cl*p: VFMLA; x = p - cl*c: VFMLS.
//
// The 1 to 3 rows left after the full blocks run one at a time through the same
// instructions with a single-lane load and store (VLD1 / VST1 of lane 0), so nothing past n
// is read or written.
//
// R3 = rows left, R4 = byte offset of the block (4 per row, in every plane and in the
// coefficient tables). Every operand pointer is reloaded from its argument slot where it is
// used (R6): there are more pointers than registers. The caller guarantees n >= 1 and the
// lengths. Operand order: `VFMLA Vm, Vn, Vd` is Vd += Vn*Vm; `VFMLS Vm, Vn, Vd` is Vd -= Vn*Vm.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> allpass32_arm64_gen.go
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
	if err := os.WriteFile("allpass32_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote allpass32_arm64.s")
}

func v(i int) string { return fmt.Sprintf("V%d.S4", i) }

// block emits the operations of 4 rows (lane = false) or of the lane 0 row (lane = true).
func block(b *arm64.Builder, lane bool) {
	addr := func(arg string) {
		b.LoadArg(arg, "R6").Raw("ADD R4, R6, R7")
	}
	load := func(arg string, dst int) {
		addr(arg)
		if lane {
			b.Raw("VLD1 (R7), V%d.S[0]", dst)
		} else {
			b.Raw("VLD1 (R7), [%s]", v(dst))
		}
	}
	store := func(arg string, src int) {
		addr(arg)
		if lane {
			b.Raw("VST1 V%d.S[0], (R7)", src)
		} else {
			b.Raw("VST1 [%s], (R7)", v(src))
		}
	}
	// cmul leaves the coefficient (ar, ai) times the data (dr, di) in (or, oi).
	cmul := func(dr, di, ar, ai, or, oi int) {
		b.Raw("VFMUL %s, %s, %s", v(ar), v(dr), v(or)).
			Raw("VFMLS %s, %s, %s", v(ai), v(di), v(or)).
			Raw("VFMUL %s, %s, %s", v(ai), v(dr), v(oi)).
			Raw("VFMLA %s, %s, %s", v(ar), v(di), v(oi))
	}
	load("inre", 0)
	load("inim", 1)
	load("phire", 2)
	load("phiim", 3)
	cmul(0, 1, 2, 3, 4, 5) // x = phi*in in (4, 5)
	for l := 0; l < 3; l++ {
		s := fmt.Sprint(l)
		load("old"+s+"re", 0)
		load("old"+s+"im", 1)
		load("q"+s+"re", 2)
		load("q"+s+"im", 3)
		cmul(0, 1, 2, 3, 6, 7) // p = q*old in (6, 7)
		load("c"+s, 8)
		b.Raw("VMOV V4.B16, V9.B16").Raw("VMOV V5.B16, V10.B16").
			Raw("VFMLA %s, %s, %s", v(6), v(8), v(9)).
			Raw("VFMLA %s, %s, %s", v(7), v(8), v(10)) // c = x + cl*p in (9, 10)
		store("new"+s+"re", 9)
		store("new"+s+"im", 10)
		b.Raw("VMOV V6.B16, V4.B16").Raw("VMOV V7.B16, V5.B16").
			Raw("VFMLS %s, %s, %s", v(9), v(8), v(4)).
			Raw("VFMLS %s, %s, %s", v(10), v(8), v(5)) // x = p - cl*c
	}
	store("outre", 4)
	store("outim", 5)
}

func kernel() *arm64.Builder {
	args := []abi.Arg{abi.Scalar("n", abi.Int64)}
	names := []string{"outre", "outim", "inre", "inim", "phire", "phiim"}
	for l := 0; l < 3; l++ {
		names = append(names, fmt.Sprintf("q%dre", l), fmt.Sprintf("q%dim", l))
	}
	names = append(names, "c0", "c1", "c2")
	for l := 0; l < 3; l++ {
		names = append(names, fmt.Sprintf("old%dre", l), fmt.Sprintf("old%dim", l), fmt.Sprintf("new%dre", l), fmt.Sprintf("new%dim", l))
	}
	for _, n := range names {
		args = append(args, abi.Scalar(n, abi.Int64))
	}
	b := arm64.NewFunc("allPass3Split32NEON", abi.LayoutArgs(args, nil), 0)
	b.LoadArg("n", "R3").Raw("MOVD $0, R4").
		Label("apblk").Raw("CMP $4, R3").Raw("BLT aptail")
	block(b, false)
	b.Raw("ADD $16, R4, R4").Raw("SUB $4, R3, R3").Raw("B apblk").
		Label("aptail").Raw("CBZ R3, apdone")
	block(b, true)
	b.Raw("ADD $4, R4, R4").Raw("SUB $1, R3, R3").Raw("B aptail").
		Label("apdone").Ret()
	return b
}
