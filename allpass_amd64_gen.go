//go:build ignore

// Command gen produces allpass_amd64.s with go-asmgen: the AVX2+FMA kernel of AllPass3
// (allpass.go). Two rows per ymm register (a complex128 is a lane pair), then at most one
// row in an xmm register, each the same operations as allPass3Ref:
//
//	complex product d*a (a the coefficient): VPERMILPD swaps d, VMULPD multiplies it by
//	the imaginary part of a duplicated, VFMADDSUB231PD fuses d times the real part of a
//	duplicated (subtracting in the even lane, adding in the odd);
//	c = x + cl*p: VFMADD231PD; x = p - cl*c: VFNMADD231PD.
//
// AX = byte offset of the row (16 per row), BX = byte offset in the float64 tables (8 per
// row), DX = rows left. Every operand pointer is reloaded from its argument slot where it
// is used (R8): there are more pointers than registers. The caller guarantees n >= 1, the
// lengths, and the gate is hasAVX2FMA. Go's amd64 assembler writes AVX forms as
// `OP src2, src1, dst`.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> allpass_amd64_gen.go
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
	f.Add(kernel().Func())
	if err := os.WriteFile("allpass_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote allpass_amd64.s")
}

// row emits the operations of one block: w = 2 rows in Y registers, w = 1 row in X.
func row(b *amd64.Builder, w int) {
	r, swap, hi := "Y", 5, 15
	if w == 1 {
		r, swap, hi = "X", 1, 3
	}
	reg := func(i int) string { return fmt.Sprintf("%s%d", r, i) }
	load := func(arg string, dst int) {
		b.LoadArg(arg, "R8").Raw("VMOVUPD (R8)(AX*1), %s", reg(dst))
	}
	// cmul leaves coefficient (loaded in register ci) times data (register di) in register t.
	cmul := func(di, ci, t, u, v int) {
		b.Raw("VPERMILPD $%d, %s, %s", swap, reg(di), reg(t)).
			Raw("VPERMILPD $%d, %s, %s", hi, reg(ci), reg(u)).
			Raw("VMULPD %s, %s, %s", reg(u), reg(t), reg(t)).
			Raw("VPERMILPD $0, %s, %s", reg(ci), reg(v)).
			Raw("VFMADDSUB231PD %s, %s, %s", reg(v), reg(di), reg(t))
	}
	load("in", 0)
	load("phi", 1)
	cmul(0, 1, 2, 3, 4) // x = phi*in in 2
	for _, names := range [3][4]string{{"old0", "q0", "c0", "new0"}, {"old1", "q1", "c1", "new1"}, {"old2", "q2", "c2", "new2"}} {
		load(names[0], 5)
		load(names[1], 6)
		cmul(5, 6, 7, 8, 9) // p = q*old in 7
		b.LoadArg(names[2], "R8")
		if w == 2 {
			b.Raw("VMOVUPD (R8)(BX*1), X10").Raw("VPERMPD $0x50, Y10, Y10") // (c0 c0 c1 c1)
		} else {
			b.Raw("VMOVDDUP (R8)(BX*1), X10")
		}
		b.Raw("VMOVAPD %s, %s", reg(2), reg(11)).
			Raw("VFMADD231PD %s, %s, %s", reg(7), reg(10), reg(11)) // c = x + cl*p
		b.LoadArg(names[3], "R8").Raw("VMOVUPD %s, (R8)(AX*1)", reg(11))
		b.Raw("VMOVAPD %s, %s", reg(7), reg(2)).
			Raw("VFNMADD231PD %s, %s, %s", reg(11), reg(10), reg(2)) // x = p - cl*c
	}
	b.LoadArg("out", "R8").Raw("VMOVUPD %s, (R8)(AX*1)", reg(2))
}

func kernel() *amd64.Builder {
	args := []abi.Arg{abi.Scalar("n", abi.Int64)}
	for _, n := range []string{"out", "in", "phi", "q0", "q1", "q2", "c0", "c1", "c2", "old0", "new0", "old1", "new1", "old2", "new2"} {
		args = append(args, abi.Scalar(n, abi.Int64))
	}
	b := amd64.NewFunc("allPass3AVX2", abi.LayoutArgs(args, nil), 0)
	b.LoadArg("n", "DX").Raw("XORQ AX, AX").Raw("XORQ BX, BX").
		Label("apair").Raw("CMPQ DX, $2").Raw("JLT atail")
	row(b, 2)
	b.Raw("ADDQ $32, AX").Raw("ADDQ $16, BX").Raw("SUBQ $2, DX").Raw("JMP apair").
		Label("atail").Raw("TESTQ DX, DX").Raw("JEQ adone")
	row(b, 1)
	b.Label("adone").Raw("VZEROUPPER").Ret()
	return b
}
