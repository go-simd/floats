//go:build ignore

// Command gen produces allpass32_amd64.s with go-asmgen: the AVX2+FMA kernel of
// AllPass3Split32 (allpass32.go). Eight rows per step, one ymm register per plane (real
// and imaginary parts are separate float32 planes), every link of the chain in registers.
// The operations are the ones the portable loop names:
//
//	complex product d*a: or = dr*ar (VMULPS), or = or - di*ai (VFNMADD231PS),
//	                     oi = dr*ai (VMULPS), oi = oi + di*ar (VFMADD231PS);
//	c = x + cl*p: VFMADD231PS; x = p - cl*c: VFNMADD231PS.
//
// The rows left after the full blocks (1 to 7): when n >= 8, one more full block over the
// LAST 8 rows (n-8 .. n-1), recomputing up to 7 rows the previous block already wrote. That is
// harmless because the written operands (out, new links) never overlap what is read (the
// caller proves it), so a row's result does not depend on whether it ran once or twice, and it
// avoids VMASKMOVPS stores, which are slow on Zen 4 (measured: the masked tail doubled the
// time of a 34 row call). When n < 8, a block whose loads and stores are VMASKMOVPS under a
// mask of the first rows: a masked-off lane neither reads nor writes memory (no fault, nothing
// past n touched). The mask is read from the caller: R9 points at 8 int32, the first `rows`
// of them -1 and the rest 0.
//
// AX = byte offset of the block (4 per row, in every plane and in the coefficient tables),
// DX = rows left. Every operand pointer is reloaded from its argument slot where it is used
// (R8): there are more pointers than registers. The caller guarantees n >= 1, the lengths,
// and the gate is hasAVX2FMA. Go's amd64 assembler writes AVX forms as `OP src2, src1, dst`.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> allpass32_amd64_gen.go
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
	if err := os.WriteFile("allpass32_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote allpass32_amd64.s")
}

func y(i int) string { return fmt.Sprintf("Y%d", i) }

// block emits the operations of 8 rows; masked makes every load and store a VMASKMOVPS
// under the mask in Y15.
func block(b *amd64.Builder, masked bool) {
	load := func(arg string, dst int) {
		b.LoadArg(arg, "R8")
		if masked {
			b.Raw("VMASKMOVPS (R8)(AX*1), Y15, %s", y(dst))
		} else {
			b.Raw("VMOVUPS (R8)(AX*1), %s", y(dst))
		}
	}
	store := func(arg string, src int) {
		b.LoadArg(arg, "R8")
		if masked {
			b.Raw("VMASKMOVPS %s, Y15, (R8)(AX*1)", y(src))
		} else {
			b.Raw("VMOVUPS %s, (R8)(AX*1)", y(src))
		}
	}
	// cmul leaves the coefficient (ar, ai) times the data (dr, di) in (or, oi).
	cmul := func(dr, di, ar, ai, or, oi int) {
		b.Raw("VMULPS %s, %s, %s", y(ar), y(dr), y(or)).
			Raw("VFNMADD231PS %s, %s, %s", y(ai), y(di), y(or)).
			Raw("VMULPS %s, %s, %s", y(ai), y(dr), y(oi)).
			Raw("VFMADD231PS %s, %s, %s", y(ar), y(di), y(oi))
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
		b.Raw("VMOVAPS %s, %s", y(4), y(9)).Raw("VMOVAPS %s, %s", y(5), y(10)).
			Raw("VFMADD231PS %s, %s, %s", y(6), y(8), y(9)).
			Raw("VFMADD231PS %s, %s, %s", y(7), y(8), y(10)) // c = x + cl*p in (9, 10)
		store("new"+s+"re", 9)
		store("new"+s+"im", 10)
		b.Raw("VMOVAPS %s, %s", y(6), y(4)).Raw("VMOVAPS %s, %s", y(7), y(5)).
			Raw("VFNMADD231PS %s, %s, %s", y(9), y(8), y(4)).
			Raw("VFNMADD231PS %s, %s, %s", y(10), y(8), y(5)) // x = p - cl*c
	}
	store("outre", 4)
	store("outim", 5)
}

func kernel() *amd64.Builder {
	args := []abi.Arg{abi.Scalar("n", abi.Int64), abi.Scalar("mask", abi.Int64)}
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
	b := amd64.NewFunc("allPass3Split32AVX2", abi.LayoutArgs(args, nil), 0)
	b.LoadArg("n", "DX").LoadArg("mask", "R9").Raw("XORQ AX, AX").
		Label("apblk").Raw("CMPQ DX, $8").Raw("JLT aptail")
	block(b, false)
	b.Raw("ADDQ $32, AX").Raw("SUBQ $8, DX").Raw("JMP apblk").
		Label("aptail").Raw("TESTQ DX, DX").Raw("JEQ apdone").
		LoadArg("n", "CX").Raw("CMPQ CX, $8").Raw("JLT apmask").
		Raw("LEAQ -8(CX), AX").Raw("SHLQ $2, AX")
	block(b, false) // the last 8 rows again: the overlap is recomputed to the same values
	b.Raw("JMP apdone").Label("apmask").Raw("VMOVDQU (R9), Y15")
	block(b, true)
	b.Label("apdone").Raw("VZEROUPPER").Ret()
	return b
}
