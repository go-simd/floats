//go:build ignore

// Command gen produces round_loong64.s with go-asmgen: the loong64 LASX float64
// elementwise Abs, Neg, Square, Floor, Ceil, Trunc and RoundEven kernels
// (go-ndarray/ndarray#118).
//
// Sources: LoongArch Reference Manual, Volume 1, LASX (256-bit vector)
// floating-point and logic instructions (not opened when this was written; the
// mnemonics are those Go's assembler testdata encodes, the semantics verified by
// execution under QEMU la464, round_test.go):
//
//   - XVFRINTRM.D, XVFRINTRP.D, XVFRINTRZ.D, XVFRINTRNE.D: round each double to
//     an integral value with the rounding the name gives (toward -Inf, +Inf, zero,
//     nearest even); Go names them XVFRINTRMD, XVFRINTRPD, XVFRINTRZD, XVFRINTRNED.
//     They use the named mode, not FCSR.RM, so they are Floor, Ceil, Trunc and
//     RoundEven.
//   - XVMULD: one IEEE multiplication per lane (Square), no fusion.
//   - XVAND.V and XVXOR.V (Go: XVANDV, XVXORV): bitwise, on 256 bits. The masks are
//     built in a register, no constant is loaded from memory: XVSEQV of a register
//     with itself is all ones, XVSRLV $1 of it is 0x7FFF...F (the Abs mask),
//     XVSLLV $63 is 0x8000...0 (the Neg mask). Bit operations only: a NaN keeps its
//     payload.
//
// LASX is not on every LoongArch CPU: the Go side calls these only when
// cpu.Loong64.HasLASX (the other kernels of lanes_loong64.go).
//
// Each kernel does n/8 blocks of eight elements (n a positive multiple of 8; the
// Go caller finishes the last seven); a block is loaded, computed and stored
// before the next, so dst may alias src exactly.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> round_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
)

func main() {
	f := emit.NewFile("loong64")
	for _, k := range []struct{ name, setup, op string }{
		{"absLASX", "XVSEQV X8, X8, X8\n\tXVSRLV $1, X8, X8", "XVANDV X8, X%d, X%d"},
		{"negLASX", "XVSEQV X8, X8, X8\n\tXVSLLV $63, X8, X8", "XVXORV X8, X%d, X%d"},
		{"squareLASX", "", "XVMULD X%d, X%d, X%d"},
		{"floorLASX", "", "XVFRINTRMD X%d, X%d"},
		{"ceilLASX", "", "XVFRINTRPD X%d, X%d"},
		{"truncLASX", "", "XVFRINTRZD X%d, X%d"},
		{"rintLASX", "", "XVFRINTRNED X%d, X%d"},
	} {
		f.Add(kernel(k.name, k.setup, k.op))
	}
	if err := os.WriteFile("round_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote round_loong64.s")
}

func verbs(s string) int {
	n := 0
	for i := 0; i+1 < len(s); i++ {
		if s[i] == '%' && s[i+1] == 'd' {
			n++
		}
	}
	return n
}

// kernel builds name(dst, src *float64, n int): X0, X1 hold the source and, after
// the operation, the result; X8 the mask.
func kernel(name, setup, op string) *emit.Function {
	sig := loong64.Layout([]string{"dst", "src", "n"}, []loong64.Type{loong64.Ptr, loong64.Ptr, loong64.Int64}, nil, nil)
	b := loong64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "R4").LoadArg("src", "R6").LoadArg("n", "R5")
	if setup != "" {
		b.Raw("%s", setup)
	}
	b.Label("loop").Raw("XVMOVQ 0(R6), X0").Raw("XVMOVQ 32(R6), X1")
	for k := 0; k < 2; k++ {
		if verbs(op) == 3 {
			b.Raw(op, k, k, k)
		} else {
			b.Raw(op, k, k)
		}
	}
	b.Raw("XVMOVQ X0, 0(R4)").Raw("XVMOVQ X1, 32(R4)").
		Raw("ADDV $64, R4").Raw("ADDV $64, R6").Raw("ADDV $-8, R5").Raw("BNE R5, loop").Ret()
	return b.Func()
}
