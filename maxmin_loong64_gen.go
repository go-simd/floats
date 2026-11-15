//go:build ignore

// Command gen produces maxmin_loong64.s with go-asmgen: the loong64 LASX float64
// pairwise Maximum and Minimum kernels (go-ndarray/ndarray#98, #118).
//
// Rule (maxmin.go): a NaN b[i] wins, else a NaN a[i] wins, else the larger
// (smaller), and for two zeros the AND (OR) of the bits.
//
// Sources: LoongArch Reference Manual, Volume 1. The vector and floating-point
// compare sections were NOT reached (the fetched page is 581 KB and only its first
// 200 KB were read); a third-party intrinsics guide gives XVBITSEL.V as "b where
// the mask bit is 1, otherwise a". The encodings below are from the manual as
// known and were verified by execution under QEMU la464 (maxmin_test.go, bit for
// bit). Go's assembler has neither instruction, so each is a WORD:
//
//   - XVFCMP.cond.D xd, xj, xk (opcode 0x0ca00000, cond in bits 19:15, xk 14:10,
//     xj 9:5, xd 4:0): per doubleword all ones when the condition holds, zero
//     otherwise. cond 0x2 = CLT (ordered less than: false for a NaN), 0x4 = CEQ
//     (false for a NaN), 0x8 = CUN (unordered: true for a NaN).
//
//   - XVBITSEL.V xd, xj, xk, xa (opcode 0x0d200000, xa in bits 19:15): xd takes
//     xk where the bit of xa is 1 and xj where it is 0.
//
//   - XVAND.V / XVOR.V (Go: XVANDV, XVORV).
//
//     t = a < b  [Minimum: b < a] ; t = t OR (b is a NaN)      takeB
//     r = takeB ? b : a
//     r = (a == b) ? (a AND b) : r      [Minimum: a OR b]      two zeros
//
// Every selected element is a copy of a or b, so NaNs keep their bits. LASX is not
// on every LoongArch CPU: the Go side calls these only when cpu.Loong64.HasLASX.
// Each kernel does n/8 blocks of eight elements (n a positive multiple of 8; the Go
// caller finishes the last seven); a block is computed before it is stored, so dst
// may alias a or b exactly.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> maxmin_loong64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/loong64"
)

func main() {
	f := emit.NewFile("loong64")
	f.Add(kernel("maximumLASX", true))
	f.Add(kernel("minimumLASX", false))
	if err := os.WriteFile("maxmin_loong64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote maxmin_loong64.s")
}

func fcmp(cond uint32, name string, d, j, k int) string {
	return fmt.Sprintf("WORD $0x%08x // xvfcmp.%s.d X%d, X%d, X%d", 0x0ca00000|cond<<15|uint32(k)<<10|uint32(j)<<5|uint32(d), name, d, j, k)
}

func bitsel(d, j, k, a int) string {
	return fmt.Sprintf("WORD $0x%08x // xvbitsel.v X%d, X%d, X%d, X%d", 0x0d200000|uint32(a)<<15|uint32(k)<<10|uint32(j)<<5|uint32(d), d, j, k, a)
}

// kernel: per vector of a block (two 4-double vectors), a in X0/X1, b in X2/X3,
// t in X4/X5, u in X6/X7, the result in X4/X5.
func kernel(name string, isMax bool) *emit.Function {
	sig := loong64.Layout([]string{"dst", "a", "b", "n"},
		[]loong64.Type{loong64.Ptr, loong64.Ptr, loong64.Ptr, loong64.Int64}, nil, nil)
	bl := loong64.NewFunc(name, sig, 0)
	bl.LoadArg("dst", "R4").LoadArg("a", "R6").LoadArg("b", "R7").LoadArg("n", "R5")
	bl.Label("loop").Raw("XVMOVQ 0(R6), X0").Raw("XVMOVQ 32(R6), X1").
		Raw("XVMOVQ 0(R7), X2").Raw("XVMOVQ 32(R7), X3")
	for k := 0; k < 2; k++ {
		a, b, t, u := k, 2+k, 4+k, 6+k
		zop := "XVANDV"
		if isMax {
			bl.Raw("%s", fcmp(0x2, "clt", t, a, b)) // t = a < b
		} else {
			zop = "XVORV"
			bl.Raw("%s", fcmp(0x2, "clt", t, b, a)) // t = b < a
		}
		bl.Raw("%s", fcmp(0x8, "cun", u, b, b)). // u = b is a NaN
								Raw("XVORV X%d, X%d, X%d", u, t, t).   // t = takeB
								Raw("%s", bitsel(t, a, b, t)).         // t = takeB ? b : a
								Raw("%s", fcmp(0x4, "ceq", u, a, b)).  // u = a == b
								Raw("%s X%d, X%d, X%d", zop, b, a, a). // a = a op b
								Raw("%s", bitsel(t, t, a, u))          // t = u ? a : t
	}
	bl.Raw("XVMOVQ X4, 0(R4)").Raw("XVMOVQ X5, 32(R4)").
		Raw("ADDV $64, R4").Raw("ADDV $64, R6").Raw("ADDV $64, R7").Raw("ADDV $-8, R5").Raw("BNE R5, loop").Ret()
	return bl.Func()
}
