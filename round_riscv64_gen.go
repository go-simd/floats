//go:build ignore

// Command gen produces round_riscv64.s with go-asmgen: the riscv64 RVV 1.0
// float64 elementwise Abs, Neg and Square kernels (go-ndarray/ndarray#118).
//
// Sources: RISC-V "V" Vector Extension 1.0, chapter 13, vector floating-point
// sign-injection (the spec's relevant section was not reached when this was
// written; semantics verified by execution under QEMU at VLEN 128, 256, 512):
//
//   - vfsgnjx.vv vd, vs2, vs1: the magnitude of vs2 with the sign bit
//     sign(vs2) XOR sign(vs1). With vs1 = vs2 the sign is always 0: Abs (Go:
//     VFSGNJXVV V0, V0, V16).
//   - vfsgnjn.vv: the magnitude of vs2 with the sign bit NOT sign(vs1). With
//     vs1 = vs2 it is the complement of the sign: Neg (Go: VFSGNJNVV).
//     Sign injection does not canonicalize a NaN and never raises a flag: a NaN
//     keeps its payload.
//   - vfmul.vv: one IEEE multiplication per element, no fusion (Square). Like
//     every RISC-V floating-point operation it returns the canonical NaN for a NaN
//     input.
//
// The elementwise loops are vector-length agnostic (VSETVLI with LMUL = 8 hands
// out as many elements as the hardware holds, including a short last strip), so
// there is no scalar tail. The V extension is optional, so the Go side calls
// these only when cpu.RISCV64.HasV.
//
// Floor, Ceil, Trunc and RoundEven have NO riscv64 kernel. RVV 1.0 has no vector
// round-to-integral instruction (there is no vfrint), and a float to int to float
// round trip (vfcvt.x.f.v, vfcvt.f.x.v) is wrong for |x| >= 2^63 (saturates), for
// NaN (becomes the largest integer) and for negative values that round to -0
// (the integer zero has no sign). Guarding every one of those cases exactly (pass
// |x| >= 2^52 and NaN through, restore the sign of a zero, fix the floor and ceil
// of a truncation with a compare) is possible but was not attempted: it could be
// tested only under QEMU, which says nothing about speed, so the portable loop
// stays (round_riscv64.go).
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> round_riscv64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
)

func main() {
	f := emit.NewFile("riscv64")
	for _, k := range []struct{ name, op string }{
		{"absRVV", "VFSGNJXVV V0, V0, V16"},
		{"negRVV", "VFSGNJNVV V0, V0, V16"},
		{"squareRVV", "VFMULVV V0, V0, V16"},
	} {
		f.Add(unaryKernel(k.name, k.op))
	}
	if err := os.WriteFile("round_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote round_riscv64.s")
}

// unaryKernel builds NAME(dst, src *float64, n int) for n >= 1: op reads V0 and
// writes V16. Each strip is loaded, computed and stored before the next, so dst
// may alias src exactly.
func unaryKernel(name, op string) *emit.Function {
	sig := riscv64.Layout([]string{"dst", "src", "n"},
		[]riscv64.Type{riscv64.Ptr, riscv64.Ptr, riscv64.Int64}, nil, nil)
	b := riscv64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "X10").LoadArg("src", "X11").LoadArg("n", "X13")
	b.Raw("loop:").
		Raw("VSETVLI X13, E64, M8, TA, MA, X14").
		Raw("VLE64V (X11), V0").
		Raw("%s", op).
		Raw("VSE64V V16, (X10)").
		Raw("SLLI $3, X14, X17").
		Raw("ADD X17, X10, X10").Raw("ADD X17, X11, X11").
		Raw("SUB X14, X13, X13").Raw("BNEZ X13, loop").Ret()
	return b.Func()
}
