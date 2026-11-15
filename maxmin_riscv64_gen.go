//go:build ignore

// Command gen produces maxmin_riscv64.s with go-asmgen: the riscv64 RVV 1.0
// float64 pairwise Maximum and Minimum kernels (go-ndarray/ndarray#98, #118).
//
// Rule (maxmin.go): a NaN b[i] wins, else a NaN a[i] wins, else the larger
// (smaller), and for two zeros the AND (OR) of the bits.
//
// Sources: RISC-V "V" Vector Extension 1.0 (v-spec.adoc, opened at
// raw.githubusercontent.com/riscv/riscv-v-spec/master/v-spec.adoc): "vmfeq and
// vmfne raise the invalid operation exception only on signaling NaN inputs", vmflt
// on any NaN; "vmfne writes 1 to the destination element when either operand is
// NaN" and the other comparisons write 0; vmerge.vvm: vd[i] = v0.mask[i] ? vs1[i]
// : vs2[i]; vmor.mm: vd.mask[i] = vs2.mask[i] || vs1.mask[i]. vfmax is not used: it
// returns the number when one operand is a NaN. Verified by execution under QEMU
// at VLEN 128, 256, 512 (maxmin_test.go).
//
//	V0 = (a < b)  [Minimum: b < a]
//	V1 = b is a NaN   (vmfne b, b)
//	V0 = V0 | V1                      takeB
//	V12 = takeB ? b : a
//	V0 = (a == b)
//	V16 = a & b   [Minimum: a | b]    (integer ops on the bits)
//	V12 = (a == b) ? V16 : V12
//
// Every selected element is a copy of a or b, so NaNs keep their bits. The loops
// are vector-length agnostic (LMUL = 4, the last strip short): no scalar tail. The V
// extension is optional, so the Go side calls these only when cpu.RISCV64.HasV. A
// strip is loaded, computed and stored before the next: dst may alias a or b
// exactly.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> maxmin_riscv64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
)

func main() {
	f := emit.NewFile("riscv64")
	f.Add(kernel("maximumRVV", "VMFLTVV V8, V4, V0", "VANDVV V8, V4, V16"))
	f.Add(kernel("minimumRVV", "VMFLTVV V4, V8, V0", "VORVV V8, V4, V16"))
	if err := os.WriteFile("maxmin_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote maxmin_riscv64.s")
}

// kernel builds NAME(dst, a, b *float64, n int) for n >= 1: a in V4 (group), b in
// V8, the result in V12.
func kernel(name, lt, zop string) *emit.Function {
	sig := riscv64.Layout([]string{"dst", "a", "b", "n"},
		[]riscv64.Type{riscv64.Ptr, riscv64.Ptr, riscv64.Ptr, riscv64.Int64}, nil, nil)
	b := riscv64.NewFunc(name, sig, 0)
	b.LoadArg("dst", "X10").LoadArg("a", "X11").LoadArg("b", "X12").LoadArg("n", "X13")
	b.Raw("loop:").
		Raw("VSETVLI X13, E64, M4, TA, MA, X14").
		Raw("VLE64V (X11), V4").
		Raw("VLE64V (X12), V8").
		Raw("%s", lt).
		Raw("VMFNEVV V8, V8, V1").
		Raw("VMORMM V1, V0, V0").
		Raw("VMERGEVVM V8, V4, V0, V12").
		Raw("VMFEQVV V8, V4, V0").
		Raw("%s", zop).
		Raw("VMERGEVVM V16, V12, V0, V12").
		Raw("VSE64V V12, (X10)").
		Raw("SLLI $3, X14, X17").
		Raw("ADD X17, X10, X10").Raw("ADD X17, X11, X11").Raw("ADD X17, X12, X12").
		Raw("SUB X14, X13, X13").Raw("BNEZ X13, loop").Ret()
	return b.Func()
}
