//go:build ignore

// Command gen produces extrema32_riscv64.s with go-asmgen: the RVV float32 Max
// and Min kernels max32RVV/min32RVV (go-ndarray's, moved here so the kernel
// lives once), for n >= 1, the NaN-propagating maximum (or minimum):
//
//   - The accumulator group V16 (LMUL=4) is seeded with a[0] in every lane.
//   - Each strip of vl elements is folded in with vfmax.vv/vfmin.vv under the
//     tail-undisturbed policy, so a short last strip leaves the other lanes alone.
//   - At the end the group is reduced with vfredmax.vs/vfredmin.vs.
//
// RISC-V's fmax/fmin order -0 below +0, as Go does, but return the non-NaN
// operand when one is NaN. So each strip also counts its NaN lanes (vmfne.vv
// x, x, then vcpop.m), and a non-zero count makes the result a quiet NaN. The V
// extension is optional: the Go side calls the kernels only when hasRVV.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> extrema32_riscv64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
)

func main() {
	f := emit.NewFile("riscv64")
	f.Add(extremum32Kernel("max32RVV", "VFMAXVV", "VFREDMAXVS"))
	f.Add(extremum32Kernel("min32RVV", "VFMINVV", "VFREDMINVS"))
	if err := os.WriteFile("extrema32_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote extrema32_riscv64.s")
}

func extremum32Kernel(name, vop, red string) *emit.Function {
	sig := riscv64.Layout([]string{"a", "n"}, []riscv64.Type{riscv64.Ptr, riscv64.Int64},
		[]string{"ret"}, []riscv64.Type{riscv64.Float32})
	b := riscv64.NewFunc(name, sig, 0)
	b.LoadArg("a", "X10").LoadArg("n", "X11")
	b.Raw("MOVF (X10), F0").
		Raw("VSETVLI X0, E32, M4, TA, MA, X16"). // vl = VLMAX
		Raw("VFMVVF F0, V16").
		Raw("MOV $0, X17"). // NaN lanes seen
		Raw("loop:").
		Raw("VSETVLI X11, E32, M4, TU, MA, X14").
		Raw("VLE32V (X10), V8").
		Raw("VMFNEVV V8, V8, V4"). // V4 = the NaN lanes of the strip
		Raw("VCPOPM V4, X15").
		Raw("ADD X15, X17, X17").
		Raw("%s V8, V16, V16", vop).
		Raw("SLLI $2, X14, X18").
		Raw("ADD X18, X10, X10").
		Raw("SUB X14, X11, X11").
		Raw("BNEZ X11, loop").
		Raw("VSETVLI X0, E32, M4, TA, MA, X16").
		Raw("VFMVSF F0, V1").
		Raw("%s V1, V16, V1", red). // V1[0] = extremum of V1[0] and every lane of V16
		Raw("VFMVFS V1, F0").
		Raw("BEQZ X17, done").
		Raw("MOV $0x7fc00000, X5").
		Raw("FMVWX X5, F0").
		Raw("done:").
		StoreRet("F0", "ret").
		Ret()
	return b.Func()
}
