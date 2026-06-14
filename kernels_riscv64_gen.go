//go:build ignore

// Command gen produces kernels_riscv64.s with go-asmgen: RVV (vector extension)
// reduction kernels for the float32/float64 dot product, sum and squared-diffs.
//
// RVV is length-agnostic: VSETVLI sets the active vector length for the current
// element width each iteration, so the kernels need no separate scalar tail —
// the final short chunk is handled by the same loop with a smaller VL. Each
// chunk multiplies into a vector (VFMULVV: vd = vs1*vs2) and then folds it into
// the running scalar with an ordered reduction (VFREDOSUMVS seed, vec, vd →
// vd[0] = seed[0] + Σ vec — the summed vector is the SECOND operand, the scalar
// seed the first), carrying the running sum in F0 (VFMVSF/VFMVFS move
// scalar↔vector lane 0). Ordered reduction keeps the result deterministic.
//
// The V extension is detected at runtime (the dispatcher gates on
// cpu.RISCV64.HasV); without it the lane-blocked scalar reference is used.
//
// Run: go run kernels_riscv64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/riscv64"
)

func dotSig(elem abi.Type) abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("a"), abi.Slice("b")},
		[]abi.Arg{abi.Scalar("ret", elem)},
	)
}

func sumSig(elem abi.Type) abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Slice("a")},
		[]abi.Arg{abi.Scalar("ret", elem)},
	)
}

func main() {
	f := emit.NewFile("riscv64")
	f.Add(dotF64().Func())
	f.Add(sumF64().Func())
	f.Add(ssdF64().Func())
	f.Add(dotF32().Func())
	f.Add(sumF32().Func())
	f.Add(ssdF32().Func())
	if err := os.WriteFile("kernels_riscv64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_riscv64.s")
}

// Register plan (all kernels):
//   X10 a_base, X11 a_len (the remaining count, decremented by VL each step),
//   X12 b_base                          (loaded by LoadArg)
//   X14 current VL (VSETVLI output), X15 byte-stride scratch
//   V1  a chunk, V2 b chunk, V4 product/diff, V6 reduction target
//   V8  scalar-carry vector (lane0 = running sum)
//   F0  running sum (returned)
//
// VSETVLI's Plan 9 form is `VSETVLI Rs1(AVL-in), vtype…, Rd(vl-out)` — the
// requested length is the FIRST operand and the granted VL the LAST. The float
// reduction VFREDOSUMVS takes (seedVec, sumVec, dstVec): vd[0] = seedVec[0] + Σ
// sumVec.

// dotF64: F0 += Σ a*b, chunked.
func dotF64() *riscv64.Builder {
	b := riscv64.NewFunc("dotRVV", dotSig(abi.Float64), 0)
	b.LoadArg("a_base", "X10").LoadArg("a_len", "X11").LoadArg("b_base", "X12").
		Raw("FMVDX X0, F0"). // running sum = 0
		Label("loop").
		Raw("BEQZ X11, done").
		Raw("VSETVLI X11, E64, M1, TA, MA, X14"). // X14 = vl <= X11
		Raw("VLE64V (X10), V1").
		Raw("VLE64V (X12), V2").
		Raw("VFMULVV V1, V2, V4").     // V4 = a*b (elementwise)
		Raw("VFMVSF F0, V8").          // V8[0] = running sum
		Raw("VFREDOSUMVS V8, V4, V6"). // V6[0] = V8[0] + Σ V4
		Raw("VFMVFS V6, F0").          // running sum = V6[0]
		Raw("SLLI $3, X14, X15").      // bytes = vl*8
		Raw("ADD X15, X10, X10").
		Raw("ADD X15, X12, X12").
		Raw("SUB X14, X11, X11").
		Raw("JMP loop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func sumF64() *riscv64.Builder {
	b := riscv64.NewFunc("sumRVV", sumSig(abi.Float64), 0)
	b.LoadArg("a_base", "X10").LoadArg("a_len", "X11").
		Raw("FMVDX X0, F0").
		Label("loop").
		Raw("BEQZ X11, done").
		Raw("VSETVLI X11, E64, M1, TA, MA, X14").
		Raw("VLE64V (X10), V1").
		Raw("VFMVSF F0, V8").
		Raw("VFREDOSUMVS V8, V1, V6").
		Raw("VFMVFS V6, F0").
		Raw("SLLI $3, X14, X15").
		Raw("ADD X15, X10, X10").
		Raw("SUB X14, X11, X11").
		Raw("JMP loop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func ssdF64() *riscv64.Builder {
	b := riscv64.NewFunc("sumSqDiffRVV", dotSig(abi.Float64), 0)
	b.LoadArg("a_base", "X10").LoadArg("a_len", "X11").LoadArg("b_base", "X12").
		Raw("FMVDX X0, F0").
		Label("loop").
		Raw("BEQZ X11, done").
		Raw("VSETVLI X11, E64, M1, TA, MA, X14").
		Raw("VLE64V (X10), V1").
		Raw("VLE64V (X12), V2").
		Raw("VFSUBVV V2, V1, V4"). // V4 = a-b
		Raw("VFMULVV V4, V4, V4"). // V4 = (a-b)^2
		Raw("VFMVSF F0, V8").
		Raw("VFREDOSUMVS V8, V4, V6").
		Raw("VFMVFS V6, F0").
		Raw("SLLI $3, X14, X15").
		Raw("ADD X15, X10, X10").
		Raw("ADD X15, X12, X12").
		Raw("SUB X14, X11, X11").
		Raw("JMP loop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

// ---- float32 (E32) ----

func dotF32() *riscv64.Builder {
	b := riscv64.NewFunc("dot32RVV", dotSig(abi.Float32), 0)
	b.LoadArg("a_base", "X10").LoadArg("a_len", "X11").LoadArg("b_base", "X12").
		Raw("FMVWX X0, F0").
		Label("loop").
		Raw("BEQZ X11, done").
		Raw("VSETVLI X11, E32, M1, TA, MA, X14").
		Raw("VLE32V (X10), V1").
		Raw("VLE32V (X12), V2").
		Raw("VFMULVV V1, V2, V4").
		Raw("VFMVSF F0, V8").
		Raw("VFREDOSUMVS V8, V4, V6").
		Raw("VFMVFS V6, F0").
		Raw("SLLI $2, X14, X15"). // bytes = vl*4
		Raw("ADD X15, X10, X10").
		Raw("ADD X15, X12, X12").
		Raw("SUB X14, X11, X11").
		Raw("JMP loop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func sumF32() *riscv64.Builder {
	b := riscv64.NewFunc("sum32RVV", sumSig(abi.Float32), 0)
	b.LoadArg("a_base", "X10").LoadArg("a_len", "X11").
		Raw("FMVWX X0, F0").
		Label("loop").
		Raw("BEQZ X11, done").
		Raw("VSETVLI X11, E32, M1, TA, MA, X14").
		Raw("VLE32V (X10), V1").
		Raw("VFMVSF F0, V8").
		Raw("VFREDOSUMVS V8, V1, V6").
		Raw("VFMVFS V6, F0").
		Raw("SLLI $2, X14, X15").
		Raw("ADD X15, X10, X10").
		Raw("SUB X14, X11, X11").
		Raw("JMP loop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}

func ssdF32() *riscv64.Builder {
	b := riscv64.NewFunc("sumSqDiff32RVV", dotSig(abi.Float32), 0)
	b.LoadArg("a_base", "X10").LoadArg("a_len", "X11").LoadArg("b_base", "X12").
		Raw("FMVWX X0, F0").
		Label("loop").
		Raw("BEQZ X11, done").
		Raw("VSETVLI X11, E32, M1, TA, MA, X14").
		Raw("VLE32V (X10), V1").
		Raw("VLE32V (X12), V2").
		Raw("VFSUBVV V2, V1, V4").
		Raw("VFMULVV V4, V4, V4").
		Raw("VFMVSF F0, V8").
		Raw("VFREDOSUMVS V8, V4, V6").
		Raw("VFMVFS V6, F0").
		Raw("SLLI $2, X14, X15").
		Raw("ADD X15, X10, X10").
		Raw("ADD X15, X12, X12").
		Raw("SUB X14, X11, X11").
		Raw("JMP loop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}
