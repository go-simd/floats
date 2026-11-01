//go:build ignore

// Command gen produces nofma_arm64.s with go-asmgen: the UNFUSED float64
// elementwise kernels (AxpyNoFMA, XpbyNoFMA, Axpby3NoFMA, AxpyDiffNoFMA) and the
// eight-accumulator pairwise-sum block that SumPairwise runs on NEON.
//
// Unfused means one IEEE rounding per binary operation, in the order NumPy
// evaluates the temporaries of the equivalent expression: only VFMUL, VFADD and
// VFSUB (Arm ARM: FMUL, FADD, FSUB (vector)) appear, never VFMLA/VFMLS, and
// VFADDP (FADDP (vector): Vd.D[0] = Vn.D[0]+Vn.D[1], Vd.D[1] = Vm.D[0]+Vm.D[1])
// forms the pairs of the pairwise block.
//
// Operand order: Go's arm64 assembler writes three-operand vector forms as
// `OP Vm, Vn, Vd` meaning Vd = Vn op Vm.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> nofma_arm64_gen.go
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
	f.Add(axpyNoFMA().Func())
	f.Add(xpbyNoFMA().Func())
	f.Add(axpby3NoFMA().Func())
	f.Add(axpyDiffNoFMA().Func())
	f.Add(pwBlock().Func())
	if err := os.WriteFile("nofma_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote nofma_arm64.s")
}

// axpySig is func(<scalar> float64, x, y []float64).
func axpySig(scalar string) abi.Signature {
	return abi.LayoutArgs(
		[]abi.Arg{abi.Scalar(scalar, abi.Float64), abi.Slice("x"), abi.Slice("y")}, nil)
}

// neonElem is the loop skeleton of kernels_arm64_gen.go: input j is loaded into
// V(1+4j).. (four registers per iteration, then one, then one double), compute(k)
// turns register k of input 0 into the result, which is stored to dst (R2). Each
// block is loaded, computed and stored before the next, so dst may alias an input
// exactly. R1 = n, R3 = i.
var neonStages = []struct {
	label, next string
	elems, regs int
	arr         string
}{
	{"uloop", "vloop", 8, 4, "D2"},
	{"vloop", "sloop", 2, 1, "D2"},
	{"sloop", "done", 1, 1, "D1"},
}

func vregs(first, n int, arr string) string {
	s := "["
	for k := 0; k < n; k++ {
		if k > 0 {
			s += ", "
		}
		s += fmt.Sprintf("V%d.%s", first+k, arr)
	}
	return s + "]"
}

func neonElem(b *arm64.Builder, ptrs []string, compute func(k int)) {
	addr := []string{"R5", "R8", "R10"}
	b.Raw("MOVD $0, R3")
	for _, s := range neonStages {
		b.Label(s.label).
			Raw(fmt.Sprintf("ADD $%d, R3, R4", s.elems)).Raw("CMP R1, R4").Raw("BGT " + s.next)
		for j, p := range ptrs {
			b.Raw(fmt.Sprintf("ADD R3<<3, %s, %s", p, addr[j])).
				Raw(fmt.Sprintf("VLD1 (%s), %s", addr[j], vregs(1+4*j, s.regs, s.arr)))
		}
		for k := 0; k < s.regs; k++ {
			compute(k)
		}
		b.Raw("ADD R3<<3, R2, R6").
			Raw(fmt.Sprintf("VST1 %s, (R6)", vregs(1, s.regs, s.arr))).
			Raw(fmt.Sprintf("ADD $%d, R3, R3", s.elems)).Raw("B " + s.label)
	}
	b.Label("done").Ret()
}

func v(n, k int) string { return fmt.Sprintf("V%d.D2", n+k) }

// axpyNoFMA: y[i] = y[i] + alpha*x[i]. Input 0 is y (V1..), input 1 is x (V5..).
func axpyNoFMA() *arm64.Builder {
	b := arm64.NewFunc("axpyNoFMANEON", axpySig("alpha"), 0)
	b.LoadArg("alpha", "F31").Raw("VDUP V31.D[0], V31.D2").
		LoadArg("y_base", "R0").LoadArg("x_len", "R1").LoadArg("x_base", "R7").Raw("MOVD R0, R2")
	neonElem(b, []string{"R0", "R7"}, func(k int) {
		b.Raw("VFMUL V31.D2, " + v(5, k) + ", " + v(5, k)). // t = alpha*x
									Raw("VFADD " + v(5, k) + ", " + v(1, k) + ", " + v(1, k)) // y = y + t
	})
	return b
}

// xpbyNoFMA: y[i] = x[i] + beta*y[i]. Input 0 is y (V1..), input 1 is x (V5..).
func xpbyNoFMA() *arm64.Builder {
	b := arm64.NewFunc("xpbyNoFMANEON", axpySig("beta"), 0)
	b.LoadArg("beta", "F31").Raw("VDUP V31.D[0], V31.D2").
		LoadArg("y_base", "R0").LoadArg("x_len", "R1").LoadArg("x_base", "R7").Raw("MOVD R0, R2")
	neonElem(b, []string{"R0", "R7"}, func(k int) {
		b.Raw("VFMUL V31.D2, " + v(1, k) + ", " + v(1, k)). // t = beta*y
									Raw("VFADD " + v(5, k) + ", " + v(1, k) + ", " + v(1, k)) // r = t + x
	})
	return b
}

// axpby3NoFMA: dst[i] = (alpha*a[i] + beta*b[i]) + gamma*c[i].
func axpby3NoFMA() *arm64.Builder {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Scalar("alpha", abi.Float64), abi.Slice("a"),
			abi.Scalar("beta", abi.Float64), abi.Slice("b"),
			abi.Scalar("gamma", abi.Float64), abi.Slice("c")}, nil)
	b := arm64.NewFunc("axpby3NoFMANEON", sig, 0)
	b.LoadArg("alpha", "F29").Raw("VDUP V29.D[0], V29.D2").
		LoadArg("beta", "F30").Raw("VDUP V30.D[0], V30.D2").
		LoadArg("gamma", "F31").Raw("VDUP V31.D[0], V31.D2").
		LoadArg("dst_base", "R2").LoadArg("a_base", "R0").LoadArg("a_len", "R1").
		LoadArg("b_base", "R7").LoadArg("c_base", "R12")
	neonElem(b, []string{"R0", "R7", "R12"}, func(k int) {
		b.Raw("VFMUL V29.D2, " + v(1, k) + ", " + v(1, k)). // r = alpha*a
									Raw("VFMUL V30.D2, " + v(5, k) + ", " + v(5, k)).          // t = beta*b
									Raw("VFADD " + v(5, k) + ", " + v(1, k) + ", " + v(1, k)). // r = r + t
									Raw("VFMUL V31.D2, " + v(9, k) + ", " + v(9, k)).          // t = gamma*c
									Raw("VFADD " + v(9, k) + ", " + v(1, k) + ", " + v(1, k))  // r = r + t
	})
	return b
}

// axpyDiffNoFMA: dst[i] = a[i] + f*(b[i] - c[i]). Input 0 is b (V1..), input 1
// is c (V5..), input 2 is a (V9..).
func axpyDiffNoFMA() *arm64.Builder {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Scalar("f", abi.Float64),
			abi.Slice("b"), abi.Slice("c")}, nil)
	b := arm64.NewFunc("axpyDiffNoFMANEON", sig, 0)
	b.LoadArg("f", "F31").Raw("VDUP V31.D[0], V31.D2").
		LoadArg("dst_base", "R2").LoadArg("b_base", "R0").LoadArg("a_len", "R1").
		LoadArg("c_base", "R7").LoadArg("a_base", "R12")
	neonElem(b, []string{"R0", "R7", "R12"}, func(k int) {
		b.Raw("VFSUB " + v(5, k) + ", " + v(1, k) + ", " + v(1, k)). // r = b - c
										Raw("VFMUL V31.D2, " + v(1, k) + ", " + v(1, k)).         // r = f*r
										Raw("VFADD " + v(9, k) + ", " + v(1, k) + ", " + v(1, k)) // r = r + a
	})
	return b
}

// pwBlock is NumPy's DOUBLE_pairwise_sum for 8 <= n <= 128 (the caller
// guarantees both bounds): the eight accumulators r[0..7] live in four D2
// registers, V0 = (r0,r1), V1 = (r2,r3), V2 = (r4,r5), V3 = (r6,r7), one lane
// each, so every lane sees exactly the additions NumPy's scalar code performs on
// it. Then
//
//	res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
//
// with VFADDP forming r0+r1, r2+r3, r4+r5, r6+r7 in lane 0, and the n % 8
// remaining elements are added one at a time to res (F8).
func pwBlock() *arm64.Builder {
	sig := abi.LayoutArgs(
		[]abi.Arg{abi.Slice("a")},
		[]abi.Arg{abi.Scalar("ret", abi.Float64)})
	b := arm64.NewFunc("pwBlockNEON", sig, 0)
	b.LoadArg("a_base", "R0").LoadArg("a_len", "R1").
		Raw("VLD1 (R0), [V0.D2, V1.D2, V2.D2, V3.D2]").
		Raw("AND $-8, R1, R4"). // R4 = n - n%8
		Raw("MOVD $8, R3").
		Label("loop").
		Raw("CMP R4, R3").Raw("BGE reduce").
		Raw("ADD R3<<3, R0, R5").
		Raw("VLD1 (R5), [V4.D2, V5.D2, V6.D2, V7.D2]").
		Raw("VFADD V4.D2, V0.D2, V0.D2").
		Raw("VFADD V5.D2, V1.D2, V1.D2").
		Raw("VFADD V6.D2, V2.D2, V2.D2").
		Raw("VFADD V7.D2, V3.D2, V3.D2").
		Raw("ADD $8, R3, R3").Raw("B loop").
		Label("reduce").
		Raw("VFADDP V0.D2, V0.D2, V8.D2").  // lane 0: r0+r1
		Raw("VFADDP V1.D2, V1.D2, V9.D2").  // lane 0: r2+r3
		Raw("VFADDP V2.D2, V2.D2, V10.D2"). // lane 0: r4+r5
		Raw("VFADDP V3.D2, V3.D2, V11.D2"). // lane 0: r6+r7
		Raw("VFADD V9.D2, V8.D2, V8.D2").   // (r0+r1)+(r2+r3)
		Raw("VFADD V11.D2, V10.D2, V10.D2").
		Raw("VFADD V10.D2, V8.D2, V8.D2"). // res
		Label("tail").
		Raw("CMP R1, R3").Raw("BGE done").
		Raw("FMOVD (R0)(R3<<3), F9").
		Raw("FADDD F9, F8, F8").
		Raw("ADD $1, R3, R3").Raw("B tail").
		Label("done").StoreRet("F8", "ret").Ret()
	return b
}
