//go:build ignore

// Command gen produces binop_arm64.s with go-asmgen: the float64 elementwise
// Add, Sub, Div, Sqrt and ClampScalar kernels (K1 of the go-scikit performance
// campaign: they were private to go-ndarray's internal/kernels).
//
// VFADD / VFSUB / VFDIV / VFSQRT (Arm ARM: FADD, FSUB, FDIV, FSQRT (vector)) are
// one IEEE rounding per lane, so every result is bit-identical to the scalar Go
// loop; the Go 1.27 assembler names them. ClampScalar uses the strict compares of
// clampF64 in kernels_arm64_gen.go (FCMGT is false when unordered and on equal
// zeros, VBIT selects), not FMAX/FMIN, which treat NaN bounds and signed zeros
// differently.
//
// Operand order: Go's arm64 assembler writes three-operand vector forms as
// `OP Vm, Vn, Vd` meaning Vd = Vn op Vm.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> binop_arm64_gen.go
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
	f.Add(binop("addNEON", "VFADD").Func())
	f.Add(binop("subNEON", "VFSUB").Func())
	f.Add(binop("divNEON", "VFDIV").Func())
	f.Add(sqrt().Func())
	f.Add(clampScalar().Func())
	if err := os.WriteFile("binop_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote binop_arm64.s")
}

// neonStages and neonElem are the loop skeleton of kernels_arm64_gen.go: input j
// is loaded into V(1+4j).. (four registers per iteration, then one, then one
// double), compute(k) turns register k of input 0 into the result, which is
// stored to dst (R2). R1 = n, R3 = i. Each block is loaded, computed and stored
// before the next, so dst may alias an input exactly.
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

// binop: dst[i] = a[i] OP b[i]; func(dst, a, b []float64). Input 0 is a (V1..),
// input 1 is b (V5..); Vd = Vn op Vm, so `OP Vb, Va, Va` is a OP b.
func binop(name, vop string) *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Slice("b")}, nil)
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("dst_base", "R2").LoadArg("a_base", "R0").LoadArg("a_len", "R1").LoadArg("b_base", "R7")
	neonElem(b, []string{"R0", "R7"}, func(k int) {
		b.Raw(vop + " " + v(5, k) + ", " + v(1, k) + ", " + v(1, k))
	})
	return b
}

// sqrt: dst[i] = sqrt(src[i]); func(dst, src []float64).
func sqrt() *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("src")}, nil)
	b := arm64.NewFunc("sqrtNEON", sig, 0)
	b.LoadArg("dst_base", "R2").LoadArg("src_base", "R0").LoadArg("src_len", "R1")
	neonElem(b, []string{"R0"}, func(k int) {
		b.Raw("VFSQRT " + v(1, k) + ", " + v(1, k))
	})
	return b
}

// clampScalar: r = v; r = lo where lo > v; r = hi where r > hi, with scalar
// bounds broadcast in V29 / V30; func(dst, v []float64, lo, hi float64).
// `VFCMGT Vm, Vn, Vd` is Vd = Vn > Vm (all ones where true); `VBIT Vm, Vn, Vd`
// copies Vn's bits into Vd where Vm is set.
func clampScalar() *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("v"),
		abi.Scalar("lo", abi.Float64), abi.Scalar("hi", abi.Float64)}, nil)
	b := arm64.NewFunc("clampScalarNEON", sig, 0)
	b.LoadArg("lo", "F29").Raw("VDUP V29.D[0], V29.D2").
		LoadArg("hi", "F30").Raw("VDUP V30.D[0], V30.D2").
		LoadArg("dst_base", "R2").LoadArg("v_base", "R0").LoadArg("v_len", "R1")
	neonElem(b, []string{"R0"}, func(k int) {
		x, m := 1+k, 16+k
		b.Raw(fmt.Sprintf("VFCMGT V%d.D2, V29.D2, V%d.D2", x, m)). // m = lo > v
										Raw(fmt.Sprintf("VBIT V%d.B16, V29.B16, V%d.B16", m, x)). // v = lo where m
										Raw(fmt.Sprintf("VFCMGT V30.D2, V%d.D2, V%d.D2", x, m)).  // m = v > hi
										Raw(fmt.Sprintf("VBIT V%d.B16, V30.B16, V%d.B16", m, x))  // v = hi where m
	})
	return b
}
