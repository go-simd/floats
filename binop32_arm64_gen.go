//go:build ignore

// Command gen produces binop32_arm64.s with go-asmgen: the float32 elementwise
// Add32, Sub32, Mul32, Div32 and Sqrt32 kernels (go-ndarray kept its own copies
// privately).
//
// VFADD / VFSUB / VFMUL / VFDIV / VFSQRT on .S4 (Arm ARM: FADD, FSUB, FMUL, FDIV,
// FSQRT (vector), single precision) are one IEEE binary32 rounding per lane, and
// the scalar tail uses FADDS / FSUBS / FMULS / FDIVS / FSQRTS, so every result is
// the correctly rounded one, bit-identical to the scalar Go loop. 16 elements (4
// registers) per iteration, then 4, then one. Each block is loaded, computed and
// stored before the next, so dst may alias an input exactly.
//
// Operand order: Go's arm64 assembler writes three-operand forms as `OP Vm, Vn,
// Vd` meaning Vd = Vn op Vm (and `FSUBS Fm, Fn, Fd` is Fd = Fn - Fm).
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> binop32_arm64_gen.go
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
	for _, op := range []struct{ name, vop, sop string }{
		{"add32", "VFADD", "FADDS"},
		{"sub32", "VFSUB", "FSUBS"},
		{"mul32", "VFMUL", "FMULS"},
		{"div32", "VFDIV", "FDIVS"},
	} {
		f.Add(binop(op.name+"NEON", op.vop, op.sop).Func())
	}
	f.Add(sqrt().Func())
	if err := os.WriteFile("binop32_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote binop32_arm64.s")
}

func vregs(first, n int) string {
	s := "["
	for k := 0; k < n; k++ {
		if k > 0 {
			s += ", "
		}
		s += fmt.Sprintf("V%d.S4", first+k)
	}
	return s + "]"
}

// neonElem is the loop skeleton: input j (pointer ptrs[j]) is loaded into
// V(1+4j).., compute(k) turns register k of input 0 into the result, stored to
// dst (R2); scalar(): the one-element step on F1 (input 0) / F5 (input 1) from
// the addresses R5 / R8. R1 = n, R3 = i.
func neonElem(b *arm64.Builder, ptrs []string, compute func(k int), scalar func()) {
	addr := []string{"R5", "R8"}
	b.Raw("MOVD $0, R3")
	for _, s := range []struct {
		label, next string
		elems, regs int
	}{{"uloop", "vloop", 16, 4}, {"vloop", "sloop", 4, 1}} {
		b.Label(s.label).
			Raw(fmt.Sprintf("ADD $%d, R3, R4", s.elems)).Raw("CMP R1, R4").Raw("BGT " + s.next)
		for j, p := range ptrs {
			b.Raw(fmt.Sprintf("ADD R3<<2, %s, %s", p, addr[j])).
				Raw(fmt.Sprintf("VLD1 (%s), %s", addr[j], vregs(1+4*j, s.regs)))
		}
		for k := 0; k < s.regs; k++ {
			compute(k)
		}
		b.Raw("ADD R3<<2, R2, R6").
			Raw(fmt.Sprintf("VST1 %s, (R6)", vregs(1, s.regs))).
			Raw(fmt.Sprintf("ADD $%d, R3, R3", s.elems)).Raw("B " + s.label)
	}
	b.Label("sloop").Raw("CMP R1, R3").Raw("BGE done")
	for j, p := range ptrs {
		b.Raw(fmt.Sprintf("ADD R3<<2, %s, %s", p, addr[j])).
			Raw(fmt.Sprintf("FMOVS (%s), F%d", addr[j], 1+4*j))
	}
	scalar()
	b.Raw("ADD R3<<2, R2, R6").Raw("FMOVS F1, (R6)").
		Raw("ADD $1, R3, R3").Raw("B sloop").
		Label("done").Ret()
}

func v(n, k int) string { return fmt.Sprintf("V%d.S4", n+k) }

// binop: dst[i] = a[i] OP b[i]; func(dst, a, b []float32). a is V1.. / F1, b is
// V5.. / F5; `OP Vb, Va, Va` is a OP b.
func binop(name, vop, sop string) *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("a"), abi.Slice("b")}, nil)
	b := arm64.NewFunc(name, sig, 0)
	b.LoadArg("dst_base", "R2").LoadArg("a_base", "R0").LoadArg("a_len", "R1").LoadArg("b_base", "R7")
	neonElem(b, []string{"R0", "R7"}, func(k int) {
		b.Raw(vop + " " + v(5, k) + ", " + v(1, k) + ", " + v(1, k))
	}, func() {
		b.Raw(sop + " F5, F1, F1")
	})
	return b
}

// sqrt: dst[i] = sqrt(src[i]); func(dst, src []float32).
func sqrt() *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("src")}, nil)
	b := arm64.NewFunc("sqrt32NEON", sig, 0)
	b.LoadArg("dst_base", "R2").LoadArg("src_base", "R0").LoadArg("src_len", "R1")
	neonElem(b, []string{"R0"}, func(k int) {
		b.Raw("VFSQRT " + v(1, k) + ", " + v(1, k))
	}, func() {
		b.Raw("FSQRTS F1, F1")
	})
	return b
}
