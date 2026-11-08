//go:build ignore

// Command gen produces polyval_arm64.s with go-asmgen: Horner (Polyval) and
// Clenshaw (Chebval) evaluation of ONE series at many points, NEON.
//
// The lanes run over different points; each point sees, in its own lane, the
// scalar operations NumPy's numpy.polynomial 2.5.4 performs, in the same order,
// each rounded on its own (VFMUL, VFADD, VFSUB only: Arm ARM FMUL, FADD, FSUB
// (vector); never VFMLA/VFMLS):
//
//	polyval (polynomial/polynomial.py):
//	    c0 = c[-1] + x*0
//	    for i in range(2, len(c)+1): c0 = c[-i] + c0*x
//	chebval (polynomial/chebyshev.py), len(c) >= 2:
//	    x2 = 2*x; c0 = c[-2]; c1 = c[-1]
//	    for i in range(3, len(c)+1): tmp = c0; c0 = c[-i] - c1; c1 = tmp + c1*x2
//	    return c0 + c1*x
//
// Operand order: Go's arm64 assembler writes `OP Vm, Vn, Vd` meaning Vd = Vn op Vm.
// 2*x is x+x (exact). A block is loaded, evaluated and stored before the next, so
// dst may alias x exactly; c is only read. The last double is processed in lane 0
// of a D2 register (loaded and stored as D1); lane 1 is zero and discarded.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> polyval_arm64_gen.go
package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/arm64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("arm64")
	f.Add(polyval().Func())
	f.Add(chebval().Func())
	if err := os.WriteFile("polyval_arm64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote polyval_arm64.s")
}

type stage struct {
	label, next string
	elems, regs int
	arr         string
}

var stages = []stage{
	{"big", "one", 8, 4, "D2"},
	{"one", "sca", 2, 1, "D2"},
	{"sca", "done", 1, 1, "D1"},
}

func vr(n int) string { return fmt.Sprintf("V%d.D2", n) }

func list(first, n int, arr string) string {
	var p []string
	for k := 0; k < n; k++ {
		p = append(p, fmt.Sprintf("V%d.%s", first+k, arr))
	}
	return "[" + strings.Join(p, ", ") + "]"
}

func newFn(name string) *arm64.Builder {
	sig := abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("x"), abi.Slice("c")}, nil)
	b := arm64.NewFunc(name, sig, 0)
	// R0 dst, R2 x, R1 n, R5 c, R6 len(c), R3 i, R4 and R9 addresses, R7 coefficient index.
	b.LoadArg("dst_base", "R0").LoadArg("x_base", "R2").LoadArg("x_len", "R1").
		LoadArg("c_base", "R5").LoadArg("c_len", "R6")
	return b
}

// bcast: V31 = c[R7] in both lanes.
func bcast(b *arm64.Builder) {
	b.Raw("ADD R7<<3, R5, R8").Raw("VLD1R (R8), [V31.D2]")
}

func loop(b *arm64.Builder, body func(s stage)) {
	b.Raw("MOVD $0, R3").Raw("VEOR V0.B16, V0.B16, V0.B16")
	for _, s := range stages {
		b.Label(s.label).Raw("ADD $%d, R3, R4", s.elems).Raw("CMP R1, R4").Raw("BGT %s", s.next)
		b.Raw("ADD R3<<3, R2, R4").Raw("VLD1 (R4), %s", list(1, s.regs, s.arr))
		body(s)
		b.Raw("ADD R3<<3, R0, R9").Raw("VST1 %s, (R9)", list(resultReg, s.regs, s.arr)).
			Raw("ADD $%d, R3, R3", s.elems).Raw("B %s", s.label)
	}
	b.Label("done").Ret()
}

var resultReg int

func polyval() *arm64.Builder {
	b := newFn("polyvalNEON")
	resultReg = 5
	loop(b, func(s stage) {
		// x V1.., acc V5.., zero V0, broadcast V31.
		b.Raw("SUB $1, R6, R7")
		bcast(b)
		for k := 0; k < s.regs; k++ { // acc = c[-1] + x*0
			b.Raw("VFMUL %s, %s, %s", vr(1+k), vr(0), vr(5+k)).
				Raw("VFADD %s, %s, %s", vr(5+k), vr(31), vr(5+k))
		}
		l := s.label + "pl"
		b.Label(l).Raw("SUB $1, R7").Raw("CMP $0, R7").Raw("BLT %sd", l)
		bcast(b)
		for k := 0; k < s.regs; k++ { // acc = c[i] + acc*x
			b.Raw("VFMUL %s, %s, %s", vr(1+k), vr(5+k), vr(5+k))
		}
		for k := 0; k < s.regs; k++ {
			b.Raw("VFADD %s, %s, %s", vr(5+k), vr(31), vr(5+k))
		}
		b.Raw("B %s", l).Label(l + "d")
	})
	return b
}

// chebval requires len(c) >= 2. x V1+k, x2 V5+k, c0 V9+k, c1 V13+k, t V17+k.
func chebval() *arm64.Builder {
	b := newFn("chebvalNEON")
	resultReg = 17
	loop(b, func(s stage) {
		for k := 0; k < s.regs; k++ {
			b.Raw("VFADD %s, %s, %s", vr(1+k), vr(1+k), vr(5+k)) // x2 = x + x
		}
		b.Raw("SUB $2, R6, R7")
		bcast(b)
		for k := 0; k < s.regs; k++ {
			b.Raw("VMOV V31.B16, V%d.B16", 9+k) // c0 = c[n-2]
		}
		b.Raw("ADD $1, R7")
		bcast(b)
		for k := 0; k < s.regs; k++ {
			b.Raw("VMOV V31.B16, V%d.B16", 13+k) // c1 = c[n-1]
		}
		b.Raw("SUB $2, R7")
		l := s.label + "cl"
		b.Label(l).Raw("CMP $0, R7").Raw("BLT %sd", l)
		bcast(b)
		for k := 0; k < s.regs; k++ { // t = c1*x2
			b.Raw("VFMUL %s, %s, %s", vr(5+k), vr(13+k), vr(17+k))
		}
		for k := 0; k < s.regs; k++ { // t = c0 + t
			b.Raw("VFADD %s, %s, %s", vr(17+k), vr(9+k), vr(17+k))
		}
		for k := 0; k < s.regs; k++ { // c0 = c[i] - c1
			b.Raw("VFSUB %s, %s, %s", vr(13+k), vr(31), vr(9+k))
		}
		for k := 0; k < s.regs; k++ { // c1 = t
			b.Raw("VMOV V%d.B16, V%d.B16", 17+k, 13+k)
		}
		b.Raw("SUB $1, R7").Raw("B %s", l).Label(l + "d")
		for k := 0; k < s.regs; k++ { // t = c1*x ; t = c0 + t
			b.Raw("VFMUL %s, %s, %s", vr(1+k), vr(13+k), vr(17+k)).
				Raw("VFADD %s, %s, %s", vr(17+k), vr(9+k), vr(17+k))
		}
	})
	return b
}
