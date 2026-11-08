//go:build ignore

// Command gen produces polyval_amd64.s with go-asmgen: Horner (Polyval) and
// Clenshaw (Chebval) evaluation of ONE series at many points, AVX2 and SSE2.
//
// The lanes run over DIFFERENT points; each point sees, in its own lane, the
// scalar operations NumPy's numpy.polynomial 2.5.4 performs, in the same order
// and each rounded on its own (no VFMADD, no fused instruction anywhere):
//
//	polyval (polynomial/polynomial.py, polyval):
//	    c0 = c[-1] + x*0
//	    for i in range(2, len(c)+1): c0 = c[-i] + c0*x
//	chebval (polynomial/chebyshev.py, chebval), len(c) >= 2:
//	    x2 = 2*x; c0 = c[-2]; c1 = c[-1]
//	    for i in range(3, len(c)+1): tmp = c0; c0 = c[-i] - c1; c1 = tmp + c1*x2
//	    return c0 + c1*x
//
// Instructions (Intel SDM vol. 2): VMULPD/VADDPD/VSUBPD and VMULSD/VADDSD/VSUBSD
// (VEX), MULPD/ADDPD/SUBPD and MULSD/ADDSD/SUBSD (SSE2), VBROADCASTSD, UNPCKLPD,
// MOVUPD/MOVAPD/MOVSD, VXORPD/XORPD. 2*x is x+x (exact, identical to the product).
// The SSE2 forms are destructive two-operand, so a sum c + t is formed as t + c;
// addition is commutative bit for bit (only the payload of a NaN that appears in
// BOTH operands could differ, which no consumer reads).
//
// Each block of points is loaded, evaluated and stored before the next, so dst
// may alias x exactly. The coefficient slice c is read, never written.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.0> polyval_amd64_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/amd64"
	"github.com/go-asmgen/asmgen/emit"
)

func main() {
	f := emit.NewFile("amd64")
	f.Add(polyval(true).Func())
	f.Add(chebval(true).Func())
	f.Add(polyval(false).Func())
	f.Add(chebval(false).Func())
	if err := os.WriteFile("polyval_amd64.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote polyval_amd64.s")
}

type stage struct {
	label, next string
	elems, regs int
	scalar      bool
}

// stages: a block of 4 or 2 registers, one register, one double. per = doubles
// per register (4 for YMM, 2 for XMM); big = registers in the first block.
func stages(per, big int) []stage {
	return []stage{
		{"big", "one", per * big, big, false},
		{"one", "sca", per, 1, false},
		{"sca", "done", 1, 1, true},
	}
}

func loop(b *amd64.Builder, ss []stage, body func(s stage)) {
	b.Raw("XORQ AX, AX")
	for _, s := range ss {
		b.Label(s.label).Raw("LEAQ %d(AX), R8", s.elems).Raw("CMPQ R8, CX").Raw("JGT %s", s.next)
		body(s)
		b.Raw("ADDQ $%d, AX", s.elems).Raw("JMP %s", s.label)
	}
	b.Label("done")
}

func xm(avx bool, scalar bool, n int) string {
	if avx && !scalar {
		return fmt.Sprintf("Y%d", n)
	}
	return fmt.Sprintf("X%d", n)
}

func xmem(k, per int) string { // x[AX + k*per]
	if k == 0 {
		return "(SI)(AX*8)"
	}
	return fmt.Sprintf("%d(SI)(AX*8)", 8*per*k)
}

func dmem(k, per int) string {
	if k == 0 {
		return "(DI)(AX*8)"
	}
	return fmt.Sprintf("%d(DI)(AX*8)", 8*per*k)
}

func sig() abi.Signature {
	return abi.LayoutArgs([]abi.Arg{abi.Slice("dst"), abi.Slice("x"), abi.Slice("c")}, nil)
}

func newFn(name string) *amd64.Builder {
	b := amd64.NewFunc(name, sig(), 0)
	b.LoadArg("dst_base", "DI").LoadArg("x_base", "SI").LoadArg("x_len", "CX").
		LoadArg("c_base", "R10").LoadArg("c_len", "R11")
	return b
}

// bcast loads c[R9] into the broadcast register 15.
func bcast(b *amd64.Builder, avx, scalar bool, dst int) {
	switch {
	case avx && !scalar:
		b.Raw("VBROADCASTSD (R10)(R9*8), Y%d", dst)
	case avx:
		b.Raw("VMOVSD (R10)(R9*8), X%d", dst)
	case !scalar:
		b.Raw("MOVSD (R10)(R9*8), X%d", dst).Raw("UNPCKLPD X%d, X%d", dst, dst)
	default:
		b.Raw("MOVSD (R10)(R9*8), X%d", dst)
	}
}

func load(b *amd64.Builder, avx, scalar bool, mem string, r int) {
	switch {
	case scalar && avx:
		b.Raw("VMOVSD %s, X%d", mem, r)
	case scalar:
		b.Raw("MOVSD %s, X%d", mem, r)
	case avx:
		b.Raw("VMOVUPD %s, Y%d", mem, r)
	default:
		b.Raw("MOVUPD %s, X%d", mem, r)
	}
}

func store(b *amd64.Builder, avx, scalar bool, r int, mem string) {
	switch {
	case scalar && avx:
		b.Raw("VMOVSD X%d, %s", r, mem)
	case scalar:
		b.Raw("MOVSD X%d, %s", r, mem)
	case avx:
		b.Raw("VMOVUPD Y%d, %s", r, mem)
	default:
		b.Raw("MOVUPD X%d, %s", r, mem)
	}
}

// op emits dst = a OP b for the family (mul/add/sub) on the right width. AVX is
// three-operand (dst = a op b); SSE2 is destructive (dst must be a).
func op(b *amd64.Builder, avx, scalar bool, name string, a, bb, dst int) {
	suffix := "PD"
	if scalar {
		suffix = "SD"
	}
	if avx {
		b.Raw("V%s%s %s, %s, %s", name, suffix, xm(avx, scalar, bb), xm(avx, scalar, a), xm(avx, scalar, dst))
		return
	}
	if a != dst {
		panic("sse2 op: a != dst")
	}
	b.Raw("%s%s %s, %s", name, suffix, xm(avx, scalar, bb), xm(avx, scalar, dst))
}

func mov(b *amd64.Builder, avx, scalar bool, src, dst int) {
	if avx {
		b.Raw("VMOVAPD %s, %s", xm(avx, scalar, src), xm(avx, scalar, dst))
		return
	}
	b.Raw("MOVAPD X%d, X%d", src, dst)
}

func polyval(avx bool) *amd64.Builder {
	name, big := "polyvalSSE2", 4
	if avx {
		name = "polyvalAVX2"
	}
	b := newFn(name)
	if avx {
		b.Raw("VXORPD Y0, Y0, Y0")
	} else {
		b.Raw("XORPD X0, X0")
	}
	loop(b, stages(map[bool]int{true: 4, false: 2}[avx], big), func(s stage) {
		sc := s.scalar
		b.Raw("MOVQ R11, R9").Raw("SUBQ $1, R9")
		bcast(b, avx, sc, 15)
		for k := 0; k < s.regs; k++ {
			load(b, avx, sc, xmem(k, map[bool]int{true: 4, false: 2}[avx]), 1+k)
		}
		// acc = x*0, then acc = c[-1] + acc.
		for k := 0; k < s.regs; k++ {
			if avx {
				op(b, avx, sc, "MUL", 0, 1+k, 5+k)
				op(b, avx, sc, "ADD", 15, 5+k, 5+k)
			} else {
				mov(b, avx, sc, 1+k, 5+k)
				op(b, avx, sc, "MUL", 5+k, 0, 5+k)
				op(b, avx, sc, "ADD", 5+k, 15, 5+k)
			}
		}
		l := s.label + "pl"
		b.Label(l).Raw("SUBQ $1, R9").Raw("JLT %sd", l)
		bcast(b, avx, sc, 15)
		for k := 0; k < s.regs; k++ {
			if avx {
				op(b, avx, sc, "MUL", 5+k, 1+k, 5+k)
			} else {
				op(b, avx, sc, "MUL", 5+k, 1+k, 5+k)
			}
		}
		for k := 0; k < s.regs; k++ {
			if avx {
				op(b, avx, sc, "ADD", 15, 5+k, 5+k)
			} else {
				op(b, avx, sc, "ADD", 5+k, 15, 5+k)
			}
		}
		b.Raw("JMP %s", l).Label(l + "d")
		for k := 0; k < s.regs; k++ {
			store(b, avx, sc, 5+k, dmem(k, map[bool]int{true: 4, false: 2}[avx]))
		}
	})
	if avx {
		b.Raw("VZEROUPPER")
	}
	b.Ret()
	return b
}

// chebval requires len(c) >= 2 (the caller handles len(c) == 1). Registers:
// x 1+k, x2 3+k, c0 5+k, c1 7+k, t 9+k, broadcast 15; k < 2.
func chebval(avx bool) *amd64.Builder {
	name, per := "chebvalSSE2", 2
	if avx {
		name, per = "chebvalAVX2", 4
	}
	b := newFn(name)
	loop(b, stages(per, 2), func(s stage) {
		sc := s.scalar
		for k := 0; k < s.regs; k++ {
			load(b, avx, sc, xmem(k, per), 1+k)
		}
		for k := 0; k < s.regs; k++ { // x2 = x + x
			if avx {
				op(b, avx, sc, "ADD", 1+k, 1+k, 3+k)
			} else {
				mov(b, avx, sc, 1+k, 3+k)
				op(b, avx, sc, "ADD", 3+k, 1+k, 3+k)
			}
		}
		b.Raw("MOVQ R11, R9").Raw("SUBQ $2, R9")
		bcast(b, avx, sc, 15)
		for k := 0; k < s.regs; k++ { // c0 = c[n-2]
			mov(b, avx, sc, 15, 5+k)
		}
		b.Raw("ADDQ $1, R9")
		bcast(b, avx, sc, 15) // c1 = c[n-1]
		for k := 0; k < s.regs; k++ {
			mov(b, avx, sc, 15, 7+k)
		}
		b.Raw("SUBQ $2, R9") // first coefficient below c[n-2] is c[n-3]
		l := s.label + "cl"
		b.Label(l).Raw("CMPQ R9, $0").Raw("JLT %sd", l)
		bcast(b, avx, sc, 15)
		for k := 0; k < s.regs; k++ { // t = c1*x2
			if avx {
				op(b, avx, sc, "MUL", 7+k, 3+k, 9+k)
			} else {
				mov(b, avx, sc, 7+k, 9+k)
				op(b, avx, sc, "MUL", 9+k, 3+k, 9+k)
			}
		}
		for k := 0; k < s.regs; k++ { // t = c0 + t
			if avx {
				op(b, avx, sc, "ADD", 5+k, 9+k, 9+k)
			} else {
				op(b, avx, sc, "ADD", 9+k, 5+k, 9+k)
			}
		}
		for k := 0; k < s.regs; k++ { // c0 = c[i] - c1
			if avx {
				op(b, avx, sc, "SUB", 15, 7+k, 5+k)
			} else {
				mov(b, avx, sc, 15, 5+k)
				op(b, avx, sc, "SUB", 5+k, 7+k, 5+k)
			}
		}
		for k := 0; k < s.regs; k++ { // c1 = t
			mov(b, avx, sc, 9+k, 7+k)
		}
		b.Raw("SUBQ $1, R9").Raw("JMP %s", l).Label(l + "d")
		for k := 0; k < s.regs; k++ { // t = c1*x ; t = c0 + t
			if avx {
				op(b, avx, sc, "MUL", 7+k, 1+k, 9+k)
				op(b, avx, sc, "ADD", 5+k, 9+k, 9+k)
			} else {
				mov(b, avx, sc, 7+k, 9+k)
				op(b, avx, sc, "MUL", 9+k, 1+k, 9+k)
				op(b, avx, sc, "ADD", 9+k, 5+k, 9+k)
			}
		}
		for k := 0; k < s.regs; k++ {
			store(b, avx, sc, 9+k, dmem(k, per))
		}
	})
	if avx {
		b.Raw("VZEROUPPER")
	}
	b.Ret()
	return b
}
