//go:build ignore

// Command gen produces kernels_ppc64le.s with go-asmgen: VSX FMA reduction
// kernels for the dot product, sum and sum-of-squared-differences, in float32
// (dot32VSX, sum32VSX, sumSqDiff32VSX) and float64 (dot64VSX, sum64VSX,
// sumSqDiff64VSX).
//
// Eight independent accumulators. An xvmaddasp / xvmaddadp / xvaddsp / xvadddp
// has a latency of 6-7 cycles on POWER9 and the core issues two per cycle, so a
// single loop-carried accumulator keeps one chain in flight and runs at about a
// seventh of the issue width (measured on cfarm433, POWER9, single core: the old
// one-accumulator f32 Dot ran at 13.6 GB/s and the Go loop at 20 GB/s for f64
// n=1e6). The bulk loop therefore keeps eight accumulators (VS32..VS39), loads
// into VS40..VS47 (a) and VS48..VS55 (b), then folds them with a balanced add
// tree into VS32. A second loop at one register stride handles what is left of
// the multiple of 8 registers, a scalar loop the tail. The summation order is
// unspecified: the contract is the error bound gamma_n * sum|terms| (see
// reduce_bound_test.go).
//
// The kernels use the upper vector registers VS32..VS63 (the Altivec VRs), so
// no FPR is clobbered except the scalar tail's F0..F3. The released Go assembler
// exposes the VSX loads (LXVW4X, LXVD2X) and stores but NOT the XV* vector-FP
// arithmetic, so these are emitted as their fixed Power ISA encodings via WORD:
// XX3-form, opcode 60, T at bit 21, A at 16, B at 11, XO at bit 3 and the
// TX/AX/BX bits (1 for VS32..VS63). XO values (Power ISA v3.0B, section 7.6.2):
// xvaddsp 64, xvsubsp 72, xvmaddasp 65, xvadddp 96, xvsubdp 104, xvmaddadp 97.
// The encoder is checked against the words GNU as produced for the previous
// kernels, e.g. xvmaddasp vs2,vs0,vs1 = 0xf0400a08.
//
// Lane order: LXVD2X / LXVW4X on little-endian swap element order, but every
// kernel is an element-wise reduction with a commutative final horizontal sum, so
// lane numbering does not change the result. VSX is the POWER8 baseline.
//
// Scalar tail operand orders (verified on qemu): the Plan 9 ppc64 FMA is
// `FMADD/FMADDS A, B, C, T` -> T = A*C + B; `FSUB X, Y` -> Y = Y - X.
//
// Run: go run -modfile=<modfile requiring github.com/go-asmgen/asmgen v0.23.2> kernels_ppc64le_gen.go
package main

import (
	"fmt"
	"os"

	"github.com/go-asmgen/asmgen/abi"
	"github.com/go-asmgen/asmgen/emit"
	"github.com/go-asmgen/asmgen/ppc64"
)

const redK = 8

type kind int

const (
	kDot kind = iota
	kSum
	kSSD
)

func main() {
	f := emit.NewFile("ppc64le")
	for _, w := range []bool{true, false} { // f32 first
		for _, k := range []kind{kDot, kSum, kSSD} {
			f.Add(kernel(k, w).Func())
		}
	}
	if err := os.WriteFile("kernels_ppc64le.s", []byte(f.String()), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println("wrote kernels_ppc64le.s")
}

// xv encodes an XX3-form VSX instruction on VS registers t, a, b (0..63).
func xv(xo, t, a, b int) string {
	w := 0xF0000000 | (t&31)<<21 | (a&31)<<16 | (b&31)<<11 | xo<<3 |
		((a>>5)&1)<<2 | ((b>>5)&1)<<1 | (t>>5)&1
	return fmt.Sprintf("WORD $0x%08x", w)
}

func kernel(k kind, f32 bool) *ppc64.Builder {
	name := map[kind]string{kDot: "dot", kSum: "sum", kSSD: "sumSqDiff"}[k]
	lanes, shift, esz := 2, 4, 8 // f64: 2 lanes, 8 registers = 16 elements
	ld, st, ldF, st1 := "LXVD2X", "STXVD2X", "FMOVD", "64"
	add, sub, madd := 96, 104, 97
	scalarAdd, scalarFMA, scalarSub := "FADD", "FMADD", "FSUB"
	rt := abi.Float64
	if f32 {
		lanes, shift, esz = 4, 5, 4
		ld, st, ldF, st1 = "LXVW4X", "STXVW4X", "FMOVS", "32"
		add, sub, madd = 64, 72, 65
		scalarAdd, scalarFMA, scalarSub = "FADDS", "FMADDS", "FSUBS"
		rt = abi.Float32
	}
	_ = st1
	suffix := map[bool]string{true: "32", false: "64"}[f32]
	args := []abi.Arg{abi.Slice("a")}
	if k != kSum {
		args = append(args, abi.Slice("b"))
	}
	sig := abi.LayoutArgs(args, []abi.Arg{abi.Scalar("ret", rt)})
	b := ppc64.NewFunc(name+suffix+"VSX", sig, 48)
	b.LoadArg("a_base", "R3").LoadArg("a_len", "R4")
	if k != kSum {
		b.LoadArg("b_base", "R5")
	}
	const acc, ta, tb = 32, 40, 48
	two := k != kSum
	// off(k) is the register holding the byte offset 16*k (R0 reads as zero).
	off := func(kk int) string {
		if kk == 0 {
			return "R0"
		}
		return fmt.Sprintf("R%d", 13+kk) // R14..R20 = 16..112
	}
	// step emits the load/compute of one register k into accumulator ac.
	step := func(kk, ac int) {
		b.Raw("%s (R3)(%s), VS%d", ld, off(kk), ta+kk)
		if two {
			b.Raw("%s (R5)(%s), VS%d", ld, off(kk), tb+kk)
		}
		switch k {
		case kDot:
			b.Raw("%s", xv(madd, ac, ta+kk, tb+kk))
		case kSum:
			b.Raw("%s", xv(add, ac, ta+kk, ac))
		case kSSD:
			b.Raw("%s", xv(sub, ta+kk, ta+kk, tb+kk))
			b.Raw("%s", xv(madd, ac, ta+kk, ta+kk))
		}
	}
	b.Raw("XXLXOR VS%d, VS%d, VS%d", acc, acc, acc).
		Raw("MOVD $0, R6").
		Raw("SRD $%d, R4, R7", shift).
		Raw("CMP R7, $0").Raw("BEQ vloop")
	for kk := 1; kk < redK; kk++ {
		b.Raw("XXLXOR VS%d, VS%d, VS%d", acc+kk, acc+kk, acc+kk)
		b.Raw("MOVD $%d, R%d", 16*kk, 13+kk)
	}
	b.Label("cloop")
	// Loads first, then the arithmetic, so the loads of the whole block are in flight.
	for kk := 0; kk < redK; kk++ {
		b.Raw("%s (R3)(%s), VS%d", ld, off(kk), ta+kk)
		if two {
			b.Raw("%s (R5)(%s), VS%d", ld, off(kk), tb+kk)
		}
	}
	for kk := 0; kk < redK; kk++ {
		switch k {
		case kDot:
			b.Raw("%s", xv(madd, acc+kk, ta+kk, tb+kk))
		case kSum:
			b.Raw("%s", xv(add, acc+kk, ta+kk, acc+kk))
		case kSSD:
			b.Raw("%s", xv(sub, ta+kk, ta+kk, tb+kk))
			b.Raw("%s", xv(madd, acc+kk, ta+kk, ta+kk))
		}
	}
	b.Raw("ADD $128, R3")
	if two {
		b.Raw("ADD $128, R5")
	}
	b.Raw("ADD $-1, R7").Raw("CMP R7, $0").Raw("BNE cloop").
		Raw("SRD $%d, R4, R6", shift).Raw("SLD $%d, R6, R6", shift)
	// Balanced add tree into VS32.
	for _, p := range [][2]int{{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {4, 6}, {0, 4}} {
		b.Raw("%s", xv(add, acc+p[0], acc+p[0], acc+p[1]))
	}
	b.Label("vloop").
		Raw("ADD $%d, R6, R9", lanes).Raw("CMP R9, R4").Raw("BGT vtail")
	step(0, acc)
	b.Raw("ADD $16, R3")
	if two {
		b.Raw("ADD $16, R5")
	}
	b.Raw("ADD $%d, R6, R6", lanes).Raw("BR vloop").
		Label("vtail")
	// Spill the accumulator to the 16-byte scratch and sum its lanes into F0.
	b.Raw("MOVD $32, R11").Raw("ADD R1, R11, R11").
		Raw("%s VS%d, (R11)(R0)", st, acc)
	b.Raw("%s 0(R11), F0", ldF)
	for l := 1; l < lanes; l++ {
		b.Raw("%s %d(R11), F%d", ldF, l*esz, l)
		b.Raw("%s F%d, F0", scalarAdd, l)
	}
	b.Label("sloop").
		Raw("CMP R6, R4").Raw("BGE done").
		Raw("%s (R3), F1", ldF)
	switch k {
	case kDot:
		b.Raw("%s (R5), F2", ldF).Raw("%s F1, F0, F2, F0", scalarFMA)
	case kSum:
		b.Raw("%s F1, F0", scalarAdd)
	case kSSD:
		b.Raw("%s (R5), F2", ldF).Raw("%s F2, F1", scalarSub).Raw("%s F1, F0, F1, F0", scalarFMA)
	}
	b.Raw("ADD $%d, R3", esz)
	if two {
		b.Raw("ADD $%d, R5", esz)
	}
	b.Raw("ADD $1, R6, R6").Raw("BR sloop").
		Label("done").StoreRet("F0", "ret").Ret()
	return b
}
